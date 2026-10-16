"""Stage 3: read each merged spine and resolve it to a catalogue record.

The model only transcribes what is on the spine. Resolution to a work/edition/ISBN
is a catalogue lookup (Google Books, then Open Library) scored with fuzzy string
matching in code. Weak reads or weak matches stay "unidentified" - no guessing.

Output: runs/<id>/identified.json  {book_id: {...}}
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import cv2
import httpx
from rapidfuzz import fuzz

from . import http
from .common import LITE_MODEL, VISION_MODEL, Metrics, gemini_json, image_part, memo, write_json

STAGE = "identify"

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "verbatim": {"type": "string", "description": "Exactly the characters visible on the spine"},
        "title": {"type": "string", "description": "Title as printed on the spine, or empty"},
        "author": {"type": "string", "description": "Author as printed on the spine, or empty"},
        "publisher": {"type": "string", "description": "Publisher/imprint as printed, or empty"},
        "legible": {"type": "boolean"},
        "confidence": {"type": "number", "description": "0-1: how sure you are the transcription is exact"},
        "special_marks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Visible marks such as 'signed', 'first edition', 'limited edition', leather, gilt; empty if none",
        },
    },
    "required": ["verbatim", "title", "author", "publisher", "legible", "confidence", "special_marks"],
}

READ_PROMPT = (
    "This is a single book spine photographed on a shelf (it may be rotated). Transcribe exactly what is printed. "
    "Split it into title, author and publisher only using what is printed on this spine. Do not complete partial "
    "words from memory and do not name a book you cannot actually read: if the text is unreadable, return empty "
    "fields, legible=false and confidence 0."
)

# Matching thresholds (0-100). A spine is "identified" only above these.
TITLE_MIN = 88
AUTHOR_MIN = 75
READ_CONF_MIN = 0.5

UA = f"library-claim-agent/0.1 ({os.getenv('CONTACT_EMAIL') or 'contact unset'})"


def norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    s = re.sub(r"\b(the|a|an|novel|a novel)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def crop_spine(run_dir: Path, obs: dict, pad: float = 0.12) -> bytes:
    img = cv2.imread(str(run_dir / obs["file"]))
    x0, y0, x1, y1 = obs["box_px"]
    w, h = x1 - x0, y1 - y0
    xa, ya = max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h))
    xb, yb = min(img.shape[1], int(x1 + pad * w)), min(img.shape[0], int(y1 + pad * h))
    crop = img[ya:yb, xa:xb]
    # Upscale tiny crops so small spine type is readable.
    if min(crop.shape[:2]) < 120:
        f = 120 / max(1, min(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


async def _get(client: httpx.AsyncClient, url: str, params: dict, metrics: Metrics) -> dict | None:
    return await http.get_json(client, url, params, metrics, STAGE)


def _isbn13(ids: list[dict]) -> str:
    for t in ("ISBN_13", "ISBN_10"):
        for i in ids or []:
            if i.get("type") == t:
                return i["identifier"]
    return ""


async def google_books(client, title: str, author: str, metrics: Metrics) -> list[dict]:
    # Plain query: field operators (intitle:/inauthor:) returned 0 results from this API in testing (2026-10-15).
    # Candidates are verified by fuzzy title/author scoring in decide(), not trusted from search ranking.
    q = f"{title} {author}".strip()
    params = {"q": q, "maxResults": 8, "printType": "books"}
    if os.getenv("GOOGLE_BOOKS_API_KEY"):
        params["key"] = os.getenv("GOOGLE_BOOKS_API_KEY")
    data = await _get(client, "https://www.googleapis.com/books/v1/volumes", params, metrics)
    out = []
    for item in (data or {}).get("items", []):
        v = item.get("volumeInfo", {})
        out.append(
            {
                "source": "Google Books",
                "volume_id": item.get("id"),
                "title": v.get("title", "") + (f": {v['subtitle']}" if v.get("subtitle") else ""),
                "authors": v.get("authors", []),
                "publisher": v.get("publisher", ""),
                "published": v.get("publishedDate", ""),
                "isbn": _isbn13(v.get("industryIdentifiers")),
                "pages": v.get("pageCount"),
                "url": v.get("infoLink") or f"https://books.google.com/books?id={item.get('id')}",
            }
        )
    return out


async def open_library(client, title: str, author: str, metrics: Metrics) -> list[dict]:
    params = {
        "title": title,
        "limit": 8,
        "fields": "key,title,author_name,isbn,publisher,first_publish_year,number_of_pages_median",
    }
    if author:
        params["author"] = author
    data = await _get(client, "https://openlibrary.org/search.json", params, metrics)
    out = []
    for d in (data or {}).get("docs", []):
        isbns = d.get("isbn") or []
        out.append(
            {
                "source": "Open Library",
                "title": d.get("title", ""),
                "authors": d.get("author_name", []),
                "publisher": (d.get("publisher") or [""])[0],
                # Search results are work-level: first_publish_year is when the WORK first appeared (1813 for any
                # Pride and Prejudice), not this copy's edition, so it must not drive the antiquarian/pre-ISBN rules.
                "published": "",
                "first_published": str(d.get("first_publish_year") or ""),
                "isbn": next((i for i in isbns if len(i) == 13), isbns[0] if isbns else ""),
                "pages": d.get("number_of_pages_median"),
                "url": f"https://openlibrary.org{d.get('key', '')}",
            }
        )
    return out


def surname_on_spine(read: dict, authors: list[str]) -> bool:
    """Spines often print a truncated or partial surname ('WOO' for Woolf, 'HARPE' for Harpe(r), 'ARUND' for
    Arundhati). True if any 3+ letter token on the spine starts a candidate author's name part, or vice versa."""
    tokens = {t for t in norm(read.get("author", "") + " " + read.get("verbatim", "")).split() if len(t) >= 3}
    for a in authors:
        for part in (p for p in norm(a).split() if len(p) >= 3):
            if any(part.startswith(t) or t.startswith(part) for t in tokens):
                return True
    return False


def _coverage(words: list[str], pool: list[str]) -> float:
    """Share of `words` (3+ letters) that appear, fuzzily, among `pool`."""
    words = [w for w in words if len(w) >= 3]
    if not words:
        return 1.0
    return sum(any(fuzz.ratio(w, p) >= 85 for p in pool) for w in words) / len(words)


def title_coverage(read: dict, cand: dict) -> tuple[float, float]:
    """(main-title words of the candidate found on the spine, title words read from the spine found in the candidate).
    token_set_ratio alone scores 100 when one shared word ('Indonesia') is a subset; both directions must be covered."""
    main = re.split(r"[:/;]| - ", cand["title"])[0]
    spine = norm(read.get("title", "") + " " + read.get("verbatim", "")).split()
    return _coverage(norm(main).split(), spine), _coverage(norm(read.get("title", "")).split(), norm(cand["title"]).split())


def score(read: dict, cand: dict) -> tuple[float, float]:
    rt = norm(read.get("title") or read.get("verbatim"))
    ct = norm(cand["title"])
    title = max(fuzz.token_set_ratio(rt, ct), fuzz.partial_ratio(ct, norm(read.get("verbatim", ""))) if len(ct) > 6 else 0)
    cand_cov, read_cov = title_coverage(read, cand)
    if cand_cov < 0.7 or read_cov < 0.5:
        title = min(title, 60.0)  # shares a word or two, not the title
    ra = norm(read.get("author", ""))
    author = max((fuzz.token_set_ratio(ra, norm(a)) for a in cand["authors"]), default=0) if ra else -1
    if surname_on_spine(read, cand["authors"]):
        author = max(author, 85.0)
    return float(title), float(author)


AUDIO_EDITION = re.compile(r"\b(audio|audible|brilliance|blackstone audio|tantor|recorded books|mp3|audiobook|audio cd)\b", re.I)


def decide(read: dict, cands: list[dict]) -> dict:
    """Pick the best candidate or return an unidentified verdict with the reason."""
    if not read or not read.get("legible") or read.get("confidence", 0) < READ_CONF_MIN:
        return {"status": "unidentified", "reason": "spine not legible"}
    scored = []
    for c in cands:
        t, a = score(read, c)
        ok = t >= TITLE_MIN and (a >= AUTHOR_MIN or (a < 0 and len(norm(read.get("title", "")).split()) >= 3 and t >= 95))
        # Prefer print: an audiobook record of the same title is the wrong product to price and has no spine size.
        print_pref = 0 if AUDIO_EDITION.search(f"{c.get('publisher', '')} {c.get('title', '')}") else 1
        scored.append((ok, print_pref, t + max(a, 0), t, a, c))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    scored = [(ok, tot, t, a, c) for ok, _, tot, t, a, c in scored]
    if not scored or not scored[0][0]:
        return {"status": "unidentified", "reason": "no catalogue match above threshold"}
    _, _, t, a, best = scored[0]
    # Edition: only claim one when the printed publisher matches the candidate's publisher.
    rp = norm(read.get("publisher", ""))
    edition_match = next(
        (c for ok, _, _, _, c in scored if ok and rp and c["isbn"] and fuzz.token_set_ratio(rp, norm(c["publisher"])) >= 80),
        None,
    )
    chosen = edition_match or best
    conf = 0.5 * t / 100 + 0.3 * (a / 100 if a >= 0 else 0.6) + 0.2 * read.get("confidence", 0)
    # Other catalogue records that passed the same title/author check: other editions of the same work.
    # Used only to find dimensions and price listings when the chosen record has none.
    alternates = [c for ok, _, _, _, c in scored if ok and c is not chosen and c.get("isbn")][:6]
    return {
        "status": "identified",
        "match": chosen,
        "alternates": alternates,
        "title_score": t,
        "author_score": a,
        "edition_basis": "publisher on spine matches catalogue record" if edition_match else "representative edition; publisher not confirmed",
        "id_confidence": round(min(conf, 0.99), 3),
    }


def parse_dimensions(text: str) -> dict | None:
    """'24.2 x 16.2 x 3.9 centimeters' / '7.3 x 4.2 x 1.5 inches' -> {'height_cm','width_cm','thickness_cm'}."""
    if not text:
        return None
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return None
    unit = 2.54 if re.search(r"inch|\bin\b|\"", text.lower()) else (0.1 if "mm" in text.lower() else 1.0)
    vals = sorted((n * unit for n in nums if n > 0), reverse=True)
    if not vals:
        return None
    out = {"height_cm": round(vals[0], 2)}  # height is the largest dimension of a book
    if len(vals) >= 3:
        out["width_cm"], out["thickness_cm"] = round(vals[1], 2), round(vals[-1], 2)
    return out


async def _dims_for(client, rec: dict, metrics: Metrics) -> dict | None:
    # Google Books first: far higher rate limit than Open Library (~3 req/s).
    if rec.get("volume_id"):
        params = {"key": os.getenv("GOOGLE_BOOKS_API_KEY")} if os.getenv("GOOGLE_BOOKS_API_KEY") else {}
        data = await _get(client, f"https://www.googleapis.com/books/v1/volumes/{rec['volume_id']}", params, metrics)
        d = (data or {}).get("volumeInfo", {}).get("dimensions") or {}
        if d.get("height"):
            dims = {k + "_cm": parse_dimensions(d[k])["height_cm"] for k in ("height", "width", "thickness") if d.get(k)}
            if 10 <= dims["height_cm"] <= 50:
                return {**dims, "source": "Google Books", "url": rec["url"]}
    if rec.get("isbn"):
        data = await _get(client, f"https://openlibrary.org/isbn/{rec['isbn']}.json", {}, metrics)
        dims = parse_dimensions((data or {}).get("physical_dimensions", ""))
        if dims and 10 <= dims["height_cm"] <= 50:
            return {**dims, "source": "Open Library", "url": f"https://openlibrary.org/isbn/{rec['isbn']}"}
    return None


async def dimensions(client, match: dict, metrics: Metrics, alternates: list[dict] | None = None) -> dict | None:
    """Catalogue physical dimensions: the chosen edition first, then other editions of the same work.

    Another edition can differ in size; the book ruler's aspect-ratio check and 8% outlier rejection
    drop it from the scale if it does not fit what is seen on the shelf.
    """
    for k, rec in enumerate(([match] + (alternates or []))[:3]):
        dims = await _dims_for(client, rec, metrics)
        if dims:
            return {**dims, "isbn": rec.get("isbn", ""), "edition_basis": "chosen edition" if k == 0 else "another edition of the same work"}
    return None


QUERY_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "author": {"type": "string"}},
    "required": ["title", "author"],
}
QUERY_PROMPT = (
    "Text read from a book spine (may contain OCR errors, truncated words and library call numbers such as "
    "'796.51 Gre' or 'NE Smi'). Return the book's title and author as they would be written in a catalogue, "
    "fixing only errors and truncations whose intended words are evident from the text itself. Leave a field "
    "empty if it is not on the spine. This is used only as a search query."
)
PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer", "description": "index of the matching record, or -1 if none matches"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["choice", "confidence", "reason"],
}
PICK_PROMPT = (
    "Which catalogue record is the book whose spine text is given? A record matches only if the spine's own words "
    "(title words, author name or fragment) identify that work; another edition, translation or volume of the same "
    "work counts. Do not choose a record because it is popular or plausible on a shelf. If none clearly matches, "
    "answer -1."
)
PICK_MIN_COVERAGE = 0.5  # even after the model picks, half the record's main-title words must be on the spine


async def _search(client, title: str, author: str, metrics: Metrics, open_library_too: bool = True) -> list[dict]:
    jobs = [google_books(client, title, author, metrics)]
    if open_library_too:
        jobs.append(open_library(client, title, author, metrics))
    return [c for part in await asyncio.gather(*jobs) for c in part]


def _dedupe(cands: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in cands:
        k = (norm(c["title"]), tuple(norm(a) for a in c["authors"][:1]), c.get("isbn"))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


async def resolve(client, read: dict, metrics: Metrics) -> dict:
    """Spine read -> catalogue verdict. Cheapest path first:
    1. Google Books (fast host) with title+author; accept a strong fuzzy match.
    2. Add Open Library and title-only searches; fuzzy decision again.
    3. Hard cases only: a small model rewrites the noisy read into a search query, the catalogues are searched
       again, and the model picks among RETRIEVED records (or none). It never supplies a title itself, and the pick
       must still share half its main-title words with the spine."""
    if not read.get("legible") or read.get("confidence", 0) < READ_CONF_MIN or not (read.get("title") or read.get("verbatim")):
        return decide(read, [])
    title, author = read.get("title") or read.get("verbatim"), read.get("author", "")
    # Google Books first and alone: it allows far more requests per second than Open Library (~3/s).
    cands = await _search(client, title, author, metrics, open_library_too=False)
    v = decide(read, cands)
    if v["status"] == "identified" and v["id_confidence"] >= 0.85:
        return v | {"match_method": "fuzzy"}
    # Open Library with author, plus title-only searches in parallel (a misread author filters out the right record).
    extra = await asyncio.gather(
        open_library(client, title, author, metrics),
        *([google_books(client, read["title"], "", metrics), open_library(client, read["title"], "", metrics)] if author and read.get("title") else []),
    )
    cands = _dedupe(cands + [c for part in extra for c in part])
    v = decide(read, cands)
    if v["status"] == "identified":
        return v | {"match_method": "fuzzy"}
    # 3. small-model help for the hard cases
    q = await gemini_json(model=LITE_MODEL, parts=[QUERY_PROMPT, f"Spine text: {read.get('verbatim', '')}"], schema=QUERY_SCHEMA, metrics=metrics, stage=STAGE)
    if q and (q.get("title") or "").strip():
        cands = _dedupe(cands + await _search(client, q["title"], q.get("author", ""), metrics))
    v = decide(read, cands)
    if v["status"] == "identified":
        return v | {"match_method": "fuzzy after query rewrite", "query": q}
    ranked = sorted(cands, key=lambda c: -score(read, c)[0])[:8]
    if not ranked:
        return v | {"query": q}
    listing = "\n".join(f"{i}. {c['title']} | {', '.join(c['authors'][:2])} | {c.get('publisher', '')} {c.get('published', '')}" for i, c in enumerate(ranked))
    pick = await gemini_json(
        model=LITE_MODEL,
        parts=[PICK_PROMPT, f"Spine text: {read.get('verbatim', '')}\nRead as: title={read.get('title', '')!r}, author={read.get('author', '')!r}", "Records:\n" + listing],
        schema=PICK_SCHEMA, metrics=metrics, stage=STAGE,
    )
    i = (pick or {}).get("choice", -1)
    if not isinstance(i, int) or not 0 <= i < len(ranked):
        return v | {"query": q, "pick": pick}
    chosen = ranked[i]
    cand_cov, _ = title_coverage(read, chosen)
    if cand_cov < PICK_MIN_COVERAGE:
        return {"status": "unidentified", "reason": "model pick rejected: too few of its title words on the spine", "query": q, "pick": pick}
    conf = round(min(0.85, 0.85 * float(pick.get("confidence", 0))), 3)
    alternates = [c for c in ranked if c is not chosen and c.get("isbn") and norm(c["title"]) == norm(chosen["title"])][:6]
    return {
        "status": "identified", "match": chosen, "alternates": alternates, "title_score": score(read, chosen)[0],
        "author_score": score(read, chosen)[1], "edition_basis": "representative edition; publisher not confirmed",
        "id_confidence": conf, "match_method": "small-model pick among catalogue records", "query": q, "pick": pick,
    }


async def _one(run_dir: Path, client, book: dict, metrics: Metrics) -> tuple[str, dict]:
    """Read one spine crop and resolve it. Memoised on the crop (frame + box), so spines read during the sweep are
    not read again unless a better view became the book's evidence frame."""
    best = book["best"]
    key = {"file": best["file"], "box": [round(v) for v in best["box_px"]]}
    v = await memo(run_dir, "spine_reads", key, lambda: _read_and_resolve(run_dir, client, book, metrics))
    return book["id"], v


async def _read_and_resolve(run_dir: Path, client, book: dict, metrics: Metrics) -> dict:
    read = await gemini_json(
        model=VISION_MODEL,
        parts=[image_part(crop_spine(run_dir, book["best"])), READ_PROMPT],
        schema=READ_SCHEMA,
        metrics=metrics,
        stage=STAGE,
    )
    read = read or {}
    verdict = await resolve(client, read, metrics)
    if verdict["status"] == "identified":
        verdict["dimensions"] = await dimensions(client, verdict["match"], metrics, verdict["alternates"])
    return {"read": read, **verdict}


def work_key(idn: dict) -> str | None:
    """Same book = same ISBN, else same main title + first author."""
    if idn.get("status") != "identified":
        return None
    m = idn["match"]
    if m.get("isbn"):
        return "isbn:" + m["isbn"]
    return "work:" + norm(re.split(r"[:/;]", m["title"])[0]) + "|" + norm((m.get("authors") or [""])[0])


SAME_SPINE_TEXT = 80


def merge_same_book(books: list[dict], identified: dict) -> tuple[list[dict], dict]:
    """Second guard against counting one physical book twice, after frame tracking.

    Two spines that resolve to the same book AND sit on the same shelf within two spine widths of each other are
    one book seen twice (loose boxes, tracking drift, overlapping row crops), provided the two spine reads also agree:
    merge them, keep every view, and put a note in the review queue. The same book anywhere else may be a genuine second copy: keep both and flag them."""
    by_key: dict[str, list[dict]] = {}
    for b in books:
        k = work_key(identified.get(b["id"], {}))
        if k:
            by_key.setdefault(k, []).append(b)
    gone: set[str] = set()
    for k, group in by_key.items():
        group.sort(key=lambda b: b["seg_box"][0])
        for i, a in enumerate(group):
            if a["id"] in gone:
                continue
            for b in group[i + 1 :]:
                if b["id"] in gone or a.get("segment") != b.get("segment") or a["shelf"] != b["shelf"]:
                    continue
                wa, wb = a["seg_box"][2] - a["seg_box"][0], b["seg_box"][2] - b["seg_box"][0]
                ca, cb = (a["seg_box"][0] + a["seg_box"][2]) / 2, (b["seg_box"][0] + b["seg_box"][2]) / 2
                if abs(ca - cb) > 2 * max(wa, wb, 1.0):
                    continue
                ra, rb = (norm((identified[x["id"]].get("read") or {}).get("verbatim", "")) for x in (a, b))
                if ra and rb and fuzz.token_set_ratio(ra, rb) < SAME_SPINE_TEXT:
                    continue  # different spine wording ('50 itineraris' vs '51 passejades'): one of them is misidentified
                keep, drop = (a, b) if identified[a["id"]].get("id_confidence", 0) >= identified[b["id"]].get("id_confidence", 0) else (b, a)
                keep["observations"] = keep["observations"] + drop["observations"]
                keep.setdefault("review_notes", []).append(f"merged duplicate sighting {drop['id']} (same book, same shelf, adjacent)")
                gone.add(drop["id"])
                if drop is a:
                    break
        left = [b for b in group if b["id"] not in gone]
        if len(left) > 1:
            for b in left:
                others = ", ".join(o["id"] for o in left if o is not b)
                b.setdefault("review_notes", []).append(f"possible duplicate copy of {others} (same book seen elsewhere): confirm it is a second copy")
    return [b for b in books if b["id"] not in gone], {k: v for k, v in identified.items() if k not in gone}


async def run(run_dir: Path, books: list[dict], metrics: Metrics, tick=None) -> dict[str, dict]:
    with metrics.time(STAGE):
        async with httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True) as client:
            async def one(b):
                r = await _one(run_dir, client, b, metrics)
                if tick:
                    tick()
                return r

            results = await asyncio.gather(*(one(b) for b in books))
    out = dict(results)
    write_json(run_dir / "identified.json", out)
    return out
