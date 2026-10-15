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

from .common import VISION_MODEL, Metrics, gemini_json, image_part, write_json

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
_http_sem = asyncio.Semaphore(8)


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
    async with _http_sem:
        for attempt in range(3):
            try:
                r = await client.get(url, params=params, timeout=10)
                metrics.add_http(STAGE)
                if r.status_code == 429:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    return None
                return r.json()
            except (httpx.HTTPError, ValueError):
                await asyncio.sleep(0.5)
    return None


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
                "published": str(d.get("first_publish_year") or ""),
                "isbn": next((i for i in isbns if len(i) == 13), isbns[0] if isbns else ""),
                "pages": d.get("number_of_pages_median"),
                "url": f"https://openlibrary.org{d.get('key', '')}",
            }
        )
    return out


def score(read: dict, cand: dict) -> tuple[float, float]:
    rt = norm(read.get("title") or read.get("verbatim"))
    ct = norm(cand["title"])
    title = max(fuzz.token_set_ratio(rt, ct), fuzz.partial_ratio(ct, norm(read.get("verbatim", ""))) if len(ct) > 6 else 0)
    ra = norm(read.get("author", ""))
    author = max((fuzz.token_set_ratio(ra, norm(a)) for a in cand["authors"]), default=0) if ra else -1
    return float(title), float(author)


def decide(read: dict, cands: list[dict]) -> dict:
    """Pick the best candidate or return an unidentified verdict with the reason."""
    if not read or not read.get("legible") or read.get("confidence", 0) < READ_CONF_MIN:
        return {"status": "unidentified", "reason": "spine not legible"}
    scored = []
    for c in cands:
        t, a = score(read, c)
        ok = t >= TITLE_MIN and (a >= AUTHOR_MIN or (a < 0 and len(norm(read["title"]).split()) >= 3 and t >= 95))
        scored.append((ok, t + max(a, 0), t, a, c))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
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
    if rec.get("isbn"):
        data = await _get(client, f"https://openlibrary.org/isbn/{rec['isbn']}.json", {}, metrics)
        dims = parse_dimensions((data or {}).get("physical_dimensions", ""))
        if dims and 10 <= dims["height_cm"] <= 50:
            return {**dims, "source": "Open Library", "url": f"https://openlibrary.org/isbn/{rec['isbn']}"}
    if rec.get("volume_id"):
        params = {"key": os.getenv("GOOGLE_BOOKS_API_KEY")} if os.getenv("GOOGLE_BOOKS_API_KEY") else {}
        data = await _get(client, f"https://www.googleapis.com/books/v1/volumes/{rec['volume_id']}", params, metrics)
        d = (data or {}).get("volumeInfo", {}).get("dimensions") or {}
        if d.get("height"):
            dims = {k + "_cm": parse_dimensions(d[k])["height_cm"] for k in ("height", "width", "thickness") if d.get(k)}
            if 10 <= dims["height_cm"] <= 50:
                return {**dims, "source": "Google Books", "url": rec["url"]}
    return None


async def dimensions(client, match: dict, metrics: Metrics, alternates: list[dict] | None = None) -> dict | None:
    """Catalogue physical dimensions: the chosen edition first, then other editions of the same work.

    Another edition can differ in size; the book ruler's aspect-ratio check and 8% outlier rejection
    drop it from the scale if it does not fit what is seen on the shelf.
    """
    for k, rec in enumerate([match] + (alternates or [])):
        dims = await _dims_for(client, rec, metrics)
        if dims:
            return {**dims, "isbn": rec.get("isbn", ""), "edition_basis": "chosen edition" if k == 0 else "another edition of the same work"}
    return None


async def _one(run_dir: Path, client, book: dict, metrics: Metrics) -> tuple[str, dict]:
    read = await gemini_json(
        model=VISION_MODEL,
        parts=[image_part(crop_spine(run_dir, book["best"])), READ_PROMPT],
        schema=READ_SCHEMA,
        metrics=metrics,
        stage=STAGE,
    )
    read = read or {}
    cands: list[dict] = []
    if read.get("legible") and (read.get("title") or read.get("verbatim")):
        title = read.get("title") or read.get("verbatim")
        gb, ol = await asyncio.gather(
            google_books(client, title, read.get("author", ""), metrics),
            open_library(client, title, read.get("author", ""), metrics),
        )
        cands = gb + ol
    verdict = decide(read, cands)
    if verdict["status"] == "identified":
        verdict["dimensions"] = await dimensions(client, verdict["match"], metrics, verdict["alternates"])
    return book["id"], {"read": read, **verdict}


async def run(run_dir: Path, books: list[dict], metrics: Metrics) -> dict[str, dict]:
    with metrics.time(STAGE):
        async with httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True) as client:
            results = await asyncio.gather(*(_one(run_dir, client, b, metrics) for b in books))
    out = dict(results)
    write_json(run_dir / "identified.json", out)
    return out
