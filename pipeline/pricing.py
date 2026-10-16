"""Stage 7: local-market prices with a retrievable source for every figure.

Sources (free):
  - eBay Browse API, per-country marketplace: NEW listings -> replacement cost,
    USED listings -> used market value, by ISBN (GTIN) then by title+author.
    These are ACTIVE listings (asking prices), labelled as such.
  - Retailer product pages found by Gemini Google-Search grounding (used only to find
    URLs); the price is read by code from the page's schema.org JSON-LD / meta tags.
  - Frankfurter (ECB reference rates) when a price had to come from another market.

A language model never produces a number here. No source -> no price; the line is
excluded from totals and sent to review.

Output: runs/<id>/prices.json
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from rapidfuzz import fuzz

from .common import LOCALES, Metrics, current_failures, locale, memo, note_failure, now_iso, write_json

STAGE = "pricing"
EBAY_API = "https://api.ebay.com"
BOOKS_CATEGORY = "267"
LOT_WORDS = re.compile(r"\b(lot|lots|bundle|set of|collection|books \d|\d+ books|box set|job lot)\b", re.I)
FINE_PRESS = ["folio society", "easton press", "franklin library", "limited editions club", "heritage press", "arion press"]
SPECIAL_WORDS = re.compile(r"signed|inscribed|first edition|1st ed|limited|numbered|leather|gilt|association", re.I)
ART_CATEGORIES = {"art_portrait", "art_framed"}


def threshold() -> float:
    return float(os.getenv("APPRAISAL_THRESHOLD", "250"))


class Ebay:
    def __init__(self, client: httpx.AsyncClient, metrics: Metrics):
        self.client, self.metrics = client, metrics
        self.token: str | None = None
        self.expiry = 0.0
        self.sem = asyncio.Semaphore(12)

    @property
    def configured(self) -> bool:
        return bool(os.getenv("EBAY_CLIENT_ID") and os.getenv("EBAY_CLIENT_SECRET"))

    async def _auth(self) -> str:
        if self.token and time.time() < self.expiry - 60:
            return self.token
        basic = base64.b64encode(f"{os.environ['EBAY_CLIENT_ID']}:{os.environ['EBAY_CLIENT_SECRET']}".encode()).decode()
        r = await self.client.post(
            f"{EBAY_API}/identity/v1/oauth2/token",
            headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"},
        )
        r.raise_for_status()
        body = r.json()
        self.token, self.expiry = body["access_token"], time.time() + body["expires_in"]
        return self.token

    async def _headers(self, country: str) -> dict:
        loc = locale(country)
        return {
            "Authorization": f"Bearer {await self._auth()}",
            "X-EBAY-C-MARKETPLACE-ID": loc["ebay_marketplace"],
            "X-EBAY-C-ENDUSERCTX": f"contextualLocation=country={country}",
        }

    async def search(self, country: str, *, condition: str, gtin: str = "", q: str = "", category: str = "") -> list[dict]:
        if not self.configured:
            return []
        params = {"limit": "50", "filter": f"conditions:{{{condition}}},buyingOptions:{{FIXED_PRICE}},itemLocationCountry:{country}"}
        if gtin:
            params["gtin"] = gtin
        if q:
            params["q"] = q[:100]
        if category:
            params["category_ids"] = category
        async with self.sem:
            r = await self.client.get(f"{EBAY_API}/buy/browse/v1/item_summary/search", params=params, headers=await self._headers(country))
            self.metrics.add_http(STAGE)
        if r.status_code != 200:
            note_failure(f"eBay: HTTP {r.status_code}")
            return []
        return [_listing(x) for x in r.json().get("itemSummaries", []) if x.get("price")]

    async def search_by_image(self, country: str, jpeg: bytes, condition: str = "NEW") -> list[dict]:
        if not self.configured:
            return []
        params = {"limit": "50", "filter": f"conditions:{{{condition}}},buyingOptions:{{FIXED_PRICE}},itemLocationCountry:{country}"}
        async with self.sem:
            r = await self.client.post(
                f"{EBAY_API}/buy/browse/v1/item_summary/search_by_image",
                params=params,
                headers=await self._headers(country) | {"Content-Type": "application/json"},
                json={"image": base64.b64encode(jpeg).decode()},
            )
            self.metrics.add_http(STAGE)
        if r.status_code != 200:
            note_failure(f"eBay: HTTP {r.status_code}")
            return []
        return [_listing(x) for x in r.json().get("itemSummaries", []) if x.get("price")]


def _listing(x: dict) -> dict:
    ship = 0.0
    for opt in x.get("shippingOptions") or []:
        cost = (opt.get("shippingCost") or {}).get("value")
        if cost is not None:
            ship = float(cost)
            break
    return {
        "title": x.get("title", ""),
        "price": float(x["price"]["value"]),
        "currency": x["price"]["currency"],
        "shipping": ship,
        "total": round(float(x["price"]["value"]) + ship, 2),
        "url": x.get("itemWebUrl", ""),
        "condition": x.get("condition", ""),
    }


def iqr_filter(values: list[float]) -> list[float]:
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return [v for v in values if lo <= v <= hi]


def book_figure(listings: list[dict], currency: str, title: str = "") -> dict | None:
    """Median of the 3-5 cheapest comparable listings (price + shipping), after removing lots and outliers."""
    ok = [l for l in listings if l["currency"] == currency and not LOT_WORDS.search(l["title"])]
    if title:
        ok = [l for l in ok if fuzz.partial_ratio(title.lower(), l["title"].lower()) >= 80]
    if not ok:
        return None
    totals = iqr_filter([l["total"] for l in ok])
    ok = sorted([l for l in ok if l["total"] in totals], key=lambda l: l["total"])
    cheapest = ok[: min(5, len(ok))]
    med = float(np.median([l["total"] for l in cheapest]))
    ref = min(cheapest, key=lambda l: abs(l["total"] - med))
    return {"amount": round(med, 2), "url": ref["url"], "n_listings": len(cheapest), "evidence_urls": [l["url"] for l in cheapest]}


def range_figure(listings: list[dict], currency: str, min_n: int = 5) -> dict | None:
    """25th-75th percentile of comparable new listings, for items whose exact model is unknown."""
    ok = [l for l in listings if l["currency"] == currency and not LOT_WORDS.search(l["title"])]
    totals = iqr_filter([l["total"] for l in ok])
    if len(totals) < min_n:
        return None
    lo, hi = np.percentile(totals, [25, 75])
    ok = [l for l in ok if lo <= l["total"] <= hi]
    return {
        "low": round(float(lo), 2),
        "high": round(float(hi), 2),
        "n_listings": len(totals),
        "url": ok[0]["url"] if ok else "",
        "evidence_urls": [l["url"] for l in ok[:8]],
    }


class FX:
    def __init__(self, client: httpx.AsyncClient):
        self.client, self.cache = client, {}

    async def rate(self, base: str, quote: str) -> tuple[float, str, str] | None:
        if base == quote:
            return 1.0, "", ""
        if (base, quote) in self.cache:
            return self.cache[(base, quote)]
        for url in ("https://api.frankfurter.dev/v1/latest", "https://api.frankfurter.app/latest"):
            try:
                r = await self.client.get(url, params={"base": base, "symbols": quote}, timeout=8)
                if r.status_code == 200:
                    d = r.json()
                    out = (float(d["rates"][quote]), d.get("date", ""), str(r.url))
                    self.cache[(base, quote)] = out
                    return out
            except (httpx.HTTPError, KeyError, ValueError):
                continue
        return None


# ---------- retailer page price parser (JSON-LD / meta / microdata; kept for pages fetched directly) ----------

def extract_price(html: str, isbn: str = "") -> dict | None:
    """Price from schema.org JSON-LD (Product/Book/Offer), then product:price meta, then microdata. Never free text."""
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            data = json.loads(block.strip())
        except ValueError:
            continue
        nodes = data if isinstance(data, list) else [data]
        stack = list(nodes)
        while stack:
            n = stack.pop()
            if not isinstance(n, dict):
                continue
            stack.extend(n.get("@graph", []) if isinstance(n.get("@graph"), list) else [])
            types_ = n.get("@type")
            types_ = types_ if isinstance(types_, list) else [types_]
            if not {"Product", "Book", "Offer", "AggregateOffer"} & set(t for t in types_ if t):
                continue
            ids = " ".join(str(n.get(k, "")) for k in ("isbn", "gtin13", "gtin", "sku"))
            if isbn and ids.strip() and isbn not in ids.replace("-", ""):
                continue
            offers = n.get("offers", n if "Offer" in str(types_) else None)
            offers = offers if isinstance(offers, list) else [offers]
            for o in offers:
                if not isinstance(o, dict):
                    continue
                price = o.get("price", o.get("lowPrice"))
                cur = o.get("priceCurrency")
                if price not in (None, "") and cur:
                    try:
                        return {"amount": float(str(price).replace(",", "")), "currency": cur, "method": "json-ld"}
                    except ValueError:
                        pass
    m = re.search(r'property=["\'](?:product|og):price:amount["\'][^>]*content=["\']([\d.,]+)', html)
    c = re.search(r'property=["\'](?:product|og):price:currency["\'][^>]*content=["\']([A-Z]{3})', html)
    if m and c:
        return {"amount": float(m.group(1).replace(",", "")), "currency": c.group(1), "method": "meta product:price"}
    m = re.search(r'itemprop=["\']price["\'][^>]*content=["\']([\d.,]+)', html)
    c = re.search(r'itemprop=["\']priceCurrency["\'][^>]*content=["\']([A-Z]{3})', html)
    if m and c:
        return {"amount": float(m.group(1).replace(",", "")), "currency": c.group(1), "method": "microdata"}
    return None


# Backup source when eBay has no listing: Google Shopping results via SerpApi (free plan: 250 searches a month,
# 50 an hour). Prices are SerpApi's parsed numbers for named shops, with the shop's own product page as the source
# URL. Google has no public shopping API for new customers, and Gemini's search-grounding terms forbid using its
# links to find pages to scrape, so neither is used here.
SERPAPI_URL = "https://serpapi.com/search.json"
SERP_LOCALE = {
    "US": {"gl": "us", "hl": "en", "google_domain": "google.com"},
    "GB": {"gl": "uk", "hl": "en", "google_domain": "google.co.uk"},
    "DE": {"gl": "de", "hl": "de", "google_domain": "google.de"},
    "AU": {"gl": "au", "hl": "en", "google_domain": "google.com.au"},
    "CA": {"gl": "ca", "hl": "en", "google_domain": "google.ca"},
}
ITEM_SEARCH_CAP = int(os.getenv("SERPAPI_ITEM_LIMIT", "15"))
SERP_HOURLY = int(os.getenv("SERPAPI_HOURLY_LIMIT", "45"))  # stay under the free plan's 50/hour
_serp_calls: list[float] = []


def retailer_configured() -> bool:
    return bool(os.getenv("SERPER_API_KEY") or os.getenv("SERPAPI_API_KEY"))


# Serper.dev (preferred backup): Google Shopping in ~2 s, 2,500 free searches, no card. Its offer links point to the
# Google Shopping offer page (shop + price shown there), not the shop's own page; that link is cited as the source.
# It has no new/used flag, so an offer counts as used when its title says so or the shop is a used-book marketplace.
SERPER_URL = "https://google.serper.dev/shopping"
SERPER_GL = {"US": "us", "GB": "gb", "DE": "de", "AU": "au", "CA": "ca"}
CURRENCY_SYMBOL = {"USD": "$", "GBP": "£", "EUR": "€", "AUD": "$", "CAD": "$"}
USED_BOOK_SHOPS = re.compile(r"abebooks|biblio|world of books|wob\b|better world books|thriftbooks|alibris|oxfam|"
                             r"awesomebooks|zvab|booksrun|half price", re.I)


def parse_price(text: str, currency: str) -> float | None:
    """'$1,234.50' -> 1234.5 when the symbol matches the local currency; otherwise None (never guess a currency)."""
    t = (text or "").strip()
    sym = CURRENCY_SYMBOL.get(currency, "")
    if not sym or sym not in t or (currency == "USD" and re.search(r"[A-Z]{1,2}\$", t)):
        return None  # e.g. 'CA$' or 'A$' in a US search
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)", t)
    return float(m.group(1).replace(",", "")) if m else None


def serper_offers(data: dict | None, currency: str) -> list[dict]:
    """Serper shopping results in the shape the matchers use (title, source, extracted_price, product_link,
    second_hand_condition)."""
    out = []
    for r in (data or {}).get("shopping", []):
        price = parse_price(r.get("price", ""), currency)
        if price is None or not r.get("link"):
            continue
        used = bool(USED_WORDS.search(r.get("title", "")) or USED_BOOK_SHOPS.search(r.get("source", "")))
        out.append({"title": r.get("title", ""), "source": r.get("source", ""), "extracted_price": price,
                    "product_link": r["link"], "second_hand_condition": "pre-owned" if used else ""})
    return out


async def _serper(client: httpx.AsyncClient, q: str, country: str, metrics: Metrics) -> dict | None:
    import time as _t

    gl = SERPER_GL.get(country)
    if not gl or not os.getenv("SERPER_API_KEY"):
        return None
    t0 = _t.monotonic()
    try:
        r = await client.post(SERPER_URL, json={"q": q, "gl": gl, "hl": "en"}, timeout=15,
                              headers={"X-API-KEY": os.getenv("SERPER_API_KEY"), "Content-Type": "application/json"})
        metrics.add_http(STAGE)
    except httpx.HTTPError as e:
        note_failure(f"Serper: {type(e).__name__}")
        return None
    from . import http
    http._stat("google.serper.dev", "requests")
    http._stat("google.serper.dev", "request_s", _t.monotonic() - t0)
    if r.status_code != 200:
        note_failure(f"Serper: HTTP {r.status_code}" + (" (search quota used up)" if r.status_code in (402, 429) else ""))
        return None
    return r.json()


async def shopping_offers(client, q: str, country: str, metrics: Metrics) -> tuple[list[dict], str]:
    """(offers in matcher shape, provider label). Serper first; SerpApi only if no Serper key."""
    cur = locale(country)["currency"]
    if os.getenv("SERPER_API_KEY"):
        return serper_offers(await _serper(client, q, country, metrics), cur), "Serper"
    loc = SERP_LOCALE.get(country)
    data = await _serpapi(client, {"engine": "google_shopping", "q": q, **loc}, metrics) if loc else None
    return (data or {}).get("shopping_results", []), "SerpApi"


async def _serpapi(client: httpx.AsyncClient, params: dict, metrics: Metrics) -> dict | None:
    import time as _t

    from . import http

    now = _t.monotonic()
    _serp_calls[:] = [t for t in _serp_calls if now - t < 3600]
    if not retailer_configured():
        return None
    if len(_serp_calls) >= SERP_HOURLY:
        note_failure(f"SerpApi: hourly search limit ({SERP_HOURLY}) reached")
        return None
    _serp_calls.append(now)
    # Never cached on disk: every price must carry its own retrieval time.
    # Google Shopping via SerpApi often takes 5-20 s: one 25 s try, one retry only on 429/5xx, never on a timeout
    # (retrying timeouts tripled the wait: ~33 s per stalled call in the audited sweep).
    return await http.get_json(client, SERPAPI_URL, {**params, "api_key": os.getenv("SERPAPI_API_KEY")}, metrics, STAGE,
                               cache=False, timeout=25, attempts=2, retry_timeouts=False)


FORMAT_WORDS = re.compile(
    r"\b(paperback|hardback|hardcover|softcover|mass market|book|books|edition|ed|new|novel|a novel|reprint|"
    r"illustrated|isbn|\d{10,13})\b", re.I)
USED_WORDS = re.compile(r"\b(used|pre-owned|preowned|second[- ]hand|refurbished)\b", re.I)
NOT_A_BOOK = re.compile(
    r"\b(tote|bag|t-?shirt|shirt|hoodie|mug|poster|print|sticker|bookmark|study guide|summary|sparknotes|cliffs ?notes|"
    r"notes|workbook|analysis|lesson plans?|quiz|calendar|journal|puzzle)\b", re.I)
NOT_PRINT = re.compile(r"\b(audio ?book|audible|e-?book|kindle|epub|audio cd|mp3|dvd)\b", re.I)


def _clean_title(t: str, author: str) -> str:
    t = re.split(r"\s[-|(\[]\s?|\(|\[|:", t)[0]
    t = re.split(r"\bby\b", t, flags=re.I)[0]
    for name in re.split(r"[ ,]+", author or ""):
        if len(name) > 2:
            t = re.sub(rf"\b{re.escape(name)}\b", " ", t, flags=re.I)
    return " ".join(re.sub(r"[^\w\s]", " ", FORMAT_WORDS.sub(" ", t)).lower().split())


def _merch(offer_title: str, book_title: str) -> bool:
    """A merchandise/study-aid word in the offer that is not part of the book's own title ('Notes from a Small
    Island' is a book; 'Infinite Jest study guide' is not)."""
    own = {w.lower() for w in NOT_A_BOOK.findall(book_title)}
    return any(w.lower() not in own for w in NOT_A_BOOK.findall(offer_title))


def shopping_matches(results: list[dict], title: str, author: str = "", used: bool = False) -> list[dict]:
    """Shopping offers that are this book in print: the offer's main title equals the book's main title (format
    words and the author's name removed), audio/e-books out, new or pre-owned as asked, with a shop and a price."""
    want = _clean_title(title, author)
    out = []
    for r in results or []:
        if not r.get("extracted_price") or not r.get("title") or not r.get("source"):
            continue
        if NOT_PRINT.search(r["title"]) or NOT_PRINT.search(r.get("source", "")) or _merch(r["title"], title):
            continue
        if bool(r.get("second_hand_condition") or USED_WORDS.search(r["title"])) != used:
            continue
        if _clean_title(r["title"], author) == want:
            out.append(r)
    if len(out) >= 3:  # drop offers more than 3x away from the median of the rest (collectibles, lots, mislabels)
        prices = sorted(float(r["extracted_price"]) for r in out)
        med = prices[len(prices) // 2]
        out = [r for r in out if med / 3 <= float(r["extracted_price"]) <= med * 3]
    return out


def _shop_query(book: dict) -> str:
    return f"{book['title'].split(':')[0]} {book['author'].split(',')[0]} book".strip()


async def shopping_search(client, book: dict, country: str, metrics: Metrics) -> dict | None:
    offers, provider = await shopping_offers(client, _shop_query(book), country, metrics)
    return {"shopping_results": offers, "provider": provider}


async def retailer_price(client: httpx.AsyncClient, book: dict, country: str, metrics: Metrics, used: bool = False,
                         data: dict | None = None) -> tuple[dict | None, dict | None]:
    """New (or used) price for a book from Google Shopping (SerpApi): the median of matching offers, reported as the
    offer of the shop at that median, with that shop's own product page as the source when SerpApi gives it.
    Returns (price or None, the raw search result so the other condition can reuse it without another search)."""
    if not SERP_LOCALE.get(country):
        return None, data
    q = _shop_query(book)
    if data is None:
        data = await shopping_search(client, book, country, metrics)
    matches = sorted(shopping_matches((data or {}).get("shopping_results", []), book["title"], book.get("author", ""), used),
                     key=lambda r: r["extracted_price"])
    if not matches:
        return None, data
    pick = matches[(len(matches) - 1) // 2]
    url, shop, amount = pick.get("product_link", ""), pick["source"], float(pick["extracted_price"])
    token = pick.get("immersive_product_page_token")  # SerpApi only; Serper offers cite the Google offer page
    if token:  # one more search for the shop's own page (product_link points to Google)
        det = await _serpapi(client, {"engine": "google_immersive_product", "page_token": token}, metrics) or {}
        raw = (det.get("product_results") or {}).get("stores") or det.get("stores") or []
        stores = [st for st in raw if st.get("link") and st.get("extracted_price")]
        same = [st for st in stores if (st.get("name") or "").lower() == shop.lower()] or stores
        if same:
            st = same[0]
            url, shop, amount = st["link"], st.get("name", shop), float(st["extracted_price"])
    kind = "pre-owned" if used else "new"
    basis = f"median of {len(matches)} matching {kind} offers" if len(matches) > 1 else f"single {kind} offer found"
    return {
        "amount": amount, "url": url,
        "source": f"Google Shopping via {(data or {}).get('provider', 'SerpApi')}: {shop} ({kind}; {basis} for '{q}')",
        "evidence_urls": [m.get("product_link", "") for m in matches[:5]],
    }, data


# ---------- books ----------

def appraisal_reasons(book: dict, read: dict) -> list[str]:
    reasons = []
    marks = " ".join(read.get("special_marks") or []) + " " + read.get("verbatim", "")
    if SPECIAL_WORDS.search(marks):
        reasons.append(f"spine shows special marks: {SPECIAL_WORDS.search(marks).group(0)}")
    year = re.search(r"\d{4}", book.get("published") or "")
    if year and int(year.group(0)) < 1900:
        reasons.append(f"published {year.group(0)} (antiquarian)")
    if not book.get("isbn") and year and int(year.group(0)) < 1970:
        reasons.append("pre-ISBN edition")
    if any(fp in (book.get("publisher") or "").lower() for fp in FINE_PRESS):
        reasons.append(f"fine-press publisher ({book['publisher']})")
    if book.get("override_first_edition"):
        reasons.append("policyholder states first edition")
    return reasons


async def price_book(ebay: Ebay, fx: FX, client, book: dict, country: str, metrics: Metrics) -> dict:
    """Replacement (new) and used figures for one identified book in `country`."""
    cur = locale(country)["currency"]
    stamp = now_iso()
    out: dict = {"replacement_cost": None, "used_value": None, "notes": []}

    async def from_market(mkt: str, condition: str) -> tuple[dict | None, str]:
        listings, basis = [], "ISBN match"
        for isbn in (book.get("isbns") or [book.get("isbn")])[:3]:
            if isbn:
                listings = await ebay.search(mkt, condition=condition, gtin=isbn)
                if listings:
                    basis = f"ISBN {isbn} match"
                    break
        if not listings:
            listings = await ebay.search(mkt, condition=condition, q=f"{book['title']} {book['author']}", category=BOOKS_CATEGORY)
            basis = "title+author match"
            fig = book_figure(listings, locale(mkt)["currency"], title=book["title"])
        else:
            fig = book_figure(listings, locale(mkt)["currency"])
        return fig, basis

    if ebay.configured:
        for condition, key in (("NEW", "replacement_cost"), ("USED", "used_value")):
            fig, basis = await from_market(country, condition)
            market, fx_info = country, None
            if not fig:
                # No local listing: try other markets and convert at ECB rates, labelled as converted.
                for other in [c for c in LOCALES if c != country][:2]:
                    cand, basis = await from_market(other, condition)
                    rate = await fx.rate(locale(other)["currency"], cur) if cand else None
                    if cand and rate:
                        cand["amount"] = round(cand["amount"] * rate[0], 2)
                        fig, market = cand, other
                        fx_info = {"from": locale(other)["currency"], "rate": rate[0], "rate_date": rate[1], "source": rate[2]}
                        break
            if fig:
                kind = "new" if condition == "NEW" else "used"
                out[key] = {
                    "amount": fig["amount"],
                    "currency": cur,
                    "source": f"eBay {locale(market)['ebay_marketplace']} active listings ({kind}, asking price incl. shipping; "
                    f"median of {fig['n_listings']} cheapest; {basis})",
                    "url": fig["url"],
                    "evidence_urls": fig["evidence_urls"],
                    "retrieved_at": stamp,
                    "converted": fx_info is not None,
                    **({"fx": fx_info} if fx_info else {}),
                    **({"condition_assumed": "used, good (median of the cheapest comparable used listings)"} if condition == "USED" else {}),
                }
    need = [k for k in ("replacement_cost", "used_value") if out[k] is None]
    if need and retailer_configured():
        data = await shopping_search(client, book, country, metrics)  # one search serves new and used
        got = await asyncio.gather(*(retailer_price(client, book, country, metrics, used=(k == "used_value"), data=data) for k in need))
        found = {k: rp for k, (rp, _) in zip(need, got)}
        rp = found.get("replacement_cost")
        if rp:
            out["replacement_cost"] = {**rp, "currency": cur, "retrieved_at": now_iso(), "converted": False}
        rp = found.get("used_value")
        if rp:
            out["used_value"] = {**rp, "currency": cur, "retrieved_at": now_iso(), "converted": False,
                                 "condition_assumed": "pre-owned, condition as listed by the shop"}
    failed = current_failures()
    why = f" (price search failed: {'; '.join(sorted(set(failed)))}; will retry)" if failed else ""
    if out["replacement_cost"] is None:
        out["notes"].append("no new-price source found" + why)
    if out["used_value"] is None:
        out["notes"].append("no used-price source found" + why)
    return out


# ---------- items ----------

def _item_crop(run_dir: Path, item: dict) -> bytes:
    img = cv2.imread(str(run_dir / item["frame_ref"]))
    if img is None:
        return b""
    ymin, xmin, ymax, xmax = item["box_2d"]
    h, w = img.shape[:2]
    crop = img[int(ymin / 1000 * h) : int(ymax / 1000 * h), int(xmin / 1000 * w) : int(xmax / 1000 * w)]
    if crop.size == 0:
        crop = img
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


async def price_item(run_dir: Path, ebay: Ebay, item: dict, country: str, overrides: dict) -> dict:
    cur = locale(country)["currency"]
    stamp = now_iso()
    if item["category"] in ART_CATEGORIES and overrides.get("is_print") is not True:
        why = "art/portrait: original vs print not confirmed by policyholder" if "is_print" not in overrides else "policyholder states original"
        return {"status": "needs_appraisal", "replacement_cost": None, "reason": why}
    if not ebay.configured and not retailer_configured():
        return {"status": "needs_appraisal", "replacement_cost": None, "reason": "price source not configured"}
    if item.get("brand_model"):
        listings = await ebay.search(country, condition="NEW", q=item["brand_model"])
        listings = [l for l in listings if fuzz.partial_ratio(item["brand_model"].lower(), l["title"].lower()) >= 75]
        fig = book_figure(listings, cur)
        if fig:
            return {
                "status": "priced",
                "replacement_cost": {
                    "low": fig["amount"], "high": fig["amount"], "currency": cur,
                    "source": f"eBay {locale(country)['ebay_marketplace']} new listings for '{item['brand_model']}' (asking; median of {fig['n_listings']} cheapest)",
                    "url": fig["url"], "evidence_urls": fig["evidence_urls"], "retrieved_at": stamp,
                },
            }
    # Unknown model: visual search first, then a descriptive keyword search; report a sourced range.
    listings = await ebay.search_by_image(country, _item_crop(run_dir, item)) if ebay.configured else []
    method = "eBay image search on the item crop"
    if len(listings) < 5 and ebay.configured:
        desc = f"{item.get('material', '')} {item['category'].replace('_', ' ')} {item.get('description', '')}"
        if item.get("is_print"):
            desc = f"framed print {item.get('description', '')}"
        listings = await ebay.search(country, condition="NEW", q=re.sub(r"\s+", " ", desc).strip())
        method = f"eBay keyword search '{desc.strip()[:60]}'"
    rng = range_figure(listings, cur)
    if not rng and retailer_configured() and item["category"] not in ART_CATEGORIES and item.get("_search_ok", True):
        q = item.get("brand_model") or re.sub(r"\s+", " ", f"{item.get('material', '')} {item.get('description', '')}").strip()
        found, provider = await shopping_offers(ebay.client, q, country, ebay.metrics)
        offers = [r for r in found if r.get("extracted_price") and not r.get("second_hand_condition")]
        if len(offers) >= 5:
            v = sorted(float(r["extracted_price"]) for r in offers)
            lo, hi = float(np.percentile(v, 25)), float(np.percentile(v, 75))
            mid = min(offers, key=lambda r: abs(float(r["extracted_price"]) - (lo + hi) / 2))
            return {
                "status": "range",
                "replacement_cost": {
                    "low": round(lo, 2), "high": round(hi, 2), "currency": cur,
                    "source": f"Google Shopping via {provider} '{q[:60]}'; 25th-75th percentile of {len(offers)} new offers",
                    "url": mid.get("product_link", ""), "evidence_urls": [r.get("product_link", "") for r in offers[:5]], "retrieved_at": now_iso(),
                },
            }
    if not rng:
        why = "fewer than 5 comparable new listings found"
        if not item.get("_search_ok", True):
            why += f" (shopping search skipped: per-sweep limit of {ITEM_SEARCH_CAP} items, most confident first)"
        if current_failures():
            why += f" (price search failed: {'; '.join(sorted(set(current_failures())))}; will retry)"
        return {"status": "needs_appraisal", "replacement_cost": None, "reason": why}
    return {
        "status": "range",
        "replacement_cost": {
            "low": rng["low"], "high": rng["high"], "currency": cur,
            "source": f"{method}; 25th-75th percentile of {rng['n_listings']} new listings (asking, incl. shipping)",
            "url": rng["url"], "evidence_urls": rng["evidence_urls"], "retrieved_at": stamp,
        },
    }


async def run(run_dir: Path, books: list[dict], items: list[dict], overrides: dict, metrics: Metrics, country: str, tick=None) -> dict:
    """books: [{id, title, author, isbn, publisher, published, read}] identified only."""
    with metrics.time(STAGE):
        async with httpx.AsyncClient(follow_redirects=True, timeout=12) as client:
            ebay, fx = Ebay(client, metrics), FX(client)

            async def one_book(b):
                reasons = appraisal_reasons(b, b.get("read", {}))
                p = await memo_book_price(run_dir, ebay, fx, client, b, country, metrics)
                for key in ("replacement_cost", "used_value"):
                    if p[key] and p[key]["amount"] >= threshold():
                        reasons.append(f"{key.replace('_', ' ')} {p[key]['amount']} {p[key]['currency']} >= appraisal threshold {threshold()}")
                if p["replacement_cost"] and p["used_value"] and p["used_value"]["amount"] >= 2 * p["replacement_cost"]["amount"]:
                    reasons.append("used asking price is at least twice the new price (possible collectible)")
                if tick:
                    tick()
                return b["id"], {**p, "appraisal_reasons": reasons}

            # Shopping searches are a scarce free quota: only the most confident items may use one.
            ranked = sorted(items, key=lambda it: -float(it.get("confidence", 0)))
            ok_ids = {it["id"] for it in ranked[:ITEM_SEARCH_CAP]}
            items = [{**it, "_search_ok": it["id"] in ok_ids} for it in items]
            # Books and items at once: their lookups are independent and mostly waiting on the network.
            n_b = len(books)
            res = await asyncio.gather(
                *(one_book(b) for b in books),
                *(_pair(it["id"], memo_item_price(run_dir, ebay, it, country, overrides.get("items", {}).get(it["id"], {})), tick)
                  for it in items),
            )
            book_prices, item_prices = dict(res[:n_b]), dict(res[n_b:])
    out = {"country": country, "currency": locale(country)["currency"], "books": book_prices, "items": item_prices}
    write_json(run_dir / f"prices_{country}.json", out)
    return out


def book_price_key(b: dict, country: str) -> dict:
    return {"country": country, **{k: b.get(k) for k in ("title", "author", "isbn", "isbns", "publisher", "published")}}


async def memo_book_price(run_dir: Path, ebay, fx, client, b: dict, country: str, metrics: Metrics) -> dict:
    """A book priced during the sweep keeps that price and its real retrieved_at time. Not memoised while no
    price source is configured, so adding keys later still prices it."""
    if not ebay.configured and not retailer_configured():
        return await price_book(ebay, fx, client, b, country, metrics)
    return await memo(run_dir, "book_prices", book_price_key(b, country), lambda: price_book(ebay, fx, client, b, country, metrics))


async def memo_item_price(run_dir: Path, ebay, item: dict, country: str, ov: dict) -> dict:
    if not ebay.configured and not retailer_configured():
        return await price_item(run_dir, ebay, item, country, ov)
    key = {"country": country, "override": ov, **{k: item.get(k) for k in ("category", "description", "material", "brand_model", "frame_ref", "box_2d")}}
    return await memo(run_dir, "item_prices", key, lambda: price_item(run_dir, ebay, item, country, ov))


async def _pair(key, coro, tick=None):
    r = await coro
    if tick:
        tick()
    return key, r
