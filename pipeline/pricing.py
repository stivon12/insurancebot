"""Prices with sources: ISBN shop pages, Google Shopping (Serper/SerpApi), eBay; FX; appraisal rules."""

from __future__ import annotations

import asyncio
import base64
import cv2
import httpx
import json
import numpy as np
import os
import re
import time
from pathlib import Path
from rapidfuzz import fuzz

from .items import product
from .common import LOCALES, Metrics, _stat, current_failures, get_json, locale, memo, note_failure, now_iso, write_json


STAGE = "pricing"
EBAY_API = "https://api.ebay.com"
BOOKS_CATEGORY = "267"
LOT_WORDS = re.compile(r"\b(lot|lots|bundle|set of|collection|books \d|\d+ books|box set|job lot)\b", re.I)
FINE_PRESS = ["folio society", "easton press", "franklin library", "limited editions club", "heritage press", "arion press"]
SPECIAL_WORDS = re.compile(r"signed|inscribed|first edition|1st ed|limited|numbered|leather|gilt|association", re.I)
ART_CATEGORIES = {"art"}
def threshold() -> float:
    return float(os.getenv("APPRAISAL_THRESHOLD", "250"))
class EbayUnavailable(Exception):
    pass
class Ebay:
    def __init__(self, client: httpx.AsyncClient, metrics: Metrics):
        self.client, self.metrics = client, metrics
        self.token: str | None = None
        self.expiry = 0.0
        self.sem = asyncio.Semaphore(12)
        self.rejected = False  # keys refused once: no more eBay calls this run

    @property
    def configured(self) -> bool:
        return bool(os.getenv("EBAY_CLIENT_ID") and os.getenv("EBAY_CLIENT_SECRET")) and not self.rejected

    async def _auth(self) -> str:
        if self.token and time.time() < self.expiry - 60:
            return self.token
        basic = base64.b64encode(f"{os.environ['EBAY_CLIENT_ID']}:{os.environ['EBAY_CLIENT_SECRET']}".encode()).decode()
        r = await self.client.post(
            f"{EBAY_API}/identity/v1/oauth2/token",
            headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"},
        )
        if r.status_code != 200:
            self.rejected = True
            note_failure(f"eBay: keys rejected (HTTP {r.status_code})")
            raise EbayUnavailable()
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
            try:
                headers = await self._headers(country)
            except EbayUnavailable:
                return []
            r = await self.client.get(f"{EBAY_API}/buy/browse/v1/item_summary/search", params=params, headers=headers)
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
            try:
                headers = await self._headers(country)
            except EbayUnavailable:
                return []
            r = await self.client.post(
                f"{EBAY_API}/buy/browse/v1/item_summary/search_by_image",
                params=params,
                headers=headers | {"Content-Type": "application/json"},
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
# Backup source when eBay has no listing: Google Shopping results via SerpApi (free plan: 250 searches a month,
# 50 an hour). Prices are SerpApi's parsed numbers for named shops, with the shop's own product page as the source
SERPAPI_URL = "https://serpapi.com/search.json"
SERP_LOCALE = {
    "US": {"gl": "us", "hl": "en", "google_domain": "google.com"},
    "GB": {"gl": "uk", "hl": "en", "google_domain": "google.co.uk"},
    "DE": {"gl": "de", "hl": "de", "google_domain": "google.de"},
    "AU": {"gl": "au", "hl": "en", "google_domain": "google.com.au"},
    "CA": {"gl": "ca", "hl": "en", "google_domain": "google.ca"},
}
def item_search_cap() -> int:
    """Shopping searches per sweep for items. SerpApi's free plan allows 50 an hour, so 15; Serper is paid per search
    (about $0.001), so 40 covers a furnished room for about 4 cents."""
    return int(os.getenv("SERPAPI_ITEM_LIMIT") or (40 if os.getenv("SERPER_API_KEY") else 15))
SERP_HOURLY = int(os.getenv("SERPAPI_HOURLY_LIMIT", "45"))  # stay under the free plan's 50/hour
_serp_calls: list[float] = []
def retailer_configured() -> bool:
    return bool(os.getenv("SERPER_API_KEY") or os.getenv("SERPAPI_API_KEY"))
# Serper.dev (preferred backup): Google Shopping in ~2 s, 2,500 free searches, no card. Its offer links point to the
# Google Shopping offer page (shop + price shown there), not the shop's own page; that link is cited as the source.
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
    _stat("google.serper.dev", "requests")
    _stat("google.serper.dev", "request_s", _t.monotonic() - t0)
    if r.status_code != 200:
        note_failure(f"Serper: HTTP {r.status_code}" + (" (search quota used up)" if r.status_code in (402, 429) else ""))
        return None
    return r.json()
async def shopping_offers(client, q: str, country: str, metrics: Metrics) -> tuple[list[dict], str]:
    """(offers in matcher shape, provider label). Serper first; SerpApi if there is no Serper key or Serper failed
    (quota used up, timeout): one provider's outage must not leave the whole sweep unpriced."""
    cur = locale(country)["currency"]
    if os.getenv("SERPER_API_KEY"):
        data = await _serper(client, q, country, metrics)
        if data is not None or not os.getenv("SERPAPI_API_KEY"):
            return serper_offers(data, cur), "Serper"
    loc = SERP_LOCALE.get(country)
    data = await _serpapi(client, {"engine": "google_shopping", "q": q, **loc}, metrics) if loc else None
    return (data or {}).get("shopping_results", []), "SerpApi"
async def _serpapi(client: httpx.AsyncClient, params: dict, metrics: Metrics) -> dict | None:
    import time as _t

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
    return await get_json(client, SERPAPI_URL, {**params, "api_key": os.getenv("SERPAPI_API_KEY")}, metrics, STAGE,
                               cache=False, timeout=25, attempts=2, retry_timeouts=False)
FORMAT_WORDS = re.compile(
    r"\b(paperback|hardback|hardcover|softcover|mass market|book|books|edition|ed|new|novel|a novel|reprint|"
    r"illustrated|isbn|\d{10,13})\b", re.I)
USED_WORDS = re.compile(r"\b(used|pre-owned|preowned|second[- ]hand|refurbished)\b", re.I)
NOT_A_BOOK = re.compile(
    r"\b(tote|bag|t-?shirt|shirt|hoodie|mug|poster|print|sticker|bookmark|study guide|summary|sparknotes|cliffs ?notes|"
    r"notes|workbook|analysis|lesson plans?|quiz|calendar|journal|puzzle)\b", re.I)
MARKETPLACE_SHOPS = re.compile(r"\b(etsy|ebay|mercari|poshmark|depop|facebook)\b", re.I)
NOT_PRINT = re.compile(r"\b(audio ?book|audible|e-?book|kindle|epub|audio cd|mp3|dvd)\b", re.I)
def _clean_title(t: str, author: str) -> str:
    t = re.split(r"\s[-|(\[]\s?|\(|\[|:", t)[0]
    t = re.sub(r"^\s*(the|a|an)\s+", " ", t, flags=re.I)  # 'Hobbit' == 'The Hobbit'
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
        if MARKETPLACE_SHOPS.search(r.get("source", "")) or COLLECTIBLE.search(r["title"]):
            continue  # one-off collectible listings (an Etsy Infinite Jest at $103.69), not the replacement product
        if bool(r.get("second_hand_condition") or USED_WORDS.search(r["title"])) != used:
            continue
        if _clean_title(r["title"], author) == want:
            out.append(r)
    if len(out) >= 3:  # drop offers more than 3x away from the median of the rest (collectibles, lots, mislabels)
        prices = sorted(float(r["extracted_price"]) for r in out)
        med = prices[len(prices) // 2]
        out = [r for r in out if med / 3 <= float(r["extracted_price"]) <= med * 3]
    if len(out) < 2:
        return []  # a single offer is not a market price
    if len(out) == 2:
        lo, hi = sorted(float(r["extracted_price"]) for r in out)
        if hi > 1.25 * lo:
            return []  # two offers that disagree by more than 25 %
    return out
def _shop_query(book: dict) -> str:
    return f"{book['title'].split(':')[0]} {book['author'].split(',')[0]} book".strip()
async def shopping_search(client, book: dict, country: str, metrics: Metrics) -> dict | None:
    offers, provider = await shopping_offers(client, _shop_query(book), country, metrics)
    return {"shopping_results": offers, "provider": provider}
async def retailer_price(client: httpx.AsyncClient, book: dict, country: str, metrics: Metrics, used: bool = False,
                         data: dict | None = None) -> tuple[dict | None, dict | None]:
    """New (or used) price for a book from Google Shopping (SerpApi): the median of matching offers, reported as the offer of the shop at that median, with that..."""
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
    basis = f"median of {len(matches)} matching {kind} offers"
    return {
        "amount": amount, "url": url,
        "source": f"Google Shopping via {(data or {}).get('provider', 'SerpApi')}: {shop} ({kind}; {basis} for '{q}')",
        "evidence_urls": [m.get("product_link", "") for m in matches[:5]],
    }, data
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
async def isbn_shop_prices(client, book: dict, country: str, need: list[str]) -> dict:
    """New and used figures from shops' own listings for this ISBN (pipeline/py): Barnes & Noble list price
    (US new), AbeBooks offers (new and used, US and UK). Tries the matched edition's ISBN first, then alternates."""
    cur = locale(country)["currency"]
    if country not in ABEBOOKS:
        return {}
    out: dict = {}
    for isbn in [i for i in (book.get("isbns") or [book.get("isbn")]) if i][:3]:
        bn_job = barnes_noble(client, isbn) if country == "US" and "replacement_cost" in need and "replacement_cost" not in out else None
        abe, bn = await asyncio.gather(abebooks(client, isbn, book["title"], country, cur),
                                       bn_job if bn_job else asyncio.sleep(0, None))
        if "replacement_cost" in need and "replacement_cost" not in out:
            if bn:
                out["replacement_cost"] = {"amount": bn["amount"], "url": bn["url"], "evidence_urls": [bn["url"]],
                                           "source": f"Barnes & Noble list price, new {bn['format'].lower()} (ISBN {isbn})"}
            else:
                fig = new_figure(abe)
                if not fig and country == "GB":  # the mixed page is mostly used offers: ask for new ones
                    fig = new_figure(await abebooks(client, isbn, book["title"], country, cur, new_only=True))
                if fig:
                    out["replacement_cost"] = {
                        "amount": fig["amount"], "url": fig["pick"]["url"], "evidence_urls": fig["evidence"],
                        "source": f"AbeBooks new offers for ISBN {isbn}: median of {fig['n']} (seller {fig['pick']['seller']}; "
                                  "asking price, shipping not included)"}
        if "used_value" in need and "used_value" not in out:
            fig = used_figure(abe)
            if fig:
                out["used_value"] = {
                    "amount": fig["amount"], "url": fig["pick"]["url"], "evidence_urls": fig["evidence"],
                    "condition_assumed": "used, as graded by the seller (median of the cheaper half of offers)",
                    "source": f"AbeBooks used offers for ISBN {isbn}: median of the cheaper half of {fig['n']} "
                              f"(seller {fig['pick']['seller']}; asking price, shipping not included)"}
        if all(k in out for k in need):
            break
    return out
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
    if need:
        found = await isbn_shop_prices(client, book, country, need)
        for k, v in found.items():
            out[k] = {**v, "currency": cur, "retrieved_at": now_iso(), "converted": False}
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
# Typical replacement cost, highest first: which items get the limited shopping searches. Unlisted categories last.
VALUE_ORDER = ["electronics", "furniture", "shelving", "coffee_machine", "rug", "lamp", "art", "decor"]
def search_order(items: list[dict]) -> list[dict]:
    """Items in the order they may use the limited shopping searches: one per category per round, the more valuable category first, the most confident sighting..."""
    def rank(cat):
        return VALUE_ORDER.index(cat) if cat in VALUE_ORDER else len(VALUE_ORDER)

    seen: dict[str, int] = {}
    keyed = []
    for it in sorted(items, key=lambda it: -float(it.get("confidence", 0))):
        cat = it.get("category", "other")
        keyed.append(((seen.get(cat, 0), rank(cat)), it))
        seen[cat] = seen.get(cat, 0) + 1
    return [it for _, it in sorted(keyed, key=lambda x: x[0])]
def item_query(item: dict) -> str:
    """Brand plus what the object is ('HP printer', not 'HP': a brand alone matched Scott bicycles to a tissue box).
    Material is left out: leading 'glass, plastic' words only dilute the product search."""
    desc = re.sub(r"\s+", " ", item.get("description", "")).strip()
    brand = (item.get("brand_model") or "").strip()
    if brand and brand.lower() not in desc.lower():
        desc = f"{brand} {desc}"
    return (desc or item.get("category", "").replace("_", " ")).strip()
# Words in a listing title that mark an accessory or a part, not the product itself ('Skin for PS5', 'MacBook Pro case').
ACCESSORY = re.compile(r"\b(case|cases|skin|skins|cover|covers|sleeve|decal|sticker|stand|mount|holder|dock|charger|adapter|"
                       r"cable|cord|battery|replacement|repair|parts?|screen protector|protector|keyboard cover|faceplate|shell|"
                       r"controller|remote|box only|empty box|manual|for parts|broken|compatible|fits|bag|strap|charging)\b|\bfor (the )?\w",
                       re.I)
TOO_CHEAP = 1 / 3  # a price under a third of the scanner's own rough value is the wrong product (an accessory)


def _product(item: dict) -> tuple[str, str]:
    return product(f"{item.get('brand_model', '')} {item.get('description', '')}")


def _products_only(listings: list[dict], item: dict) -> list[dict]:
    """Electronics and appliances drown in accessory listings; for a bag or a stand those words are the product."""
    if item["category"] not in ("electronics", "appliance", "coffee_machine") and not _product(item)[0]:
        return listings
    return [l for l in listings if not ACCESSORY.search(l["title"])]


def _plausible(low: float, item: dict) -> bool:
    est = item.get("est_value_usd")
    return not est or low >= TOO_CHEAP * est


async def price_item(run_dir: Path, ebay: Ebay, item: dict, country: str, overrides: dict) -> dict:
    cur = locale(country)["currency"]
    stamp = now_iso()
    if item["category"] in ART_CATEGORIES and overrides.get("is_print") is not True:
        why = "art/portrait: original vs print not confirmed by policyholder" if "is_print" not in overrides else "policyholder states original"
        return {"status": "needs_appraisal", "replacement_cost": None, "reason": why}
    if not ebay.configured and not retailer_configured():
        return {"status": "needs_appraisal", "replacement_cost": None, "reason": "price source not configured"}
    noun, cat = _product(item)
    rejected = []

    def result(rng: dict, source: str) -> dict:
        return {"status": "range", "replacement_cost": {"low": rng["low"], "high": rng["high"], "currency": cur, "source": source,
                                                         "url": rng["url"], "evidence_urls": rng["evidence_urls"], "retrieved_at": stamp}}

    if ebay.configured:
        tries = []
        if item.get("brand_model"):  # the product itself: brand + what it is, inside its eBay category
            q = item["brand_model"] if noun and noun in item["brand_model"].lower() else f"{item['brand_model']} {noun}".strip()
            tries.append((q, cat, f"eBay {locale(country)['ebay_marketplace']} new listings for '{q}'"))
        tries.append((item_query(item), cat, f"eBay keyword search '{item_query(item)[:60]}'"))
        for q, c, label in tries:
            listings = _products_only(await ebay.search(country, condition="NEW", q=q, category=c), item)
            if item.get("brand_model"):
                listings = [l for l in listings if fuzz.partial_ratio(item["brand_model"].lower(), l["title"].lower()) >= 75] or listings
            rng = range_figure(listings, cur)
            if rng and _plausible(rng["low"], item):
                return result(rng, f"{label}; 25th-75th percentile of {rng['n_listings']} new listings, accessories removed (asking, incl. shipping)")
            if rng:
                rejected.append(f"{label} gave {rng['low']}-{rng['high']} {cur}, far below the item's rough value")
        if not item.get("brand_model"):  # an unknown model: a visual search on the item's crop
            listings = _products_only(await ebay.search_by_image(country, _item_crop(run_dir, item)), item)
            rng = range_figure(listings, cur)
            if rng and _plausible(rng["low"], item):
                return result(rng, f"eBay image search on the item crop; 25th-75th percentile of {rng['n_listings']} new listings, accessories removed")
            if rng:
                rejected.append(f"eBay image search gave {rng['low']}-{rng['high']} {cur}, far below the item's rough value")
    if retailer_configured() and item["category"] not in ART_CATEGORIES and item.get("_search_ok", True):
        q = f"{item_query(item)} {noun}".strip() if noun and noun not in item_query(item).lower() else item_query(item)
        found, provider = await shopping_offers(ebay.client, q, country, ebay.metrics)
        offers = [r for r in found if r.get("extracted_price") and not r.get("second_hand_condition")]
        offers = _products_only([{**r, "title": r.get("title", "")} for r in offers], item)
        if len(offers) >= 5:
            v = sorted(float(r["extracted_price"]) for r in offers)
            lo, hi = float(np.percentile(v, 25)), float(np.percentile(v, 75))
            if _plausible(lo, item):
                mid = min(offers, key=lambda r: abs(float(r["extracted_price"]) - (lo + hi) / 2))
                return result({"low": round(lo, 2), "high": round(hi, 2), "url": mid.get("product_link", ""),
                               "evidence_urls": [r.get("product_link", "") for r in offers[:5]]},
                              f"Google Shopping via {provider} '{q[:60]}'; 25th-75th percentile of {len(offers)} new offers, accessories removed")
            rejected.append(f"Google Shopping gave {lo:.0f}-{hi:.0f} {cur}, far below the item's rough value")
    why = "; ".join(rejected) or "fewer than 5 comparable new listings found"
    if not item.get("_search_ok", True):
        why += f" (shopping search skipped: per-sweep limit of {item_search_cap()} items, highest-value categories first)"
    if current_failures():
        why += f" (price search failed: {'; '.join(sorted(set(current_failures())))}; will retry)"
    return {"status": "needs_appraisal", "replacement_cost": None, "reason": why}


async def run(run_dir: Path, books: list[dict], items: list[dict], overrides: dict, metrics: Metrics, country: str, tick=None) -> dict:
    """books: [{id, title, author, isbn, publisher, published, read}] identified only."""
    with metrics.time(STAGE):
        from .books import UA

        async with httpx.AsyncClient(headers={"User-Agent": UA}, follow_redirects=True, timeout=12) as client:
            ebay, fx = Ebay(client, metrics), FX(client)

            async def one_book(b):
                reasons = appraisal_reasons(b, b.get("read", {}))
                p = await memo_book_price(run_dir, ebay, fx, client, b, country, metrics)
                for key in ("replacement_cost", "used_value"):
                    if p[key] and p[key]["amount"] >= threshold():
                        reasons.append(f"{key.replace('_', ' ')} {p[key]['amount']} {p[key]['currency']} >= appraisal threshold {threshold()}")
                if p["replacement_cost"] and p["used_value"] and p["used_value"]["amount"] >= 2 * p["replacement_cost"]["amount"]:
                    reasons.append("used asking price is at least twice the new price (possible collectible)")
                elif p["replacement_cost"] and p["used_value"] and p["used_value"]["amount"] > p["replacement_cost"]["amount"]:
                    # Actual cash value cannot exceed replacement cost: a used copy is never worth more than a new one.
                    p = {**p, "used_value": {**p["used_value"], "amount": p["replacement_cost"]["amount"],
                                             "capped_at_new": True, "uncapped_amount": p["used_value"]["amount"]}}
                if tick:
                    tick()
                return b["id"], {**p, "appraisal_reasons": reasons}

            # Shopping searches are a scarce free quota: spend them where the money is (big-ticket categories first),
            # then on the most confident sightings.
            if not all("_search_ok" in it for it in items):  # the live sweep already decided per item
                ranked = search_order(items)
                ok_ids = {it["id"] for it in ranked[:item_search_cap()]}
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
    # The query and whether this item may use a shopping search both decide the answer, so both are in the key (an item
    # skipped by the per-sweep limit once must not stay unpriced when it is ranked inside the limit later).
    key = {"country": country, "override": ov, "q": item_query(item), "search": item.get("_search_ok", True),
           **{k: item.get(k) for k in ("category", "description", "material", "brand_model", "frame_ref", "box_2d")}}
    return await memo(run_dir, "item_prices", key, lambda: price_item(run_dir, ebay, item, country, ov))
async def _pair(key, coro, tick=None):
    r = await coro
    if tick:
        tick()
    return key, r


ABEBOOKS = {"US": "https://www.abebooks.com", "GB": "https://www.abebooks.co.uk"}
ABE_NEW_ONLY = "100121501"  # AbeBooks' "new" condition filter (n=)
COLLECTIBLE = re.compile(r"\b(signed|inscribed|first edition|1st edition|1st ed|first printing|limited edition|"
                         r"numbered|association copy|proof|arc|advance reading|uncorrected|leather|folio society)\b", re.I)
NOT_PRINT_FORMAT = re.compile(r"(EBook|Audiobook|AudiobookFormat|GraphicNovel)$", re.I)
_HOST_GAP = {"www.abebooks.com": 0.4, "www.abebooks.co.uk": 0.4, "www.barnesandnoble.com": 0.5}
_host_state: dict[tuple[int, str], tuple[asyncio.Semaphore, asyncio.Lock, list[float]]] = {}
async def _fetch(client: httpx.AsyncClient, url: str, params: dict | None = None) -> httpx.Response | None:
    host = httpx.URL(url).host
    key = (id(asyncio.get_running_loop()), host)
    if key not in _host_state:
        _host_state[key] = (asyncio.Semaphore(3), asyncio.Lock(), [0.0])
    sem, lock, nxt = _host_state[key]
    async with sem:
        async with lock:
            wait = nxt[0] - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            nxt[0] = time.monotonic() + _HOST_GAP.get(host, 0.3)
        try:
            r = await client.get(url, params=params, timeout=15, follow_redirects=True)
        except httpx.HTTPError as e:
            note_failure(f"{host}: {type(e).__name__}")
            return None
    if r.status_code == 404:
        return None  # not stocked: an answer, not a failure
    if r.status_code != 200:
        note_failure(f"{host}: HTTP {r.status_code}")
        return None
    return r
def _ld_blocks(html: str) -> list:
    out = []
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            out.append(json.loads(block.strip()))
        except ValueError:
            continue
    return out
def _main_title(t: str) -> str:
    t = re.split(r"[:(\[]| - ", t or "")[0].lower()
    t = re.sub(r"^\s*(the|a|an)\s+|,\s*(the|a|an)\s*$|\.\s*the\s*$", " ", t.strip())
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())
def same_title(listing: str, book: str) -> bool:
    """Listing title is this book ('Hobbit, The', 'THE HOBBIT', 'Infinite Jest: A Novel'), not a mislisted ISBN."""
    a, b = _main_title(listing), _main_title(book)
    return bool(a and b) and fuzz.token_set_ratio(a, b) >= 85 and fuzz.partial_ratio(a, b) >= 85
def parse_abebooks(html: str, isbn: str, title: str, currency: str) -> list[dict]:
    offers = []
    for d in _ld_blocks(html):
        if not isinstance(d, dict) or d.get("@type") != "ItemList":
            continue
        for el in d.get("itemListElement", []):
            it = (el or {}).get("item") or {}
            o = it.get("offers") or {}
            if str(it.get("isbn", "")).replace("-", "") != isbn or o.get("priceCurrency") != currency:
                continue
            if "InStock" not in str(o.get("availability", "InStock")) or NOT_PRINT_FORMAT.search(str(it.get("bookFormat", ""))):
                continue
            if not same_title(it.get("name", ""), title) or COLLECTIBLE.search(it.get("name", "")):
                continue
            try:
                price = float(o["price"])
            except (KeyError, TypeError, ValueError):
                continue
            if price <= 0:
                continue
            cond = "new" if "NewCondition" in str(o.get("itemCondition")) else "used" if "UsedCondition" in str(o.get("itemCondition")) else ""
            if not cond:
                continue
            offers.append({"amount": price, "currency": currency, "condition": cond, "url": o.get("url") or it.get("url", ""),
                           "seller": ((o.get("seller") or {}).get("name") or ""), "format": str(it.get("bookFormat", "")).rsplit("/", 1)[-1]})
    return offers
def parse_bn(html: str, isbn: str) -> dict | None:
    """The exact-ISBN print variant's new price from a B&N ProductGroup."""
    for d in _ld_blocks(html):
        nodes = d.get("@graph", [d]) if isinstance(d, dict) else d if isinstance(d, list) else []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            for v in n.get("hasVariant", []) or ([n] if n.get("offers") else []):
                o = v.get("offers") or {}
                o = o[0] if isinstance(o, list) and o else o
                if not isinstance(o, dict) or f"ean={isbn}" not in str(o.get("url", "")):
                    continue
                if NOT_PRINT_FORMAT.search(str(v.get("bookFormat", ""))) or o.get("priceCurrency", "USD") != "USD":
                    return None
                try:
                    price = float(o["price"])
                except (KeyError, TypeError, ValueError):
                    return None
                if price > 0:
                    return {"amount": price, "currency": "USD", "condition": "new", "url": o["url"], "seller": "Barnes & Noble",
                            "format": str(v.get("bookFormat", "")).rsplit("/", 1)[-1]}
    return None
async def abebooks(client, isbn: str, title: str, country: str, currency: str, new_only: bool = False) -> list[dict]:
    base = ABEBOOKS.get(country)
    if not base:
        return []
    params = {"isbn": isbn, **({"n": ABE_NEW_ONLY} if new_only else {})}
    r = await _fetch(client, f"{base}/servlet/SearchResults", params)
    return parse_abebooks(r.text, isbn, title, currency) if r else []
async def barnes_noble(client, isbn: str) -> dict | None:
    r = await _fetch(client, f"https://www.barnesandnoble.com/s/{isbn}")
    return parse_bn(r.text, isbn) if r else None
def used_figure(offers: list[dict]) -> dict | None:
    """Used value: median of the cheaper half of >= 3 in-stock used offers, after dropping offers > 3x the median."""
    used = sorted((o for o in offers if o["condition"] == "used"), key=lambda o: o["amount"])
    if len(used) < 3:
        return None
    med = used[len(used) // 2]["amount"]
    used = [o for o in used if med / 3 <= o["amount"] <= med * 3]
    half = used[: max(3, (len(used) + 1) // 2)]
    pick = half[(len(half) - 1) // 2]
    return {"amount": pick["amount"], "n": len(used), "pick": pick, "evidence": [o["url"] for o in half[:5]]}
def new_figure(offers: list[dict]) -> dict | None:
    """New price from marketplace offers: median of >= 3 new offers, or 2 that agree within 25 %."""
    new = sorted((o for o in offers if o["condition"] == "new"), key=lambda o: o["amount"])
    if len(new) >= 3:
        med = new[len(new) // 2]["amount"]
        new = [o for o in new if med / 3 <= o["amount"] <= med * 3]
    elif len(new) == 2 and new[1]["amount"] > 1.25 * new[0]["amount"]:
        return None
    if len(new) < 2:
        return None
    pick = new[(len(new) - 1) // 2]
    return {"amount": pick["amount"], "n": len(new), "pick": pick, "evidence": [o["url"] for o in new[:5]]}
