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

from .common import LOCALES, Metrics, locale, now_iso, write_json

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
        self.sem = asyncio.Semaphore(6)

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
            return []
        return [_listing(x) for x in r.json().get("itemSummaries", []) if x.get("price")]

    async def search_by_image(self, country: str, jpeg: bytes, condition: str = "NEW") -> list[dict]:
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


# ---------- retailer page fallback (URL discovery by grounded search, price parsed by code) ----------

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


async def retailer_urls(title: str, author: str, isbn: str, country: str, metrics: Metrics) -> list[str]:
    """Candidate retailer product URLs via Gemini Google Search grounding (URLs only, never prices)."""
    if not os.getenv("GEMINI_API_KEY"):
        return []
    from google.genai import types

    from .common import gemini

    model = os.getenv("GROUNDING_MODEL", "gemini-2.5-flash-lite")
    prompt = f"Find bookshop product pages in {LOCALES[country]['name']} selling the new book '{title}' by {author} (ISBN {isbn})."
    try:
        resp = await gemini().aio.models.generate_content(
            model=model, contents=prompt, config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        )
        metrics.add_llm(STAGE, model, resp.usage_metadata)
        chunks = resp.candidates[0].grounding_metadata.grounding_chunks or []
        return [c.web.uri for c in chunks if c.web and c.web.uri][:4]
    except Exception:
        return []


async def retailer_price(client: httpx.AsyncClient, book: dict, country: str, metrics: Metrics) -> dict | None:
    cur = locale(country)["currency"]
    for url in await retailer_urls(book["title"], book["author"], book["isbn"], country, metrics):
        try:
            r = await client.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0 (compatible; library-claim-agent/0.1)"})
            metrics.add_http(STAGE)
        except httpx.HTTPError:
            continue
        if r.status_code != 200:
            continue
        p = extract_price(r.text, book["isbn"])
        if p and p["currency"] == cur:
            host = httpx.URL(str(r.url)).host
            return {"amount": p["amount"], "url": str(r.url), "source": f"{host} product page ({p['method']})"}
    return None


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
    if out["replacement_cost"] is None:
        rp = await retailer_price(client, book, country, metrics)
        if rp:
            out["replacement_cost"] = {**rp, "currency": cur, "retrieved_at": now_iso(), "converted": False}
    if out["replacement_cost"] is None:
        out["notes"].append("no new-price source found")
    if out["used_value"] is None:
        out["notes"].append("no used-price source found")
    return out


# ---------- items ----------

def _item_crop(run_dir: Path, item: dict) -> bytes:
    img = cv2.imread(str(run_dir / item["frame_ref"]))
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
    if not ebay.configured:
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
    listings = await ebay.search_by_image(country, _item_crop(run_dir, item))
    method = "eBay image search on the item crop"
    if len(listings) < 5:
        desc = f"{item.get('material', '')} {item['category'].replace('_', ' ')} {item.get('description', '')}"
        if item.get("is_print"):
            desc = f"framed print {item.get('description', '')}"
        listings = await ebay.search(country, condition="NEW", q=re.sub(r"\s+", " ", desc).strip())
        method = f"eBay keyword search '{desc.strip()[:60]}'"
    rng = range_figure(listings, cur)
    if not rng:
        return {"status": "needs_appraisal", "replacement_cost": None, "reason": "fewer than 5 comparable new listings found"}
    return {
        "status": "range",
        "replacement_cost": {
            "low": rng["low"], "high": rng["high"], "currency": cur,
            "source": f"{method}; 25th-75th percentile of {rng['n_listings']} new listings (asking, incl. shipping)",
            "url": rng["url"], "evidence_urls": rng["evidence_urls"], "retrieved_at": stamp,
        },
    }


async def run(run_dir: Path, books: list[dict], items: list[dict], overrides: dict, metrics: Metrics, country: str) -> dict:
    """books: [{id, title, author, isbn, publisher, published, read}] identified only."""
    with metrics.time(STAGE):
        async with httpx.AsyncClient(follow_redirects=True, timeout=12) as client:
            ebay, fx = Ebay(client, metrics), FX(client)

            async def one_book(b):
                reasons = appraisal_reasons(b, b.get("read", {}))
                p = await price_book(ebay, fx, client, b, country, metrics)
                for key in ("replacement_cost", "used_value"):
                    if p[key] and p[key]["amount"] >= threshold():
                        reasons.append(f"{key.replace('_', ' ')} {p[key]['amount']} {p[key]['currency']} >= appraisal threshold {threshold()}")
                if p["replacement_cost"] and p["used_value"] and p["used_value"]["amount"] >= 2 * p["replacement_cost"]["amount"]:
                    reasons.append("used asking price is at least twice the new price (possible collectible)")
                return b["id"], {**p, "appraisal_reasons": reasons}

            book_prices = dict(await asyncio.gather(*(one_book(b) for b in books)))
            item_prices = dict(
                await asyncio.gather(
                    *(
                        _pair(it["id"], price_item(run_dir, ebay, it, country, overrides.get("items", {}).get(it["id"], {})))
                        for it in items
                    )
                )
            )
    out = {"country": country, "currency": locale(country)["currency"], "books": book_prices, "items": item_prices}
    write_json(run_dir / f"prices_{country}.json", out)
    return out


async def _pair(key, coro):
    return key, await coro
