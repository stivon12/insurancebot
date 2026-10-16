"""Book prices keyed by ISBN, read from shops' own schema.org product data (JSON-LD), cited by the listing's URL.

  - Barnes & Noble (US, new): /s/<isbn> redirects to the product page; its ProductGroup lists every format, and the
    variant whose offer URL carries ean=<isbn> is this exact edition's list price.
  - AbeBooks (US .com, UK .co.uk; new and used): the ISBN search page lists ~30 offers, each with price, currency,
    NewCondition/UsedCondition, format, seller and listing URL. Seller prices exclude shipping.

Both pages are allowed by the sites' robots.txt; requests are few (one or two per book), spaced per host, and send
an identifying User-Agent. Prices are never cached: each carries its own retrieval time.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import httpx
from rapidfuzz import fuzz

from .common import note_failure

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
