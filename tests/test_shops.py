"""Shop page parsers, on JSON-LD captured from the real pages (2026-10-16)."""

from pathlib import Path

from pipeline import shops

FX = Path(__file__).parent / "fixtures"


def test_abebooks_us_conditions_and_mislisted_isbn():
    html = (FX / "abebooks_us_9780316066525.html").read_text()
    offers = shops.parse_abebooks(html, "9780316066525", "Infinite Jest", "USD")
    assert offers and all(o["url"].startswith("https://www.abebooks.com/") for o in offers)
    assert not any("Country Towns" in o["url"] for o in offers)  # same ISBN, different book: dropped
    assert {o["condition"] for o in offers} == {"new", "used"}
    used = shops.used_figure(offers)
    assert 8 <= used["amount"] <= 20 and used["n"] >= 20


def test_abebooks_uk_title_variants_match():
    html = (FX / "abebooks_uk_9780261102217.html").read_text()
    offers = shops.parse_abebooks(html, "9780261102217", "The Hobbit", "GBP")
    assert len(offers) >= 25  # 'Hobbit', 'THE HOBBIT', 'Hobbit : Or There and Back Again' all count
    assert shops.parse_abebooks(html, "9780261102217", "The Hobbit", "USD") == []  # never mix currencies


def test_bn_picks_the_exact_isbn_variant():
    html = (FX / "bn_9780316066525.html").read_text()
    assert shops.parse_bn(html, "9780316066525")["amount"] == 24.99  # paperback, not the $40 hardcover
    assert shops.parse_bn(html, "9780316073851") is None  # the ebook variant is not a print price


def test_new_figure_needs_agreement():
    o = lambda a: {"amount": a, "condition": "new", "url": "u", "seller": "s"}
    assert shops.new_figure([o(10)]) is None
    assert shops.new_figure([o(10), o(20)]) is None
    assert shops.new_figure([o(10), o(11)])["amount"] == 10
    assert shops.new_figure([o(10), o(11), o(12), o(300)])["amount"] == 11
