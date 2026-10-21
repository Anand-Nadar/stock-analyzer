from stockpilot import db
from stockpilot.ai import facts as F, filings, report as R
from stockpilot.config import Settings
from stockpilot.data import universe as uni
from stockpilot.data.prices import store_prices
from stockpilot.fundamentals.importer import upsert_statement
import test_swing as ts


def _world():
    conn = db.connect(":memory:"); db.init_db(conn)
    uni.upsert_stocks(conn, uni.parse_equity_list("SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER\nAAA,A Ltd,EQ,X1\nBBB,B Ltd,EQ,X2\n"))
    sid = conn.execute("SELECT id FROM stocks WHERE symbol='AAA'").fetchone()["id"]
    store_prices(conn, sid, ts._series())
    for yr, rev, np_ in [(2023, 10000, 1200), (2024, 11500, 1450), (2025, 13000, 1700)]:
        upsert_statement(conn, sid, f"{yr}-03-31", "A",
                         dict(revenue=rev, operating_profit=rev * .2, ebitda=rev * .25, interest_expense=rev * .01, net_profit=np_,
                              total_assets=rev * 1.5, current_liabilities=rev * .3, total_debt=rev * .3, equity=rev * .8,
                              cash=rev * .1, operating_cash_flow=np_ * 1.1, shares_outstanding=50.0), "test", 1)
    conn.commit()
    return conn


def test_facts_cover_all_sections_and_flag_missing():
    conn = _world()
    f = F.build_facts(conn, "AAA", Settings(), today=__import__("datetime").date(2025, 1, 6))
    assert {"price", "swing", "long_term"} <= set(f) and f["long_term"]["metrics"]["pe"] is not None
    g = F.build_facts(conn, "BBB", Settings())
    assert "price" not in g and "long_term" not in g and len(g["missing"]) == 2
    text = R.plain_report(f)
    assert "Swing screen" in text and "Long-term screen" in text and "not investment advice" in text


def test_flatten_ids_unique_and_sequential():
    flat = F.flatten_facts({"a": 1, "b": {"c": 2.5, "d": [3, "x"]}, "n": None})
    assert [x["id"] for x in flat] == ["F1", "F2", "F3", "F4"]
    assert flat[1]["path"] == "b.c"


def test_verification_catches_invented_numbers_and_bad_citations():
    flat = F.flatten_facts({"price": {"close": 482.35, "rsi14": 61.2}, "pe": 14.7})
    good = "Close is 482.35 [F1] with RSI 61.2 [F2]; P/E of 14.7 [F3]."
    v = R.verify_report(good, flat)
    assert not v["unverified_numbers"] and not v["unknown_citations"] and v["has_citations"]
    bad = "The stock will reach 650 soon [F9]. Close 482.4 [F1]. Price target 700."
    v = R.verify_report(bad, flat)
    assert 650.0 in v["unverified_numbers"] and 700.0 in v["unverified_numbers"] and 482.4 not in v["unverified_numbers"]
    assert v["unknown_citations"] == ["F9"] and v["forbidden_phrases"]
    assert R.verification_notes(v)


def test_llm_report_uses_facts_and_verifies(monkeypatch=None):
    conn = _world()
    facts = F.build_facts(conn, "AAA", Settings())
    seen = {}

    def fake_chat(messages, model=None, temperature=0.2, s=None):
        seen["user"] = messages[-1]["content"]
        return "Close was 999.99 [F1]."          # invented number

    orig = R.ollama.chat
    R.ollama.chat = fake_chat
    try:
        out = R.llm_report(facts)
    finally:
        R.ollama.chat = orig
    assert "F1:" in seen["user"] and 999.99 in out["verification"]["unverified_numbers"] and out["notes"]


def test_filing_chunking_and_number_check():
    text = ("Revenue was Rs 12,450.5 crore in FY25.\n\n" * 400)
    chunks = filings.chunk_text(text, 3000)
    assert len(chunks) > 3 and all(len(c) <= 3100 for c in chunks)
    assert filings.unverified_numbers("Revenue 12,450.5 crore; profit 3,333 crore.", text) == ["3333"]
