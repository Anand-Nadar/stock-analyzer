from datetime import date

import pandas as pd

from stockpilot import db
from stockpilot.data import universe as uni
from stockpilot.data.prices import load_price_frame, store_prices
from stockpilot.data.validation import clean_price_frame, find_gaps, freshness_report

CSV = """SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE
RELIANCE,Reliance Industries Limited,EQ,29-NOV-1995,10,1,INE002A01018,10
TCS,Tata Consultancy Services Limited,EQ,25-AUG-2004,1,1,INE467B01029,1
SOMEBE,Some BE Co,BE,01-JAN-2010,10,1,INE000000000,10
"""


def _frame(n=5, start="2025-01-01"):
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "adj_close": 10.5, "volume": 1000}, index=idx)


def test_clean_price_frame_drops_bad_rows():
    df = _frame()
    df.iloc[1, df.columns.get_loc("high")] = 5.0      # high below close -> inconsistent
    df.iloc[2, df.columns.get_loc("close")] = float("nan")
    clean, issues = clean_price_frame(df)
    assert len(clean) == 3 and issues


def test_parse_master_filters_series():
    df = uni.parse_equity_list(CSV)
    assert list(df["symbol"]) == ["RELIANCE", "TCS"]


def test_db_roundtrip_universe_freshness_gaps():
    conn = db.connect(":memory:")
    db.init_db(conn)
    uni.upsert_stocks(conn, uni.parse_equity_list(CSV))
    uni.upsert_stocks(conn, uni.parse_equity_list(CSV))          # idempotent
    assert conn.execute("SELECT COUNT(*) c FROM stocks").fetchone()["c"] == 2
    n, unknown = uni.set_universe(conn, ["RELIANCE", "NOPE"])
    assert n == 1 and unknown == ["NOPE"]

    sid = conn.execute("SELECT id FROM stocks WHERE symbol='RELIANCE'").fetchone()["id"]
    store_prices(conn, sid, _frame(5, "2025-01-01"))
    store_prices(conn, sid, _frame(5, "2025-01-01"))              # duplicates upsert, not duplicate
    assert len(load_price_frame(conn, sid)) == 5

    rep = freshness_report(conn, today=date(2025, 1, 20))
    assert rep.symbols_stale == ["RELIANCE"]
    store_prices(conn, sid, _frame(3, "2025-02-10"))
    assert find_gaps(conn, "RELIANCE") == [("2025-01-07", "2025-02-10")]


INDEX_CSV = """Company Name,Industry,Symbol,Series,ISIN Code
Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018
HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01034
Gone Ltd.,Services,GONE,EQ,INE000000001
"""


def test_index_constituents_set_universe_and_sector():
    conn = db.connect(":memory:")
    db.init_db(conn)
    uni.upsert_stocks(conn, uni.parse_equity_list(CSV + "HDFCBANK,HDFC Bank,EQ,01-JAN-1995,2,1,INE040A01034,2\n"))
    df = uni.parse_index_constituents(INDEX_CSV)
    n, unknown, sectors = uni.apply_index_universe(conn, df)
    assert n == 2 and unknown == ["GONE"] and sectors == 2
    row = conn.execute("SELECT sector, in_universe FROM stocks WHERE symbol='HDFCBANK'").fetchone()
    assert row["sector"] == "Financial Services" and row["in_universe"] == 1
    assert conn.execute("SELECT in_universe FROM stocks WHERE symbol='TCS'").fetchone()["in_universe"] == 0


def test_backfill_only_when_years_explicit(monkeypatch=None):
    from datetime import timedelta
    from stockpilot.data import prices as P
    conn = db.connect(":memory:")
    db.init_db(conn)
    uni.upsert_stocks(conn, uni.parse_equity_list(CSV))
    uni.set_universe(conn, ["RELIANCE"])
    sid = conn.execute("SELECT id FROM stocks WHERE symbol='RELIANCE'").fetchone()["id"]
    start = pd.Timestamp(date.today() - timedelta(days=100))
    P.store_prices(conn, sid, _frame(60, start.date().isoformat()))
    seen = []
    orig = P.fetch_history
    P.fetch_history = lambda tk, st, end=None: (seen.append(st), pd.DataFrame())[1]
    try:
        P.update_prices(conn, include_benchmark=False, pause=0)                 # incremental
        P.update_prices(conn, include_benchmark=False, years=3, pause=0)        # explicit -> backfill
    finally:
        P.fetch_history = orig
    last = conn.execute("SELECT MAX(date) d FROM daily_prices").fetchone()["d"]
    assert (date.fromisoformat(last) - seen[0]).days == 5
    assert (date.today() - seen[1]).days in (365 * 3, 365 * 3 + 1)
