import sqlite3
from datetime import date

from stockpilot import db
from stockpilot.config import Settings
from stockpilot.data import universe as uni
from stockpilot.fundamentals.importer import import_csv, mark_validated
from stockpilot.fundamentals.scoring import compute_metrics, is_financial, score_long_term

S = Settings()


def _stmt(year, rev, np_, **kw):
    base = dict(period_end=f"{year}-03-31", period_type="A", revenue=rev, operating_profit=rev * 0.2, ebitda=rev * 0.25,
                interest_expense=rev * 0.01, net_profit=np_, total_assets=rev * 1.5, total_liabilities=rev * 0.7,
                current_liabilities=rev * 0.3, total_debt=rev * 0.3, equity=rev * 0.8, cash=rev * 0.1,
                operating_cash_flow=np_ * 1.1, shares_outstanding=50.0, validated=1)
    base.update(kw)
    return base


GOOD = [_stmt(2023, 10000, 1200), _stmt(2024, 11500, 1450), _stmt(2025, 13000, 1700)]


def test_metrics_known_values():
    m = compute_metrics(GOOD, price=300.0)
    assert abs(m["revenue_cagr"] - ((13000 / 10000) ** 0.5 - 1) * 100) < 1e-6
    assert abs(m["roe"] - 1700 / (13000 * 0.8) * 100) < 1e-6
    assert abs(m["pe"] - 300.0 / (1700 / 50)) < 1e-6           # EPS = 34 -> P/E 8.82
    assert m["revenue_up_frac"] == 1.0 and m["profit_pos_frac"] == 1.0


def test_healthy_company_is_candidate():
    m = compute_metrics(GOOD, price=500.0)                     # P/E ~14.7
    r = score_long_term("GOOD", m, {"excess_12m": 10.0, "above_200dma": True}, None, False, date(2025, 9, 1), S)
    assert r["status"] == "CANDIDATE" and r["score"] >= 60 and r["confidence"] == 1.0


def test_missing_data_lowers_confidence_not_score():
    full = score_long_term("A", compute_metrics(GOOD, 500.0), {}, None, False, date(2025, 9, 1), S)
    no_val = [dict(s, shares_outstanding=None, ebitda=None) for s in GOOD]
    part = score_long_term("A", compute_metrics(no_val, 500.0), {}, None, False, date(2025, 9, 1), S)
    assert part["pillars"]["valuation"]["points"] is None
    assert part["confidence"] < full["confidence"]
    assert part["score"] > 40                                   # not dragged to zero by the missing pillar


def test_serious_flag_blocks_candidate_even_with_high_score():
    levered = [dict(s, total_debt=s["equity"] * 3.0) for s in GOOD]    # debt/equity 3
    r = score_long_term("LEV", compute_metrics(levered, 500.0), {"excess_12m": 20.0, "above_200dma": True}, None, False, date(2025, 9, 1), S)
    assert r["serious_flags"] and r["status"] == "WATCH_SERIOUS_WARNING"


def test_loss_and_stale_and_no_data():
    loss = GOOD[:2] + [_stmt(2025, 13000, -300)]
    r = score_long_term("L", compute_metrics(loss, 500.0), {}, None, False, date(2025, 9, 1), S)
    assert "latest year net loss" in r["serious_flags"]
    stale = score_long_term("S", compute_metrics(GOOD, 500.0), {}, None, False, date(2027, 6, 1), S)
    assert any("stale" in f for f in stale["serious_flags"])
    assert score_long_term("N", compute_metrics([], None), {}, None, False, date(2025, 9, 1), S)["status"] == "NO_DATA"


def test_financials_skip_leverage():
    assert is_financial("Financial Services") and not is_financial("Technology")
    bank = [dict(s, total_debt=s["equity"] * 8.0) for s in GOOD]
    r = score_long_term("BANK", compute_metrics(bank, 500.0), {}, None, True, date(2025, 9, 1), S)
    assert not any("leverage" in f for f in r["serious_flags"])
    assert r["pillars"]["financial_health"]["points"] is None


def test_csv_import_and_migration(tmp_path):
    # an "old" database without the new columns must be migrated in place
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("CREATE TABLE financial_statements (id INTEGER PRIMARY KEY, stock_id INTEGER NOT NULL, period_end TEXT NOT NULL,"
                      " period_type TEXT NOT NULL, revenue REAL, net_profit REAL, total_assets REAL, total_liabilities REAL, total_debt REAL,"
                      " operating_cash_flow REAL, source TEXT, validated INTEGER NOT NULL DEFAULT 0, UNIQUE (stock_id, period_end, period_type));")
    old.commit(); old.close()
    conn = db.connect(path); conn.row_factory = sqlite3.Row
    db.init_db(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(financial_statements)")}
    assert {"equity", "shares_outstanding", "operating_profit"} <= cols

    conn = db.connect(":memory:"); db.init_db(conn)
    uni.upsert_stocks(conn, uni.parse_equity_list("SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER\nAAA,A,EQ,X\n"))
    csv = tmp_path / "f.csv"
    csv.write_text("symbol,period_end,revenue,net_profit,equity,shares_outstanding\nAAA,2025-03-31,\"1,000\",100,500,10\nZZZ,2025-03-31,1,1,1,1\nAAA,bad,1,1,1,1\n")
    n, errs = import_csv(conn, csv)
    assert n == 1 and len(errs) == 2
    row = conn.execute("SELECT revenue, validated FROM financial_statements").fetchone()
    assert row["revenue"] == 1000.0 and row["validated"] == 1
    assert mark_validated(conn, "AAA") == 1
