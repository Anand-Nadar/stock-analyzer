import json
from datetime import date

import pandas as pd

from stockpilot import db
from stockpilot.config import Settings
from stockpilot.data import universe as uni
from stockpilot.data.prices import store_prices
from stockpilot.paper import open_paper_trade, paper_report, update_paper_trades
from stockpilot.portfolio import add_transaction, compute_book, portfolio_summary


def _t(side, qty, price, d, fees=0.0, sym="AAA"):
    return {"symbol": sym, "trade_date": d, "side": side, "quantity": qty, "price": price, "fees": fees}


def test_fifo_realized_and_remaining_lots():
    book = compute_book([_t("BUY", 10, 100, "2025-01-01"), _t("BUY", 10, 120, "2025-02-01"), _t("SELL", 15, 130, "2025-03-01", fees=15)])
    r = book["realized"][0]
    assert r["cost"] == 10 * 100 + 5 * 120 and r["proceeds"] == 15 * 130 - 15
    assert r["pnl"] == (1950 - 15) - 1600
    assert book["lots"]["AAA"] == [[5, 120.0, "2025-02-01"]]


def test_buy_fees_raise_cost_basis_and_oversell_rejected():
    book = compute_book([_t("BUY", 10, 100, "2025-01-01", fees=50)])
    assert book["lots"]["AAA"][0][1] == 105.0
    try:
        compute_book([_t("BUY", 5, 100, "2025-01-01"), _t("SELL", 6, 100, "2025-01-02")])
        assert False, "should raise"
    except ValueError:
        pass


def _db_with_prices():
    conn = db.connect(":memory:"); db.init_db(conn)
    uni.upsert_stocks(conn, uni.parse_equity_list("SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER\nAAA,A,EQ,X1\nBBB,B,EQ,X2\n"))
    conn.execute("UPDATE stocks SET sector='IT' WHERE symbol='AAA'"); conn.commit()
    return conn


def _bars(start, closes, high_pad=1.0, low_pad=1.0):
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + high_pad, "low": c - low_pad, "close": c, "adj_close": c, "volume": 1000})


def test_summary_and_ledger_validation():
    conn = _db_with_prices()
    ids = {r["symbol"]: r["id"] for r in conn.execute("SELECT id, symbol FROM stocks")}
    store_prices(conn, ids["AAA"], _bars("2025-01-01", [110.0] * 5))
    store_prices(conn, ids["BBB"], _bars("2025-01-01", [50.0] * 5))
    add_transaction(conn, "AAA", "BUY", 100, 100.0, "2025-01-01")
    add_transaction(conn, "BBB", "BUY", 100, 40.0, "2025-01-01")
    add_transaction(conn, "AAA", "SELL", 40, 105.0, "2025-01-03")
    try:                                   # a back-dated sell before the buy must be refused, nothing written
        add_transaction(conn, "BBB", "SELL", 10, 50.0, "2024-12-01")
        assert False
    except ValueError:
        pass
    assert conn.execute("SELECT COUNT(*) c FROM portfolio_transactions").fetchone()["c"] == 3
    p = portfolio_summary(conn, Settings(max_position_pct=50.0), today=date(2025, 1, 8))
    aaa = next(x for x in p["positions"] if x["symbol"] == "AAA")
    assert aaa["quantity"] == 60 and aaa["avg_cost"] == 100.0 and aaa["unrealized_pnl"] == 600.0
    assert p["realized_pnl"] == 200.0
    assert abs(sum(x["weight_pct"] for x in p["positions"]) - 100) < 0.2
    assert any("Unclassified" in w or "no sector" in w for w in p["warnings"])     # BBB has no sector


def _seed_run(conn, symbol, entry, stop, target, as_of, setup="BREAKOUT"):
    cand = {"symbol": symbol, "setup": setup, "score": 70, "plan": {"entry": entry, "stop_loss": stop, "target": target,
            "position": {"quantity": 10}}}
    conn.execute("INSERT INTO recommendation_runs (engine, data_as_of, config_json, results_json) VALUES ('SWING', ?, '{}', ?)",
                 (as_of, json.dumps({"candidates": [cand], "rejected": []})))
    conn.commit()


def test_paper_trade_lifecycle():
    conn = _db_with_prices()
    sid = conn.execute("SELECT id FROM stocks WHERE symbol='AAA'").fetchone()["id"]
    bars = _bars("2025-01-01", [100, 101, 103, 106, 111, 112])          # high = close+1 -> target 110 hit on 5th bar
    store_prices(conn, sid, bars)
    _seed_run(conn, "AAA", 100.0, 95.0, 110.0, "2025-01-01")
    t = open_paper_trade(conn, "AAA")
    assert t["qty"] == 10
    try:
        open_paper_trade(conn, "AAA"); assert False
    except ValueError:
        pass
    closed = update_paper_trades(conn, Settings(bt_max_hold_days=20))
    assert closed and closed[0]["reason"] == "TARGET"
    rep = paper_report(conn, Settings())
    assert rep["closed"] == 1 and rep["trades"][0]["net_pnl"] > 0


def test_near_breakout_cannot_be_paper_traded():
    conn = _db_with_prices()
    _seed_run(conn, "AAA", 100.0, 95.0, 110.0, "2025-01-01", setup="NEAR_BREAKOUT")
    try:
        open_paper_trade(conn, "AAA"); assert False
    except ValueError as e:
        assert "near-breakout" in str(e)
