"""Paper trading: simulated entries from a saved recommendation run, exited by the same rules as the backtester."""
from __future__ import annotations

import json
import sqlite3

from .config import SETTINGS, Settings
from .data.prices import load_price_frame
from .trade_sim import find_exit


def open_paper_trade(conn: sqlite3.Connection, symbol: str, run_id: int | None = None, quantity: int | None = None) -> dict:
    row = (conn.execute("SELECT * FROM recommendation_runs WHERE id=? AND engine='SWING'", (run_id,)).fetchone() if run_id
           else conn.execute("SELECT * FROM recommendation_runs WHERE engine='SWING' ORDER BY id DESC LIMIT 1").fetchone())
    if not row:
        raise ValueError("no swing screening run found; run `screen-swing` first")
    cand = next((c for c in json.loads(row["results_json"])["candidates"] if c["symbol"] == symbol.upper()), None)
    if not cand:
        raise ValueError(f"{symbol.upper()} is not a candidate in run #{row['id']}")
    if cand["setup"] != "BREAKOUT":
        raise ValueError(f"{symbol.upper()} is a near-breakout (buy-stop trigger {cand['plan']['entry']}); paper-trade it after it actually breaks out")
    st = conn.execute("SELECT id FROM stocks WHERE symbol=?", (symbol.upper(),)).fetchone()
    if conn.execute("SELECT 1 FROM paper_trades WHERE stock_id=? AND closed_at IS NULL", (st["id"],)).fetchone():
        raise ValueError(f"{symbol.upper()} already has an open paper trade")
    p = cand["plan"]
    qty = quantity if quantity is not None else p["position"]["quantity"]
    if qty < 1:
        raise ValueError("quantity must be at least 1")
    cur = conn.execute("INSERT INTO paper_trades (stock_id, opened_at, entry_price, quantity, stop_loss, target, recommendation_run_id) VALUES (?,?,?,?,?,?,?)",
                       (st["id"], row["data_as_of"], p["entry"], qty, p["stop_loss"], p["target"], row["id"]))
    conn.commit()
    return {"id": cur.lastrowid, "symbol": symbol.upper(), "entry": p["entry"], "qty": qty, "stop": p["stop_loss"], "target": p["target"]}


def update_paper_trades(conn: sqlite3.Connection, s: Settings = SETTINGS) -> list[dict]:
    """Check every open paper trade against bars after its entry date and close those that hit stop/target/time."""
    closed = []
    for t in conn.execute("SELECT t.*, s.symbol FROM paper_trades t JOIN stocks s ON s.id=t.stock_id WHERE t.closed_at IS NULL").fetchall():
        df = load_price_frame(conn, t["stock_id"])
        bars = df[df.index > t["opened_at"]]
        if bars.empty:
            continue
        ex = find_exit(bars["open"].to_numpy(), bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy(),
                       t["stop_loss"], t["target"], s.bt_max_hold_days)
        if ex:
            k, price, reason = ex
            d = bars.index[k].date().isoformat()
            conn.execute("UPDATE paper_trades SET closed_at=?, exit_price=?, exit_reason=? WHERE id=?", (d, price, reason, t["id"]))
            closed.append({"symbol": t["symbol"], "closed_at": d, "exit": price, "reason": reason})
    conn.commit()
    return closed


def paper_report(conn: sqlite3.Connection, s: Settings = SETTINGS) -> dict:
    rows = conn.execute("SELECT t.*, s.symbol FROM paper_trades t JOIN stocks s ON s.id=t.stock_id ORDER BY t.id").fetchall()
    out, net_total, wins, n_closed = [], 0.0, 0, 0
    for t in rows:
        df = load_price_frame(conn, t["stock_id"])
        last = float(df["close"].iloc[-1]) if not df.empty else t["entry_price"]
        exit_px = t["exit_price"] if t["closed_at"] else last
        cost = t["quantity"] * t["entry_price"] * s.round_trip_cost_pct / 100.0
        pnl = t["quantity"] * (exit_px - t["entry_price"]) - cost
        out.append({"id": t["id"], "symbol": t["symbol"], "opened_at": t["opened_at"], "entry": t["entry_price"], "qty": t["quantity"],
                    "stop": t["stop_loss"], "target": t["target"], "status": t["exit_reason"] if t["closed_at"] else "OPEN",
                    "closed_at": t["closed_at"], "mark_or_exit": round(exit_px, 2), "net_pnl": round(pnl, 2)})
        if t["closed_at"]:
            n_closed += 1
            net_total += pnl
            wins += pnl > 0
    return {"trades": out, "closed": n_closed, "win_rate_pct": round(wins / n_closed * 100, 1) if n_closed else None,
            "closed_net_pnl": round(net_total, 2)}
