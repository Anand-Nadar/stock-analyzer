"""Manual portfolio tracking from a transaction ledger (FIFO lots).

Taxes are NOT calculated (Indian STCG/LTCG rules depend on holding period, grandfathering and your situation) - use a tax professional.
"""
from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path

from .config import SETTINGS, Settings


def compute_book(txns: list[dict]) -> dict:
    """Pure FIFO engine. `txns` need: symbol, trade_date, side, quantity, price, fees (sorted by date/id).
    Buy fees are added to the lot cost; sell fees reduce proceeds. Raises ValueError on overselling."""
    lots: dict[str, deque] = defaultdict(deque)       # symbol -> deque of [qty, cost_per_share, buy_date]
    realized = []
    for t in txns:
        sym, qty = t["symbol"], int(t["quantity"])
        fees = float(t.get("fees") or 0.0)
        if t["side"] == "BUY":
            lots[sym].append([qty, (qty * t["price"] + fees) / qty, t["trade_date"]])
        else:
            if sum(l[0] for l in lots[sym]) < qty:
                raise ValueError(f"cannot sell {qty} {sym} on {t['trade_date']}: only {sum(l[0] for l in lots[sym])} held")
            remaining, cost, bought_dates = qty, 0.0, []
            while remaining:
                lot = lots[sym][0]
                take = min(remaining, lot[0])
                cost += take * lot[1]
                bought_dates.append((lot[2], take))
                lot[0] -= take
                remaining -= take
                if lot[0] == 0:
                    lots[sym].popleft()
            proceeds = qty * t["price"] - fees
            realized.append({"symbol": sym, "sell_date": t["trade_date"], "quantity": qty, "proceeds": round(proceeds, 2),
                             "cost": round(cost, 2), "pnl": round(proceeds - cost, 2), "lots": bought_dates})
    return {"lots": {k: [list(l) for l in v] for k, v in lots.items() if v}, "realized": realized}


def _ledger(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""SELECT t.*, s.symbol FROM portfolio_transactions t JOIN stocks s ON s.id=t.stock_id
                           ORDER BY t.trade_date, t.id""").fetchall()
    return [dict(r) for r in rows]


def add_transaction(conn: sqlite3.Connection, symbol: str, side: str, quantity: int, price: float,
                    trade_date: str | None = None, fees: float = 0.0, notes: str = "") -> int:
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    if quantity <= 0 or price <= 0 or fees < 0:
        raise ValueError("quantity and price must be positive, fees non-negative")
    trade_date = trade_date or date.today().isoformat()
    datetime.strptime(trade_date, "%Y-%m-%d")
    st = conn.execute("SELECT id FROM stocks WHERE symbol=? AND security_type='EQUITY'", (symbol.upper(),)).fetchone()
    if not st:
        raise ValueError(f"{symbol.upper()} not in stock master (run load-master)")
    # validate the whole ledger (including back-dated entries) before writing
    compute_book(sorted(_ledger(conn) + [{"symbol": symbol.upper(), "trade_date": trade_date, "side": side, "quantity": quantity,
                                          "price": price, "fees": fees, "id": 10**12}], key=lambda t: (t["trade_date"], t["id"])))
    cur = conn.execute("INSERT INTO portfolio_transactions (stock_id, trade_date, side, quantity, price, fees, notes) VALUES (?,?,?,?,?,?,?)",
                       (st["id"], trade_date, side, quantity, price, fees, notes))
    conn.commit()
    return cur.lastrowid


def import_transactions_csv(conn: sqlite3.Connection, path: Path | str) -> tuple[int, list[str]]:
    """CSV columns: symbol, date, side, quantity, price, [fees], [notes]"""
    n, errs = 0, []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            try:
                add_transaction(conn, row["symbol"], row["side"], int(float(row["quantity"])), float(row["price"]),
                                row.get("date") or row.get("trade_date"), float(row.get("fees") or 0), row.get("notes", ""))
                n += 1
            except (ValueError, KeyError) as exc:
                errs.append(f"line {i}: {exc}")
    return n, errs


def portfolio_summary(conn: sqlite3.Connection, s: Settings = SETTINGS, today: date | None = None) -> dict:
    today = today or date.today()
    book = compute_book(_ledger(conn))
    positions, total_value, total_cost = [], 0.0, 0.0
    for sym, lots in book["lots"].items():
        st = conn.execute("SELECT id, sector FROM stocks WHERE symbol=?", (sym,)).fetchone()
        last = conn.execute("SELECT date, close FROM daily_prices WHERE stock_id=? ORDER BY date DESC LIMIT 1", (st["id"],)).fetchone()
        qty = sum(l[0] for l in lots)
        cost = sum(l[0] * l[1] for l in lots)
        price, pdate = (round(last["close"], 2), last["date"]) if last else (None, None)
        value = qty * price if price is not None else None
        positions.append({"symbol": sym, "sector": st["sector"] or "Unclassified", "quantity": qty, "avg_cost": round(cost / qty, 2), "cost": round(cost, 2),
                          "price": price, "price_date": pdate, "value": round(value, 2) if value is not None else None,
                          "unrealized_pnl": round(value - cost, 2) if value is not None else None,
                          "unrealized_pct": round((value / cost - 1) * 100, 2) if value is not None else None,
                          "first_buy": min(l[2] for l in lots)})
        total_cost += cost
        total_value += value if value is not None else cost       # unpriced positions valued at cost
    sectors: dict[str, float] = defaultdict(float)
    warnings = []
    for p in positions:
        v = p["value"] if p["value"] is not None else p["cost"]
        p["weight_pct"] = round(v / total_value * 100, 1) if total_value else 0.0
        sectors[p["sector"]] += v
        if p["weight_pct"] > s.max_position_pct:
            warnings.append(f"{p['symbol']} is {p['weight_pct']}% of the portfolio (limit {s.max_position_pct:.0f}%)")
        if p["price_date"] is None:
            warnings.append(f"{p['symbol']}: no price data, valued at cost")
        elif (today - date.fromisoformat(p["price_date"])).days > s.max_stale_days:
            warnings.append(f"{p['symbol']}: price is stale ({p['price_date']})")
    sector_pct = {k: round(v / total_value * 100, 1) for k, v in sorted(sectors.items(), key=lambda kv: -kv[1])} if total_value else {}
    for k, v in sector_pct.items():
        if v > s.max_sector_pct and k != "Unclassified":
            warnings.append(f"sector {k} is {v}% of the portfolio (limit {s.max_sector_pct:.0f}%)")
    if "Unclassified" in sector_pct:
        warnings.append("some holdings have no sector (use `import-sectors`), so sector concentration is understated")
    realized = sum(r["pnl"] for r in book["realized"])
    return {"positions": sorted(positions, key=lambda p: -(p["value"] or p["cost"])), "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2), "unrealized_pnl": round(total_value - total_cost, 2),
            "realized_pnl": round(realized, 2), "sector_pct": sector_pct, "realized": book["realized"], "warnings": warnings}
