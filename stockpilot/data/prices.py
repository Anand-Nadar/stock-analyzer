"""Daily OHLCV download (yfinance - an UNOFFICIAL source) and storage."""
from __future__ import annotations

import sqlite3
import time
from datetime import date, timedelta

import pandas as pd

from ..config import SETTINGS
from ..db import log_refresh
from .validation import clean_price_frame


def fetch_history(yf_ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
    """Fetch daily bars for one ticker. Returns a frame indexed by date (naive)."""
    import yfinance as yf  # imported lazily so the rest of the app works without it

    raw = yf.Ticker(yf_ticker).history(start=start.isoformat(), end=end.isoformat() if end else None,
                                       auto_adjust=False, actions=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df[["open", "high", "low", "close", "adj_close", "volume"]]


def store_prices(conn: sqlite3.Connection, stock_id: int, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [
        (stock_id, idx.date().isoformat(), float(r.open), float(r.high), float(r.low), float(r.close),
         None if pd.isna(r.adj_close) else float(r.adj_close),
         0 if pd.isna(r.volume) else int(r.volume))
        for idx, r in df.iterrows()
    ]
    conn.executemany(
        """INSERT INTO daily_prices (stock_id, date, open, high, low, close, adj_close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(stock_id, date) DO UPDATE SET
             open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
             adj_close=excluded.adj_close, volume=excluded.volume, fetched_at=datetime('now')""",
        rows,
    )
    conn.commit()
    return len(rows)


def update_prices(conn: sqlite3.Connection, symbols: list[str] | None = None, include_benchmark: bool = True,
                  years: int | None = None, pause: float = 0.4, retries: int = 2) -> dict:
    """Incrementally download prices for the universe (or the given symbols).
    Passing `years` explicitly also back-fills older history for stocks you already have."""
    backfill = years is not None    # an explicit --years means: make sure history goes back that far
    years = years or SETTINGS.price_history_years
    if symbols:
        q = f"SELECT id, symbol, yf_ticker FROM stocks WHERE symbol IN ({','.join('?' * len(symbols))})"
        targets = conn.execute(q, [s.upper() for s in symbols]).fetchall()
    else:
        targets = conn.execute("SELECT id, symbol, yf_ticker FROM stocks WHERE in_universe=1").fetchall()
    if include_benchmark:
        targets += conn.execute("SELECT id, symbol, yf_ticker FROM stocks WHERE symbol=?",
                                (SETTINGS.benchmark_symbol,)).fetchall()

    summary = {"ok": 0, "failed": [], "empty": [], "rows": 0, "issues": {}}
    for t in targets:
        row = conn.execute("SELECT MIN(date) a, MAX(date) d FROM daily_prices WHERE stock_id=?", (t["id"],)).fetchone()
        first, last = row["a"], row["d"]
        wanted = date.today() - timedelta(days=365 * years)
        if last and not (backfill and first and date.fromisoformat(first) > wanted + timedelta(days=30)):
            # incremental: re-fetch the last 5 days so late corrections/adjustments overwrite stored values
            start = date.fromisoformat(last) - timedelta(days=5)
        else:
            start = wanted      # first download, or a longer history was requested (backfill)
        df = pd.DataFrame()
        for attempt in range(retries + 1):
            try:
                df = fetch_history(t["yf_ticker"], start)
                break
            except Exception as exc:  # network / rate-limit errors
                if attempt == retries:
                    summary["failed"].append(f"{t['symbol']}: {exc}")
                time.sleep(pause * (attempt + 2))
        if df.empty:
            if t["symbol"] not in [f.split(":")[0] for f in summary["failed"]]:
                summary["empty"].append(t["symbol"])
            continue
        df, issues = clean_price_frame(df)
        if issues:
            summary["issues"][t["symbol"]] = issues
        summary["rows"] += store_prices(conn, t["id"], df)
        summary["ok"] += 1
        time.sleep(pause)

    status = "OK" if not summary["failed"] and not summary["empty"] else ("PARTIAL" if summary["ok"] else "FAILED")
    log_refresh(conn, "download_prices", status,
                f"ok={summary['ok']} failed={len(summary['failed'])} empty={len(summary['empty'])} rows={summary['rows']}")
    return summary


def load_price_frame(conn: sqlite3.Connection, stock_id: int, as_of: str | None = None) -> pd.DataFrame:
    q = "SELECT date, open, high, low, close, adj_close, volume FROM daily_prices WHERE stock_id=?"
    params: list = [stock_id]
    if as_of:
        q += " AND date<=?"
        params.append(as_of)
    q += " ORDER BY date"
    df = pd.read_sql_query(q, conn, params=params, parse_dates=["date"]).set_index("date")
    return df
