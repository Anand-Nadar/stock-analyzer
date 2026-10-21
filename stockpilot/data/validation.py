"""Price-data validation: OHLC sanity, duplicates, gaps and freshness."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from datetime import date, datetime

import pandas as pd

from ..config import SETTINGS


def clean_price_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop invalid rows from a frame indexed by date with open/high/low/close/volume columns.

    Returns (clean_df, issues). Rows are never silently altered; bad rows are dropped and reported.
    """
    issues: list[str] = []
    if df.empty:
        return df, issues
    df = df.copy()
    dup = df.index.duplicated(keep="last")
    if dup.any():
        issues.append(f"{int(dup.sum())} duplicate dates removed")
        df = df[~dup]
    num = ["open", "high", "low", "close"]
    bad_nan = df[num].isna().any(axis=1)
    bad_nonpos = (df[num] <= 0).any(axis=1)
    bad_ohlc = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    bad = bad_nan | bad_nonpos | bad_ohlc
    if bad.any():
        issues.append(f"{int(bad.sum())} rows with NaN/non-positive/inconsistent OHLC dropped")
        df = df[~bad]
    return df.sort_index(), issues


@dataclass
class FreshnessReport:
    latest_date: str | None
    stale_days: int | None
    symbols_tracked: int
    symbols_with_data: int
    symbols_stale: list[str]
    symbols_missing: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def freshness_report(conn: sqlite3.Connection, today: date | None = None, universe_only: bool = True) -> FreshnessReport:
    today = today or datetime.now().date()
    where = "s.in_universe=1" if universe_only else "s.security_type='EQUITY'"
    rows = conn.execute(
        f"""SELECT s.symbol, MAX(p.date) AS last_date
            FROM stocks s LEFT JOIN daily_prices p ON p.stock_id = s.id
            WHERE {where} GROUP BY s.id"""
    ).fetchall()
    stale, missing, dates = [], [], []
    for r in rows:
        if r["last_date"] is None:
            missing.append(r["symbol"])
            continue
        d = date.fromisoformat(r["last_date"])
        dates.append(d)
        if (today - d).days > SETTINGS.max_stale_days:
            stale.append(r["symbol"])
    latest = max(dates) if dates else None
    return FreshnessReport(
        latest_date=latest.isoformat() if latest else None,
        stale_days=(today - latest).days if latest else None,
        symbols_tracked=len(rows),
        symbols_with_data=len(dates),
        symbols_stale=sorted(stale),
        symbols_missing=sorted(missing),
    )


def find_gaps(conn: sqlite3.Connection, symbol: str, max_gap_days: int = 7) -> list[tuple[str, str]]:
    """Return (from_date, to_date) pairs where consecutive bars are more than `max_gap_days` apart."""
    rows = conn.execute(
        """SELECT p.date FROM daily_prices p JOIN stocks s ON s.id=p.stock_id
           WHERE s.symbol=? ORDER BY p.date""",
        (symbol,),
    ).fetchall()
    dates = pd.to_datetime([r["date"] for r in rows])
    gaps = []
    for a, b in zip(dates[:-1], dates[1:]):
        if (b - a).days > max_gap_days:
            gaps.append((a.date().isoformat(), b.date().isoformat()))
    return gaps
