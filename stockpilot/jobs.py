"""Scheduled maintenance: daily update pipeline and database backups.

Run once a day after market close (NSE closes 15:30 IST; data providers need a little time), e.g. with cron:
    30 19 * * 1-5  cd /home/dell/workspace/stock-analyzer && .venv/bin/python -m stockpilot daily-update >> data/daily.log 2>&1
Every step is isolated: a failure is logged to data_refresh_log (visible on the dashboard) and does not stop later steps.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from . import db
from .config import SETTINGS, Settings


def backup_db(conn: sqlite3.Connection, s: Settings = SETTINGS) -> Path:
    dest_dir = s.db_path.parent / "backups"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"stockpilot-{datetime.now():%Y%m%d-%H%M%S}.db"
    out = sqlite3.connect(str(dest))
    try:
        conn.backup(out)              # consistent snapshot even if the DB is in use
    finally:
        out.close()
    for old in sorted(dest_dir.glob("stockpilot-*.db"))[:-s.backup_keep]:
        old.unlink()
    return dest


def daily_update(conn: sqlite3.Connection, s: Settings = SETTINGS) -> dict:
    from .data.prices import update_prices
    from .data.validation import freshness_report
    from .fundamentals.scoring import screen_long
    from .paper import update_paper_trades
    from .swing import screen_swing

    results: dict[str, str] = {}

    def step(name, fn):
        try:
            detail = fn()
            results[name] = "OK"
            db.log_refresh(conn, name, "OK", str(detail or "")[:500])
        except Exception as exc:       # noqa: BLE001 - we want every failure surfaced, not raised
            results[name] = f"FAILED: {exc}"
            db.log_refresh(conn, name, "FAILED", str(exc)[:500])

    def prices():
        r = update_prices(conn)
        if r["failed"] or r["empty"]:
            raise RuntimeError(f"{len(r['failed'])} failed, {len(r['empty'])} empty: " + "; ".join((r["failed"] + r["empty"])[:5]))
        return f"{r['ok']} symbols, {r['rows']} rows"

    def fresh():
        rep = freshness_report(conn)
        if rep.symbols_stale or rep.symbols_missing or rep.latest_date is None:
            raise RuntimeError(f"stale: {rep.symbols_stale[:8]} missing: {rep.symbols_missing[:8]}")
        return f"latest {rep.latest_date}"

    step("download_prices", prices)
    step("freshness_check", fresh)
    step("screen_swing", lambda: f"{len(screen_swing(conn, s)['candidates'])} candidates")
    step("screen_long", lambda: f"{len(screen_long(conn, s)['candidates'])} candidates")
    step("paper_update", lambda: f"{len(update_paper_trades(conn, s))} closed")
    step("backup", lambda: str(backup_db(conn, s)))
    results["ok"] = all(v == "OK" for k, v in results.items())
    return results
