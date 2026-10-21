"""SQLite schema and connection helpers."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import SETTINGS

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS stocks (
    id            INTEGER PRIMARY KEY,
    symbol        TEXT NOT NULL UNIQUE,          -- NSE symbol, e.g. RELIANCE
    company_name  TEXT,
    exchange      TEXT NOT NULL DEFAULT 'NSE',
    isin          TEXT,
    series        TEXT,
    security_type TEXT NOT NULL DEFAULT 'EQUITY', -- EQUITY | INDEX
    yf_ticker     TEXT,                           -- e.g. RELIANCE.NS
    sector        TEXT,
    industry      TEXT,
    status        TEXT NOT NULL DEFAULT 'ACTIVE',
    in_universe   INTEGER NOT NULL DEFAULT 0,     -- 1 = part of the tracked universe
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS daily_prices (
    stock_id  INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date      TEXT NOT NULL,                      -- YYYY-MM-DD
    open      REAL, high REAL, low REAL, close REAL,
    adj_close REAL,
    volume    INTEGER,
    source    TEXT NOT NULL DEFAULT 'yfinance',
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (stock_id, date)
);

CREATE TABLE IF NOT EXISTS financial_statements (
    id INTEGER PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    period_end TEXT NOT NULL, period_type TEXT NOT NULL,   -- Q | A
    revenue REAL, net_profit REAL, total_assets REAL, total_liabilities REAL,
    total_debt REAL, operating_cash_flow REAL, source TEXT, validated INTEGER NOT NULL DEFAULT 0,
    operating_profit REAL, ebitda REAL, interest_expense REAL, equity REAL,
    current_liabilities REAL, cash REAL, shares_outstanding REAL,   -- amounts in Rs crore, shares in crore
    UNIQUE (stock_id, period_end, period_type)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    id INTEGER PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    ex_date TEXT NOT NULL, action_type TEXT NOT NULL,       -- DIVIDEND | SPLIT | BONUS | ...
    ratio_or_amount REAL, details TEXT, source TEXT,
    UNIQUE (stock_id, ex_date, action_type)
);

CREATE TABLE IF NOT EXISTS fundamental_metrics (
    id INTEGER PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    period_end TEXT NOT NULL, metric TEXT NOT NULL, value REAL,
    calculated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (stock_id, period_end, metric)
);

CREATE TABLE IF NOT EXISTS technical_indicators (
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date TEXT NOT NULL, indicator TEXT NOT NULL, value REAL,
    calculated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (stock_id, date, indicator)
);

CREATE TABLE IF NOT EXISTS recommendation_runs (
    id INTEGER PRIMARY KEY,
    engine TEXT NOT NULL,                 -- SWING | LONG_TERM
    run_at TEXT NOT NULL DEFAULT (datetime('now')),
    data_as_of TEXT,                      -- latest price date used
    config_json TEXT NOT NULL,            -- scoring weights, gates, risk settings
    results_json TEXT NOT NULL,           -- full per-stock scores, reasons, rejections
    notes TEXT
);

CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    list_name TEXT NOT NULL DEFAULT 'default', notes TEXT,
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (stock_id, list_name)
);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id INTEGER PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    buy_date TEXT NOT NULL, quantity INTEGER NOT NULL, buy_price REAL NOT NULL,
    fees REAL NOT NULL DEFAULT 0, notes TEXT
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    opened_at TEXT NOT NULL, entry_price REAL NOT NULL, quantity INTEGER NOT NULL,
    stop_loss REAL NOT NULL, target REAL,
    closed_at TEXT, exit_price REAL, exit_reason TEXT,
    recommendation_run_id INTEGER REFERENCES recommendation_runs(id)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY,
    strategy TEXT NOT NULL, config_json TEXT NOT NULL,
    run_at TEXT NOT NULL DEFAULT (datetime('now')),
    costs_json TEXT, metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS data_refresh_log (
    id INTEGER PRIMARY KEY,
    run_at TEXT NOT NULL DEFAULT (datetime('now')),
    task TEXT NOT NULL, status TEXT NOT NULL,   -- OK | PARTIAL | FAILED
    details TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_transactions (
    id INTEGER PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    trade_date TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    price REAL NOT NULL CHECK (price > 0),
    fees REAL NOT NULL DEFAULT 0, notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_prices_date ON daily_prices(date);
"""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else SETTINGS.db_path
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(db_path: Path | str | None = None):
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added after the first release; applied to databases created earlier.
_ADDED_COLUMNS = {
    "financial_statements": {
        "operating_profit": "REAL", "ebitda": "REAL", "interest_expense": "REAL", "equity": "REAL",
        "current_liabilities": "REAL", "cash": "REAL", "shares_outstanding": "REAL",
    },
}


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    for table, cols in _ADDED_COLUMNS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, typ in cols.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
    conn.commit()


def log_refresh(conn: sqlite3.Connection, task: str, status: str, details: str = "") -> None:
    conn.execute("INSERT INTO data_refresh_log (task, status, details) VALUES (?, ?, ?)", (task, status, details))
    conn.commit()
