"""NSE security master import and universe selection."""
from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pandas as pd

from ..config import SETTINGS

NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) StockPilot/0.1", "Accept": "text/csv,*/*"}


def parse_equity_list(csv_text: str) -> pd.DataFrame:
    """Parse NSE's EQUITY_L.csv into normalised columns. Keeps only the EQ series."""
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip().upper() for c in df.columns]
    required = {"SYMBOL", "NAME OF COMPANY", "SERIES"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected NSE file format, missing columns: {sorted(missing)}")
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
    df = df[df["SERIES"] == "EQ"].drop_duplicates("SYMBOL")
    isin_col = "ISIN NUMBER" if "ISIN NUMBER" in df.columns else None
    out = pd.DataFrame(
        {
            "symbol": df["SYMBOL"],
            "company_name": df["NAME OF COMPANY"].astype(str).str.strip(),
            "series": df["SERIES"],
            "isin": df[isin_col].astype(str).str.strip() if isin_col else None,
        }
    )
    return out.reset_index(drop=True)


def download_equity_list() -> str:
    import requests

    resp = requests.get(NSE_EQUITY_LIST_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def upsert_stocks(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    rows = [
        (r.symbol, r.company_name, r.isin, r.series, f"{r.symbol}.NS")
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        """INSERT INTO stocks (symbol, company_name, isin, series, yf_ticker)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(symbol) DO UPDATE SET
             company_name=excluded.company_name, isin=excluded.isin,
             series=excluded.series, yf_ticker=excluded.yf_ticker,
             updated_at=datetime('now')""",
        rows,
    )
    conn.commit()
    return len(rows)


def ensure_benchmark(conn: sqlite3.Connection) -> int:
    conn.execute(
        """INSERT INTO stocks (symbol, company_name, security_type, yf_ticker, exchange)
           VALUES (?, 'Nifty 50 Index', 'INDEX', ?, 'NSE')
           ON CONFLICT(symbol) DO NOTHING""",
        (SETTINGS.benchmark_symbol, SETTINGS.benchmark_yf_ticker),
    )
    conn.commit()
    return conn.execute("SELECT id FROM stocks WHERE symbol=?", (SETTINGS.benchmark_symbol,)).fetchone()["id"]


def read_symbol_file(path: Path | str) -> list[str]:
    symbols = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip().upper()
        if line:
            symbols.append(line)
    return list(dict.fromkeys(symbols))


def set_universe(conn: sqlite3.Connection, symbols: list[str]) -> tuple[int, list[str]]:
    """Mark exactly `symbols` as the tracked universe. Returns (count_marked, unknown_symbols)."""
    known = {r["symbol"] for r in conn.execute("SELECT symbol FROM stocks WHERE security_type='EQUITY'")}
    unknown = [s for s in symbols if s not in known]
    valid = [s for s in symbols if s in known]
    conn.execute("UPDATE stocks SET in_universe=0 WHERE security_type='EQUITY'")
    conn.executemany("UPDATE stocks SET in_universe=1 WHERE symbol=?", [(s,) for s in valid])
    conn.commit()
    return len(valid), unknown


# ---------------------------- index constituents (wider universe + sectors) ----------------------------
INDEX_FILES = {"nifty50": "ind_nifty50list.csv", "nifty100": "ind_nifty100list.csv",
               "nifty200": "ind_nifty200list.csv", "nifty500": "ind_nifty500list.csv"}
INDEX_BASE_URL = "https://www.niftyindices.com/IndexConstituent/"


def parse_index_constituents(csv_text: str) -> pd.DataFrame:
    """Parse a niftyindices.com constituent file: Company Name, Industry, Symbol, Series, ISIN Code."""
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns:
        raise ValueError("Unexpected index file format: no 'Symbol' column")
    out = pd.DataFrame({
        "symbol": df["SYMBOL"].astype(str).str.strip().str.upper(),
        "industry": df["INDUSTRY"].astype(str).str.strip() if "INDUSTRY" in df.columns else None,
    })
    return out.drop_duplicates("symbol").reset_index(drop=True)


def download_index_list(name: str) -> str:
    import requests
    if name not in INDEX_FILES:
        raise ValueError(f"unknown index '{name}'; choose from {', '.join(INDEX_FILES)}")
    resp = requests.get(INDEX_BASE_URL + INDEX_FILES[name], headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def apply_index_universe(conn: sqlite3.Connection, df: pd.DataFrame) -> tuple[int, list[str], int]:
    """Make the index's stocks the tracked universe and record each stock's industry as its sector.
    Returns (stocks_marked, unknown_symbols, sectors_set)."""
    n, unknown = set_universe(conn, list(df["symbol"]))
    sectors = 0
    if "industry" in df.columns:
        for r in df.itertuples(index=False):
            if r.industry and str(r.industry) not in ("None", "nan"):
                sectors += conn.execute("UPDATE stocks SET sector=? WHERE symbol=? AND security_type='EQUITY'",
                                        (r.industry, r.symbol)).rowcount
    conn.commit()
    return n, unknown, sectors
