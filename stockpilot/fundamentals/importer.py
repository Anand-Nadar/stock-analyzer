"""Import annual/quarterly financial statements.

Units convention (so valuation maths is consistent): all money amounts in Rs CRORE, shares in CRORE.
Two sources:
  * CSV  - your own curated numbers (see config/financials_template.csv). Stored as validated.
  * yfinance - convenient but UNOFFICIAL and sometimes incomplete; stored as NOT validated.
    Always cross-check against the company's filings, then run `validate-financials SYMBOL`.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

CRORE = 1e7
FIELDS = ["revenue", "operating_profit", "ebitda", "interest_expense", "net_profit", "total_assets",
          "total_liabilities", "current_liabilities", "total_debt", "equity", "cash",
          "operating_cash_flow", "shares_outstanding"]


def _num(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def upsert_statement(conn: sqlite3.Connection, stock_id: int, period_end: str, period_type: str,
                     values: dict, source: str, validated: int) -> None:
    cols = ["stock_id", "period_end", "period_type", "source", "validated"] + FIELDS
    vals = [stock_id, period_end, period_type, source, validated] + [values.get(f) for f in FIELDS]
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols[3:])
    conn.execute(
        f"INSERT INTO financial_statements ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
        f"ON CONFLICT(stock_id, period_end, period_type) DO UPDATE SET {updates}", vals)


def import_csv(conn: sqlite3.Connection, path: Path | str) -> tuple[int, list[str]]:
    """Columns: symbol, period_end, [period_type], [source], [validated] + any of FIELDS."""
    n, errors = 0, []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            row = {k.strip().lower(): v for k, v in row.items() if k}
            sym = (row.get("symbol") or "").strip().upper()
            pe = (row.get("period_end") or "").strip()
            st = conn.execute("SELECT id FROM stocks WHERE symbol=?", (sym,)).fetchone()
            if not st:
                errors.append(f"line {i}: unknown symbol '{sym}' (run load-master first)")
                continue
            if len(pe) != 10 or pe[4] != "-":
                errors.append(f"line {i}: period_end must be YYYY-MM-DD, got '{pe}'")
                continue
            ptype = (row.get("period_type") or "A").strip().upper()
            if ptype not in ("A", "Q"):
                errors.append(f"line {i}: period_type must be A or Q")
                continue
            values = {f: _num(row.get(f)) for f in FIELDS}
            validated = 0 if (row.get("validated") or "1").strip() in ("0", "false", "no") else 1
            upsert_statement(conn, st["id"], pe, ptype, values, row.get("source") or "csv", validated)
            n += 1
    conn.commit()
    return n, errors


def import_sectors_csv(conn: sqlite3.Connection, path: Path | str) -> int:
    """CSV columns: symbol, sector, [industry]."""
    n = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip().lower(): v for k, v in row.items() if k}
            cur = conn.execute("UPDATE stocks SET sector=?, industry=? WHERE symbol=?",
                               ((row.get("sector") or "").strip() or None, (row.get("industry") or "").strip() or None,
                                (row.get("symbol") or "").strip().upper()))
            n += cur.rowcount
    conn.commit()
    return n


def mark_validated(conn: sqlite3.Connection, symbol: str) -> int:
    cur = conn.execute("UPDATE financial_statements SET validated=1 WHERE stock_id=(SELECT id FROM stocks WHERE symbol=?)",
                       (symbol.upper(),))
    conn.commit()
    return cur.rowcount


# ---------------------------- yfinance ----------------------------
_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "operating_profit": ["Operating Income", "EBIT"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "net_profit": ["Net Income", "Net Income Common Stockholders"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "current_liabilities": ["Current Liabilities"],
    "total_debt": ["Total Debt"],
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "shares_outstanding": ["Ordinary Shares Number", "Share Issued"],
}


def _pick(frames, field: str, col):
    for df in frames:
        if df is None or getattr(df, "empty", True) or col not in df.columns:
            continue
        for label in _ROWS[field]:
            if label in df.index:
                v = df.at[label, col]
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v == v:  # not NaN
                    return v
    return None


def fetch_yfinance(conn: sqlite3.Connection, symbols: list[str]) -> dict:
    """Fetch annual statements (typically the last 3-4 fiscal years) for each symbol."""
    import yfinance as yf

    out = {"ok": [], "empty": [], "failed": []}
    for sym in symbols:
        st = conn.execute("SELECT id, yf_ticker FROM stocks WHERE symbol=?", (sym.upper(),)).fetchone()
        if not st:
            out["failed"].append(f"{sym}: not in stock master")
            continue
        try:
            t = yf.Ticker(st["yf_ticker"])
            inc, bal, cf = t.financials, t.balance_sheet, t.cashflow
            if inc is None or inc.empty:
                out["empty"].append(sym)
                continue
            for col in inc.columns:
                vals = {}
                for f in FIELDS:
                    v = _pick([inc, bal, cf], f, col)
                    if v is not None:
                        vals[f] = abs(v) / CRORE if f == "interest_expense" else v / CRORE
                upsert_statement(conn, st["id"], col.date().isoformat(), "A", vals, "yfinance", 0)
            try:
                info = t.info or {}
                if info.get("sector"):
                    conn.execute("UPDATE stocks SET sector=COALESCE(sector, ?), industry=COALESCE(industry, ?) WHERE id=?",
                                 (info.get("sector"), info.get("industry"), st["id"]))
            except Exception:
                pass
            conn.commit()
            out["ok"].append(sym)
        except Exception as exc:
            out["failed"].append(f"{sym}: {exc}")
    return out
