"""Collect verified facts about a stock from the local database.

Everything the AI may talk about comes from here. The model never fetches or calculates numbers itself.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime

from ..config import SETTINGS, Settings
from ..data.prices import load_price_frame
from ..fundamentals.scoring import screen_long
from ..indicators import add_indicators
from ..swing import evaluate_swing


def _r(x, n=2):
    return None if x is None or x != x else round(float(x), n)


def build_facts(conn: sqlite3.Connection, symbol: str, s: Settings = SETTINGS, today: date | None = None,
                long_run: dict | None = None) -> dict:
    symbol = symbol.upper()
    st = conn.execute("SELECT * FROM stocks WHERE symbol=?", (symbol,)).fetchone()
    if not st:
        raise ValueError(f"{symbol} is not in the stock master (run load-master)")
    today = today or datetime.now().date()
    df = load_price_frame(conn, st["id"])
    bench_row = conn.execute("SELECT id FROM stocks WHERE symbol=?", (s.benchmark_symbol,)).fetchone()
    bench = load_price_frame(conn, bench_row["id"]) if bench_row else None
    facts: dict = {"symbol": symbol, "company": st["company_name"], "sector": st["sector"], "missing": []}

    if df.empty:
        facts["missing"].append("no price data")
    else:
        d = add_indicators(df)
        last = d.iloc[-1]
        facts["price"] = {"as_of": df.index[-1].date().isoformat(), "age_days": (today - df.index[-1].date()).days,
                          "close": _r(last["close"]), "sma20": _r(last["sma20"]), "sma50": _r(last["sma50"]),
                          "sma200": _r(last["sma200"]), "rsi14": _r(last["rsi14"], 1), "atr14": _r(last["atr14"]),
                          "return_20d_pct": _r(last["ret20"], 1), "return_60d_pct": _r(last["ret60"], 1),
                          "volume_ratio_20d": _r(last["vol_ratio20"]), "bars_available": len(df)}
        if facts["price"]["age_days"] > s.max_stale_days:
            facts["missing"].append(f"price data is {facts['price']['age_days']} days old")
        sw = evaluate_swing(symbol, df, bench, s)
        swing = {"qualifies": sw["qualifies"], "score": sw["score"], "setup": sw.get("setup"),
                 "rejections": sw["rejections"], "warnings": sw["warnings"]}
        if "components" in sw:
            swing["components"] = {k: {"points": v["points"], "max": v["max"], "detail": v["detail"]} for k, v in sw["components"].items()}
            pl = sw["plan"]
            swing["plan"] = {"entry": pl["entry"], "stop_loss": pl["stop_loss"], "target": pl["target"],
                             "net_risk_reward": pl["net_risk_reward"], "stop_distance_pct": pl["stop_distance_pct"],
                             "quantity": pl["position"]["quantity"], "risk_at_stop": pl["position"]["risk_at_stop"]}
        facts["swing"] = swing

    lr = long_run or screen_long(conn, s, persist=False, extra_symbols=[symbol])
    lt = next((x for x in lr["results"] if x["symbol"] == symbol), None)
    if lt and lt["status"] != "NO_DATA":
        facts["long_term"] = {
            "status": lt["status"], "score": lt["score"], "confidence": lt["confidence"],
            "pillars": {k: {"points": v["points"], "weight": v["weight"], "detail": v["detail"]} for k, v in lt["pillars"].items()},
            "serious_flags": lt["serious_flags"], "flags": lt["flags"], "warnings": lt["warnings"],
            "metrics": {k: (_r(v) if isinstance(v, float) else v) for k, v in lt["metrics"].items() if v is not None},
        }
    else:
        facts["missing"].append("no annual financial statements: long-term view unavailable (import-financials / fetch-financials)")
    return facts


def flatten_facts(facts: dict) -> list[dict]:
    """Numbered scalar facts (F1, F2, ...) the model can cite."""
    out: list[dict] = []

    def walk(prefix: str, v):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else k, x)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(f"{prefix}[{i}]", x)
        elif v is not None:
            out.append({"id": f"F{len(out) + 1}", "path": prefix, "value": v})

    walk("", facts)
    return out
