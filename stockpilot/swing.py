"""Transparent swing-trading screen (long-only, end-of-day).

Score (0-100), initial weights are a *hypothesis*, not proven predictive factors:
  Trend 25 | Momentum 20 | Breakout 20 | Volume 15 | Relative strength 10 | Trade quality 10

A stock is a candidate only if it has a breakout/near-breakout setup, passes every hard gate
(history, liquidity, price, stop distance, risk/reward) AND scores >= min_score.
Every evaluation (qualified or rejected) is stored in recommendation_runs for reproducibility.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings
from .data.prices import load_price_frame
from .indicators import add_indicators
from .risk import position_size

WEIGHTS = {"trend": 25, "momentum": 20, "breakout": 20, "volume": 15, "relative_strength": 10, "trade_quality": 10}


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else float(x)


def bench_ret60_series(bench: pd.DataFrame | None, index: pd.Index) -> pd.Series | None:
    """Benchmark 60-day return (%), forward-filled onto `index`. None if benchmark is unusable."""
    if bench is None or len(bench) <= 70:
        return None
    return (bench["close"].pct_change(60) * 100.0).reindex(index, method="ffill")


def evaluate_swing(symbol: str, df: pd.DataFrame, bench: pd.DataFrame | None = None,
                   s: Settings = SETTINGS) -> dict:
    """Evaluate one stock using only data in `df` (so as-of backtests cannot look ahead)."""
    if len(df) < s.min_history_bars:
        return {"symbol": symbol, "qualifies": False, "score": None, "warnings": [],
                "rejections": [f"insufficient history ({len(df)} bars < {s.min_history_bars})"]}
    d = add_indicators(df)
    series = bench_ret60_series(bench, d.index)
    b60 = _f(series.iloc[-1]) if series is not None else None
    return evaluate_prepared(symbol, d, b60, s)


def evaluate_prepared(symbol: str, d: pd.DataFrame, b60: float | None, s: Settings = SETTINGS) -> dict:
    """Evaluate the LAST row of `d`, a frame that already has indicator columns (all causal, so
    slicing a full-history frame up to day t gives the same answer as recomputing on data up to t).
    `b60` is the benchmark's 60-day return (%) as of that row, or None."""
    res: dict = {"symbol": symbol, "qualifies": False, "score": None, "rejections": [], "warnings": []}
    last = d.iloc[-1]
    close = float(last["close"])
    res["as_of"] = d.index[-1].date().isoformat()
    res["close"] = round(close, 2)

    needed = ["sma20", "sma50", "sma200", "rsi14", "atr14", "resistance20", "base_low20", "vol_ratio20", "traded_value_med20"]
    if any(_f(last[k]) is None for k in needed):
        res["rejections"].append("indicators incomplete (missing/NaN in recent data)")
        return res

    # ---------------- scoring ----------------
    pts: dict[str, dict] = {}

    conds = [close > last["sma20"], close > last["sma50"], close > last["sma200"],
             last["sma20"] > last["sma50"], last["sma50"] > last["sma200"]]
    pts["trend"] = {"points": 5.0 * sum(bool(c) for c in conds), "max": WEIGHTS["trend"],
                    "detail": f"{sum(bool(c) for c in conds)}/5 trend conditions (price>20/50/200 DMA, 20>50, 50>200)"}

    rsi_v = float(last["rsi14"])
    rsi_pts = 10.0 if 55 <= rsi_v <= 70 else 6.0 if (50 <= rsi_v < 55 or 70 < rsi_v <= 78) else 2.0 if 45 <= rsi_v < 50 else 0.0
    r20, r60 = _f(last["ret20"]), _f(last["ret60"])
    mom = rsi_pts + (5.0 if (r20 or 0) > 0 else 0.0) + (5.0 if (r60 or 0) > 0 else 0.0)
    pts["momentum"] = {"points": mom, "max": WEIGHTS["momentum"],
                       "detail": f"RSI14={rsi_v:.1f}, 20d={r20:+.1f}%, 60d={r60:+.1f}%" if r20 is not None and r60 is not None else f"RSI14={rsi_v:.1f}"}

    resistance, base_low = float(last["resistance20"]), float(last["base_low20"])
    if close > resistance:
        setup, brk = "BREAKOUT", 20.0
    elif close >= 0.98 * resistance:
        setup, brk = "NEAR_BREAKOUT", 10.0
    else:
        setup, brk = "NONE", 0.0
    pts["breakout"] = {"points": brk, "max": WEIGHTS["breakout"],
                       "detail": f"{setup}: close {close:.2f} vs 20-day resistance {resistance:.2f}"}

    vr = float(last["vol_ratio20"])
    vp = 15.0 if vr >= 1.5 else 10.0 if vr >= 1.2 else 5.0 if vr >= 1.0 else 0.0
    pts["volume"] = {"points": vp, "max": WEIGHTS["volume"], "detail": f"volume = {vr:.2f}x 20-day average"}

    rs_avail = b60 is not None and r60 is not None
    if rs_avail:
        diff = r60 - b60
        pts["relative_strength"] = {"points": 10.0 if diff >= 10 else 7.0 if diff >= 5 else 4.0 if diff > 0 else 0.0,
                                    "max": WEIGHTS["relative_strength"],
                                    "detail": f"60d return {r60:+.1f}% vs benchmark {b60:+.1f}% ({diff:+.1f} pts)"}
    if not rs_avail:
        pts["relative_strength"] = {"points": 0.0, "max": WEIGHTS["relative_strength"], "detail": "benchmark data unavailable"}
        res["warnings"].append("benchmark data missing: relative strength not scored (confidence reduced)")

    # ---------------- trade plan ----------------
    atr_v = float(last["atr14"])
    entry = close if setup == "BREAKOUT" else round(resistance * 1.001, 2)
    stop = round(entry - s.atr_stop_multiple * atr_v, 2)
    target = round(entry + (resistance - base_low), 2)   # measured move: height of the 20-day base
    cost = entry * s.round_trip_cost_pct / 100.0
    risk, reward = entry - stop, target - entry
    rr_net = (reward - cost) / (risk + cost) if risk > 0 else 0.0
    stop_pct = risk / entry * 100.0 if entry > 0 else 0.0
    med_tv = float(last["traded_value_med20"])

    tq = (4.0 if med_tv >= 5 * s.min_median_traded_value else 2.0 if med_tv >= s.min_median_traded_value else 0.0)
    tq += 4.0 if rr_net >= 3 else 3.0 if rr_net >= 2 else 2.0 if rr_net >= s.min_risk_reward else 0.0
    atr_pct = atr_v / close * 100.0
    tq += 2.0 if atr_pct <= 4 else 1.0 if atr_pct <= 6 else 0.0
    pts["trade_quality"] = {"points": tq, "max": WEIGHTS["trade_quality"],
                            "detail": f"median traded value Rs {med_tv/1e7:.1f} cr, net R:R {rr_net:.2f}, ATR {atr_pct:.1f}% of price"}

    score = round(sum(p["points"] for p in pts.values()), 1)
    res.update(score=score, data_confidence=round((100 - (0 if rs_avail else WEIGHTS["relative_strength"])) / 100, 2),
               components=pts, setup=setup)

    # ---------------- hard gates ----------------
    if setup == "NONE":
        res["rejections"].append("no breakout / near-breakout setup")
    if setup == "NEAR_BREAKOUT" and s.breakout_only:
        res["rejections"].append("near-breakout (buy-stop) setups excluded by breakout_only")
    if close < s.min_price:
        res["rejections"].append(f"price below Rs {s.min_price:.0f}")
    if med_tv < s.min_median_traded_value:
        res["rejections"].append(f"illiquid: median traded value Rs {med_tv/1e7:.1f} cr < Rs {s.min_median_traded_value/1e7:.1f} cr")
    if stop_pct > s.max_stop_distance_pct:
        res["rejections"].append(f"stop too far ({stop_pct:.1f}% > {s.max_stop_distance_pct:.1f}%)")
    if rr_net < s.min_risk_reward:
        res["rejections"].append(f"net risk/reward {rr_net:.2f} < {s.min_risk_reward}")
    if score < s.min_score:
        res["rejections"].append(f"score {score} below threshold {s.min_score}")

    size = position_size(s.capital, s.risk_pct_per_trade, entry, stop, s.max_position_pct)
    res["plan"] = {
        "entry": round(entry, 2), "entry_type": "at close (confirmed)" if setup == "BREAKOUT" else "buy-stop trigger above resistance",
        "stop_loss": stop, "invalidation": f"close below {stop}", "target": target,
        "risk_per_share": round(risk, 2), "stop_distance_pct": round(stop_pct, 2),
        "net_risk_reward": round(rr_net, 2), "position": size.as_dict(),
    }
    if size.quantity == 0:
        res["rejections"].append("position size is zero under current risk settings")
    res["qualifies"] = not res["rejections"]
    return res


def screen_swing(conn: sqlite3.Connection, s: Settings = SETTINGS, as_of: str | None = None, persist: bool = True) -> dict:
    bench_row = conn.execute("SELECT id FROM stocks WHERE symbol=?", (s.benchmark_symbol,)).fetchone()
    bench = load_price_frame(conn, bench_row["id"], as_of) if bench_row else None
    if bench is not None and bench.empty:
        bench = None

    stocks = conn.execute("SELECT id, symbol FROM stocks WHERE in_universe=1 ORDER BY symbol").fetchall()
    results, latest = [], []
    for st in stocks:
        df = load_price_frame(conn, st["id"], as_of)
        if df.empty:
            results.append({"symbol": st["symbol"], "qualifies": False, "score": None,
                            "rejections": ["no price data"], "warnings": []})
            continue
        r = evaluate_swing(st["symbol"], df, bench, s)
        results.append(r)
        if "as_of" in r:
            latest.append(r["as_of"])

    results.sort(key=lambda r: (not r["qualifies"], -(r["score"] or -1)))
    data_as_of = max(latest) if latest else None
    run = {
        "engine": "SWING", "data_as_of": data_as_of,
        "config": {"weights": WEIGHTS, "settings": {k: (str(v) if not isinstance(v, (int, float, str)) else v)
                                                      for k, v in asdict(s).items()}},
        "candidates": [r for r in results if r["qualifies"]],
        "rejected": [r for r in results if not r["qualifies"]],
    }
    if persist:
        cur = conn.execute(
            "INSERT INTO recommendation_runs (engine, data_as_of, config_json, results_json, notes) VALUES (?,?,?,?,?)",
            ("SWING", data_as_of, json.dumps(run["config"]), json.dumps({"candidates": run["candidates"], "rejected": run["rejected"]}),
             "Conditional setups only; not investment advice."))
        conn.commit()
        run["run_id"] = cur.lastrowid
    return run
