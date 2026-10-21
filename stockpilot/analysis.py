"""Diagnostics on backtest trades: WHERE does the strategy make or lose money?

The key question is whether the score means anything: if higher-scoring setups do not produce better R-multiples
(R = profit divided by the amount risked), the score is not predictive and tuning its weights is guesswork.
"""
from __future__ import annotations

import math
from collections import defaultdict


def _stats(trades: list[dict]) -> dict:
    rs = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    n = len(trades)
    if not n:
        return {"n": 0}
    wins = sum(t["pnl"] > 0 for t in trades)
    out = {"n": n, "win_pct": round(wins / n * 100, 1), "net_pnl": round(sum(t["pnl"] for t in trades), 0)}
    if rs:
        mean = sum(rs) / len(rs)
        sd = math.sqrt(sum((r - mean) ** 2 for r in rs) / (len(rs) - 1)) if len(rs) > 1 else float("nan")
        se = sd / math.sqrt(len(rs)) if len(rs) > 1 else float("nan")
        out.update(avg_r=round(mean, 2), r_low=round(mean - 1.96 * se, 2), r_high=round(mean + 1.96 * se, 2))
    return out


def _group(trades: list[dict], key) -> dict:
    g = defaultdict(list)
    for t in trades:
        g[key(t)].append(t)
    return {k: _stats(v) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


def _bucket(score) -> str:
    if score is None:
        return "n/a"
    return "<60" if score < 60 else "60-69" if score < 70 else "70-79" if score < 80 else "80+"


def analyze_trades(trades: list[dict]) -> dict:
    overall = _stats(trades)
    gross = sum(t.get("gross_pnl", t["pnl"]) for t in trades)
    costs = sum(t.get("costs", 0.0) for t in trades)
    notes = []
    if overall.get("avg_r") is not None and overall["r_low"] <= 0 <= overall["r_high"]:
        notes.append(f"average R of {overall['avg_r']} has a 95% range of {overall['r_low']} to {overall['r_high']}: "
                     "statistically indistinguishable from zero, i.e. no proven edge")
    if costs and gross > 0 and costs > 0.5 * gross:
        notes.append("costs eat more than half of the gross profit: trading less often or in bigger moves would matter more than tuning scores")
    return {"overall": overall, "gross_pnl": round(gross, 0), "costs": round(costs, 0), "by_exit": _group(trades, lambda t: t["reason"]),
            "by_score": _group(trades, lambda t: _bucket(t.get("score"))), "by_setup": _group(trades, lambda t: t.get("setup")),
            "by_year": _group(trades, lambda t: t["entry_date"][:4]), "notes": notes}
