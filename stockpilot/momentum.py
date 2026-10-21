"""Momentum rotation backtest: once a month, hold the strongest N stocks (by past return), otherwise sit still.

Why test this: the swing strategy traded ~24x capital per period and costs ate most of its gross edge.
This one trades a few times a year per stock, so costs matter far less.

Rules (all deliberately simple, nothing tuned):
  * On the last trading day of each month rank every eligible stock by its return from `lookback` bars ago to `skip` bars ago
    (the classic 12-1 month momentum: the most recent month is skipped because very recent winners often reverse).
  * Eligible = traded that day, price >= min_price, median daily traded value >= min_median_traded_value.
  * Next trading day at the OPEN: sell holdings that dropped out of the top N, buy new entrants with equal money
    (equity / N each). Holdings that stay are left alone (less trading).
  * Optional regime filter: if the Nifty is below its 200-day average, hold only cash.
  * Costs: round_trip_cost_pct (half per side) plus bt_slippage_pct per side on all traded value.
A baseline holding ALL eligible stocks equally (no ranking) is run the same way: if ranking adds nothing, momentum isn't helping.
"""
from __future__ import annotations

import json
import math
import sqlite3

import pandas as pd

from .backtest import _load, _max_dd
from .config import SETTINGS, Settings

MOMENTUM_LIMITATIONS = [
    "SURVIVORSHIP / SELECTION BIAS IS SEVERE HERE: today's index members were chosen partly because they did well, "
    "and a momentum strategy buys past winners, so this test is flattered more than most.",
    "Only one market history is tested; momentum has long stretches of underperformance and sharp reversals.",
    "Dividends excluded; costs are assumptions; yfinance data is unofficial. Taxes (short-term gains) are not modelled.",
]


def _frames(data: dict[str, pd.DataFrame]):
    closes = pd.DataFrame({k: v["close"] for k, v in data.items()}).sort_index()
    opens = pd.DataFrame({k: v["open"] for k, v in data.items()}).reindex(closes.index)
    tv = pd.DataFrame({k: (v["close"] * v["volume"]).rolling(20, min_periods=20).median() for k, v in data.items()}).reindex(closes.index)
    return closes, opens, tv


def rank_momentum(closes: pd.DataFrame, tv: pd.DataFrame, i: int, s: Settings, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Momentum (%) of every eligible stock as of row `i` (descending). Empty if there isn't enough history."""
    if i < lookback:
        return pd.Series(dtype=float)
    now, recent, past = closes.iloc[i], closes.iloc[i - skip], closes.iloc[i - lookback]
    ok = now.notna() & recent.notna() & past.notna() & (now >= s.min_price) & (tv.iloc[i] >= s.min_median_traded_value)
    return ((recent / past - 1.0) * 100.0)[ok].sort_values(ascending=False)


def run_momentum(data: dict[str, pd.DataFrame], bench: pd.DataFrame | None, s: Settings = SETTINGS, start: str | None = None,
                 end: str | None = None, top_n: int = 10, lookback: int = 252, skip: int = 21, regime_filter: bool = False) -> dict:
    out = {"start": None, "end": None, "metrics": {}, "equity": [], "limitations": MOMENTUM_LIMITATIONS, "target": []}
    if not data:
        out["metrics"] = {"error": "no data"}
        return out
    closes, opens, tv = _frames(data)
    ff = closes.ffill()
    opens_f = opens.fillna(ff)
    pos = {d: i for i, d in enumerate(closes.index)}
    dates = [d for d in closes.index if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]
    if len(dates) < 2:
        out["metrics"] = {"error": "not enough data in the selected period"}
        return out
    month_end = set(pd.Series(dates, index=dates).groupby([pd.Index(dates).year, pd.Index(dates).month]).max())
    regime = None
    if regime_filter:
        if bench is None:
            raise ValueError("regime filter needs benchmark data")
        bc = bench["close"]
        regime = (bc > bc.rolling(200, min_periods=200).mean()).reindex(closes.index, method="ffill").fillna(False)

    side = s.round_trip_cost_pct / 200.0 + s.bt_slippage_pct / 100.0       # fraction of traded value, per side
    cash, shares = float(s.capital), {}
    pending, curve, expo, traded, costs, opened = None, [], [], 0.0, 0.0, 0

    for t in dates:
        i = pos[t]
        if pending is not None:                                           # rebalance at today's open
            px = opens_f.iloc[i]
            eq_open = cash + sum(n * px[sym] for sym, n in shares.items())
            for sym in [x for x in shares if x not in pending]:
                v = shares.pop(sym) * px[sym]
                cash += v * (1 - side); traded += v; costs += v * side
            for sym in pending:
                if sym in shares or len(shares) >= top_n:
                    continue
                budget = min(eq_open / top_n, cash)
                n = math.floor(budget / (px[sym] * (1 + side)))
                if n >= 1:
                    v = n * px[sym]
                    cash -= v * (1 + side); traded += v; costs += v * side
                    shares[sym] = n; opened += 1
            pending = None
        mv = sum(n * ff.iloc[i][sym] for sym, n in shares.items())
        eq = cash + mv
        curve.append((t, eq)); expo.append(mv / eq if eq > 0 else 0.0)
        if t in month_end and t != dates[-1]:
            ranked = rank_momentum(closes, tv, i, s, lookback, skip)
            blocked = regime is not None and not bool(regime.iloc[i])
            pending = [] if blocked else list(ranked.head(top_n).index)
            out["target"] = pending

    eqs = pd.Series({d: e for d, e in curve})
    days = max((dates[-1] - dates[0]).days, 1)
    rets = eqs.pct_change().dropna()
    by_year = {}
    prev = float(s.capital)
    for yr, grp in eqs.groupby(eqs.index.year):
        by_year[int(yr)] = round((float(grp.iloc[-1]) / prev - 1) * 100, 1)
        prev = float(grp.iloc[-1])
    m = {"total_return_pct": round((eqs.iloc[-1] / s.capital - 1) * 100, 2),
         "cagr_pct": round(((eqs.iloc[-1] / s.capital) ** (365.25 / days) - 1) * 100, 2) if eqs.iloc[-1] > 0 else None,
         "max_drawdown_pct": round(_max_dd(eqs.values), 2),
         "sharpe_rf0": round(float(rets.mean() / rets.std() * math.sqrt(252)), 2) if len(rets) > 2 and rets.std() > 0 else None,
         "avg_exposure_pct": round(sum(expo) / len(expo) * 100, 1), "turnover_x": round(traded / float(eqs.mean()), 2),
         "costs_paid": round(costs, 0), "positions_opened": opened, "returns_by_year_pct": by_year}
    if bench is not None:
        b = bench["close"].loc[dates[0]:dates[-1]]
        if len(b) > 1:
            m["benchmark_return_pct"] = round((b.iloc[-1] / b.iloc[0] - 1) * 100, 2)
            m["benchmark_max_drawdown_pct"] = round(_max_dd(b.values), 2)
    out.update(start=dates[0].date().isoformat(), end=dates[-1].date().isoformat(), metrics=m,
               equity=[(d.date().isoformat(), round(e, 2)) for d, e in curve])
    return out


def current_picks(data: dict[str, pd.DataFrame], s: Settings = SETTINGS, top_n: int = 10, lookback: int = 252, skip: int = 21) -> tuple[str | None, pd.Series]:
    closes, _, tv = _frames(data)
    if closes.empty:
        return None, pd.Series(dtype=float)
    r = rank_momentum(closes, tv, len(closes) - 1, s, lookback, skip)
    return closes.index[-1].date().isoformat(), r.head(top_n)


def run_and_store_momentum(conn: sqlite3.Connection, s: Settings = SETTINGS, start=None, end=None, oos_from=None,
                           top_n: int = 10, lookback: int = 252, skip: int = 21, regime_filter: bool = False) -> dict:
    data, bench = _load(conn, s)
    periods = [("full", start, end)] if not oos_from else [
        ("in_sample", start, (pd.Timestamp(oos_from) - pd.Timedelta(days=1)).date().isoformat()), ("out_of_sample", oos_from, end)]
    report = {}
    for name, n in (("momentum", top_n), ("equal_weight_all", max(len(data), 1))):
        for label, a, b in periods:
            res = run_momentum(data, bench, s, a, b, n, lookback, skip, regime_filter)
            report[f"{name}:{label}"] = res
            conn.execute("INSERT INTO backtest_runs (strategy, config_json, costs_json, metrics_json) VALUES (?,?,?,?)",
                         (f"{name}:{label}", json.dumps({"start": res["start"], "end": res["end"], "capital": s.capital, "top_n": n, "lookback": lookback,
                                                         "skip": skip, "regime_filter": regime_filter, "universe_size": len(data)}),
                          json.dumps({"round_trip_cost_pct": s.round_trip_cost_pct, "slippage_pct_per_side": s.bt_slippage_pct}),
                          json.dumps({"metrics": res["metrics"], "trades": []})))
    conn.commit()
    asof, picks = current_picks(data, s, top_n, lookback, skip)
    report["_picks"] = {"as_of": asof, "picks": {k: round(float(v), 1) for k, v in picks.items()}}
    return report
