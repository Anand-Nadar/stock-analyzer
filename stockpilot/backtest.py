"""Portfolio-level backtest of the swing strategy on stored daily data.

Design choices (all aimed at NOT flattering the strategy):
  * Signals are generated from data up to the close of day t and executed on day t+1:
    confirmed breakouts at the next OPEN; near-breakouts as a one-day buy-stop at the trigger price.
  * Indicators are causal, so no future data leaks into a decision (unit-tested).
  * Costs: `round_trip_cost_pct` (brokerage/STT/charges, an assumption) plus `bt_slippage_pct` per side.
  * Same-bar stop/target ambiguity resolves to the stop. Gaps fill at the open.
  * Position size follows the live risk rules (risk % of current equity, concentration and cash caps).
Known limitations (also printed with every report): the universe is today's survivors (survivorship bias),
dividends are not included, results depend on yfinance data quality, and few trades means weak evidence.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import SETTINGS, Settings
from .data.prices import load_price_frame
from .indicators import add_indicators
from .risk import position_size
from .swing import bench_ret60_series, evaluate_prepared, _f
from .trade_sim import find_exit

LIMITATIONS = [
    "Universe is today's list of stocks (survivorship bias): delisted or collapsed names are absent, which flatters results.",
    "Dividends are not included; prices are split-adjusted yfinance data (unofficial) and unaudited.",
    "Costs are assumptions (see round_trip_cost_pct, bt_slippage_pct). Taxes beyond the cost assumption are not modelled.",
    "Past performance does not predict future returns.",
]


def baseline_settings(s: Settings) -> Settings:
    """A deliberately simple comparison: any 20-day-high breakout that is liquid and sizeable -
    no score threshold, no risk/reward or stop-distance gate."""
    return replace(s, min_score=0.0, min_risk_reward=-1e9, max_stop_distance_pct=1e9)


def _load(conn: sqlite3.Connection, s: Settings):
    bench_row = conn.execute("SELECT id FROM stocks WHERE symbol=?", (s.benchmark_symbol,)).fetchone()
    bench = load_price_frame(conn, bench_row["id"]) if bench_row else None
    if bench is not None and bench.empty:
        bench = None
    data = {}
    for st in conn.execute("SELECT id, symbol FROM stocks WHERE in_universe=1 ORDER BY symbol"):
        df = load_price_frame(conn, st["id"])
        if len(df) >= s.min_history_bars + 5:
            data[st["symbol"]] = df
    return data, bench


def run_backtest(data: dict[str, pd.DataFrame], bench: pd.DataFrame | None, s: Settings = SETTINGS,
                 start: str | None = None, end: str | None = None, strategy: str = "swing",
                 regime_filter: bool = False) -> dict:
    ss = baseline_settings(s) if strategy == "baseline" else s
    ind = {sym: add_indicators(df) for sym, df in data.items()}
    arrs = {sym: {k: df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")} for sym, df in data.items()}
    pos_of = {sym: {d: i for i, d in enumerate(df.index)} for sym, df in data.items()}
    b60 = {sym: bench_ret60_series(bench, df.index) for sym, df in data.items()}

    regime_ok: dict = {}
    if regime_filter:
        if bench is None:
            raise ValueError("regime filter needs benchmark data (download-prices includes it by default)")
        bc = bench["close"]
        regime_ok = (bc > bc.rolling(200, min_periods=200).mean()).to_dict()   # True only while the Nifty is above its 200-day average

    calendar = sorted(set().union(*[set(df.index) for df in data.values()])) if data else []
    if start:
        calendar = [d for d in calendar if d >= pd.Timestamp(start)]
    if end:
        calendar = [d for d in calendar if d <= pd.Timestamp(end)]
    out = {"strategy": strategy, "start": None, "end": None, "trades": [], "equity": [], "metrics": {}, "limitations": LIMITATIONS}
    if len(calendar) < 2:
        out["metrics"] = {"error": "not enough data in the selected period"}
        return out

    cash = equity = s.capital
    open_pos: list[dict] = []
    trades: list[dict] = []
    pending: list[dict] = []
    curve: list[tuple] = []
    invested_frac: list[float] = []
    cost_rate = s.round_trip_cost_pct / 100.0
    slip = s.bt_slippage_pct / 100.0

    def close_due(t):
        nonlocal cash
        for p in [p for p in open_pos if p["exit_date"] <= t]:
            open_pos.remove(p)
            exit_fill = p["exit_price"] * (1 - slip)
            gross = p["qty"] * (exit_fill - p["entry_fill"])
            costs = p["qty"] * p["entry_fill"] * cost_rate
            slip_amt = p["qty"] * (p["entry_fill"] - p["entry_raw"]) + p["qty"] * (p["exit_price"] - exit_fill)
            pnl = gross - costs
            cash += p["qty"] * exit_fill - costs
            risk_amt = p["qty"] * (p["entry_fill"] - p["stop"])
            trades.append({"symbol": p["symbol"], "entry_date": p["entry_date"].date().isoformat(),
                           "exit_date": p["exit_date"].date().isoformat(), "entry": round(p["entry_fill"], 2),
                           "exit": round(exit_fill, 2), "qty": p["qty"], "stop": p["stop"], "target": p["target"],
                           "reason": p["reason"], "pnl": round(pnl, 2),
                           "gross_pnl": round(gross + slip_amt, 2), "costs": round(costs + slip_amt, 2), "r_multiple": round(pnl / risk_amt, 2) if risk_amt > 0 else None,
                           "bars_held": p["bars_held"], "score": p["score"], "setup": p["setup"]})

    for t in calendar:
        # 1. exits dated BEFORE today free their slots/cash
        close_due(t - pd.Timedelta(days=1))
        # 2. execute orders raised at yesterday's close
        for o in pending:
            sym = o["symbol"]
            i = pos_of[sym].get(t)
            if i is None or len(open_pos) >= s.bt_max_positions or any(p["symbol"] == sym for p in open_pos):
                continue
            a = arrs[sym]
            if o["setup"] == "BREAKOUT":
                fill = a["open"][i]
            elif a["high"][i] >= o["entry"]:
                fill = max(a["open"][i], o["entry"])
            else:
                continue
            entry_fill = fill * (1 + slip)
            if not (o["stop"] < entry_fill < o["target"]):
                continue   # gapped through the plan: skip rather than chase
            ps = position_size(equity, s.risk_pct_per_trade, entry_fill, o["stop"], s.max_position_pct, available_cash=cash)
            if ps.quantity < 1:
                continue
            ex = find_exit(a["open"][i:], a["high"][i:], a["low"][i:], a["close"][i:], o["stop"], o["target"], s.bt_max_hold_days)
            idx = data[sym].index
            j_end = int(idx.searchsorted(calendar[-1], side="right")) - 1   # last bar inside the tested period
            if ex is None or i + ex[0] > j_end:
                k, price, reason = j_end - i, float(a["close"][j_end]), "PERIOD_END"   # never use data past the period
            else:
                k, price, reason = ex
            cash -= ps.quantity * entry_fill
            open_pos.append({"symbol": sym, "qty": ps.quantity, "entry_fill": entry_fill, "entry_raw": fill, "entry_date": t, "stop": o["stop"],
                             "target": o["target"], "exit_date": idx[i + k], "exit_price": price, "reason": reason,
                             "bars_held": k + 1, "score": o["score"], "setup": o["setup"]})
        close_due(t)   # same-day exits
        # 3. mark to market at today's close
        mv = 0.0
        for p in open_pos:
            i = pos_of[p["symbol"]].get(t)
            px = arrs[p["symbol"]]["close"][i] if i is not None else p["entry_fill"]
            mv += p["qty"] * px
        equity = cash + mv
        curve.append((t, equity))
        invested_frac.append(mv / equity if equity > 0 else 0.0)
        # 4. look for signals at today's close for tomorrow
        pending = []
        sd = replace(ss, capital=equity)
        if t != calendar[-1] and (not regime_filter or regime_ok.get(t, False)):
            for sym in data:
                i = pos_of[sym].get(t)
                if i is None or i + 1 < s.min_history_bars or i + 1 >= len(data[sym]):
                    continue
                d = ind[sym].iloc[: i + 1]
                bs = b60[sym]
                r = evaluate_prepared(sym, d, _f(bs.iloc[i]) if bs is not None else None, sd)
                if r["qualifies"] and r["setup"] in ("BREAKOUT", "NEAR_BREAKOUT"):
                    pl = r["plan"]
                    pending.append({"symbol": sym, "setup": r["setup"], "entry": pl["entry"], "stop": pl["stop_loss"],
                                    "target": pl["target"], "score": r["score"]})
            pending.sort(key=lambda x: -x["score"])

    close_due(calendar[-1] + pd.Timedelta(days=3650))   # any PERIOD_END positions
    out.update(start=calendar[0].date().isoformat(), end=calendar[-1].date().isoformat(), trades=trades,
               equity=[(d.date().isoformat(), round(e, 2)) for d, e in curve])
    out["metrics"] = compute_metrics(trades, curve, invested_frac, bench, s.capital, calendar[0], calendar[-1])
    return out


def _max_dd(values) -> float:
    peak, dd = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        dd = min(dd, (v / peak - 1) * 100.0)
    return float(dd)


def compute_metrics(trades, curve, invested_frac, bench, capital, t0, t1) -> dict:
    eq = pd.Series({d: e for d, e in curve})
    final = float(eq.iloc[-1])
    days = max((t1 - t0).days, 1)
    rets = eq.pct_change().dropna()
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    gp, gl = sum(wins), -sum(losses)
    rs = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
    m = {
        "trades": len(trades),
        "total_return_pct": round((final / capital - 1) * 100, 2),
        "cagr_pct": round(((final / capital) ** (365.25 / days) - 1) * 100, 2) if final > 0 else None,
        "max_drawdown_pct": round(_max_dd(eq.values), 2),
        "sharpe_rf0": round(float(rets.mean() / rets.std() * math.sqrt(252)), 2) if len(rets) > 2 and rets.std() > 0 else None,
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "profit_factor": (round(gp / gl, 2) if gl > 0 else (None if gp == 0 else float("inf"))) if trades else None,
        "avg_win": round(gp / len(wins), 2) if wins else None,
        "avg_loss": round(-gl / len(losses), 2) if losses else None,
        "avg_r": round(sum(rs) / len(rs), 2) if rs else None,
        "avg_bars_held": round(sum(t["bars_held"] for t in trades) / len(trades), 1) if trades else None,
        "avg_exposure_pct": round(sum(invested_frac) / len(invested_frac) * 100, 1),
        "turnover_x": round(sum(t["qty"] * t["entry"] for t in trades) / float(eq.mean()), 2) if trades else 0.0,
        "net_pnl": round(sum(t["pnl"] for t in trades), 2),
    }
    if bench is not None:
        b = bench["close"].loc[t0:t1]
        if len(b) > 1:
            m["benchmark_return_pct"] = round((b.iloc[-1] / b.iloc[0] - 1) * 100, 2)
            m["benchmark_max_drawdown_pct"] = round(_max_dd(b.values), 2)
    notes = []
    if len(trades) < 30:
        notes.append(f"only {len(trades)} trades: far too few to draw statistical conclusions (aim for 30+ across different market regimes)")
    if m.get("benchmark_return_pct") is not None and m["total_return_pct"] < m["benchmark_return_pct"]:
        notes.append("strategy did not beat buy-and-hold on the benchmark over this period")
    m["notes"] = notes
    return m


def run_and_store(conn: sqlite3.Connection, s: Settings = SETTINGS, start: str | None = None, end: str | None = None,
                  oos_from: str | None = None, strategies=("swing", "baseline"), regime_filter: bool = False) -> dict:
    data, bench = _load(conn, s)
    periods = [("full", start, end)] if not oos_from else [
        ("in_sample", start, (pd.Timestamp(oos_from) - pd.Timedelta(days=1)).date().isoformat()),
        ("out_of_sample", oos_from, end)]
    report: dict = {}
    for strat in strategies:
        for label, a, b in periods:
            res = run_backtest(data, bench, s, a, b, strat, regime_filter)
            report[f"{strat}:{label}"] = res
            conn.execute("INSERT INTO backtest_runs (strategy, config_json, costs_json, metrics_json) VALUES (?,?,?,?)",
                         (f"{strat}:{label}",
                          json.dumps({"start": res["start"], "end": res["end"], "capital": s.capital, "risk_pct": s.risk_pct_per_trade,
                                      "max_positions": s.bt_max_positions, "max_hold_days": s.bt_max_hold_days,
                                      "min_score": s.min_score, "universe_size": len(data), "regime_filter": regime_filter}),
                          json.dumps({"round_trip_cost_pct": s.round_trip_cost_pct, "slippage_pct_per_side": s.bt_slippage_pct}),
                          json.dumps({"metrics": res["metrics"], "trades": res["trades"]})))
    conn.commit()
    return report
