import numpy as np
import pandas as pd

from stockpilot.backtest import baseline_settings, run_backtest
from stockpilot.config import Settings
from stockpilot.trade_sim import find_exit


def _arr(rows):
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]   # o, h, l, c


def test_find_exit_stop_target_time_and_gaps():
    # target hit
    assert find_exit(*_arr([(100, 101, 99, 100), (101, 112, 100, 111)]), stop=95, target=110, max_hold=10)[2] == "TARGET"
    # stop and target same bar -> stop wins
    k, px, why = find_exit(*_arr([(100, 115, 90, 100)]), stop=95, target=110, max_hold=10)
    assert why == "STOP" and px == 95
    # gap down through stop fills at the open (worse than stop)
    k, px, why = find_exit(*_arr([(100, 101, 99, 100), (90, 92, 88, 91)]), stop=95, target=120, max_hold=10)
    assert why == "STOP" and px == 90 and k == 1
    # time exit at close of last allowed bar
    k, px, why = find_exit(*_arr([(100, 102, 99, 101)] * 5), stop=90, target=120, max_hold=3)
    assert why == "TIME" and k == 2 and px == 101
    # still open
    assert find_exit(*_arr([(100, 102, 99, 101)] * 2), stop=90, target=120, max_hold=5) is None


def _breakout_world(n=420, seed=3):
    """Steady uptrend with repeated tight bases and breakouts on rising volume."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n, name="date")
    close = np.empty(n)
    price = 100.0
    for i in range(n):
        phase = i % 40
        price *= 1.0 + (0.0 if phase < 30 else 0.012) + rng.normal(0, 0.0015)
        close[i] = price
    close = pd.Series(close, index=idx)
    high, low = close * 1.004, close * 0.996
    vol = pd.Series(3_000_000.0, index=idx)
    vol[[(i % 40) >= 30 for i in range(n)]] = 7_000_000.0
    return pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "high": high, "low": low, "close": close,
                         "adj_close": close, "volume": vol})


S = Settings(capital=300_000, risk_pct_per_trade=1.0, min_median_traded_value=1e6)


def test_backtest_invariants():
    data = {"AAA": _breakout_world(seed=1), "BBB": _breakout_world(seed=2)}
    res = run_backtest(data, None, S, strategy="baseline")
    eq = [e for _, e in res["equity"]]
    assert all(e > 0 for e in eq)
    for t in res["trades"]:
        assert t["exit_date"] >= t["entry_date"]
        assert t["qty"] >= 1
        # entries only after the history warm-up
        assert t["entry_date"] > data["AAA"].index[S.min_history_bars - 1].date().isoformat()
    # final equity reconciles with starting capital + realised pnl (all positions closed at period end)
    assert abs(eq[-1] - (S.capital + sum(t["pnl"] for t in res["trades"]))) < 1.0 or res["trades"] == []
    assert res["metrics"]["max_drawdown_pct"] <= 0


def test_costs_reduce_pnl():
    data = {"AAA": _breakout_world(seed=1)}
    cheap = run_backtest(data, None, Settings(**{**S.__dict__, "round_trip_cost_pct": 0.0, "bt_slippage_pct": 0.0}), strategy="baseline")
    dear = run_backtest(data, None, Settings(**{**S.__dict__, "round_trip_cost_pct": 1.0, "bt_slippage_pct": 0.3}), strategy="baseline")
    if cheap["trades"]:
        assert dear["metrics"]["net_pnl"] < cheap["metrics"]["net_pnl"]


def test_period_split_does_not_leak():
    data = {"AAA": _breakout_world(seed=1)}
    cut = "2024-03-01"
    res = run_backtest(data, None, S, start=None, end=cut, strategy="baseline")
    assert res["end"] <= cut
    assert all(t["exit_date"] <= cut for t in res["trades"])      # exits are forced inside the period


def test_baseline_is_looser_than_swing():
    b = baseline_settings(S)
    assert b.min_score == 0 and b.min_risk_reward < 0


def test_regime_filter_blocks_entries_in_a_falling_market():
    world = _breakout_world(seed=1)
    falling = world[["close"]].copy()
    falling["close"] = np.linspace(200, 100, len(world))          # benchmark always below its 200-day average
    res = run_backtest({"AAA": world}, falling, S, strategy="baseline", regime_filter=True)
    assert res["trades"] == []
    rising = world[["close"]].copy()
    rising["close"] = np.linspace(100, 200, len(world))
    on = run_backtest({"AAA": world}, rising, S, strategy="baseline", regime_filter=True)
    off = run_backtest({"AAA": world}, rising, S, strategy="baseline", regime_filter=False)
    assert 0 < len(on["trades"]) <= len(off["trades"])


def test_trades_record_costs_and_analysis_runs():
    from stockpilot.analysis import analyze_trades
    res = run_backtest({"AAA": _breakout_world(seed=1)}, None, S, strategy="baseline")
    assert res["trades"] and all(t["costs"] > 0 for t in res["trades"])
    assert all(abs((t["gross_pnl"] - t["costs"]) - t["pnl"]) < 0.02 for t in res["trades"])
    an = analyze_trades(res["trades"])
    assert an["overall"]["n"] == len(res["trades"]) and set(an["by_exit"]) <= {"STOP", "TARGET", "TIME", "PERIOD_END"}


def test_breakout_only_excludes_near_breakouts():
    import test_swing as ts
    from stockpilot.swing import evaluate_swing
    df = ts._series(breakout=False)
    df.iloc[-1, df.columns.get_loc("close")] = df["high"].iloc[-21:-1].max() * 0.99        # just under the ceiling
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-1] + 1
    r_off = evaluate_swing("T", df, None, Settings(capital=300_000))
    r_on = evaluate_swing("T", df, None, Settings(capital=300_000, breakout_only=True))
    if r_off.get("setup") == "NEAR_BREAKOUT":
        assert any("breakout_only" in x for x in r_on["rejections"]) and not r_on["qualifies"]
