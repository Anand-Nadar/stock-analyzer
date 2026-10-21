import numpy as np
import pandas as pd

from stockpilot.config import Settings
from stockpilot.momentum import rank_momentum, run_momentum, _frames

S = Settings(capital=300_000, min_median_traded_value=1e6)


def _stock(drift, n=700, seed=0, vol=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n, name="date")
    c = pd.Series(100 * np.cumprod(1 + drift + rng.normal(0, vol, n)), index=idx)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.005, "low": c * 0.995, "close": c, "adj_close": c,
                         "volume": 5_000_000.0})


def _world():
    return {"UP": _stock(0.0020, seed=1), "MID": _stock(0.0005, seed=2), "FLAT": _stock(0.0, seed=3), "DOWN": _stock(-0.0015, seed=4)}


def test_rank_prefers_strongest_and_needs_history():
    closes, _, tv = _frames(_world())
    r = rank_momentum(closes, tv, len(closes) - 1, S)
    assert r.index[0] == "UP" and r.index[-1] == "DOWN"
    assert rank_momentum(closes, tv, 100, S).empty                      # not enough history yet


def test_momentum_beats_holding_the_loser_and_costs_matter():
    world = _world()
    top = run_momentum(world, None, S, top_n=2)
    assert top["metrics"]["total_return_pct"] > 0 and top["metrics"]["positions_opened"] >= 2
    eq = [e for _, e in top["equity"]]
    assert all(e > 0 for e in eq)
    dear = run_momentum(world, None, Settings(**{**S.__dict__, "round_trip_cost_pct": 3.0, "bt_slippage_pct": 1.0}), top_n=2)
    assert dear["metrics"]["total_return_pct"] < top["metrics"]["total_return_pct"]
    assert dear["metrics"]["costs_paid"] > top["metrics"]["costs_paid"]


def test_regime_filter_in_falling_market_holds_cash():
    world = _world()
    falling = world["UP"][["close"]].copy()
    falling["close"] = np.linspace(200, 100, len(falling))
    res = run_momentum(world, falling, S, top_n=2, regime_filter=True)
    assert res["metrics"]["positions_opened"] == 0 and res["metrics"]["total_return_pct"] == 0.0


def test_period_split_and_calendar_returns():
    res = run_momentum(_world(), None, S, start="2023-06-01", end="2024-03-01", top_n=2)
    assert res["start"] >= "2023-06-01" and res["end"] <= "2024-03-01"
    assert set(res["metrics"]["returns_by_year_pct"]) == {2023, 2024}
