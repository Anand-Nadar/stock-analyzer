import numpy as np
import pandas as pd

from stockpilot.config import Settings
from stockpilot.swing import evaluate_swing


def _series(n=260, breakout=True, seed=1, volume=2_000_000):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n, name="date")
    drift = np.linspace(100, 180, n) + rng.normal(0, 0.8, n)
    close = pd.Series(drift, index=idx)
    # tight consolidation in the last 20 bars below a ceiling, then optional breakout on the last bar
    close.iloc[-21:-1] = 180 + rng.normal(0, 0.4, 20)
    close.iloc[-1] = close.iloc[-21:-1].max() + 3.0 if breakout else 170.0   # well below the ceiling
    high, low = close + 1.0, close - 1.0
    vol = pd.Series(volume, index=idx, dtype=float)
    vol.iloc[-1] = volume * 2.0 if breakout else volume
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "adj_close": close, "volume": vol})


S = Settings(capital=300_000, risk_pct_per_trade=0.5)


def test_breakout_scores_and_plan():
    df = _series()
    bench = df[["close"]] * 0.5 + 1000   # flat-ish benchmark relative to stock
    r = evaluate_swing("TEST", df, bench, S)
    assert r["setup"] == "BREAKOUT"
    assert r["components"]["breakout"]["points"] == 20
    assert r["components"]["volume"]["points"] == 15
    p = r["plan"]
    assert p["stop_loss"] < p["entry"] < p["target"]
    assert p["position"]["risk_at_stop"] <= 1_500 + 1e-6      # never exceeds the risk budget
    assert 0 <= r["score"] <= 100


def test_no_setup_is_rejected():
    r = evaluate_swing("TEST", _series(breakout=False), None, S)
    assert not r["qualifies"]
    assert any("no breakout" in x for x in r["rejections"])
    assert any("benchmark" in w for w in r["warnings"])        # missing data is flagged, confidence reduced
    assert r["data_confidence"] < 1.0


def test_illiquid_rejected():
    r = evaluate_swing("TEST", _series(volume=100), None, S)
    assert not r["qualifies"] and any("illiquid" in x for x in r["rejections"])


def test_insufficient_history():
    r = evaluate_swing("TEST", _series().iloc[:100], None, S)
    assert not r["qualifies"] and "insufficient history" in r["rejections"][0]


def test_no_lookahead():
    df = _series()
    full = evaluate_swing("T", df, None, S)
    trimmed = evaluate_swing("T", df.iloc[:-1], None, S)       # as-of previous day
    assert trimmed["as_of"] < full["as_of"]
    assert trimmed["setup"] != "BREAKOUT"
