import numpy as np
import pandas as pd

from stockpilot.indicators import atr, ema, macd, rsi, sma, volume_ratio


def test_sma_and_ema():
    s = pd.Series(range(1, 11), dtype=float)
    assert sma(s, 5).iloc[-1] == 8.0
    assert np.isnan(sma(s, 5).iloc[3])
    assert ema(s, 3).iloc[-1] > sma(s, 3).iloc[2]


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi(up).iloc[-1] == 100.0
    assert rsi(down).iloc[-1] < 1e-6
    noisy = pd.Series(100 + np.sin(np.arange(200) / 3) * 5)
    r = rsi(noisy).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_known_value():
    # Classic Wilder example series (first 14 changes), RSI ~ 70.46
    c = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                   45.89, 46.03, 45.61, 46.28, 46.28])
    assert abs(rsi(c, 14).iloc[-1] - 70.46) < 0.1


def test_atr_constant_range():
    idx = pd.RangeIndex(40)
    high, low, close = pd.Series(11.0, index=idx), pd.Series(9.0, index=idx), pd.Series(10.0, index=idx)
    assert abs(atr(high, low, close).iloc[-1] - 2.0) < 1e-9


def test_macd_columns():
    m = macd(pd.Series(np.linspace(100, 150, 120)))
    assert list(m.columns) == ["macd", "signal", "hist"]
    assert m["macd"].iloc[-1] > 0


def test_volume_ratio_uses_previous_days_only():
    v = pd.Series([100.0] * 20 + [300.0])
    assert volume_ratio(v, 20).iloc[-1] == 3.0
