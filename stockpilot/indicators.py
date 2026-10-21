"""Technical indicators implemented with pandas/numpy only (no extra dependencies).

All functions take pandas Series indexed by date and return Series of the same index.
RSI and ATR use Wilder's smoothing (the standard definition).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(x: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, seeded with the simple average of the first `period` valid values."""
    valid = x.dropna()
    out = pd.Series(np.nan, index=x.index)
    if len(valid) < period:
        return out
    seed_pos = x.index.get_loc(valid.index[period - 1])
    seeded = x.copy()
    seeded.iloc[:seed_pos] = np.nan
    seeded.iloc[seed_pos] = valid.iloc[:period].mean()
    return seeded.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    gain[delta.isna()] = np.nan
    loss[delta.isna()] = np.nan
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    # no losses at all in the window -> RSI is 100 (where gains exist)
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan  # first bar has no previous close
    return _wilder(tr, period)


def pct_return(close: pd.Series, periods: int) -> pd.Series:
    return close.pct_change(periods) * 100.0


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Today's volume divided by the average of the *previous* `window` days."""
    return volume / volume.shift(1).rolling(window, min_periods=window).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with the standard indicator columns added."""
    out = df.copy()
    c = out["close"]
    out["sma20"], out["sma50"], out["sma200"] = sma(c, 20), sma(c, 50), sma(c, 200)
    out["rsi14"] = rsi(c, 14)
    m = macd(c)
    out["macd"], out["macd_signal"], out["macd_hist"] = m["macd"], m["signal"], m["hist"]
    out["atr14"] = atr(out["high"], out["low"], c, 14)
    out["ret20"], out["ret60"] = pct_return(c, 20), pct_return(c, 60)
    out["vol_ratio20"] = volume_ratio(out["volume"], 20)
    out["traded_value_med20"] = (c * out["volume"]).rolling(20, min_periods=20).median()
    out["resistance20"] = out["high"].shift(1).rolling(20, min_periods=20).max()
    out["base_low20"] = out["low"].rolling(20, min_periods=20).min()
    return out
