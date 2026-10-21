"""Central configuration. Override any value with an environment variable.

Example: STOCKPILOT_CAPITAL=300000 STOCKPILOT_RISK_PCT=0.5 python -m stockpilot screen-swing
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default, cast=str):
    raw = os.environ.get(f"STOCKPILOT_{name}")
    return cast(raw) if raw is not None else default


@dataclass
class Settings:
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", PROJECT_ROOT / "data" / "stockpilot.db")))

    # --- data ---
    benchmark_symbol: str = "NIFTY50"
    benchmark_yf_ticker: str = "^NSEI"
    price_history_years: int = field(default_factory=lambda: _env("HISTORY_YEARS", 3, int))
    max_stale_days: int = 4  # calendar days; covers weekends + one holiday

    # --- risk / capital (all user-configurable) ---
    capital: float = field(default_factory=lambda: _env("CAPITAL", 300_000.0, float))
    risk_pct_per_trade: float = field(default_factory=lambda: _env("RISK_PCT", 0.5, float))
    max_position_pct: float = field(default_factory=lambda: _env("MAX_POSITION_PCT", 20.0, float))

    # --- swing screen gates ---
    min_history_bars: int = 220
    min_median_traded_value: float = 5e7  # Rs 5 crore median daily traded value (20d)
    min_price: float = 20.0
    min_risk_reward: float = 1.5
    max_stop_distance_pct: float = 8.0
    atr_stop_multiple: float = 2.0
    min_score: float = 55.0
    # Rough all-in round-trip cost (brokerage, STT, exchange/DP charges, slippage) as % of trade value.
    # This is an ASSUMPTION - set it from your own broker's contract note.
    round_trip_cost_pct: float = field(default_factory=lambda: _env("COST_PCT", 0.25, float))

    # --- long-term (fundamental) screen ---
    lt_min_score: float = 60.0
    lt_min_confidence: float = 0.6        # share of score weight that had real data behind it
    lt_stale_months: int = 20             # latest annual statement older than this => stale
    max_sector_pct: float = 40.0          # portfolio sector concentration warning

    # --- backtest / paper trading ---
    bt_max_positions: int = 5
    bt_max_hold_days: int = 20            # trading bars
    bt_slippage_pct: float = 0.05         # per side, on top of round_trip_cost_pct

    # --- local AI (Ollama) ---
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1"))
    ollama_timeout: int = 300

    # --- housekeeping ---
    backup_keep: int = 14


SETTINGS = Settings()
