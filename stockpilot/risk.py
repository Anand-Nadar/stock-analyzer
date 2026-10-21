"""Risk-based position sizing."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass
class PositionSize:
    quantity: int
    position_value: float
    risk_budget: float
    risk_per_share: float
    risk_at_stop: float          # loss if the stop fills exactly at the stop price
    limited_by: str              # RISK | CONCENTRATION | CASH | INVALID
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def position_size(capital: float, risk_pct: float, entry: float, stop: float,
                  max_position_pct: float = 20.0, available_cash: float | None = None) -> PositionSize:
    """Shares to buy so that (entry - stop) * shares <= capital * risk_pct%.

    The result is further capped by a concentration limit and available cash.
    A stop-loss does NOT guarantee the loss is limited to the risk budget (gaps, slippage).
    """
    risk_budget = capital * risk_pct / 100.0
    risk_per_share = entry - stop
    if capital <= 0 or entry <= 0 or risk_per_share <= 0:
        return PositionSize(0, 0.0, risk_budget, max(risk_per_share, 0.0), 0.0, "INVALID",
                            "Entry must be above stop-loss and capital positive (long trades only).")

    by_risk = math.floor(risk_budget / risk_per_share)
    by_concentration = math.floor(capital * max_position_pct / 100.0 / entry)
    cash = capital if available_cash is None else available_cash
    by_cash = math.floor(cash / entry)

    qty = min(by_risk, by_concentration, by_cash)
    limited = "RISK"
    if qty == by_concentration and by_concentration < by_risk:
        limited = "CONCENTRATION"
    if qty == by_cash and by_cash < min(by_risk, by_concentration):
        limited = "CASH"
    qty = max(qty, 0)
    return PositionSize(qty, round(qty * entry, 2), round(risk_budget, 2), round(risk_per_share, 2),
                        round(qty * risk_per_share, 2), limited,
                        "" if qty > 0 else "Risk budget too small for even one share at this stop distance.")
