"""Shared exit logic used by both the backtester and paper trading (so they can never disagree)."""
from __future__ import annotations

import numpy as np


def find_exit(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, stop: float,
              target: float | None, max_hold: int) -> tuple[int, float, str] | None:
    """Scan bars in order (index 0 = first bar the position is exposed to) for the first exit.

    Conservative rules:
      * if a bar touches both stop and target, the STOP is assumed to have hit first;
      * a gap through the stop fills at the open (worse than the stop); a gap above target fills at the open;
      * after `max_hold` bars without a stop/target the position exits at that bar's close (TIME).
    Returns (bar_index, price, reason) or None if still open when the data ends.
    """
    for k in range(len(c)):
        if l[k] <= stop:
            return k, float(min(o[k], stop)), "STOP"
        if target is not None and h[k] >= target:
            return k, float(max(o[k], target)), "TARGET"
        if k + 1 >= max_hold:
            return k, float(c[k]), "TIME"
    return None
