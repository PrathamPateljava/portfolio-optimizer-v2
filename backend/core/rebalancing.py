"""Automated rebalancing.

Given a portfolio's CURRENT (drifted) weights and freshly optimized TARGET
weights, decide what to trade under a drift-band policy:
    delta_i  = target_i - current_i     (>0 buy, <0 sell)
    act only when |delta_i| > band       (avoids churn on tiny drifts)
    turnover = sum(|delta_i|) / 2         (fraction of the portfolio traded)
"""
from __future__ import annotations

from ..schemas import RebalanceAction


def compute_rebalance(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    drift_band: float,
) -> tuple[list[RebalanceAction], float]:
    """Return (actions, turnover)."""
    tickers = sorted(set(current_weights) | set(target_weights))
    actions: list[RebalanceAction] = []
    total_abs = 0.0

    for t in tickers:
        cur = current_weights.get(t, 0.0)
        tgt = target_weights.get(t, 0.0)
        delta = tgt - cur
        total_abs += abs(delta)
        if abs(delta) > drift_band:
            actions.append(
                RebalanceAction(ticker=t, current_weight=cur, target_weight=tgt, delta=delta)
            )

    return actions, total_abs / 2.0
