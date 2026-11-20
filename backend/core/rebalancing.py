"""Automated rebalancing.

Given a portfolio's CURRENT (drifted) weights and freshly optimized TARGET
weights, decide what to trade. Two bands gate each holding (the "5/25" rule):

    delta_i    = target_i - current_i          (>0 buy, <0 sell)
    threshold_i = max(min(drift_band, relative_band * target_i), DUST_FLOOR)
    act only when |delta_i| > threshold_i
    turnover   = sum(|delta_i|) / 2             (fraction of the portfolio traded)

The DUST_FLOOR stops no-op trades: a holding with a ~0 target gets a ~0 relative
threshold and would otherwise trip on floating-point dust (a delta that rounds to
0.00%). A genuine exit still has a meaningful delta and trades through.

WHY TWO BANDS: a single absolute band treats a 40% position and a 1.7% position
identically — 5 points is nothing for the big one and impossible to breach for
the small one, so small holdings never rebalance. The relative band
(`relative_band` as a fraction of each holding's OWN target weight) fixes that:
big positions are still governed by the absolute band, small positions by the
proportional one, whichever is tighter.

  * big  (target 40%): min(5%, 25% x 40% = 10%) = 5%   -> absolute band governs
  * small(target 1.7%): min(5%, 25% x 1.7% = 0.42%) = 0.42% -> relative governs

`relative_band = 0` disables the relative arm (pure absolute band). A holding
the optimizer now wants at 0% has threshold 0, so an existing position in it is
flagged for a full exit rather than frozen by the absolute band.

NOTE: rebalance PERIODICALLY, not daily. Monthly or quarterly is the sweet spot;
quarterly is a good default. Checking daily/weekly just adds trading costs and
chases noise — the bands already suppress tiny trades, and fundamentals barely
move on a weekly horizon anyway.
"""
from __future__ import annotations

from ..schemas import RebalanceAction

# Minimum |delta| worth a trade (fraction of the portfolio). Guards the relative
# band from flagging dust on holdings whose target is ~0. 0.002 = 0.2 points.
_DUST_FLOOR = 0.002


def compute_rebalance(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    drift_band: float,
    relative_band: float = 0.0,
) -> tuple[list[RebalanceAction], float]:
    """Return (actions, turnover) using the absolute + relative (5/25) bands."""
    tickers = sorted(set(current_weights) | set(target_weights))
    actions: list[RebalanceAction] = []
    total_abs = 0.0

    for t in tickers:
        cur = current_weights.get(t, 0.0)
        tgt = target_weights.get(t, 0.0)
        delta = tgt - cur
        total_abs += abs(delta)

        # Per-holding threshold: the tighter of the absolute band and the
        # relative band (a fraction of this holding's own target weight).
        threshold = drift_band
        if relative_band > 0:
            threshold = min(drift_band, relative_band * tgt)
        threshold = max(threshold, _DUST_FLOOR)  # never trade on no-op dust

        if abs(delta) > threshold:
            actions.append(
                RebalanceAction(ticker=t, current_weight=cur, target_weight=tgt, delta=delta)
            )

    return actions, total_abs / 2.0
