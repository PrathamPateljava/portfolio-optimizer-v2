"""Performance metrics and the risk-free-rate comparison.

Every optimized portfolio reports its Sharpe (excess return per unit risk over
the market RFR) and its raw excess return vs the RFR:
    mu_p   = w . mu
    sig_p  = sqrt(w^T Sigma w)
    Sharpe = (mu_p - r_f) / sig_p
    alpha  = mu_p - r_f          (excess vs RFR)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import PortfolioMetrics
from .mpt import portfolio_performance


def portfolio_metrics(
    weights: np.ndarray, mu: pd.Series, sigma: pd.DataFrame, risk_free: float
) -> PortfolioMetrics:
    """Package return/vol/Sharpe into the response schema."""
    ret, vol, sharpe = portfolio_performance(weights, mu, sigma, risk_free)
    return PortfolioMetrics(
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        risk_free_rate=risk_free,
    )
