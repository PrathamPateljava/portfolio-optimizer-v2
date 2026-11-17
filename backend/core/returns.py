"""Expected returns and the covariance matrix — the inputs every model needs.

These feed both MPT and Black-Litterman.

KEY EQUATIONS
-------------
Daily simple returns from a price series P_t:  r_t = P_t / P_{t-1} - 1
Annualization (D = trading days/year from config.TRADING_DAYS[market]):
    mu_annual    = mean(r_daily) * D
    Sigma_annual = cov(r_daily)  * D

Sample covariance can be near-singular with few observations / correlated
assets. Ledoit-Wolf shrinkage (sklearn.covariance.LedoitWolf) is a noted upgrade
if the optimizer ever produces extreme weights.
"""
from __future__ import annotations

import pandas as pd

from ..config import TRADING_DAYS, Market


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Aligned prices -> daily simple returns (first NaN row dropped)."""
    return prices.pct_change().dropna(how="any")


def expected_annual_returns(prices: pd.DataFrame, market: Market) -> pd.Series:
    """Annualized mean return per asset. Index = tickers."""
    return daily_returns(prices).mean() * TRADING_DAYS[market]


def annual_covariance(prices: pd.DataFrame, market: Market) -> pd.DataFrame:
    """Annualized covariance matrix of returns. Symmetric, tickers x tickers."""
    return daily_returns(prices).cov() * TRADING_DAYS[market]
