"""Modern Portfolio Theory (Markowitz mean-variance) optimization.

Reused by Black-Litterman: BL only changes the inputs (mu, Sigma), then calls
these routines.

NOTATION:  w weights (sum 1), mu expected annual returns, Sigma annual cov,
           r_f annual risk-free rate.
  return   mu_p  = w . mu
  variance var_p = w^T Sigma w
  vol      sig_p = sqrt(var_p)
  Sharpe   S     = (mu_p - r_f) / sig_p

All optimizers use scipy.optimize.minimize (SLSQP) with sum(w)=1 and per-asset
bounds. Long-only => [0,1]; allow_short => [-1,1]. Note the max-Sharpe objective
is non-convex, so SLSQP finds a local optimum; the equal-weight start is a
reliable x0 for the small universes in this prototype.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_SUM_TO_ONE = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _bounds(n: int, allow_short: bool) -> list[tuple[float, float]]:
    return [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n


def _as_arrays(mu: pd.Series, sigma: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)


def portfolio_performance(
    weights: np.ndarray, mu: pd.Series, sigma: pd.DataFrame, risk_free: float
) -> tuple[float, float, float]:
    """Return (annual_return, annual_vol, sharpe) for a weight vector."""
    w = np.asarray(weights, dtype=float)
    mu_arr, sigma_arr = _as_arrays(mu, sigma)
    ret = float(w @ mu_arr)
    var = float(w @ sigma_arr @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - risk_free) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def min_variance(mu: pd.Series, sigma: pd.DataFrame, allow_short: bool = False) -> np.ndarray:
    """Weights of the global minimum-variance portfolio."""
    n = len(mu)
    _, sigma_arr = _as_arrays(mu, sigma)
    x0 = np.repeat(1.0 / n, n)
    res = minimize(
        lambda w: w @ sigma_arr @ w,
        x0,
        method="SLSQP",
        bounds=_bounds(n, allow_short),
        constraints=[_SUM_TO_ONE],
    )
    return res.x


def max_sharpe(
    mu: pd.Series, sigma: pd.DataFrame, risk_free: float, allow_short: bool = False
) -> np.ndarray:
    """Weights of the maximum-Sharpe (tangency) portfolio."""
    n = len(mu)
    mu_arr, sigma_arr = _as_arrays(mu, sigma)
    x0 = np.repeat(1.0 / n, n)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(w @ sigma_arr @ w)
        if vol <= 0:
            return 1e9
        return -(w @ mu_arr - risk_free) / vol

    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=_bounds(n, allow_short),
        constraints=[_SUM_TO_ONE],
    )
    return res.x


def efficient_return(
    mu: pd.Series, sigma: pd.DataFrame, target_return: float, allow_short: bool = False
) -> np.ndarray:
    """Min-variance weights achieving a specific target return."""
    n = len(mu)
    mu_arr, sigma_arr = _as_arrays(mu, sigma)
    x0 = np.repeat(1.0 / n, n)
    cons = [
        _SUM_TO_ONE,
        {"type": "eq", "fun": lambda w, m=mu_arr, t=target_return: w @ m - t},
    ]
    res = minimize(
        lambda w: w @ sigma_arr @ w,
        x0,
        method="SLSQP",
        bounds=_bounds(n, allow_short),
        constraints=cons,
    )
    return res.x


def efficient_frontier(
    mu: pd.Series, sigma: pd.DataFrame, n_points: int = 25, allow_short: bool = False
) -> list[tuple[float, float, np.ndarray]]:
    """Trace the frontier: list of (volatility, expected_return, weights)."""
    mu_arr, _ = _as_arrays(mu, sigma)
    targets = np.linspace(mu_arr.min(), mu_arr.max(), n_points)
    out: list[tuple[float, float, np.ndarray]] = []
    for t in targets:
        w = efficient_return(mu, sigma, float(t), allow_short)
        ret, vol, _ = portfolio_performance(w, mu, sigma, 0.0)
        # Keep only points where the target-return constraint was actually met.
        if abs(ret - t) < 1e-3:
            out.append((vol, ret, w))
    return out
