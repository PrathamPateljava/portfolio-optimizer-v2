"""Black-Litterman model.

Starts from the market-implied "equilibrium" excess returns (a stable prior)
and nudges them toward the investor's views, weighted by confidence. The output
posterior mu/Sigma go straight into the MPT optimizer.

PIPELINE
--------
  1. Prior (equilibrium excess returns):   Pi = delta * Sigma * w_mkt
       w_mkt = market weights (prototype uses equal weights)
       delta = market risk aversion (prototype default ~2.5)
  2. Views -> (P, Q, Omega):
       P     : k x n picking matrix (row per view; +1 absolute, +1/-1 relative)
       Q     : k view returns
       Omega : k x k diagonal view uncertainty. Base = diag(P (tau Sigma) P^T),
               scaled by (1-confidence)/confidence so higher confidence -> smaller
               Omega -> the view is enforced more strongly.
  3. Posterior returns:
       E[R] = [ (tau Sigma)^-1 + P^T Omega^-1 P ]^-1
              @ [ (tau Sigma)^-1 Pi + P^T Omega^-1 Q ]
  4. Posterior covariance:
       M          = [ (tau Sigma)^-1 + P^T Omega^-1 P ]^-1
       Sigma_post = Sigma + M

NOTE: Pi and E[R] are EXCESS returns (over the risk-free rate). The API route
adds r_f back to get total expected returns before optimizing/reporting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT_TAU
from ..schemas import View


def equilibrium_returns(
    sigma: pd.DataFrame, market_weights: np.ndarray, delta: float
) -> pd.Series:
    """Reverse-optimize market weights into implied excess returns Pi."""
    sigma_arr = np.asarray(sigma, dtype=float)
    w = np.asarray(market_weights, dtype=float)
    pi = delta * (sigma_arr @ w)
    return pd.Series(pi, index=sigma.index)


def build_views(
    views: list[View], tickers: list[str], sigma: pd.DataFrame, tau: float = DEFAULT_TAU
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn API View objects into (P, Q, Omega)."""
    k, n = len(views), len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    P = np.zeros((k, n))
    Q = np.zeros(k)

    for row, view in enumerate(views):
        for asset, weight in zip(view.assets, view.weights):
            if asset not in idx:
                raise ValueError(f"view references unknown ticker: {asset}")
            P[row, idx[asset]] = weight
        Q[row] = view.q

    sigma_arr = np.asarray(sigma, dtype=float)
    if k == 0:
        return P, Q, np.zeros((0, 0))

    base = np.diag(P @ (tau * sigma_arr) @ P.T).copy()
    conf = np.array([v.confidence for v in views])
    # (1-conf)/conf: 1.0 at conf=0.5, -> 0 as conf-> 1 (view enforced), large as conf->0.
    factor = (1.0 - conf) / np.clip(conf, 1e-6, 1.0)
    scaled = np.maximum(base * factor, 1e-8)  # floor keeps Omega invertible
    Omega = np.diag(scaled)
    return P, Q, Omega


def posterior(
    pi: pd.Series,
    sigma: pd.DataFrame,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = DEFAULT_TAU,
) -> tuple[pd.Series, pd.DataFrame]:
    """Combine prior + views into posterior (mu, Sigma)."""
    idx = sigma.index
    sigma_arr = np.asarray(sigma, dtype=float)
    pi_arr = np.asarray(pi, dtype=float)

    # No views -> posterior is just the prior.
    if P.shape[0] == 0:
        return pd.Series(pi_arr, index=idx), pd.DataFrame(sigma_arr, index=idx, columns=idx)

    tau_sigma_inv = np.linalg.inv(tau * sigma_arr)
    omega_inv = np.linalg.inv(Omega)

    M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    er = M @ (tau_sigma_inv @ pi_arr + P.T @ omega_inv @ Q)
    sigma_post = sigma_arr + M

    return pd.Series(er, index=idx), pd.DataFrame(sigma_post, index=idx, columns=idx)
