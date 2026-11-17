"""Starter tests to sanity-check your MPT math as you implement it.

Run with:  pytest -q

These will fail until you implement the functions. They encode a few properties
that MUST hold if your math is right — use them as a spec, not an afterthought.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.config import Market
from backend.core import mpt


@pytest.fixture
def toy_inputs():
    """Two assets, known mu/Sigma so results are hand-checkable."""
    tickers = ["A", "B"]
    mu = pd.Series([0.10, 0.20], index=tickers)
    sigma = pd.DataFrame([[0.04, 0.00], [0.00, 0.09]], index=tickers, columns=tickers)
    return mu, sigma


def test_weights_sum_to_one(toy_inputs):
    mu, sigma = toy_inputs
    w = mpt.min_variance(mu, sigma)
    assert np.isclose(w.sum(), 1.0)


def test_long_only_weights_nonnegative(toy_inputs):
    mu, sigma = toy_inputs
    w = mpt.max_sharpe(mu, sigma, risk_free=0.03, allow_short=False)
    assert (w >= -1e-8).all()


def test_min_variance_favors_lower_variance_asset(toy_inputs):
    # Asset A has lower variance (0.04 < 0.09), so min-variance should tilt to A.
    mu, sigma = toy_inputs
    w = mpt.min_variance(mu, sigma)
    assert w[0] > w[1]


def test_performance_shapes(toy_inputs):
    mu, sigma = toy_inputs
    w = np.array([0.5, 0.5])
    ret, vol, sharpe = mpt.portfolio_performance(w, mu, sigma, risk_free=0.03)
    assert vol > 0
    assert np.isclose(sharpe, (ret - 0.03) / vol)
