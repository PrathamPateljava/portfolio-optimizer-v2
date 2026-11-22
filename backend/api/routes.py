"""HTTP endpoints. Orchestration only: each route wires together core/ and db/
functions and contains almost no math itself.
"""
from __future__ import annotations

from datetime import date

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..config import Market
from ..core import black_litterman as bl
from ..core import metrics as metrics_mod
from ..core import mpt, rebalancing, returns
from ..core.data_provider import YFinanceProvider, get_risk_free_rate
from ..db import crud
from ..db.models import Portfolio, User, get_session
from .deps import get_current_user
from ..schemas import (
    Allocation,
    FrontierPoint,
    HoldingIn,
    OptimizationMethod,
    OptimizeRequest,
    OptimizeResponse,
    PortfolioCreate,
    PortfolioMetrics,
    PortfolioOut,
    PortfolioUpdate,
    PriceQuote,
    RebalanceRequest,
    RebalanceResponse,
)

router = APIRouter()

_DEFAULT_DELTA = 2.5  # market risk-aversion for the Black-Litterman prior


def _run_optimization(
    req: OptimizeRequest,
) -> tuple[np.ndarray, list[str], object, object, float, list[FrontierPoint]]:
    """Shared pipeline used by /optimize and /rebalance.

    Returns (weights, tickers, mu_used, sigma_used, risk_free, frontier_points).
    mu_used/sigma_used are the inputs the metrics should be computed against
    (historical for MPT, posterior for Black-Litterman).
    """
    provider = YFinanceProvider()
    prices = provider.get_price_history(req.tickers, req.lookback_days)

    market = req.market
    mu = returns.expected_annual_returns(prices, market)
    sigma = returns.annual_covariance(prices, market)
    rf = get_risk_free_rate(market)
    tickers = list(mu.index)
    frontier: list[FrontierPoint] = []

    if req.method == OptimizationMethod.MIN_VARIANCE:
        w = mpt.min_variance(mu, sigma, req.allow_short)
        mu_used, sigma_used = mu, sigma

    elif req.method == OptimizationMethod.MAX_SHARPE:
        w = mpt.max_sharpe(mu, sigma, rf, req.allow_short)
        mu_used, sigma_used = mu, sigma

    elif req.method == OptimizationMethod.EFFICIENT_FRONTIER:
        curve = mpt.efficient_frontier(mu, sigma, allow_short=req.allow_short)
        best_w, best_sharpe = None, float("-inf")
        for vol, ret, weights in curve:
            sharpe = (ret - rf) / vol if vol > 0 else float("-inf")
            frontier.append(FrontierPoint(volatility=vol, expected_return=ret, sharpe=sharpe))
            if sharpe > best_sharpe:
                best_sharpe, best_w = sharpe, weights
        # Fall back to a direct max-Sharpe solve if the frontier came back empty.
        w = best_w if best_w is not None else mpt.max_sharpe(mu, sigma, rf, req.allow_short)
        mu_used, sigma_used = mu, sigma

    elif req.method == OptimizationMethod.BLACK_LITTERMAN:
        n = len(tickers)
        w_mkt = np.repeat(1.0 / n, n)  # prototype shortcut: equal market weights
        pi = bl.equilibrium_returns(sigma, w_mkt, _DEFAULT_DELTA)
        P, Q, Omega = bl.build_views(req.views, tickers, sigma)
        excess, sigma_post = bl.posterior(pi, sigma, P, Q, Omega)
        mu_total = excess + rf  # BL works in excess returns; add rf back for total
        w = mpt.max_sharpe(mu_total, sigma_post, rf, req.allow_short)
        mu_used, sigma_used = mu_total, sigma_post

    else:  # pragma: no cover - enum is exhaustive
        raise HTTPException(status_code=400, detail=f"unknown method: {req.method}")

    return w, tickers, mu_used, sigma_used, rf, frontier


def _to_portfolio_out(p: Portfolio) -> PortfolioOut:
    return PortfolioOut(
        id=p.id,
        name=p.name,
        market=Market(p.market),
        holdings=[HoldingIn(ticker=h.ticker, weight=h.weight) for h in p.holdings],
    )


# --- Optimization ----------------------------------------------------------

@router.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> OptimizeResponse:
    """Run the chosen model and return weights + metrics (+ frontier)."""
    try:
        w, tickers, mu_used, sigma_used, rf, frontier = _run_optimization(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    allocations = [Allocation(ticker=t, weight=float(wi)) for t, wi in zip(tickers, w)]
    metrics = metrics_mod.portfolio_metrics(w, mu_used, sigma_used, rf)
    return OptimizeResponse(
        method=req.method, allocations=allocations, metrics=metrics, frontier=frontier
    )


# --- Prices ----------------------------------------------------------------

@router.get("/prices/{ticker}", response_model=PriceQuote)
def latest_price(ticker: str) -> PriceQuote:
    """Latest price for one ticker."""
    try:
        price = YFinanceProvider().get_latest_price(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return PriceQuote(ticker=ticker, price=price, as_of=date.today())


# --- Portfolio CRUD --------------------------------------------------------

@router.post("/portfolios", response_model=PortfolioOut)
def create_portfolio(
    data: PortfolioCreate,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> PortfolioOut:
    """Persist a portfolio owned by the current user."""
    return _to_portfolio_out(crud.create_portfolio(db, data, user.id))


@router.get("/portfolios", response_model=list[PortfolioOut])
def list_portfolios(
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> list[PortfolioOut]:
    """List the current user's saved portfolios."""
    return [_to_portfolio_out(p) for p in crud.list_portfolios(db, user.id)]


@router.get("/portfolios/{portfolio_id}", response_model=PortfolioOut)
def get_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> PortfolioOut:
    """Fetch one of the current user's portfolios; 404 if missing or not theirs."""
    portfolio = crud.get_portfolio(db, portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="portfolio not found")
    return _to_portfolio_out(portfolio)


@router.patch("/portfolios/{portfolio_id}", response_model=PortfolioOut)
def rename_portfolio(
    portfolio_id: int,
    data: PortfolioUpdate,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> PortfolioOut:
    """Rename one of the current user's portfolios; 404 if missing or not theirs."""
    portfolio = crud.rename_portfolio(db, portfolio_id, user.id, data.name)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="portfolio not found")
    return _to_portfolio_out(portfolio)


@router.delete("/portfolios/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> None:
    """Delete one of the current user's portfolios; 404 if missing or not theirs."""
    if not crud.delete_portfolio(db, portfolio_id, user.id):
        raise HTTPException(status_code=404, detail="portfolio not found")


# --- Rebalancing -----------------------------------------------------------

@router.post("/rebalance", response_model=RebalanceResponse)
def rebalance(
    req: RebalanceRequest,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> RebalanceResponse:
    """Compute rebalancing actions for one of the current user's portfolios."""
    portfolio = crud.get_portfolio(db, req.portfolio_id, user.id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="portfolio not found")

    market = Market(portfolio.market)
    stored = {h.ticker: h.weight for h in portfolio.holdings}
    tickers = list(stored.keys())
    if len(tickers) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 holdings to rebalance")

    # Derive CURRENT weights by drifting the stored weights with each asset's
    # price change over the lookback window: w_now_i ∝ w_stored_i * (P_last / P_first).
    try:
        prices = YFinanceProvider().get_price_history(tickers, 365)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    first, last = prices.iloc[0], prices.iloc[-1]
    drifted = {t: stored[t] * float(last[t] / first[t]) for t in tickers}
    total = sum(drifted.values())
    current = {t: drifted[t] / total for t in tickers}

    # Re-optimize for the TARGET weights using the shared pipeline.
    opt_req = OptimizeRequest(
        market=market, tickers=tickers, method=req.method, lookback_days=365, allow_short=False
    )
    w, opt_tickers, mu_used, sigma_used, rf, _ = _run_optimization(opt_req)
    target = {t: float(wi) for t, wi in zip(opt_tickers, w)}

    actions, turnover = rebalancing.compute_rebalance(
        current, target, req.drift_band, req.relative_band
    )
    metrics = metrics_mod.portfolio_metrics(w, mu_used, sigma_used, rf)
    return RebalanceResponse(
        portfolio_id=req.portfolio_id, actions=actions, turnover=turnover, rebalanced_metrics=metrics
    )
