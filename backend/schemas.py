"""Pydantic request/response contracts shared by the API and the frontend.

These are the interface between backend and Streamlit, so they're written for
you. When you implement the routes in `api/routes.py`, these types tell you
exactly what shape of data goes in and comes out.
"""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, EmailStr, Field

from .config import Market


class OptimizationMethod(str, Enum):
    MIN_VARIANCE = "min_variance"          # MPT: minimize portfolio variance
    MAX_SHARPE = "max_sharpe"              # MPT: tangency portfolio (max Sharpe)
    EFFICIENT_FRONTIER = "efficient_frontier"  # MPT: sweep the frontier
    BLACK_LITTERMAN = "black_litterman"    # BL posterior -> then optimize


# --- Black-Litterman views -------------------------------------------------

class View(BaseModel):
    """A single investor view for Black-Litterman.

    Two supported forms:
      * Absolute: "I expect ASSET to return `q` annually" -> assets=[A], weights=[1.0]
      * Relative: "A will outperform B by `q`"            -> assets=[A, B], weights=[1.0, -1.0]

    `confidence` in (0, 1] is your certainty; it maps to the Omega matrix.
    """
    assets: list[str] = Field(..., description="Tickers this view is about")
    weights: list[float] = Field(..., description="Picking-row entries; sum to 0 for relative views")
    q: float = Field(..., description="Expected (annualized) return implied by the view")
    confidence: float = Field(0.5, gt=0, le=1)


# --- Optimization request/response ----------------------------------------

class OptimizeRequest(BaseModel):
    market: Market
    tickers: list[str] = Field(..., min_length=2)
    method: OptimizationMethod
    lookback_days: int = Field(365, ge=60, description="History window for returns/covariance")
    allow_short: bool = False
    target_return: float | None = Field(None, description="Only for a single point on the frontier")
    views: list[View] = Field(default_factory=list, description="Only used for BLACK_LITTERMAN")


class Allocation(BaseModel):
    ticker: str
    weight: float


class PortfolioMetrics(BaseModel):
    expected_return: float   # annualized
    volatility: float        # annualized
    sharpe: float
    risk_free_rate: float


class FrontierPoint(BaseModel):
    volatility: float
    expected_return: float
    sharpe: float


class OptimizeResponse(BaseModel):
    method: OptimizationMethod
    allocations: list[Allocation]
    metrics: PortfolioMetrics
    frontier: list[FrontierPoint] = Field(default_factory=list)


# --- Persistence-facing models ---------------------------------------------

class HoldingIn(BaseModel):
    ticker: str
    weight: float


class PortfolioCreate(BaseModel):
    name: str
    market: Market
    holdings: list[HoldingIn]


class PortfolioOut(BaseModel):
    id: int
    name: str
    market: Market
    holdings: list[HoldingIn]


class PortfolioUpdate(BaseModel):
    """Editable fields on a saved portfolio (currently just the name)."""
    name: str = Field(..., min_length=1, max_length=120)


# --- Rebalancing ------------------------------------------------------------

class RebalanceRequest(BaseModel):
    portfolio_id: int
    method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
    drift_band: float = Field(0.05, ge=0, description="Absolute band: trade a holding if |current-target| exceeds this")
    relative_band: float = Field(
        0.25, ge=0,
        description="5/25 rule: also trade when a holding drifts past this fraction of its OWN "
                    "target weight (whichever band is tighter wins). 0 disables the relative arm.",
    )


class RebalanceAction(BaseModel):
    ticker: str
    current_weight: float
    target_weight: float
    delta: float           # target - current (positive = buy, negative = sell)


class RebalanceResponse(BaseModel):
    portfolio_id: int
    actions: list[RebalanceAction]
    turnover: float        # sum of |delta| / 2
    rebalanced_metrics: PortfolioMetrics


# --- Auth -------------------------------------------------------------------

class UserCreate(BaseModel):
    """Signup / login payload."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    email: EmailStr


class Token(BaseModel):
    """Returned by /auth/register and /auth/login."""
    access_token: str
    token_type: str = "bearer"
    email: EmailStr


# --- Misc -------------------------------------------------------------------

class PriceQuote(BaseModel):
    ticker: str
    price: float
    as_of: date
