"""Database CRUD operations against the models in models.py.

Thin: take a Session + typed inputs, return ORM objects. The route layer maps
ORM objects <-> Pydantic schemas.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..schemas import PortfolioCreate
from .models import Holding, Portfolio


def create_portfolio(db: Session, data: PortfolioCreate) -> Portfolio:
    """Insert a Portfolio plus its Holdings and return the persisted object."""
    portfolio = Portfolio(name=data.name, market=data.market.value)
    portfolio.holdings = [
        Holding(ticker=h.ticker, weight=h.weight) for h in data.holdings
    ]
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio


def get_portfolio(db: Session, portfolio_id: int) -> Portfolio | None:
    """Fetch one portfolio (with holdings) by id, or None."""
    return db.get(Portfolio, portfolio_id)


def list_portfolios(db: Session) -> list[Portfolio]:
    """Return all portfolios."""
    return list(db.scalars(select(Portfolio)).all())


def delete_portfolio(db: Session, portfolio_id: int) -> bool:
    """Delete a portfolio; return True if something was deleted."""
    portfolio = db.get(Portfolio, portfolio_id)
    if portfolio is None:
        return False
    db.delete(portfolio)
    db.commit()
    return True
