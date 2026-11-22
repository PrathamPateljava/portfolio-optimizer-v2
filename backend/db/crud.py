"""Database CRUD operations against the models in models.py.

Thin: take a Session + typed inputs, return ORM objects. The route layer maps
ORM objects <-> Pydantic schemas. Portfolio reads/writes are scoped to a user id
so one account can never see or touch another's books.
"""
from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..schemas import PortfolioCreate
from .models import Holding, Portfolio, User


# --- Users -----------------------------------------------------------------

def get_user_by_email(db: Session, email: str) -> User | None:
    """Look up a user by email (case-sensitive as stored), or None."""
    return db.scalar(select(User).where(User.email == email))


def get_user(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def create_user(db: Session, email: str, password_hash: str) -> User:
    """Insert a new user. Caller must have checked the email is free."""
    user = User(email=email, password_hash=password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# --- Portfolios (always scoped to a user) ----------------------------------

def create_portfolio(db: Session, data: PortfolioCreate, user_id: int) -> Portfolio:
    """Insert a Portfolio (owned by user_id) plus its Holdings."""
    portfolio = Portfolio(
        name=data.name,
        market=data.market.value,
        user_id=user_id,
        result_json=json.dumps(data.snapshot) if data.snapshot is not None else None,
    )
    portfolio.holdings = [
        Holding(ticker=h.ticker, weight=h.weight) for h in data.holdings
    ]
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio


def get_portfolio(db: Session, portfolio_id: int, user_id: int) -> Portfolio | None:
    """Fetch one portfolio by id ONLY if it belongs to user_id, else None.

    Returning None for someone else's portfolio lets the route 404 without
    leaking whether that id exists.
    """
    portfolio = db.get(Portfolio, portfolio_id)
    if portfolio is None or portfolio.user_id != user_id:
        return None
    return portfolio


def list_portfolios(db: Session, user_id: int) -> list[Portfolio]:
    """Return all portfolios owned by this user."""
    return list(db.scalars(select(Portfolio).where(Portfolio.user_id == user_id)).all())


def rename_portfolio(db: Session, portfolio_id: int, user_id: int, name: str) -> Portfolio | None:
    """Rename a portfolio owned by user_id; return it, or None if not found/owned."""
    portfolio = get_portfolio(db, portfolio_id, user_id)
    if portfolio is None:
        return None
    portfolio.name = name
    db.commit()
    db.refresh(portfolio)
    return portfolio


def delete_portfolio(db: Session, portfolio_id: int, user_id: int) -> bool:
    """Delete a portfolio owned by user_id; return True if something was deleted."""
    portfolio = get_portfolio(db, portfolio_id, user_id)
    if portfolio is None:
        return False
    db.delete(portfolio)
    db.commit()
    return True
