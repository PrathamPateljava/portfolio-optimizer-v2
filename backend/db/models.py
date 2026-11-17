"""SQLAlchemy models + engine/session for the SQLite prototype.

The schema is written for you (it's a contract). CRUD logic is in crud.py for
you to implement.

A Portfolio has many Holdings. Each Holding stores a target weight for a ticker.
"""
from __future__ import annotations

from sqlalchemy import ForeignKey, String, Float, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from ..config import settings


class Base(DeclarativeBase):
    pass


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    market: Mapped[str] = mapped_column(String(16))  # "NSE" | "NASDAQ"

    holdings: Mapped[list["Holding"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )


class Holding(Base):
    __tablename__ = "holdings"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"))
    ticker: Mapped[str] = mapped_column(String(32))
    weight: Mapped[float] = mapped_column(Float)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="holdings")


# SQLite needs check_same_thread=False when used from FastAPI's threadpool.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create tables. Called on FastAPI startup."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """FastAPI dependency: yields a session and closes it afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
