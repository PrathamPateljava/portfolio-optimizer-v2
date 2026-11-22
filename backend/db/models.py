"""SQLAlchemy models + engine/session.

A User has many Portfolios. A Portfolio has many Holdings. Each Holding stores a
target weight for a ticker. Portfolios are scoped to the user that owns them;
`/optimize` stays public, but saving/listing/rebalancing requires an account.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from ..config import settings


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    portfolios: Mapped[list["Portfolio"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    market: Mapped[str] = mapped_column(String(16))  # "NSE" | "NASDAQ"
    # Nullable so the column can be back-filled onto an existing table and so
    # legacy pre-auth rows remain valid (they just won't belong to any user).
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    # JSON snapshot of the full optimize result shown when the portfolio was
    # saved (method, allocations, metrics, frontier). Lets the UI restore the
    # exact charts/returns/risk later instead of only the raw weights. Nullable
    # so legacy rows (and pre-snapshot saves) stay valid.
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Capital the user chose to deploy at save time. Nullable (legacy rows and
    # saves that skip it). Stored so growth can be measured against it later.
    initial_amount: Mapped[float | None] = mapped_column(Float, nullable=True)

    owner: Mapped["User | None"] = relationship(back_populates="portfolios")
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


def _ensure_user_id_column() -> None:
    """Back-fill the portfolios.user_id column on an already-existing table.

    `create_all` creates missing TABLES but never alters existing ones, so a DB
    (e.g. Neon) that predates auth would keep an old portfolios table with no
    user_id. This adds the column once, idempotently, across SQLite/Postgres.
    """
    insp = inspect(engine)
    if not insp.has_table("portfolios"):
        return
    cols = {c["name"] for c in insp.get_columns("portfolios")}
    if "user_id" in cols:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE portfolios ADD COLUMN user_id INTEGER"))


def _ensure_result_json_column() -> None:
    """Back-fill the portfolios.result_json column on an already-existing table.

    Same reasoning as _ensure_user_id_column: create_all never alters existing
    tables, so a DB that predates the snapshot feature keeps the old schema.
    Adds the column once, idempotently, across SQLite/Postgres.
    """
    insp = inspect(engine)
    if not insp.has_table("portfolios"):
        return
    cols = {c["name"] for c in insp.get_columns("portfolios")}
    if "result_json" in cols:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE portfolios ADD COLUMN result_json TEXT"))


def _ensure_initial_amount_column() -> None:
    """Back-fill the portfolios.initial_amount column on an existing table.

    Same reasoning as the other migrations here: create_all never alters an
    existing table. Idempotent across SQLite/Postgres.
    """
    insp = inspect(engine)
    if not insp.has_table("portfolios"):
        return
    cols = {c["name"] for c in insp.get_columns("portfolios")}
    if "initial_amount" in cols:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE portfolios ADD COLUMN initial_amount FLOAT"))


def init_db() -> None:
    """Create tables (users/portfolios/holdings) and run light migrations."""
    Base.metadata.create_all(bind=engine)
    _ensure_user_id_column()
    _ensure_result_json_column()
    _ensure_initial_amount_column()


def get_session():
    """FastAPI dependency: yields a session and closes it afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
