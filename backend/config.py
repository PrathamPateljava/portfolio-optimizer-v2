"""Central configuration and market constants.

This file is written for you (it's the spec, not logic). Adjust the constants
as you learn what your models need. Everything else in `core/` reads from here
so you have a single source of truth for things like trading-day counts and the
Black-Litterman tau.
"""
from __future__ import annotations

from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class Market(str, Enum):
    """The two exchanges supported by the prototype."""
    NSE = "NSE"        # National Stock Exchange of India (tickers use the .NS suffix on yfinance)
    NASDAQ = "NASDAQ"  # US tech-heavy exchange (plain tickers on yfinance, e.g. AAPL)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./portfolio.db"
    api_base_url: str = "http://127.0.0.1:8000"
    rfr_us_fallback: float = 0.043
    rfr_india_fallback: float = 0.070


settings = Settings()

# ---------------------------------------------------------------------------
# Market constants used across the optimization code.
# ---------------------------------------------------------------------------

# Trading days per year, used to annualize daily returns and covariance.
# NSE and NASDAQ have slightly different calendars.
TRADING_DAYS = {
    Market.NSE: 250,
    Market.NASDAQ: 252,
}

# yfinance ticker suffix per market. NSE needs ".NS"; NASDAQ needs nothing.
TICKER_SUFFIX = {
    Market.NSE: ".NS",
    Market.NASDAQ: "",
}

# yfinance symbol that yields a live-ish risk-free proxy per market.
# ^TNX = US 10Y Treasury yield. Yahoo now quotes it as a percent (e.g. 4.28),
# though a legacy x10 form (42.8) also shows up historically. data_provider
# normalizes both to a decimal. India has no clean free yfinance symbol for the
# 10Y G-Sec; use the fallback in settings for the prototype and revisit later.
RFR_SYMBOL = {
    Market.NSE: None,        # -> use settings.rfr_india_fallback
    Market.NASDAQ: "^TNX",
}

# Black-Litterman scalar tau (uncertainty in the prior). Small by convention.
DEFAULT_TAU = 0.05

# A small starter universe per market so you have something to test with.
DEFAULT_UNIVERSE = {
    Market.NSE: ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS"],
    Market.NASDAQ: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
}
