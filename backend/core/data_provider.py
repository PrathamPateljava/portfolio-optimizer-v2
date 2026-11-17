"""Live price data access, behind a small interface so it can be swapped later.

Prototype implementation: yfinance (free, no key, covers NSE via the `.NS`
suffix and NASDAQ directly).

WHY AN INTERFACE: yfinance is behind `PriceProvider`, so a real-time provider
(Alpha Vantage, Finnhub, Zerodha Kite) can be dropped in later as a new subclass
without touching the rest of the app.

GOTCHAS handled here:
  * NSE tickers need the ".NS" suffix (RELIANCE.NS). See config.TICKER_SUFFIX.
  * NSE prices are INR, NASDAQ USD -> keep a portfolio single-market so returns
    are unit-consistent (enforced upstream by the single-market request).
  * NSE and NASDAQ have different calendars -> we align on the intersection of
    dates (dropna) before returning, so covariance is well-defined.
  * yfinance can rate-limit / return empty -> we cache briefly and raise clear
    errors on empty frames.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import yfinance as yf

from ..config import Market, RFR_SYMBOL, settings

# Tiny in-memory TTL cache so repeated Streamlit interactions don't hammer yahoo.
_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}
_CACHE_TTL = 300  # seconds


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull adjusted-close columns out of a yfinance frame (single or multi)."""
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        # Single ticker: columns are plain field names.
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]
    return close


class PriceProvider:
    """Abstract price source."""

    def get_price_history(self, tickers: list[str], lookback_days: int) -> pd.DataFrame:
        raise NotImplementedError

    def get_latest_price(self, ticker: str) -> float:
        raise NotImplementedError


class YFinanceProvider(PriceProvider):
    """yfinance-backed implementation."""

    def get_price_history(self, tickers: list[str], lookback_days: int) -> pd.DataFrame:
        if len(tickers) < 1:
            raise ValueError("need at least one ticker")

        key = (tuple(sorted(tickers)), lookback_days)
        hit = _CACHE.get(key)
        if hit and (time.time() - hit[0]) < _CACHE_TTL:
            return hit[1]

        raw = yf.download(
            tickers,
            period=f"{lookback_days}d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw is None or raw.empty:
            raise ValueError(f"no price data returned for {tickers}")

        prices = _extract_close(raw, tickers)
        # Reorder to the requested ticker order, then drop any date with a gap so
        # every column is observed on the same trading days.
        prices = prices.reindex(columns=tickers).dropna(how="any")
        if prices.empty:
            raise ValueError(f"no overlapping trading days for {tickers}")

        _CACHE[key] = (time.time(), prices)
        return prices

    def get_latest_price(self, ticker: str) -> float:
        raw = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise ValueError(f"no price data for {ticker}")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            raise ValueError(f"no valid close for {ticker}")
        return float(close.iloc[-1])


def _normalize_yield(raw_value: float) -> float:
    """Normalize a ^TNX quote to a decimal rate.

    Yahoo may report the 10Y yield as a percent (4.28) or the legacy x10 form
    (42.8). Anything above ~25 can't be a plausible raw percent, so treat it as
    the x10 form.
    """
    v = float(raw_value)
    if v > 25:
        v /= 10.0
    return v / 100.0


def get_risk_free_rate(market: Market) -> float:
    """Annual risk-free rate (decimal) for a market, with a safe fallback."""
    fallback = settings.rfr_india_fallback if market == Market.NSE else settings.rfr_us_fallback
    symbol = RFR_SYMBOL.get(market)
    if symbol is None:
        return fallback
    try:
        raw = yf.download(symbol, period="5d", auto_adjust=False, progress=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            return fallback
        return _normalize_yield(close.iloc[-1])
    except Exception:
        return fallback
