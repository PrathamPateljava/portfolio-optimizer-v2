# Portfolio Optimizer

A full-stack portfolio optimizer and manager. Build portfolios from **NSE (India)**
or **NASDAQ** stocks, optimize them with **Modern Portfolio Theory** or the
**Black-Litterman** model, get **automated rebalancing** suggestions, and compare
performance against the **risk-free rate** of the relevant market.

Prototype stack: **FastAPI** (backend/API) · **Streamlit** (frontend) ·
**yfinance** (live-ish prices) · **SQLite + SQLAlchemy** (persistence).

---

## Architecture

```
Streamlit UI  ──HTTP──►  FastAPI  ──►  core/ (finance & ML)  ──►  yfinance
(frontend/)              (backend/api)   returns, mpt,              (live prices)
                              │          black_litterman,
                              │          rebalancing, metrics
                              └──►  db/ (SQLAlchemy + SQLite)
```

The frontend only ever talks to the API over HTTP — it never imports the math
directly. The API layer (`routes.py`) is thin orchestration; all the finance/ML
lives in `core/` so it's testable in isolation. Black-Litterman reuses the MPT
optimizer — it only changes the `mu`/`Sigma` inputs.

### Layout

```
backend/
  config.py            Markets, trading-day counts, RFR sources, tau, universe  [written]
  schemas.py           Pydantic request/response contracts                      [written]
  main.py              FastAPI app wiring                                        [written]
  api/routes.py        HTTP endpoints — orchestration only                      [YOU: bodies]
  core/
    data_provider.py   yfinance access behind a swappable interface             [YOU]
    returns.py         expected returns + covariance                            [YOU]
    mpt.py             Markowitz: min-var, max-Sharpe, efficient frontier       [YOU]
    black_litterman.py equilibrium prior + views -> posterior mu/Sigma          [YOU]
    rebalancing.py     drift-band rebalancing actions + turnover                [YOU]
    metrics.py         Sharpe + excess-return-vs-RFR                            [YOU]
  db/
    models.py          SQLAlchemy Portfolio/Holding + engine/session            [written]
    crud.py            create/get/list/delete                                   [YOU]
frontend/
  streamlit_app.py     UI skeleton                                              [YOU: API calls + charts]
tests/
  test_mpt.py          starter property tests for your MPT math                 [written]
```

`[written]` = scaffolding/contracts done for you. `[YOU]` = the logic you implement.

---

## Setup

```bash
cd portfolio-optimizer-app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # tweak if you like
```

Run the backend and frontend in two terminals:

```bash
# terminal 1 — API (http://127.0.0.1:8000/docs for interactive docs)
uvicorn backend.main:app --reload

# terminal 2 — UI
streamlit run frontend/streamlit_app.py
```

Run the tests as you implement the math:

```bash
pytest -q
```

---

## Implementation roadmap (suggested order)

Each `core/` file has the exact equations, step-by-step TODOs, and gotchas in its
docstrings. Build bottom-up so you can test each layer before the next depends on it:

1. **`core/data_provider.py`** — get clean, date-aligned price history from
   yfinance. Everything depends on this.
2. **`core/returns.py`** — daily returns → annualized `mu` and `Sigma`.
3. **`core/mpt.py`** — `portfolio_performance` first, then `min_variance`,
   `max_sharpe`, `efficient_frontier`. `pytest -q` should start passing here.
4. **`core/metrics.py`** — Sharpe + excess-vs-RFR packaging.
5. **`core/black_litterman.py`** — prior → views → posterior, then reuse `mpt`.
6. **`core/rebalancing.py`** — drift-band actions + turnover.
7. **`backend/api/routes.py`** + **`backend/db/crud.py`** — wire it all together.
8. **`frontend/streamlit_app.py`** — API calls, allocation chart, efficient
   frontier plot, save/rebalance controls.

---

## Prototype scope & known gotchas

- **Single-market portfolios only.** NSE prices are INR, NASDAQ USD — don't mix
  them in one covariance matrix without FX conversion. Cross-market comes later.
- **yfinance is ~15 min delayed**, not tick-level real-time. It's behind the
  `PriceProvider` interface so you can drop in Alpha Vantage / Finnhub / Zerodha
  later without touching the rest of the app.
- **India risk-free rate** has no clean free yfinance symbol; the prototype uses
  the configurable fallback in `.env`. US uses `^TNX` (10Y Treasury).
- **Covariance can be near-singular** with few observations / correlated assets;
  Ledoit-Wolf shrinkage is a noted upgrade in `returns.py`.
- **Different trading calendars** (NSE vs NASDAQ) — align dates before computing
  covariance.
```
