"""Streamlit frontend. Talks to the FastAPI backend over HTTP (never imports
backend.core directly) so the API stays the single source of truth.

Run (with the backend up on :8000):
    streamlit run frontend/streamlit_app.py

Visual system: colors come from the dataviz reference palette (dark mode). The
categorical order is colorblind-safe; the frontier uses a single-hue blue ramp
for the Sharpe magnitude; the excess-return delta uses reserved status colors.
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")  # tolerate a trailing slash in the secret

# --- Palette (dark mode, from dataviz reference) ---------------------------
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
SURFACE = "#1a1a19"
CARD = "#212120"
GRID = "#2c2c2a"
AXIS = "#383835"
ACCENT = "#3987e5"
GOOD = "#0ca30c"
CRIT = "#d03b3b"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Categorical hues in the validated fixed order (never cycled).
CATEGORICAL = ["#3987e5", "#008300", "#d55181", "#c98500",
               "#199e70", "#d95926", "#9085e9", "#e66767"]
# Single-hue blue ramp (100 -> 400 -> 700) for sequential magnitude.
BLUE_RAMP = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]

METHOD_LABELS = {
    "max_sharpe": "MPT · Max Sharpe",
    "min_variance": "MPT · Min Variance",
    "efficient_frontier": "MPT · Efficient Frontier",
    "black_litterman": "Black–Litterman",
}

st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")


# --- Global styling --------------------------------------------------------

st.markdown(
    f"""
    <style>
      .stApp {{ background: {'#0d0d0d'}; }}
      [data-testid="stHeader"] {{ background: transparent; }}
      #MainMenu, footer {{ visibility: hidden; }}
      .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1300px; }}
      html, body, [class*="css"] {{ font-family: {FONT}; }}

      /* Sidebar */
      [data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid rgba(255,255,255,0.06); }}
      .brand {{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.15rem;
                color:{INK}; letter-spacing:-0.01em; margin: 0 0 4px 2px; }}
      .brand .dot {{ width:9px; height:9px; border-radius:50%; background:{ACCENT};
                     box-shadow:0 0 12px {ACCENT}; }}
      .brand small {{ display:block; color:{MUTED}; font-weight:500; font-size:0.72rem; letter-spacing:0.02em; }}

      /* Hero */
      .hero-title {{ font-size:1.9rem; font-weight:750; color:{INK}; letter-spacing:-0.02em; margin:0; }}
      .hero-sub {{ color:{INK_2}; font-size:0.95rem; margin-top:4px; }}

      /* Section headers */
      .sec {{ display:flex; align-items:baseline; gap:12px; margin: 26px 0 12px; }}
      .sec-bar {{ width:4px; height:19px; border-radius:3px; background:{ACCENT}; transform:translateY(3px); }}
      .sec-title {{ font-size:1.2rem; font-weight:680; color:{INK}; letter-spacing:-0.01em; }}
      .sec-sub {{ color:{MUTED}; font-size:0.85rem; }}

      /* Results head + pills */
      .rhead {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }}
      .pills {{ display:flex; gap:8px; }}
      .pill {{ font-size:0.78rem; font-weight:600; padding:5px 12px; border-radius:999px;
               color:{INK_2}; background:{CARD}; border:1px solid rgba(255,255,255,0.08); }}
      .pill-accent {{ color:{'#dbe9fb'}; background:rgba(57,135,229,0.16); border-color:rgba(57,135,229,0.4); }}

      /* KPI cards */
      .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:6px; }}
      .kpi {{ position:relative; flex:1; min-width:150px; background:{CARD};
              border:1px solid rgba(255,255,255,0.07); border-radius:16px;
              padding:18px 18px 16px; overflow:hidden; }}
      .kpi-accent {{ position:absolute; top:0; left:0; right:0; height:3px; }}
      .kpi-label {{ color:{MUTED}; font-size:0.72rem; font-weight:600; letter-spacing:0.06em;
                    text-transform:uppercase; }}
      .kpi-value {{ color:{INK}; font-size:1.72rem; font-weight:720; margin-top:7px; letter-spacing:-0.02em; }}

      /* Empty state */
      .empty {{ text-align:center; padding:64px 24px; background:{CARD};
                border:1px solid rgba(255,255,255,0.07); border-radius:20px; margin-top:16px; }}
      .empty .big {{ font-size:2.6rem; }}
      .empty h3 {{ color:{INK}; margin:14px 0 6px; font-weight:700; }}
      .empty p {{ color:{MUTED}; margin:0; }}

      /* Buttons */
      .stButton > button {{ border-radius:10px; font-weight:650; border:1px solid rgba(255,255,255,0.1); }}
      .stButton > button[kind="primary"] {{ background:{ACCENT}; border:none;
                box-shadow:0 4px 16px rgba(57,135,229,0.35); }}

      /* Card wrapper for charts */
      .card {{ background:{CARD}; border:1px solid rgba(255,255,255,0.07);
               border-radius:16px; padding:8px 14px 4px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helpers ---------------------------------------------------------------

def _auth_headers() -> dict:
    tok = st.session_state.get("token")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def post(path: str, payload: dict, auth: bool = False) -> dict:
    r = httpx.post(f"{API}{path}", json=payload,
                   headers=_auth_headers() if auth else {}, timeout=60)
    r.raise_for_status()
    return r.json()


def get(path: str, auth: bool = False) -> dict:
    r = httpx.get(f"{API}{path}", headers=_auth_headers() if auth else {}, timeout=60)
    r.raise_for_status()
    return r.json()


def delete(path: str, auth: bool = False) -> None:
    r = httpx.request("DELETE", f"{API}{path}", headers=_auth_headers() if auth else {}, timeout=60)
    r.raise_for_status()


def patch(path: str, payload: dict, auth: bool = False) -> dict:
    r = httpx.request("PATCH", f"{API}{path}", json=payload,
                      headers=_auth_headers() if auth else {}, timeout=60)
    r.raise_for_status()
    return r.json()


def _error_detail(exc: httpx.HTTPStatusError) -> str:
    try:
        return exc.response.json().get("detail", str(exc))
    except Exception:
        return str(exc)


def _logout() -> None:
    st.session_state.pop("token", None)
    st.session_state.pop("user_email", None)
    _cookie_delete(_TOKEN_COOKIE)


def authenticate(mode: str, email: str, password: str) -> None:
    """mode is 'login' or 'register'. Stores the token or raises an httpx error."""
    data = post(f"/auth/{mode}", {"email": email, "password": password})
    st.session_state["token"] = data["access_token"]
    st.session_state["user_email"] = data["email"]
    _cookie_set(_TOKEN_COOKIE, data["access_token"])


def section(title: str, sub: str = "") -> None:
    sub_html = f'<span class="sec-sub">{sub}</span>' if sub else ""
    st.markdown(f'<div class="sec"><span class="sec-bar"></span>'
                f'<span class="sec-title">{title}</span>{sub_html}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, accent: str) -> str:
    return (f'<div class="kpi"><div class="kpi-accent" style="background:{accent}"></div>'
            f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>')


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=13),
        margin=dict(l=8, r=8, t=34, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor=CARD, bordercolor="rgba(255,255,255,0.12)", font=dict(color=INK)),
        title=dict(text="", font=dict(color=INK, size=15)),  # text="" or Streamlit's theme renders "undefined"
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=AXIS)
    return fig


PLOTLY_CFG = {"displayModeBar": False}


# --- "Stay logged in" via a browser cookie (degrades to a no-op if the ------
#     component is unavailable, e.g. in headless tests) -------------------------

try:
    import extra_streamlit_components as stx
    _cookie_mgr = stx.CookieManager(key="pf_cookies")
except Exception:
    _cookie_mgr = None

_TOKEN_COOKIE = "pf_token"


def _cookie_get(name: str):
    if _cookie_mgr is None:
        return None
    try:
        return _cookie_mgr.get(name)
    except Exception:
        return None


def _cookie_set(name: str, value: str, days: int = 7) -> None:
    if _cookie_mgr is None:
        return
    try:
        from datetime import datetime, timedelta
        _cookie_mgr.set(name, value,
                        expires_at=datetime.now() + timedelta(days=days), key=f"ck_set_{name}")
    except Exception:
        pass


def _cookie_delete(name: str) -> None:
    if _cookie_mgr is None:
        return
    try:
        _cookie_mgr.delete(name, key=f"ck_del_{name}")
    except Exception:
        pass


# Restore a prior session from the cookie so a hard refresh doesn't log you out.
# esc returns None on the very first run, then reruns once its component mounts,
# so this resolves on the following run. Validating via /auth/me also refreshes
# the stored email and drops the cookie if the token has expired.
if not st.session_state.get("token"):
    _saved_token = _cookie_get(_TOKEN_COOKIE)
    if _saved_token:
        st.session_state["token"] = _saved_token
        try:
            st.session_state["user_email"] = get("/auth/me", auth=True)["email"]
        except Exception:
            _logout()


# --- Sidebar: build the optimization request -------------------------------

with st.sidebar:
    st.markdown('<div class="brand"><span class="dot"></span>'
                '<div>Portfolio Optimizer<small>MPT · Black–Litterman</small></div></div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # --- Account ---
    if st.session_state.get("token"):
        st.markdown(
            f'<div style="color:{MUTED};font-size:0.72rem;font-weight:600;'
            f'letter-spacing:0.06em;text-transform:uppercase;">Signed in as</div>'
            f'<div style="color:{INK};font-weight:650;margin:2px 0 8px;word-break:break-all;">'
            f'{st.session_state.get("user_email", "")}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Log out", width="stretch"):
            _logout()
            st.rerun()
    else:
        with st.expander("🔒 Sign in / Create account", expanded=False):
            tab_login, tab_signup = st.tabs(["Log in", "Sign up"])
            with tab_login:
                le = st.text_input("Email", key="login_email")
                lp = st.text_input("Password", type="password", key="login_pw")
                if st.button("Log in", key="btn_login", type="primary", width="stretch"):
                    try:
                        authenticate("login", le, lp)
                        st.rerun()
                    except httpx.HTTPStatusError as exc:
                        st.error(_error_detail(exc))
                    except httpx.HTTPError as exc:
                        st.error(f"Could not reach the API: {exc}")
            with tab_signup:
                se = st.text_input("Email", key="signup_email")
                sp = st.text_input("Password (min 8 chars)", type="password", key="signup_pw")
                if st.button("Create account", key="btn_signup", type="primary", width="stretch"):
                    try:
                        authenticate("register", se, sp)
                        st.rerun()
                    except httpx.HTTPStatusError as exc:
                        st.error(_error_detail(exc))
                    except httpx.HTTPError as exc:
                        st.error(f"Could not reach the API: {exc}")
    st.markdown("---")

    market = st.selectbox("Market", ["NASDAQ", "NSE"])
    default_tickers = ("AAPL, MSFT, GOOGL, AMZN, NVDA" if market == "NASDAQ"
                       else "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ITC.NS")
    tickers_raw = st.text_area("Tickers (comma-separated)", value=default_tickers,
                               help="NSE tickers need the .NS suffix, e.g. RELIANCE.NS")
    tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]

    method = st.selectbox("Optimization method", list(METHOD_LABELS),
                          format_func=lambda m: METHOD_LABELS[m])
    lookback = st.slider("Lookback (days)", 60, 1095, 365)
    allow_short = st.checkbox("Allow short positions", value=False)

    views: list[dict] = []
    if method == "black_litterman":
        st.markdown("**Your views**")
        n_views = st.number_input("Number of views", min_value=0, max_value=5, value=1, step=1)
        for i in range(int(n_views)):
            with st.expander(f"View {i + 1}", expanded=(i == 0)):
                vtype = st.radio("Type", ["Absolute", "Relative"], key=f"vt{i}", horizontal=True)
                if vtype == "Absolute":
                    asset = st.selectbox("Asset", tickers, key=f"va{i}")
                    q = st.number_input("Expected annual return", value=0.15, step=0.01, key=f"vq{i}")
                    conf = st.slider("Confidence", 0.05, 1.0, 0.5, key=f"vc{i}")
                    views.append({"assets": [asset], "weights": [1.0], "q": q, "confidence": conf})
                else:
                    out = st.selectbox("Outperformer", tickers, key=f"vo{i}")
                    und = st.selectbox("Underperformer", tickers, key=f"vu{i}")
                    q = st.number_input("Outperformance (annual)", value=0.05, step=0.01, key=f"vrq{i}")
                    conf = st.slider("Confidence", 0.05, 1.0, 0.5, key=f"vrc{i}")
                    views.append({"assets": [out, und], "weights": [1.0, -1.0], "q": q, "confidence": conf})

    run = st.button("⚡ Optimize", type="primary", width="stretch")


# --- Account status (main view) --------------------------------------------

if st.session_state.get("token"):
    st.markdown(
        f'<div style="display:flex;justify-content:flex-end;margin:-8px 0 2px;">'
        f'<span class="pill pill-accent">👤 {st.session_state.get("user_email", "")} · signed in</span>'
        f'</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div style="display:flex;justify-content:flex-end;margin:-8px 0 2px;">'
        '<span class="pill">👤 Guest · sign in from the sidebar to save</span></div>',
        unsafe_allow_html=True)


# --- Hero ------------------------------------------------------------------

st.markdown('<div class="hero-title">Portfolio Optimizer</div>'
            '<div class="hero-sub">Build, optimize &amp; rebalance NSE / NASDAQ portfolios — '
            'Modern Portfolio Theory and Black–Litterman, benchmarked against the market risk-free rate.</div>',
            unsafe_allow_html=True)


# --- Run optimization ------------------------------------------------------

if run:
    if len(tickers) < 2:
        st.error("Enter at least 2 tickers.")
    else:
        payload = {"market": market, "tickers": tickers, "method": method,
                   "lookback_days": lookback, "allow_short": allow_short, "views": views}
        try:
            with st.spinner("Fetching live prices and optimizing…"):
                st.session_state["result"] = post("/optimize", payload)
                st.session_state["result_market"] = market
                st.session_state["result_method"] = method
        except httpx.HTTPStatusError as exc:
            st.error(f"Optimization failed: {_error_detail(exc)}")
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the API at {API}: {exc}")


# --- Show results ----------------------------------------------------------

result = st.session_state.get("result")

if not result:
    st.markdown(
        '<div class="empty"><div class="big">📈</div>'
        '<h3>Design your first portfolio</h3>'
        '<p>Pick a market, enter tickers, choose a method, and hit <b>Optimize</b> in the sidebar.</p></div>',
        unsafe_allow_html=True,
    )
else:
    m = result["metrics"]
    alloc_df = pd.DataFrame(result["allocations"])
    res_method = st.session_state.get("result_method", "max_sharpe")
    res_market = st.session_state.get("result_market", market)

    # Results header + method/market pills
    st.markdown(
        f'<div class="rhead" style="margin-top:22px;">'
        f'<span class="sec-title" style="font-size:1.3rem;">Optimized portfolio</span>'
        f'<div class="pills"><span class="pill pill-accent">{METHOD_LABELS[res_method]}</span>'
        f'<span class="pill">{res_market}</span></div></div>',
        unsafe_allow_html=True,
    )

    # KPI row
    er, vol, sh, rf = m["expected_return"], m["volatility"], m["sharpe"], m["risk_free_rate"]
    ex = er - rf
    cards = "".join([
        kpi_card("Expected Return", f"{er * 100:.2f}%", ACCENT),
        kpi_card("Volatility", f"{vol * 100:.2f}%", "#c98500"),
        kpi_card("Sharpe Ratio", f"{sh:.2f}", "#199e70"),
        kpi_card("Risk-Free Rate", f"{rf * 100:.2f}%", MUTED),
        kpi_card("Excess vs RFR", f"{'▲' if ex >= 0 else '▼'} {abs(ex) * 100:.2f}%",
                 GOOD if ex >= 0 else CRIT),
    ])
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1, 1], gap="medium")

    with left:
        section("Allocation")
        top = alloc_df.sort_values("weight", ascending=False).iloc[0]
        # Don't draw dust slices in the pie (the table below still lists every
        # holding). Drop anything under 0.01% of the book (weight fraction 1e-4).
        _MIN_PIE_WEIGHT = 0.0001  # 0.01%
        pie_df = alloc_df[alloc_df["weight"] >= _MIN_PIE_WEIGHT]
        if pie_df.empty:  # safety: never pass an empty frame to px.pie
            pie_df = alloc_df.loc[[top.name]]
        donut = px.pie(pie_df, names="ticker", values="weight", hole=0.64,
                       color_discrete_sequence=CATEGORICAL)
        donut.update_traces(
            textposition="outside", texttemplate="%{label} · %{percent:.1%}",
            marker=dict(line=dict(color=SURFACE, width=2)),  # 2px surface gap between fills
            hovertemplate="%{label}<br>%{percent:.1%}<extra></extra>", sort=False,
        )
        donut.update_layout(showlegend=False, annotations=[dict(
            text=f"<b>{top['ticker']}</b><br><span style='color:{MUTED}'>{top['weight'] * 100:.1f}%</span>",
            x=0.5, y=0.5, font=dict(size=17, color=INK), showarrow=False)])
        st.plotly_chart(style_fig(donut, 360), width="stretch", config=PLOTLY_CFG)

        show = alloc_df.assign(weight=lambda d: d["weight"] * 100)
        st.dataframe(
            show, hide_index=True, width="stretch",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "weight": st.column_config.ProgressColumn(
                    "Weight", format="%.2f%%", min_value=0, max_value=100),
            },
        )

    with right:
        frontier = result.get("frontier") or []
        if frontier:
            section("Efficient frontier", "colored by Sharpe ratio")
            fdf = pd.DataFrame(frontier)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fdf["volatility"] * 100, y=fdf["expected_return"] * 100, mode="lines+markers",
                name="Frontier", line=dict(color=ACCENT, width=2),
                marker=dict(size=9, color=fdf["sharpe"], colorscale=BLUE_RAMP, line=dict(color=SURFACE, width=1),
                            showscale=True, colorbar=dict(title="Sharpe", outlinewidth=0, thickness=12)),
                hovertemplate="Vol %{x:.2f}%<br>Return %{y:.2f}%<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=[vol * 100], y=[er * 100], mode="markers", name="Chosen",
                marker=dict(color=CRIT, size=16, symbol="star", line=dict(color=SURFACE, width=1.5)),
                hovertemplate="Chosen<br>Vol %{x:.2f}%<br>Return %{y:.2f}%<extra></extra>"))
            fig.update_layout(xaxis_title="Volatility (%)", yaxis_title="Expected return (%)",
                              legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(style_fig(fig, 430), width="stretch", config=PLOTLY_CFG)
        else:
            section("Risk vs return")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[vol * 100], y=[er * 100], mode="markers+text", text=["Your portfolio"],
                textposition="top center", textfont=dict(color=INK_2),
                marker=dict(color=ACCENT, size=20, line=dict(color=SURFACE, width=2)),
                hovertemplate="Vol %{x:.2f}%<br>Return %{y:.2f}%<extra></extra>"))
            fig.add_hline(y=rf * 100, line=dict(color=MUTED, width=1, dash="dot"),
                          annotation_text=f"Risk-free {rf * 100:.2f}%", annotation_font_color=MUTED)
            fig.update_layout(xaxis_title="Volatility (%)", yaxis_title="Expected return (%)")
            st.plotly_chart(style_fig(fig, 430), width="stretch", config=PLOTLY_CFG)
            st.caption("Switch to the Efficient Frontier method to trace the full risk/return curve.")

    # --- Save this portfolio ---
    section("Save portfolio")
    if not st.session_state.get("token"):
        st.info("🔒 **Sign in** from the sidebar to save this portfolio and rebalance it later. "
                "Optimizing is free — only saving needs an account.")
    else:
        sc1, sc2 = st.columns([3, 1])
        save_name = sc1.text_input("Name", value="My Portfolio", label_visibility="collapsed",
                                   placeholder="Portfolio name")
        if sc2.button("💾 Save", width="stretch"):
            try:
                saved = post("/portfolios", {
                    "name": save_name, "market": res_market,
                    "holdings": [{"ticker": a["ticker"], "weight": a["weight"]} for a in result["allocations"]]},
                    auth=True)
                st.success(f"Saved portfolio #{saved['id']} — {saved['name']}")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    _logout()
                    st.warning("Your session expired — please log in again from the sidebar.")
                else:
                    st.error(f"Save failed: {_error_detail(exc)}")
            except httpx.HTTPError as exc:
                st.error(f"Save failed: {exc}")


# --- Saved portfolios & rebalancing ----------------------------------------

section("Saved portfolios & rebalancing", "recompute targets from live prices")

if not st.session_state.get("token"):
    st.info("🔒 **Sign in** from the sidebar to view and rebalance your saved portfolios.")
else:
    try:
        portfolios = get("/portfolios", auth=True)
    except httpx.HTTPStatusError as exc:
        portfolios = []
        if exc.response.status_code == 401:
            _logout()
            st.warning("Your session expired — please log in again from the sidebar.")
        else:
            st.caption("Could not load your portfolios.")
    except httpx.HTTPError:
        portfolios = []
        st.caption("Start the backend to see saved portfolios.")

    if st.session_state.get("token") and not portfolios:
        st.caption("You haven't saved any portfolios yet. Optimize one above and hit 💾 Save.")

    if portfolios:
        st.dataframe(
            pd.DataFrame([{"ID": p["id"], "Name": p["name"], "Market": p["market"],
                           "Holdings": len(p["holdings"])} for p in portfolios]),
            hide_index=True, width="stretch",
        )

        labels = {f"#{p['id']} — {p['name']}": p["id"] for p in portfolios}
        rc1, rc2, rc3, rc4 = st.columns([2, 1.3, 1, 1])
        chosen = rc1.selectbox("Portfolio", list(labels.keys()))
        rb_method = rc2.selectbox("Target method", ["max_sharpe", "min_variance"],
                                  format_func=lambda m: METHOD_LABELS[m], key="rb_method")
        band = rc3.slider("Drift band (abs)", 0.0, 0.25, 0.05, step=0.01,
                          help="Absolute band: trade a holding if |current − target| exceeds this.")
        rel_band = rc4.slider("Relative band", 0.0, 1.0, 0.25, step=0.05,
                              help="5/25 rule: also trade when a holding drifts past this fraction of "
                                   "its OWN target weight, so small positions aren't frozen. 0 = off.")
        st.caption("↻ Rebalance **periodically — monthly or quarterly** (quarterly is a good default), "
                   "not daily or weekly. Frequent checks just add trading costs and chase noise; the "
                   "bands already suppress tiny trades, and fundamentals barely move week to week.")

        if st.button("♻️ Compute rebalance", type="primary"):
            try:
                with st.spinner("Recomputing targets from live prices…"):
                    rb = post("/rebalance", {"portfolio_id": labels[chosen],
                                             "method": rb_method, "drift_band": band,
                                             "relative_band": rel_band}, auth=True)
                tc = rb["turnover"]
                st.markdown(f'<div class="kpi-row"><div class="kpi" style="max-width:220px">'
                            f'<div class="kpi-accent" style="background:{ACCENT}"></div>'
                            f'<div class="kpi-label">Turnover</div>'
                            f'<div class="kpi-value">{tc * 100:.2f}%</div></div></div>', unsafe_allow_html=True)
                st.write("")
                if rb["actions"]:
                    adf = pd.DataFrame(rb["actions"])
                    adf["Action"] = adf["delta"].apply(lambda d: "🟢 BUY" if d > 0 else "🔴 SELL")
                    for c in ("current_weight", "target_weight", "delta"):
                        adf[c] = adf[c] * 100
                    adf = adf[["ticker", "Action", "current_weight", "target_weight", "delta"]]
                    st.dataframe(
                        adf, hide_index=True, width="stretch",
                        column_config={
                            "ticker": st.column_config.TextColumn("Ticker"),
                            "current_weight": st.column_config.NumberColumn("Current", format="%.2f%%"),
                            "target_weight": st.column_config.NumberColumn("Target", format="%.2f%%"),
                            "delta": st.column_config.NumberColumn("Δ", format="%+.2f%%"),
                        },
                    )
                else:
                    st.success("No trades needed — all holdings are within their bands.")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    _logout()
                    st.warning("Your session expired — please log in again from the sidebar.")
                else:
                    st.error(f"Rebalance failed: {_error_detail(exc)}")
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the API: {exc}")

        # --- Manage the selected portfolio (rename / delete) ---
        st.write("")
        _sel_id = labels[chosen]
        _cur_name = next((p["name"] for p in portfolios if p["id"] == _sel_id), "")
        mc1, mc2, _ = st.columns([1, 1, 2])

        with mc1:
            with st.popover("✏️ Rename"):
                new_name = st.text_input("New name", value=_cur_name,
                                         key=f"rename_input_{_sel_id}", max_chars=120)
                if st.button("Save name", key="rename_confirm"):
                    new_name = new_name.strip()
                    if not new_name:
                        st.info("Enter a name.")
                    elif new_name == _cur_name:
                        st.info("That's already the name.")
                    else:
                        try:
                            patch(f"/portfolios/{_sel_id}", {"name": new_name}, auth=True)
                            st.success(f"Renamed to {new_name}.")
                            st.rerun()
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code == 401:
                                _logout()
                                st.warning("Your session expired — please log in again from the sidebar.")
                            else:
                                st.error(f"Rename failed: {_error_detail(exc)}")
                        except httpx.HTTPError as exc:
                            st.error(f"Could not reach the API: {exc}")

        with mc2:
            with st.popover("🗑 Delete"):
                st.caption(f"Permanently delete **{chosen}**. This can't be undone.")
                if st.button("Delete permanently", key="del_confirm"):
                    try:
                        delete(f"/portfolios/{_sel_id}", auth=True)
                        st.success(f"Deleted {chosen}.")
                        st.rerun()
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 401:
                            _logout()
                            st.warning("Your session expired — please log in again from the sidebar.")
                        else:
                            st.error(f"Delete failed: {_error_detail(exc)}")
                    except httpx.HTTPError as exc:
                        st.error(f"Could not reach the API: {exc}")
