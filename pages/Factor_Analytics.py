import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    compute_asset_betas,
    compute_attribution,
    compute_beta_alpha,
    compute_factor_proxies,
    fetch_prices,
    parse_weights,
)


@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    return fetch_prices(tickers, start, end)


st.title("Factor and Beta Analytics")
st.caption("Market beta, alpha attribution, and risk factor proxies")

with st.sidebar:
    st.header("Portfolio Inputs")
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, AMZN, GOOGL, NVDA",
    )
    weights_input = st.text_input(
        "Weights (optional, comma-separated)",
        value="",
        help="Example: 0.2, 0.2, 0.2, 0.2, 0.2",
    )
    benchmark = st.text_input("Benchmark ticker", value="SPY")
    start_date = st.date_input("Start date", value=dt.date(2019, 1, 1))
    end_date = st.date_input("End date", value=dt.date.today())
    rf_rate = st.number_input("Risk-free rate (annual)", value=0.03, step=0.005)
    beta_window = st.slider("Rolling beta window (days)", 40, 252, 120)
    run = st.button("Run analysis")

if not run:
    st.info("Configure inputs and click 'Run analysis' to generate the analytics.")
    st.stop()

raw_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if len(raw_tickers) < 2:
    st.error("Please enter at least two tickers")
    st.stop()

try:
    weights, weight_msg = parse_weights(weights_input, raw_tickers)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.spinner("Fetching market data..."):
    prices = load_prices(raw_tickers + [benchmark], start_date, end_date)

if prices.empty:
    st.error("No data returned. Try a different date range or tickers.")
    st.stop()

portfolio_prices = prices[raw_tickers].dropna(how="all")
benchmark_prices = prices[[benchmark]].dropna(how="all")
portfolio_prices = portfolio_prices.ffill().dropna()
benchmark_prices = benchmark_prices.ffill().dropna()

common_index = portfolio_prices.index.intersection(benchmark_prices.index)
portfolio_prices = portfolio_prices.loc[common_index]
benchmark_prices = benchmark_prices.loc[common_index]

returns = portfolio_prices.pct_change().dropna()
port_returns = returns.dot(weights)
bench_returns = benchmark_prices.pct_change().dropna().iloc[:, 0]

st.caption(weight_msg)

beta_stats = compute_beta_alpha(port_returns, bench_returns, rf_rate)

kpi_cols = st.columns(3)
kpi_cols[0].metric("Portfolio Beta", f"{beta_stats['beta']:.2f}")
kpi_cols[1].metric("Alpha (annual)", f"{beta_stats['alpha']:.2%}")
kpi_cols[2].metric("R²", f"{beta_stats['r2']:.2f}")

st.subheader("Portfolio vs Benchmark Regression")
aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
if not aligned.empty:
    x = aligned.iloc[:, 1]
    y = aligned.iloc[:, 0]
    slope, intercept = np.polyfit(x, y, 1)
    reg_line = intercept + slope * x
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Daily returns"))
    fig_scatter.add_trace(
        go.Scatter(x=x, y=reg_line, mode="lines", name="Fit", line=dict(color="#EF553B"))
    )
    fig_scatter.update_layout(
        xaxis_title=f"{benchmark} daily return",
        yaxis_title="Portfolio daily return",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Not enough aligned data for regression chart.")

st.subheader("Asset Betas")
asset_betas = compute_asset_betas(returns, bench_returns, rf_rate)
if not asset_betas.empty:
    fig_betas = px.bar(
        asset_betas,
        title="Beta by Asset",
        labels={"value": "Beta", "index": "Ticker"},
    )
    fig_betas.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_betas, use_container_width=True)

    asset_stats = pd.DataFrame(
        {
            "Beta": asset_betas,
            "Annual Return": returns.mean() * 252,
            "Annual Volatility": returns.std() * np.sqrt(252),
        }
    ).dropna()
    if not asset_stats.empty:
        fig_beta_rr = px.scatter(
            asset_stats,
            x="Beta",
            y="Annual Return",
            size="Annual Volatility",
            text=asset_stats.index,
            title="Beta vs Return (bubble size = volatility)",
        )
        fig_beta_rr.update_traces(textposition="top center")
        fig_beta_rr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_beta_rr, use_container_width=True)
else:
    st.info("Not enough data to calculate asset betas.")

st.subheader("Rolling Portfolio Beta")
aligned_roll = pd.concat([port_returns, bench_returns], axis=1).dropna()
if len(aligned_roll) > beta_window:
    rolling_beta = (
        aligned_roll.iloc[:, 0]
        .rolling(beta_window)
        .cov(aligned_roll.iloc[:, 1])
        / aligned_roll.iloc[:, 1].rolling(beta_window).var()
    )
    fig_roll = px.line(
        rolling_beta,
        title=f"Rolling Beta ({beta_window}d)",
        labels={"value": "Beta"},
    )
    fig_roll.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_roll, use_container_width=True)
else:
    st.info("Not enough data for rolling beta.")

st.subheader("Risk Factor Proxies")
factor_returns = returns.copy()
factor_returns["Portfolio"] = port_returns
factor_returns["Benchmark"] = bench_returns
factors = compute_factor_proxies(factor_returns)
if not factors.empty:
    st.dataframe(
        factors.style.format({
            "Momentum (6M)": "{:.2%}",
            "Volatility (3M)": "{:.2%}",
            "Max Drawdown (1Y)": "{:.2%}",
        }),
        use_container_width=True,
    )
    st.caption("These are proxy factors derived from price behavior, not fundamental factors.")
else:
    st.info("Not enough data to compute factor proxies.")

st.subheader("Performance Attribution")
attribution = compute_attribution(returns, weights)
if attribution.empty:
    st.info("Not enough data to compute attribution.")
else:
    st.dataframe(
        attribution.style.format(
            {
                "Return Contribution": "{:.2%}",
                "Volatility Contribution": "{:.2%}",
                "Risk-Adjusted Contribution": "{:.2f}",
            }
        ),
        use_container_width=True,
    )
    st.caption("Volatility contribution shows percentage of total portfolio variance.")
