import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    compute_attribution,
    compute_health_score,
    compute_metrics,
    build_pdf_report,
    fetch_prices,
    generate_insights,
    make_monthly_heatmap,
    parse_weights,
)


st.set_page_config(
    page_title="Stock Portfolio Analytics",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.5rem;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    [data-testid="stMetric"] {background: #FFFFFF; border-radius: 14px; padding: 12px; border: 1px solid #E2E8F0;}
    .stDataFrame {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_prices(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    return fetch_prices(tickers, start, end)


st.title("Stock Portfolio Analytics")
st.caption("Industry-style portfolio dashboard with risk and performance analytics")

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
    rolling_window = st.slider("Rolling volatility window (days)", 20, 120, 60)
    run = st.button("Run analysis")
    st.caption("Tip: See the Factor & Beta page in the top-left page selector.")

if not run:
    st.info("Configure inputs and click 'Run analysis' to generate the dashboard.")
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

raw_portfolio_prices = prices[raw_tickers].dropna(how="all")
portfolio_prices = raw_portfolio_prices.copy()
benchmark_prices = prices[[benchmark]].dropna(how="all")
portfolio_prices = portfolio_prices.ffill().dropna()
benchmark_prices = benchmark_prices.ffill().dropna()

common_index = portfolio_prices.index.intersection(benchmark_prices.index)
portfolio_prices = portfolio_prices.loc[common_index]
benchmark_prices = benchmark_prices.loc[common_index]

returns = portfolio_prices.pct_change().dropna()
port_returns = returns.dot(weights)
bench_returns = benchmark_prices.pct_change().dropna().iloc[:, 0]

metrics = compute_metrics(port_returns, rf_rate)
bench_metrics = compute_metrics(bench_returns, rf_rate)

kpi_cols = st.columns(4)
kpi_cols[0].metric("Annual Return", f"{metrics['annual_return']:.2%}")
kpi_cols[1].metric("Annual Volatility", f"{metrics['annual_vol']:.2%}")
kpi_cols[2].metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
kpi_cols[3].metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
st.caption(weight_msg)

st.subheader("Performance Summary")
summary = pd.DataFrame(
    {
        "Annual Return": [metrics["annual_return"], bench_metrics["annual_return"]],
        "Annual Volatility": [metrics["annual_vol"], bench_metrics["annual_vol"]],
        "Sharpe": [metrics["sharpe"], bench_metrics["sharpe"]],
        "Max Drawdown": [metrics["max_drawdown"], bench_metrics["max_drawdown"]],
    },
    index=["Portfolio", benchmark],
)
st.dataframe(
    summary.style.format(
        {
            "Annual Return": "{:.2%}",
            "Annual Volatility": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max Drawdown": "{:.2%}",
        }
    ),
    use_container_width=True,
)

score, score_notes = compute_health_score(metrics)
st.subheader("Portfolio Health Score")
if np.isnan(score):
    st.info("Not enough data to compute health score.")
else:
    score_col, note_col = st.columns([1, 2])
    score_col.metric("Health Score", f"{score:.1f}/100")
    note_col.markdown("\n".join([f"- {note}" for note in score_notes]))

st.subheader("Data Quality")
coverage = raw_portfolio_prices.notna().mean().mean()
missing = raw_portfolio_prices.isna().sum().sum()
total_cells = raw_portfolio_prices.shape[0] * raw_portfolio_prices.shape[1]
quality_cols = st.columns(3)
quality_cols[0].metric("Coverage", f"{coverage:.1%}")
quality_cols[1].metric("Missing values", f"{int(missing):,}")
quality_cols[2].metric("Total observations", f"{int(total_cells):,}")

st.subheader("Key Insights")
insights = generate_insights(returns, port_returns, bench_returns)
st.markdown("\n".join([f"- {item}" for item in insights]))

col_left, col_right = st.columns([2, 1])

with col_left:
    normalized = portfolio_prices / portfolio_prices.iloc[0]
    fig_prices = px.line(
        normalized,
        title="Normalized Price Performance",
        labels={"value": "Normalized Price", "variable": "Ticker"},
    )
    fig_prices.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_prices, use_container_width=True)

with col_right:
    alloc = pd.Series(weights, index=raw_tickers)
    fig_alloc = px.pie(
        alloc,
        values=alloc.values,
        names=alloc.index,
        title="Portfolio Allocation",
        hole=0.5,
    )
    fig_alloc.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_alloc, use_container_width=True)

st.subheader("Portfolio vs Benchmark")
compare = pd.DataFrame(
    {
        "Portfolio": metrics["cumulative"],
        benchmark: bench_metrics["cumulative"],
    }
)
fig_compare = px.line(compare, labels={"value": "Cumulative Growth", "variable": "Series"})
fig_compare.update_layout(margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_compare, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    rolling_vol = port_returns.rolling(rolling_window).std() * np.sqrt(252)
    fig_vol = px.line(
        rolling_vol,
        title=f"Rolling Volatility ({rolling_window}d)",
        labels={"value": "Volatility"},
    )
    fig_vol.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_vol, use_container_width=True)

with col_b:
    rolling_sharpe = (
        port_returns.rolling(rolling_window).mean()
        / port_returns.rolling(rolling_window).std()
        * np.sqrt(252)
    )
    fig_sharpe = px.line(
        rolling_sharpe,
        title=f"Rolling Sharpe ({rolling_window}d)",
        labels={"value": "Sharpe"},
    )
    fig_sharpe.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_sharpe, use_container_width=True)

st.subheader("Drawdown Curve")
fig_dd = px.area(
    metrics["drawdown"],
    labels={"value": "Drawdown"},
)
fig_dd.update_layout(margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_dd, use_container_width=True)

st.subheader("Correlation and Return Distribution")

col_c, col_d = st.columns(2)
with col_c:
    corr = returns.corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Asset Correlation",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

with col_d:
    fig_hist = px.histogram(
        port_returns,
        nbins=50,
        title="Daily Return Distribution",
        labels={"value": "Daily Return"},
    )
    fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Risk vs Return")
asset_stats = pd.DataFrame(
    {
        "Annual Return": returns.mean() * 252,
        "Annual Volatility": returns.std() * np.sqrt(252),
    }
)
fig_rr = px.scatter(
    asset_stats,
    x="Annual Volatility",
    y="Annual Return",
    text=asset_stats.index,
    title="Asset Risk-Return Scatter",
)
fig_rr.update_traces(textposition="top center")
fig_rr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_rr, use_container_width=True)

st.subheader("Monthly Returns Heatmap")
heatmap = make_monthly_heatmap(port_returns)
if heatmap.data:
    st.plotly_chart(heatmap, use_container_width=True)
else:
    st.info("Not enough data to render monthly heatmap")

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

    total_return = attribution["Return Contribution"].sum()
    waterfall = go.Figure(
        go.Waterfall(
            name="Return Attribution",
            orientation="v",
            measure=["relative"] * len(attribution.index) + ["total"],
            x=[*attribution.index.tolist(), "Total"],
            y=[*attribution["Return Contribution"].tolist(), total_return],
            increasing=dict(marker=dict(color="#16A34A")),
            decreasing=dict(marker=dict(color="#EF4444")),
            totals=dict(marker=dict(color="#2563EB")),
        )
    )
    waterfall.update_layout(
        title="Return Contribution Waterfall",
        yaxis_title="Annualized Contribution",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(waterfall, use_container_width=True)

st.subheader("Data Exports")
export_cols = st.columns(2)
export_cols[0].download_button(
    "Download price data (CSV)",
    data=portfolio_prices.to_csv().encode("utf-8"),
    file_name="portfolio_prices.csv",
    mime="text/csv",
)
export_cols[1].download_button(
    "Download returns (CSV)",
    data=returns.to_csv().encode("utf-8"),
    file_name="portfolio_returns.csv",
    mime="text/csv",
)

st.subheader("PDF Report")
date_range = f"{start_date} to {end_date}"
report_bytes = build_pdf_report(
    tickers=raw_tickers,
    benchmark=benchmark,
    date_range=date_range,
    weights=weights,
    metrics=metrics,
    bench_metrics=bench_metrics,
    insights=insights,
    attribution=attribution,
    health_score=score,
)
st.download_button(
    "Download portfolio report (PDF)",
    data=report_bytes,
    file_name="portfolio_report.pdf",
    mime="application/pdf",
)
