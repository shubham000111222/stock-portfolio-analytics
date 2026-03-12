import datetime as dt
from typing import Dict, List, Tuple

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


def fetch_prices(tickers: List[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data.to_frame(name=tickers[0])
    prices = prices.dropna(how="all")
    return prices


def parse_weights(raw_weights: str, tickers: List[str]) -> Tuple[np.ndarray, str]:
    if not raw_weights.strip():
        weights = np.array([1.0 / len(tickers)] * len(tickers))
        return weights, "Equal weights applied"

    parts = [p.strip() for p in raw_weights.split(",") if p.strip()]
    if len(parts) != len(tickers):
        raise ValueError("Number of weights must match number of tickers")
    weights = np.array([float(p) for p in parts], dtype=float)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("Sum of weights must be greater than 0")
    weights = weights / total
    return weights, "Weights normalized to sum to 1.0"


def compute_metrics(port_returns: pd.Series, rf_rate: float) -> Dict[str, pd.Series]:
    if port_returns.empty:
        return {
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "drawdown": pd.Series(dtype=float),
            "cumulative": pd.Series(dtype=float),
        }
    daily_rf = (1 + rf_rate) ** (1 / 252) - 1
    ann_return = (1 + port_returns).prod() ** (252 / len(port_returns)) - 1
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = (port_returns.mean() - daily_rf) / port_returns.std() * np.sqrt(252)
    cum = (1 + port_returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum / peak) - 1
    max_dd = drawdown.min()
    return {
        "annual_return": ann_return,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "drawdown": drawdown,
        "cumulative": cum,
    }


def make_monthly_heatmap(port_returns: pd.Series) -> go.Figure:
    monthly = port_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    if monthly.empty:
        return go.Figure()
    data = (
        monthly.to_frame("ret")
        .assign(year=lambda d: d.index.year, month=lambda d: d.index.month)
        .pivot(index="year", columns="month", values="ret")
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale="RdYlGn",
            colorbar=dict(title="Return"),
            zmid=0,
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Month", yaxis_title="Year")
    return fig


def compute_beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_rate: float,
) -> Dict[str, float]:
    daily_rf = (1 + rf_rate) ** (1 / 252) - 1
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return {"beta": np.nan, "alpha": np.nan, "r2": np.nan}
    port_excess = aligned.iloc[:, 0] - daily_rf
    bench_excess = aligned.iloc[:, 1] - daily_rf
    cov = np.cov(bench_excess, port_excess)[0, 1]
    var = np.var(bench_excess)
    beta = cov / var if var > 0 else np.nan
    alpha_daily = port_excess.mean() - beta * bench_excess.mean()
    alpha = alpha_daily * 252
    corr = np.corrcoef(bench_excess, port_excess)[0, 1]
    r2 = corr**2 if not np.isnan(corr) else np.nan
    return {"beta": beta, "alpha": alpha, "r2": r2}


def compute_asset_betas(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    rf_rate: float,
) -> pd.Series:
    daily_rf = (1 + rf_rate) ** (1 / 252) - 1
    aligned = returns.join(benchmark_returns.rename("benchmark")).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    bench_excess = aligned["benchmark"] - daily_rf
    betas = {}
    for col in returns.columns:
        asset_excess = aligned[col] - daily_rf
        cov = np.cov(bench_excess, asset_excess)[0, 1]
        var = np.var(bench_excess)
        betas[col] = cov / var if var > 0 else np.nan
    return pd.Series(betas).sort_values(ascending=False)


def compute_factor_proxies(returns: pd.DataFrame) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    momentum = (1 + returns).rolling(126).apply(np.prod, raw=True) - 1
    vol = returns.rolling(63).std() * np.sqrt(252)
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum / peak) - 1
    max_dd_1y = drawdown.rolling(252).min()
    latest = pd.DataFrame(
        {
            "Momentum (6M)": momentum.iloc[-1],
            "Volatility (3M)": vol.iloc[-1],
            "Max Drawdown (1Y)": max_dd_1y.iloc[-1],
        }
    )
    return latest


def compute_attribution(returns: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    weights = np.asarray(weights, dtype=float)
    if weights.size != returns.shape[1]:
        raise ValueError("Weights must match number of assets")

    mean_daily = returns.mean()
    ann_return_contrib = mean_daily * weights * 252
    asset_vol = returns.std() * np.sqrt(252)
    risk_adj = ann_return_contrib / asset_vol.replace(0, np.nan)

    cov_ann = returns.cov() * 252
    port_var = float(weights.T @ cov_ann.values @ weights)
    if port_var <= 0:
        vol_contrib = pd.Series(np.nan, index=returns.columns)
    else:
        marginal = cov_ann.values @ weights
        vol_contrib = pd.Series(weights * marginal / port_var, index=returns.columns)

    attribution = pd.DataFrame(
        {
            "Return Contribution": ann_return_contrib,
            "Volatility Contribution": vol_contrib,
            "Risk-Adjusted Contribution": risk_adj,
        }
    )
    return attribution.sort_values("Return Contribution", ascending=False)


def generate_insights(
    returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> List[str]:
    if portfolio_returns.empty or returns.empty:
        return ["Not enough data to generate insights."]

    total_returns = (1 + returns).prod() - 1
    best_asset = total_returns.idxmax()
    worst_asset = total_returns.idxmin()
    port_total = (1 + portfolio_returns).prod() - 1
    bench_total = (1 + benchmark_returns).prod() - 1
    win_rate = (portfolio_returns > 0).mean()

    corr = returns.corr()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    avg_corr = corr.where(mask).stack().mean()

    cum = (1 + portfolio_returns).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd_date = drawdown.idxmin().date()

    delta = port_total - bench_total
    outperform = "outperformed" if delta >= 0 else "underperformed"

    insights = [
        f"Portfolio {outperformed} benchmark by {delta:.2%} over the period.",
        f"Best asset: {best_asset} ({total_returns[best_asset]:.2%}); worst asset: {worst_asset} ({total_returns[worst_asset]:.2%}).",
        f"Win rate: {win_rate:.1%} of days were positive returns.",
        f"Average pairwise correlation: {avg_corr:.2f} (lower is better for diversification).",
        f"Max drawdown occurred on {max_dd_date} at {drawdown.min():.2%}.",
    ]
    return insights


def compute_health_score(metrics: Dict[str, float]) -> Tuple[float, List[str]]:
    ann_return = metrics.get("annual_return", np.nan)
    ann_vol = metrics.get("annual_vol", np.nan)
    sharpe = metrics.get("sharpe", np.nan)
    max_dd = metrics.get("max_drawdown", np.nan)

    if any(pd.isna(val) for val in [ann_return, ann_vol, sharpe, max_dd]):
        return np.nan, ["Not enough data to compute health score."]

    return_score = np.clip((ann_return + 0.1) / 0.3, 0, 1)
    vol_score = np.clip(1 - (ann_vol / 0.4), 0, 1)
    sharpe_score = np.clip((sharpe + 0.5) / 2.5, 0, 1)
    dd_score = np.clip(1 - (abs(max_dd) / 0.5), 0, 1)

    score = (return_score * 0.35 + sharpe_score * 0.35 + vol_score * 0.2 + dd_score * 0.1) * 100
    notes = [
        f"Return score: {return_score * 100:.0f}/100",
        f"Sharpe score: {sharpe_score * 100:.0f}/100",
        f"Volatility score: {vol_score * 100:.0f}/100",
        f"Drawdown score: {dd_score * 100:.0f}/100",
    ]
    return score, notes


def build_pdf_report(
    tickers: List[str],
    benchmark: str,
    date_range: str,
    weights: np.ndarray,
    metrics: Dict[str, float],
    bench_metrics: Dict[str, float],
    insights: List[str],
    attribution: pd.DataFrame,
    health_score: float,
) -> bytes:
    # Lazy import reportlab - only load when PDF is actually generated
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="Portfolio Analytics Report")
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Stock Portfolio Analytics Report", styles["Title"]),
        Paragraph(date_range, styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"Tickers: {', '.join(tickers)}", styles["Normal"]),
        Paragraph(f"Benchmark: {benchmark}", styles["Normal"]),
        Paragraph(f"Weights: {', '.join([f'{w:.2%}' for w in weights])}", styles["Normal"]),
        Spacer(1, 12),
    ]

    summary_data = [
        ["Metric", "Portfolio", "Benchmark"],
        ["Annual Return", f"{metrics['annual_return']:.2%}", f"{bench_metrics['annual_return']:.2%}"],
        ["Annual Volatility", f"{metrics['annual_vol']:.2%}", f"{bench_metrics['annual_vol']:.2%}"],
        ["Sharpe", f"{metrics['sharpe']:.2f}", f"{bench_metrics['sharpe']:.2f}"],
        ["Max Drawdown", f"{metrics['max_drawdown']:.2%}", f"{bench_metrics['max_drawdown']:.2%}"],
    ]
    summary_table = Table(summary_data, hAlign="LEFT")
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.extend([Paragraph("Performance Summary", styles["Heading2"]), summary_table, Spacer(1, 12)])

    story.append(Paragraph(f"Portfolio Health Score: {health_score:.1f}/100", styles["Heading2"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Key Insights", styles["Heading2"]))
    for item in insights:
        story.append(Paragraph(f"- {item}", styles["Normal"]))
    story.append(Spacer(1, 12))

    if not attribution.empty:
        top = attribution.head(6)
        attrib_data = [["Asset", "Return Contrib", "Volatility Contrib", "Risk-Adj Contrib"]]
        for idx, row in top.iterrows():
            attrib_data.append(
                [
                    idx,
                    f"{row['Return Contribution']:.2%}",
                    f"{row['Volatility Contribution']:.2%}",
                    f"{row['Risk-Adjusted Contribution']:.2f}",
                ]
            )
        attrib_table = Table(attrib_data, hAlign="LEFT")
        attrib_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.extend([Paragraph("Top Attribution", styles["Heading2"]), attrib_table])

    doc.build(story)
    return buffer.getvalue()
