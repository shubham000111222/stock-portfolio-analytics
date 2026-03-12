# Stock Portfolio Analytics Dashboard

An industry-style analytics project that evaluates a multi-asset portfolio with risk, return, drawdown, and correlation insights. The dashboard uses Yahoo Finance data and presents interactive visualizations suitable for demos and resume showcases.

## Highlights
- Portfolio vs benchmark performance
- Risk metrics: volatility, Sharpe ratio, max drawdown
- Key insights summary panel
- Correlation heatmap and allocation breakdown
- Rolling volatility and drawdown curves
- Monthly returns heatmap
- Factor and beta analytics page
- Performance attribution table + waterfall chart
- Data export (prices + returns)
- Portfolio health score
- PDF report export

## Tech Stack
- Python, Streamlit
- Pandas, NumPy
- Plotly
- Yahoo Finance (yfinance)

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage
- Enter tickers and optional weights in the sidebar.
- Choose date range, benchmark, and risk-free rate.
- Explore portfolio analytics and compare with SPY or any benchmark ticker.

## Notes
- Data comes from Yahoo Finance and may have missing values for some tickers.
- This project is intended for education and portfolio demonstration only.
