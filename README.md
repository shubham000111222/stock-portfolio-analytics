# Stock Portfolio Analytics Dashboard

📈 An industry-style analytics platform that evaluates multi-asset portfolios with comprehensive risk analysis, performance tracking, and factor insights. The dashboard uses Yahoo Finance data and presents interactive visualizations suitable for demos and portfolio management.

**[Live Demo](https://stock-portfolio-analytics-5ustsrbxfa4aqzcmnehqwt.streamlit.app/)**

## ✨ Features

- **Performance Metrics**: Annual return, volatility, Sharpe ratio, max drawdown
- **Benchmark Comparison**: Compare against SPY, QQQ, or any ticker
- **Risk Analysis**: Rolling volatility, drawdown curves, correlation matrices
- **Factor & Beta Analytics**: Compute alpha, beta, and factor proxies
- **Attribution Analysis**: Understand asset-level contributions
- **Portfolio Health Score**: AI-powered assessment
- **Monthly Heatmaps**: Visualize monthly returns patterns
- **PDF Reports**: Download comprehensive reports
- **Data Export**: Export prices and returns

## 🚀 Deploy on Streamlit Community Cloud

1. Create a [Streamlit Community Cloud](https://share.streamlit.io/) account.
2. Click **New app**
3. Select your GitHub repository for `stock-portfolio-analytics`
4. Main file path: `app.py`
5. Click **Deploy**

Access it at: `https://stock-portfolio-analytics-5ustsrbxfa4aqzcmnehqwt.streamlit.app/`

## 💻 Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`

## 📊 Usage

1. **Input Configuration** (sidebar):
   - Enter tickers: `AAPL, MSFT, AMZN, GOOGL, NVDA`
   - Set weights (optional, auto-equal if empty)
   - Choose benchmark and date range
   - Adjust risk-free rate

2. **Click "Run Analysis"** to compute metrics

3. **Explore Results**:
   - Main dashboard: KPIs, charts, insights
   - Factor & Beta page: Alpha, beta, momentum analysis

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Financial Data**: Yahoo Finance (yfinance)
- **Reports**: ReportLab

## ⚙️ Performance Optimizations

- Data caching: 1 hour TTL (reduces API calls)
- Lazy-loaded PDF generation
- Minimal Streamlit toolbar
- Optimized dependencies

## ⚠️ Notes

- Data from Yahoo Finance (historical ~2000+)
- Real-time prices updated daily
- Minimum 252 trading days recommended for factor analysis
- Education and demonstration purposes

## 📝 License

MIT License
