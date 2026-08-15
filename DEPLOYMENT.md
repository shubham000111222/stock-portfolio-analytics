# Streamlit Community Cloud Deployment Guide

## Prerequisites
- [GitHub Account](https://github.com)
- [Streamlit Community Cloud Account](https://share.streamlit.io/)

## Step-by-Step Deployment

### 1. Push Code to GitHub

Make sure your repository is pushed to your GitHub account:

```bash
git add .
git commit -m "Initial commit: Stock Portfolio Analytics"
git push
```

### 2. Create a Streamlit Cloud App

1. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app** (top-right).
3. Select the repository containing your project.
4. Set the **Main file path** to `app.py`.
5. Click **Deploy**.

### 3. Wait for Deployment

- Streamlit Cloud will automatically install dependencies from `requirements.txt`.
- Your app deploys in ~2 minutes automatically!
- Access it at: `https://stock-portfolio-analytics-5ustsrbxfa4aqzcmnehqwt.streamlit.app/`

## Updating Your Deployment

To push updates to your Streamlit Cloud app:

```bash
git add .
git commit -m "Update: Your change description"
git push
```

The app will automatically rebuild and redeploy within minutes.

## Performance Tips

- Data caching is used to minimize Yahoo Finance API calls.
- Initial loads might take a few seconds as the cache populates.
- Be aware that cloud platforms may occasionally experience rate limiting from Yahoo Finance. If this occurs, wait a few minutes and try again.

---

**Now your Stock Portfolio Analytics is live! 🎉**

Share the link: `https://stock-portfolio-analytics-5ustsrbxfa4aqzcmnehqwt.streamlit.app/`
