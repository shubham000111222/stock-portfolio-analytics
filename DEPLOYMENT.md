# Hugging Face Spaces Deployment Guide

## Prerequisites
- [Hugging Face Account](https://huggingface.co) (free)
- Git installed locally
- Your HF username

## Step-by-Step Deployment

### 1. Create a Hugging Face Space

1. Visit [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"** (top-right)
3. Fill in the form:
   - **Owner**: Your username
   - **Space name**: `stock-portfolio-analytics`
   - **License**: MIT (or your preference)
   - **Space SDK**: Select **Streamlit**
   - **Visibility**: Public or Private
4. Click **"Create Space"**

### 2. Connect Your Local Repository

In your terminal, navigate to the project directory:

```bash
cd stock-portfolio-analytics
```

Initialize git (if not already done):

```bash
git init
git add .
git commit -m "Initial commit: Stock Portfolio Analytics"
```

Add the Hugging Face remote:

```bash
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/stock-portfolio-analytics
```

Replace `YOUR-USERNAME` with your actual Hugging Face username.

Push to Hugging Face:

```bash
git push hf main
```

**Note**: If you're on the `master` branch instead of `main`, use:
```bash
git push hf master:main
```

### 3. Wait for Deployment

- HF Spaces will automatically detect the Streamlit app
- Deployment takes ~2 minutes
- You'll see a "Building" status → "Running" when complete
- Your app will be at: `https://huggingface.co/spaces/YOUR-USERNAME/stock-portfolio-analytics`

## Updating Your Deployment

To push updates to your HF Space:

```bash
git add .
git commit -m "Update: Your change description"
git push hf main
```

The app will automatically rebuild and redeploy.

## HF Spaces Advantages

✅ **Free hosting** (with some compute limits)  
✅ **Automatic SSL** (HTTPS)  
✅ **Auto-scaling** for traffic  
✅ **Easy updates** (just git push)  
✅ **No credit card** required (for free tier)  
✅ **Community sharing** on HF Hub  

## Performance Tips

- App data is cached for **1 hour** (reduces API calls)
- HF Spaces free tier has ~16GB RAM
- First load might be slow (app initializes), subsequent loads are faster
- Stock data is fetched from Yahoo Finance (may have occasional delays)

## Accessing Secrets

If you need API keys in the future, add them via HF Space Settings:

1. Go to your Space page
2. Click **"Settings"** (gear icon)
3. Under **"Secrets"**, add your keys:
   - Key name: `API_KEY`
   - Value: `your-secret-value`

Then access in code:
```python
import os
api_key = os.getenv("API_KEY")
```

## Troubleshooting

**App won't start?**
- Check the "Logs" tab in your HF Space
- Ensure all packages in `requirements.txt` are compatible
- Verify Python version (HF uses 3.10+)

**Slow performance?**
- Clear Streamlit cache or wait for 1-hour cache expiry
- Reduce date range or number of tickers
- Contact HF support for compute upgrades

**Need to delete/recreate Space?**
- Go to Space Settings → Danger Zone → Delete this Space
- Create a new Space and push again

---

**Now your Stock Portfolio Analytics is live! 🎉**

Share the link: `https://huggingface.co/spaces/YOUR-USERNAME/stock-portfolio-analytics`
