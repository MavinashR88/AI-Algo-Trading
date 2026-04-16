# Setup Guide

Step-by-step instructions for setting up the AI Algo Trading System.

## 1. Local Development Setup

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Git

### Install

```bash
# Clone repository
git clone <repo-url>
cd AI-Algo-Trading

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Configure Environment Variables

Edit `.env` with your credentials. See sections below for how to obtain each.

### Run Locally

```bash
# Start the bot
python main.py

# Start with debug logging
python main.py --log-level DEBUG

# Run tests
pytest

# Emergency stop
python kill.py
```

## 2. Alpaca Account Setup (US Markets)

Alpaca provides free paper trading for US stocks.

1. Go to [alpaca.markets](https://alpaca.markets/) and create an account
2. Navigate to the Paper Trading dashboard
3. Generate API keys (Paper Trading environment)
4. Add to your `.env`:

```
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
```

### Notes
- Paper trading is free and unlimited
- Live trading requires a funded account ($0 commissions on stocks)
- API keys for paper and live are separate — the system uses the correct endpoint based on `config.yaml` market mode

## 3. Zerodha Kite Connect Setup (India Markets)

Zerodha Kite Connect provides API access to NSE/BSE trading.

### Paper Mode (Free)
The system has a built-in paper trading simulator for Indian markets. No Zerodha account needed for paper trading. Just leave the Zerodha env vars empty.

### Live Mode
1. Create a Zerodha account at [zerodha.com](https://zerodha.com/)
2. Subscribe to Kite Connect API at [kite.trade](https://kite.trade/)
   - One-time fee: Rs 2000
   - Monthly charges: None (pay per API call beyond free tier)
3. Create a Kite Connect app to get your API key
4. Implement the login flow to get access token (expires daily)
5. Add to your `.env`:

```
ZERODHA_API_KEY=your_api_key
ZERODHA_ACCESS_TOKEN=your_access_token
ZERODHA_API_SECRET=your_api_secret
```

### Access Token Refresh
Kite access tokens expire at 7:00 AM IST daily. For automated trading, you need to refresh the token daily. The system will handle this in the agent loop.

## 4. Anthropic API Setup (AI Sentiment Analysis)

1. Create an account at [console.anthropic.com](https://console.anthropic.com/)
2. Generate an API key
3. Add to your `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### Pricing
- Claude Sonnet: ~$3/M input tokens, ~$15/M output tokens
- Sentiment analysis uses ~500 tokens per batch of headlines
- Estimated cost: ~$1-5/day depending on news volume

## 5. NewsAPI Setup (News Feed)

1. Register at [newsapi.org](https://newsapi.org/)
2. Get your free API key (100 requests/day on free tier)
3. Add to `.env`:

```
NEWSAPI_KEY=your_key
```

## 6. Email Notifications (Gmail)

1. Enable 2-Step Verification on your Google account
2. Go to [Google App Passwords](https://myaccount.google.com/apppasswords)
3. Generate an App Password for "Mail"
4. Add to `.env`:

```
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_16_char_app_password
EMAIL_RECIPIENTS=you@email.com,other@email.com
```

## 7. SMS Notifications (Twilio)

1. Create a free account at [twilio.com](https://www.twilio.com/)
2. Get your Account SID and Auth Token from the dashboard
3. Get a Twilio phone number (free trial includes one)
4. Add to `.env`:

```
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1234567890
TWILIO_TO_NUMBER=+1234567890
```

### Free Tier Limits
- Trial accounts can only send to verified numbers
- Upgrade to paid ($1/month per number) for production use

## 8. Railway Deployment

### First-Time Setup

1. Create an account at [railway.app](https://railway.app/)
2. Install Railway CLI: `npm install -g @railway/cli`
3. Login: `railway login`
4. Link your GitHub repository:
   - Railway Dashboard > New Project > Deploy from GitHub Repo
5. Set environment variables in Railway Dashboard:
   - Go to your service > Variables tab
   - Add all variables from your `.env` file

### Deploy

Railway auto-deploys on push to main. The `Procfile` and `railway.toml` are already configured.

```bash
# Manual deploy via CLI
railway up
```

### Railway Free Tier Limits
- 500 hours/month execution time
- 512 MB RAM
- 1 GB disk
- The bot is lightweight and fits well within these limits

### Monitoring
- Railway provides built-in logs (stdout/stderr)
- The bot writes structured JSON logs for easy parsing
- Dashboard (Phase 5) provides real-time monitoring

## 9. Running the Full System

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Verify config
python -c "from config import get_config; c = get_config(); print('Config OK')"

# 3. Initialize database
python -c "from db import Database; from config import get_config; c = get_config(); Database(c.database.full_path).initialize(); print('DB OK')"

# 4. Run tests
pytest

# 5. Start the bot
python main.py
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `ALPACA_API_KEY not set` | Check your `.env` file |
| Database locked | Kill any running bot instances: `python kill.py` |
| Email not sending | Verify Gmail App Password (not regular password) |
| SMS not sending | Verify Twilio number is verified on trial accounts |
