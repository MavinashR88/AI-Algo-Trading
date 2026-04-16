# AI Algo Trading System

Fully autonomous AI algorithmic trading bot for US markets (NYSE/NASDAQ via Alpaca) and Indian markets (NSE/BSE via Zerodha Kite Connect).

## What It Does

- **Dual-market trading**: Simultaneously trades US and Indian markets with independent strategies and risk management
- **AI-powered decisions**: Uses Claude AI for news sentiment analysis combined with 8 technical strategies
- **Paper-first approach**: Paper trades, learns, and only goes live after statistically proving >=70% win rate
- **Risk-first design**: Enforced drawdown limits, position sizing, psychology guards, and kill switches
- **Real-time dashboard**: Web-based monitoring with live charts, trade journal, and performance metrics
- **Full notifications**: Email + SMS alerts for every trade, drawdown, and critical event

## Architecture

```
main.py                    Entry point (async event loop)
  ├── config/              YAML config + holiday calendars
  ├── brokers/             Alpaca (US) + Zerodha (India) adapters
  ├── data/                Market data + news + sentiment engine
  ├── strategy/            8 strategies + backtester + signal generator
  ├── risk/                Risk manager + money manager + psychology guard
  ├── agent/               LangGraph stateful trading agent
  ├── dashboard/           FastAPI + React web dashboard
  ├── notifications/       Email (Gmail SMTP) + SMS (Twilio)
  ├── db/                  SQLite with WAL mode (PostgreSQL-ready schema)
  └── logs/                Structured JSON logging (structlog)
```

## Quick Start

### Prerequisites
- Python 3.11+
- Alpaca account (free paper trading)
- Optional: Zerodha Kite Connect, NewsAPI key, Twilio account

### Setup

```bash
# Clone and enter project
git clone <repo-url>
cd AI-Algo-Trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the bot
python main.py
```

### Run Tests

```bash
pytest                     # all tests
pytest -m "not slow"       # skip slow tests
pytest --cov=. --cov-report=html  # with coverage
```

## Configuration

All parameters live in `config/config.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `markets` | Watchlists, fees, trading hours per market |
| `risk` | Max drawdown, position limits, stop-loss settings |
| `psychology` | Anti-revenge trading, overtrading guards |
| `strategy` | Strategy parameters and weights |
| `graduation` | Paper-to-live promotion criteria |
| `notifications` | Alert settings and events |

## Safety Features

- **Kill switch**: `python kill.py` stops everything immediately
- **Per-market halt**: Dashboard toggles per market
- **Daily drawdown limit**: 3% — auto-halt
- **Weekly drawdown limit**: 8% — auto-halt for the week
- **No revenge trading**: 30-min cooldown after 2 consecutive losses
- **No FOMO**: Only trades signals with confidence >= 0.65
- **Graduation gate**: Human confirmation required before live trading

## Deployment (Railway)

```bash
# Push to GitHub, then connect Railway to your repo
# Railway auto-detects Procfile and deploys
# Set all .env variables in Railway dashboard
```

See [SETUP.md](SETUP.md) for detailed deployment guide.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Trading Agent | LangGraph |
| AI Model | Claude Sonnet (Anthropic API) |
| US Broker | Alpaca |
| India Broker | Zerodha Kite Connect |
| Data | yfinance, NewsAPI, Google News RSS |
| Database | SQLite (WAL mode) |
| Backend | FastAPI |
| Frontend | React |
| Hosting | Railway |
| Logging | structlog (JSON) |

## Project Status

- [x] Phase 1: Foundation & Infrastructure
- [ ] Phase 2: Data, News & Sentiment Engine
- [ ] Phase 3: Strategy Engine & Backtesting
- [ ] Phase 4: LangGraph Trading Agent
- [ ] Phase 5: Web Dashboard
- [ ] Phase 6: Notifications & Tax Export
- [ ] Phase 7: Security Hardening

## License

See [LICENSE](LICENSE) for details.
