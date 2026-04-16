# ADR 001: Architecture & Technology Decisions

**Status:** Accepted
**Date:** 2026-04-16

## Context

We need to build a fully autonomous AI trading bot that trades US and Indian markets simultaneously, runs 24/7, paper trades first, and graduates to live trading only after statistically significant performance.

## Decisions

### Language: Python 3.11+

**Why:** Python has the strongest ecosystem for financial data (yfinance, pandas, ta), AI/ML (anthropic, langchain), and async web (FastAPI). Type hints (3.11+) give us safety without Java-level verbosity. Every broker SDK we need (alpaca-py, kiteconnect) is Python-first.

**Alternatives considered:** Node.js (weaker data science ecosystem), Rust (overkill for I/O-bound trading, broker SDKs don't exist), Go (missing financial/AI libraries).

### Trading Agent: LangGraph

**Why:** LangGraph provides stateful graph execution with checkpointing, branching, and human-in-the-loop support. This maps perfectly to a trading workflow: market data -> signals -> risk check -> execute -> monitor. State persistence means the bot survives restarts. The graph structure makes it easy to add/remove nodes (strategies) without touching other parts.

**Alternatives considered:** Plain asyncio loops (loses state management, harder to debug), Temporal (too heavy for this use case), custom FSM (reinventing the wheel).

### AI Model: Claude Sonnet via Anthropic API

**Why:** Claude Sonnet offers the best cost/quality ratio for sentiment analysis. It handles nuanced financial news better than smaller models. The Anthropic SDK is clean and well-maintained. Costs ~$1-5/day at our volume.

**Alternatives considered:** GPT-4 (more expensive, similar quality), local LLMs (too slow for real-time sentiment, need GPU), FinBERT (good for sentiment only, can't explain reasoning).

### US Broker: Alpaca

**Why:** Free paper trading, $0 commissions on stocks, excellent Python SDK (alpaca-py), WebSocket streaming for real-time data, REST API for orders. Supports fractional shares. Well-documented.

**Alternatives considered:** Interactive Brokers (complex API, higher barrier), TD Ameritrade (API being deprecated post-Schwab merger), Robinhood (no official API).

### India Broker: Zerodha Kite Connect

**Why:** Most popular broker API in India. INR 20 flat per trade (cheapest). Good Python SDK (kiteconnect). WebSocket for live data. Supports NSE and BSE.

**Decision:** For paper trading, we build a local simulator (no Kite subscription needed). Kite API only needed for live trading.

**Alternatives considered:** Upstox (less mature API), Angel Broking (fewer features), Fyers (smaller community).

### Database: SQLite with WAL mode

**Why:** Zero setup, zero dependencies, perfect for Railway deployment (single file). WAL (Write-Ahead Logging) mode gives us concurrent reads during writes — critical since the dashboard reads while the bot writes. Schema uses standard SQL and TEXT timestamps, making PostgreSQL migration trivial when needed.

**Migration path:** When we outgrow SQLite (>10GB data or need concurrent writers), swap `db/database.py` to use asyncpg. Schema stays the same.

**Alternatives considered:** PostgreSQL (needs separate service on Railway, costs money), MongoDB (schema-less is wrong for financial data), Redis (no persistence guarantees).

### Web Framework: FastAPI

**Why:** Native async support, automatic OpenAPI docs, WebSocket support for real-time dashboard, type validation via Pydantic. Fastest Python web framework for our use case.

**Alternatives considered:** Flask (no native async, no WebSocket), Django (too heavy, ORM we don't need), Starlette (FastAPI is built on it, adds value).

### Frontend: React

**Why:** Component model fits our dashboard layout (multiple independent cards). Huge ecosystem for charting (Recharts). Can be served as static files from FastAPI.

**Alternatives considered:** Svelte (smaller ecosystem for financial charts), HTMX (too simple for real-time charts), Vue (similar to React, less financial tooling).

### Hosting: Railway

**Why:** Free tier (500 hrs/month) is enough for a single-process Python app. Auto-restart on crash. GitHub integration for auto-deploy. Simple environment variable management. No Kubernetes complexity.

**Alternatives considered:** Heroku (free tier removed), AWS Lambda (wrong model for persistent bot), DigitalOcean (costs money from day 1), self-hosted VPS (maintenance burden).

### Logging: structlog (JSON)

**Why:** Structured JSON logs are machine-parseable (Railway, Datadog, etc.) and human-readable with pretty-printing. Context binding lets us attach trade_id, market, asset to every log line without passing it around. Better than stdlib logging for traceability.

### Configuration: YAML

**Why:** More readable than JSON for nested config. Supports comments. PyYAML is battle-tested. All strategy parameters, risk thresholds, and graduation criteria live in one file.

## Architectural Principles

1. **Broker-agnostic core**: All broker calls go through `BaseBroker` interface. Adding a new broker = one new file.
2. **Config-driven behavior**: No magic numbers in code. Everything tunable in `config.yaml`.
3. **Fail-safe defaults**: If anything is unclear, don't trade. Risk checks reject ambiguous signals.
4. **Audit everything**: Every decision, trade, and error is logged to the database.
5. **Independent markets**: US and India run as separate state machines. One market's issues never affect the other.

## Consequences

- SQLite limits us to a single writer process (fine for now, migrate to PostgreSQL if needed)
- Railway free tier limits execution to ~16 hrs/day (upgrade to paid when going live)
- Kite Connect access token expires daily (need automated refresh for live India trading)
