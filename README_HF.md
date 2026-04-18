---
title: AI Algo Trading Dashboard
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Autonomous trading system dashboard (US + India markets)
---

# AI Algo Trading System — Live Dashboard

Autonomous algorithmic trading dashboard running the full FastAPI backend.

## Features

- **Live FastAPI backend** with 13 REST endpoints + WebSocket
- **Dual-market support**: US (Alpaca) + India (Zerodha Kite)
- **8 trading strategies**: momentum, mean reversion, breakout, MA crossover, RSI divergence, volume-price, sentiment, hybrid
- **AI sentiment analysis** using Claude API
- **Risk management + psychology guard**
- **Backtester** with Sharpe ratio, max drawdown, profit factor

## Demo Mode

This Space runs in **paper/demo mode** — no live broker keys are required.
The dashboard is pre-seeded with sample trades and sentiment data so you can
explore the UI immediately.

## Repo

Source: [github.com/MavinashR88/AI-Algo-Trading](https://github.com/MavinashR88/AI-Algo-Trading)
