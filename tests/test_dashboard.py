"""Tests for FastAPI dashboard endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dashboard.api import app, set_database
from db.database import Database
from db import queries


@pytest.fixture
def client(test_db: Database):
    """Create test client with initialized DB."""
    set_database(test_db)
    return TestClient(app)


class TestDashboardRoot:
    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "AI Algo Trading Dashboard" in resp.text


class TestStatusEndpoint:
    def test_get_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "us" in data
        assert "india" in data
        assert data["us"]["mode"] == "paper"

    def test_status_has_timestamp(self, client):
        resp = client.get("/api/status")
        data = resp.json()
        assert "timestamp" in data


class TestTradesEndpoint:
    def test_get_trades_empty(self, client):
        resp = client.get("/api/trades?market=us")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_get_trades_with_data(self, client, test_db):
        queries.insert_trade(
            test_db, market="us", asset="AAPL", direction="long",
            entry_price=150, quantity=10, entry_time="2026-04-16T10:00:00",
            strategy="momentum",
        )
        resp = client.get("/api/trades?market=us")
        data = resp.json()
        assert data["count"] == 1
        assert data["trades"][0]["asset"] == "AAPL"

    def test_get_open_trades(self, client, test_db):
        queries.insert_trade(
            test_db, market="us", asset="MSFT", direction="long",
            entry_price=400, quantity=5, entry_time="2026-04-16T10:00:00",
            strategy="breakout",
        )
        resp = client.get("/api/trades/open")
        data = resp.json()
        assert data["count"] == 1


class TestSignalsEndpoint:
    def test_get_signals_empty(self, client):
        resp = client.get("/api/signals?market=us")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_get_signals_with_data(self, client, test_db):
        queries.insert_signal(
            test_db, market="us", asset="AAPL", direction="long",
            confidence=0.8, timeframe="15m", strategy="momentum",
        )
        resp = client.get("/api/signals?market=us")
        data = resp.json()
        assert data["count"] == 1


class TestSentimentEndpoint:
    def test_get_sentiment_empty(self, client):
        resp = client.get("/api/sentiment/AAPL?market=us")
        assert resp.status_code == 200
        data = resp.json()
        assert data["latest"] is None

    def test_get_sentiment_with_data(self, client, test_db):
        queries.insert_sentiment(
            test_db, market="us", asset="AAPL", score=0.7,
            headline="Good news", source="test",
        )
        resp = client.get("/api/sentiment/AAPL?market=us")
        data = resp.json()
        assert data["latest"]["score"] == 0.7


class TestPerformanceEndpoint:
    def test_get_performance_empty(self, client):
        resp = client.get("/api/performance?market=us")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0


class TestBacktestEndpoint:
    def test_get_backtest_empty(self, client):
        resp = client.get("/api/backtest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_get_backtest_with_filter(self, client, test_db):
        queries.insert_backtest_result(
            test_db, market="us", strategy="momentum",
            start_date="2025-01-01", end_date="2025-12-31",
            total_trades=100, winning_trades=65, losing_trades=35,
            win_rate=0.65, sharpe_ratio=1.2, max_drawdown_pct=0.08,
            profit_factor=1.5, avg_trade_duration="2h",
            max_consecutive_losses=4,
            total_pnl=5000, total_fees=100, net_pnl=4900,
        )
        resp = client.get("/api/backtest?market=us&strategy=momentum")
        data = resp.json()
        assert data["count"] == 1


class TestErrorsEndpoint:
    def test_get_errors_empty(self, client):
        resp = client.get("/api/errors")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0


class TestControlEndpoints:
    def test_halt_market(self, client, test_db):
        resp = client.post("/api/halt?market=us&reason=Test+halt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "halted"

        state = queries.get_bot_state(test_db, "us")
        assert state["halted"] == 1

    def test_resume_market(self, client, test_db):
        queries.update_bot_state(test_db, "us", halted=1)
        resp = client.post("/api/resume?market=us")
        assert resp.status_code == 200

        state = queries.get_bot_state(test_db, "us")
        assert state["halted"] == 0


class TestConfigEndpoint:
    def test_get_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        # May return config or error depending on config loading
        data = resp.json()
        assert isinstance(data, dict)
