"""Tests for database layer: schema creation, queries, and data integrity."""

from __future__ import annotations

import pytest

from db.database import Database
from db import queries


class TestDatabaseInitialization:
    """Test database setup and schema creation."""

    def test_initialize_creates_tables(self, test_db: Database):
        """All expected tables exist after initialization."""
        rows = test_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {r["name"] for r in rows}

        expected = {
            "schema_version", "bot_state", "trades", "signals",
            "sentiment_scores", "market_data_cache", "error_logs",
            "daily_summaries", "backtest_results", "audit_log",
        }
        assert expected.issubset(table_names)

    def test_initialize_is_idempotent(self, test_db: Database):
        """Calling initialize() twice doesn't raise or duplicate data."""
        test_db.initialize()  # second call
        rows = test_db.execute("SELECT COUNT(*) as cnt FROM bot_state")
        assert rows[0]["cnt"] == 2  # still just 'us' and 'india'

    def test_bot_state_initialized_for_both_markets(self, test_db: Database):
        """Bot state rows exist for both US and India."""
        us = test_db.execute_one("SELECT * FROM bot_state WHERE market = 'us'")
        india = test_db.execute_one("SELECT * FROM bot_state WHERE market = 'india'")
        assert us is not None
        assert india is not None
        assert us["mode"] == "paper"
        assert india["mode"] == "paper"

    def test_wal_journal_mode(self, tmp_db_path):
        """WAL journal mode is set on connection."""
        db = Database(db_path=tmp_db_path, journal_mode="WAL")
        db.initialize()
        row = db.execute_one("PRAGMA journal_mode")
        assert row[0] == "wal"

    def test_schema_version_tracked(self, test_db: Database):
        """Schema version is recorded in schema_version table."""
        row = test_db.execute_one("SELECT MAX(version) as v FROM schema_version")
        assert row["v"] == 1

    def test_indexes_created(self, test_db: Database):
        """Indexes are created for common query patterns."""
        rows = test_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        index_names = {r["name"] for r in rows}
        assert "idx_trades_market" in index_names
        assert "idx_trades_asset" in index_names
        assert "idx_signals_market" in index_names


class TestTradeQueries:
    """Test trade CRUD operations."""

    def test_insert_and_retrieve_trade(self, test_db: Database):
        """Insert a trade and retrieve it."""
        trade_id = queries.insert_trade(
            test_db,
            market="us",
            asset="AAPL",
            direction="long",
            entry_price=150.0,
            quantity=10,
            entry_time="2026-04-16T10:00:00",
            strategy="momentum",
            confidence=0.75,
            mode="paper",
        )
        assert trade_id > 0

        open_trades = queries.get_open_trades(test_db, "us")
        assert len(open_trades) == 1
        assert open_trades[0]["asset"] == "AAPL"
        assert open_trades[0]["direction"] == "long"
        assert open_trades[0]["entry_price"] == 150.0

    def test_close_trade(self, test_db: Database):
        """Close a trade and verify P&L is recorded."""
        trade_id = queries.insert_trade(
            test_db,
            market="us",
            asset="MSFT",
            direction="long",
            entry_price=400.0,
            quantity=5,
            entry_time="2026-04-16T10:00:00",
            strategy="breakout",
        )

        queries.close_trade(
            test_db,
            trade_id,
            exit_price=410.0,
            exit_time="2026-04-16T14:00:00",
            pnl=50.0,
            pnl_pct=0.025,
            net_pnl=49.5,
            fees=0.5,
        )

        open_trades = queries.get_open_trades(test_db, "us")
        assert len(open_trades) == 0

        recent = queries.get_recent_trades(test_db, "us", limit=1)
        assert len(recent) == 1
        assert recent[0]["status"] == "closed"
        assert recent[0]["pnl"] == 50.0
        assert recent[0]["net_pnl"] == 49.5

    def test_count_trades_today(self, test_db: Database):
        """Count trades for today."""
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        queries.insert_trade(
            test_db, market="us", asset="AAPL", direction="long",
            entry_price=100, quantity=1, entry_time=today, strategy="test",
        )
        queries.insert_trade(
            test_db, market="us", asset="MSFT", direction="long",
            entry_price=200, quantity=1, entry_time=today, strategy="test",
        )

        count = queries.count_trades_today(test_db, "us")
        assert count == 2

    def test_open_trades_filtered_by_market(self, test_db: Database):
        """Open trades are filtered by market."""
        queries.insert_trade(
            test_db, market="us", asset="AAPL", direction="long",
            entry_price=100, quantity=1, entry_time="2026-04-16T10:00:00",
            strategy="test",
        )
        queries.insert_trade(
            test_db, market="india", asset="RELIANCE", direction="long",
            entry_price=2500, quantity=1, entry_time="2026-04-16T10:00:00",
            strategy="test",
        )

        us_trades = queries.get_open_trades(test_db, "us")
        india_trades = queries.get_open_trades(test_db, "india")
        all_trades = queries.get_open_trades(test_db)

        assert len(us_trades) == 1
        assert len(india_trades) == 1
        assert len(all_trades) == 2


class TestSignalQueries:
    """Test signal operations."""

    def test_insert_and_reject_signal(self, test_db: Database):
        """Insert a signal and mark it rejected."""
        signal_id = queries.insert_signal(
            test_db,
            market="us",
            asset="TSLA",
            direction="long",
            confidence=0.55,
            timeframe="15m",
            strategy="momentum",
            reasoning="Low confidence",
        )

        queries.mark_signal_rejected(signal_id=signal_id, db=test_db, reason="Below confidence threshold")

        row = test_db.execute_one("SELECT * FROM signals WHERE id = ?", (signal_id,))
        assert row["acted_on"] == 0
        assert row["reject_reason"] == "Below confidence threshold"

    def test_insert_and_act_on_signal(self, test_db: Database):
        """Insert a signal and mark it as acted upon."""
        signal_id = queries.insert_signal(
            test_db,
            market="india",
            asset="TCS",
            direction="long",
            confidence=0.80,
            timeframe="15m",
            strategy="breakout",
        )

        queries.mark_signal_acted(test_db, signal_id)

        row = test_db.execute_one("SELECT * FROM signals WHERE id = ?", (signal_id,))
        assert row["acted_on"] == 1


class TestSentimentQueries:
    """Test sentiment score operations."""

    def test_insert_and_get_sentiment(self, test_db: Database):
        """Insert sentiment and retrieve latest."""
        queries.insert_sentiment(
            test_db,
            market="us",
            asset="AAPL",
            score=0.65,
            headline="Apple beats earnings",
            source="newsapi",
        )
        queries.insert_sentiment(
            test_db,
            market="us",
            asset="AAPL",
            score=0.80,
            headline="Apple announces new product",
            source="google_news",
        )

        latest = queries.get_latest_sentiment(test_db, "us", "AAPL")
        assert latest is not None
        assert latest["score"] == 0.80

    def test_get_sentiment_for_nonexistent_asset(self, test_db: Database):
        """No sentiment returns None."""
        result = queries.get_latest_sentiment(test_db, "us", "NONEXISTENT")
        assert result is None


class TestBotStateQueries:
    """Test bot state operations."""

    def test_get_bot_state(self, test_db: Database):
        """Get bot state for a market."""
        state = queries.get_bot_state(test_db, "us")
        assert state is not None
        assert state["market"] == "us"
        assert state["mode"] == "paper"
        assert state["capital"] == 0.0

    def test_update_bot_state(self, test_db: Database):
        """Update bot state fields."""
        queries.update_bot_state(
            test_db, "us",
            capital=100000.0,
            win_rate=0.72,
            total_trades=50,
        )

        state = queries.get_bot_state(test_db, "us")
        assert state["capital"] == 100000.0
        assert state["win_rate"] == 0.72
        assert state["total_trades"] == 50


class TestAuditLog:
    """Test audit logging."""

    def test_log_audit(self, test_db: Database):
        """Audit log entries are stored."""
        test_db.log_audit("test_action", "test_module", {"key": "value"})

        rows = test_db.execute(
            "SELECT * FROM audit_log WHERE action = 'test_action'"
        )
        assert len(rows) == 1
        assert rows[0]["module"] == "test_module"


class TestErrorLogQueries:
    """Test error log operations."""

    def test_insert_error_log(self, test_db: Database):
        """Insert and retrieve error logs."""
        err_id = queries.insert_error_log(
            test_db,
            level="ERROR",
            module="brokers.alpaca",
            message="Connection timeout",
            context={"retry_count": 3},
        )
        assert err_id > 0

        rows = test_db.execute("SELECT * FROM error_logs WHERE id = ?", (err_id,))
        assert len(rows) == 1
        assert rows[0]["level"] == "ERROR"
        assert rows[0]["module"] == "brokers.alpaca"
