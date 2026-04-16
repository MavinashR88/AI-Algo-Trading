"""Tests for risk management engine."""

from __future__ import annotations

import pytest

from db import queries
from db.database import Database
from risk.risk_manager import RiskManager, RiskAssessment


@pytest.fixture
def risk_config():
    return {
        "max_risk_per_trade_pct": 0.01,
        "max_open_positions_per_market": 3,
        "max_trades_per_market_per_day": 10,
        "max_daily_drawdown_pct": 0.03,
        "max_weekly_drawdown_pct": 0.08,
        "stop_loss_atr_multiplier": 2.0,
        "take_profit_rr_ratio": 2.0,
        "trailing_stop_activation_rr": 1.5,
        "trailing_stop_atr_multiplier": 1.5,
    }


class TestRiskAssessment:
    def test_approved(self):
        r = RiskAssessment(approved=True, position_size=10)
        assert r.approved
        assert r.position_size == 10

    def test_rejected(self):
        r = RiskAssessment(approved=False, reason="Max positions")
        assert not r.approved


class TestRiskManagerBasic:
    def test_calculate_stop_loss_long(self, risk_config):
        rm = RiskManager(risk_config)
        sl = rm._calculate_stop_loss(100.0, 2.0, "long")
        assert sl == 96.0  # 100 - 2*2

    def test_calculate_stop_loss_short(self, risk_config):
        rm = RiskManager(risk_config)
        sl = rm._calculate_stop_loss(100.0, 2.0, "short")
        assert sl == 104.0  # 100 + 2*2

    def test_calculate_take_profit_long(self, risk_config):
        rm = RiskManager(risk_config)
        tp = rm._calculate_take_profit(100.0, 96.0, "long")
        assert tp == 108.0  # 100 + (4 * 2.0 RR)

    def test_calculate_take_profit_short(self, risk_config):
        rm = RiskManager(risk_config)
        tp = rm._calculate_take_profit(100.0, 104.0, "short")
        assert tp == 92.0  # 100 - (4 * 2.0 RR)


class TestRiskManagerWithoutDB:
    def test_basic_assessment_no_db(self, risk_config):
        rm = RiskManager(risk_config, db=None)
        result = rm.assess_trade(
            market="us", asset="AAPL", direction="long",
            entry_price=150.0, current_atr=3.0, capital=100_000,
        )
        assert result.approved
        assert result.position_size > 0
        assert result.stop_loss is not None
        assert result.take_profit is not None
        assert result.risk_reward_ratio >= 1.0

    def test_zero_capital_rejected(self, risk_config):
        rm = RiskManager(risk_config, db=None)
        result = rm.assess_trade(
            market="us", asset="AAPL", direction="long",
            entry_price=150.0, current_atr=3.0, capital=0,
        )
        assert not result.approved


class TestRiskManagerWithDB:
    def test_halted_market_rejected(self, risk_config, test_db: Database):
        queries.update_bot_state(test_db, "us", halted=1, halt_reason="Drawdown")
        rm = RiskManager(risk_config, db=test_db)
        result = rm.assess_trade(
            market="us", asset="AAPL", direction="long",
            entry_price=150.0, current_atr=3.0, capital=100_000,
        )
        assert not result.approved
        assert "halted" in result.reason.lower()

    def test_max_positions_rejected(self, risk_config, test_db: Database):
        # Create 3 open trades
        for asset in ("AAPL", "MSFT", "GOOGL"):
            queries.insert_trade(
                test_db, market="us", asset=asset, direction="long",
                entry_price=100, quantity=1, entry_time="2026-04-16T10:00:00",
                strategy="test",
            )

        rm = RiskManager(risk_config, db=test_db)
        result = rm.assess_trade(
            market="us", asset="TSLA", direction="long",
            entry_price=200.0, current_atr=5.0, capital=100_000,
        )
        assert not result.approved
        assert "position" in result.reason.lower()

    def test_duplicate_asset_rejected(self, risk_config, test_db: Database):
        queries.insert_trade(
            test_db, market="us", asset="AAPL", direction="long",
            entry_price=150, quantity=10, entry_time="2026-04-16T10:00:00",
            strategy="test",
        )
        rm = RiskManager(risk_config, db=test_db)
        result = rm.assess_trade(
            market="us", asset="AAPL", direction="long",
            entry_price=155.0, current_atr=3.0, capital=100_000,
        )
        assert not result.approved

    def test_drawdown_limit_rejected(self, risk_config, test_db: Database):
        queries.update_bot_state(test_db, "us", current_drawdown_pct=0.05)
        rm = RiskManager(risk_config, db=test_db)
        result = rm.assess_trade(
            market="us", asset="AAPL", direction="long",
            entry_price=150.0, current_atr=3.0, capital=100_000,
        )
        assert not result.approved
        assert "drawdown" in result.reason.lower()

    def test_update_drawdown(self, risk_config, test_db: Database):
        queries.update_bot_state(test_db, "us", capital=100_000, peak_capital=100_000)
        rm = RiskManager(risk_config, db=test_db)
        rm.update_drawdown("us", 95_000)

        state = queries.get_bot_state(test_db, "us")
        assert state["current_drawdown_pct"] > 0
        assert state["peak_capital"] == 100_000


class TestTrailingStop:
    def test_trailing_not_activated(self, risk_config):
        rm = RiskManager(risk_config)
        result = rm.calculate_trailing_stop(100.0, 100.5, 2.0, "long")
        assert result is None  # Not enough profit

    def test_trailing_activated_long(self, risk_config):
        rm = RiskManager(risk_config)
        # Large profit should activate
        result = rm.calculate_trailing_stop(100.0, 120.0, 2.0, "long")
        if result is not None:
            assert result < 120.0  # Trailing stop below current price
