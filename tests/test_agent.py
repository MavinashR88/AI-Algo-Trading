"""Tests for trading agent: psychology guard, nodes, and graph."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from db import queries
from db.database import Database
from agent.psychology_guard import PsychologyGuard, PsychologyCheck
from agent.state import AgentState
from agent.nodes import analyze_market, check_risk, check_psychology, should_trade, should_execute
from agent.graph import build_trading_graph, create_trading_agent
from strategy.signal_generator import SignalGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_df(rows: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    rng = np.random.RandomState(42)
    prices = 100.0 + np.cumsum(rng.uniform(0.1, 0.8, rows))
    return pd.DataFrame({
        "Open": prices - rng.uniform(0.1, 0.5, rows),
        "High": prices + rng.uniform(0.5, 2.0, rows),
        "Low": prices - rng.uniform(0.5, 2.0, rows),
        "Close": prices,
        "Volume": rng.uniform(1e6, 5e6, rows),
    }, index=dates)


@pytest.fixture
def psych_config():
    return {
        "revenge_trade_lookback": 2,
        "revenge_trade_cooldown_minutes": 30,
        "min_signal_confidence": 0.65,
        "sentiment_override_threshold": -0.7,
        "no_averaging_down": True,
    }


# ---------------------------------------------------------------------------
# Psychology Guard
# ---------------------------------------------------------------------------

class TestPsychologyCheck:
    def test_approved(self):
        c = PsychologyCheck(approved=True)
        assert c.approved

    def test_rejected(self):
        c = PsychologyCheck(approved=False, reason="test", guard_type="test")
        assert not c.approved


class TestPsychologyGuard:
    def test_low_confidence_rejected(self, psych_config):
        guard = PsychologyGuard(psych_config)
        result = guard.check("us", "AAPL", "long", confidence=0.3)
        assert not result.approved
        assert result.guard_type == "low_confidence"

    def test_high_confidence_approved(self, psych_config):
        guard = PsychologyGuard(psych_config)
        result = guard.check("us", "AAPL", "long", confidence=0.80)
        assert result.approved

    def test_sentiment_override_blocks_long(self, psych_config):
        guard = PsychologyGuard(psych_config)
        result = guard.check("us", "AAPL", "long", confidence=0.80, sentiment_score=-0.9)
        assert not result.approved
        assert result.guard_type == "sentiment_override"

    def test_sentiment_override_allows_short(self, psych_config):
        guard = PsychologyGuard(psych_config)
        result = guard.check("us", "AAPL", "short", confidence=0.80, sentiment_score=-0.9)
        assert result.approved

    def test_neutral_sentiment_ok(self, psych_config):
        guard = PsychologyGuard(psych_config)
        result = guard.check("us", "AAPL", "long", confidence=0.80, sentiment_score=0.0)
        assert result.approved


class TestPsychologyGuardWithDB:
    def test_no_averaging_down(self, psych_config, test_db: Database):
        queries.insert_trade(
            test_db, market="us", asset="AAPL", direction="long",
            entry_price=150, quantity=10, entry_time="2026-04-16T10:00:00",
            strategy="test",
        )
        guard = PsychologyGuard(psych_config, db=test_db)
        result = guard.check("us", "AAPL", "long", confidence=0.80)
        assert not result.approved
        assert result.guard_type == "no_averaging_down"

    def test_different_asset_ok(self, psych_config, test_db: Database):
        queries.insert_trade(
            test_db, market="us", asset="AAPL", direction="long",
            entry_price=150, quantity=10, entry_time="2026-04-16T10:00:00",
            strategy="test",
        )
        guard = PsychologyGuard(psych_config, db=test_db)
        result = guard.check("us", "MSFT", "long", confidence=0.80)
        assert result.approved

    def test_consecutive_losses_cooldown(self, psych_config, test_db: Database):
        queries.update_bot_state(test_db, "us", consecutive_losses=3)
        # Insert a recent closed trade
        tid = queries.insert_trade(
            test_db, market="us", asset="TSLA", direction="long",
            entry_price=200, quantity=5, entry_time=datetime.utcnow().isoformat(),
            strategy="test",
        )
        queries.close_trade(
            test_db, tid, exit_price=190,
            exit_time=datetime.utcnow().isoformat(),
            pnl=-50, pnl_pct=-0.05, net_pnl=-51, fees=1,
        )

        guard = PsychologyGuard(psych_config, db=test_db)
        result = guard.check("us", "NVDA", "long", confidence=0.80)
        assert not result.approved
        assert result.guard_type == "revenge_trade"

    def test_record_win_resets_counter(self, psych_config, test_db: Database):
        queries.update_bot_state(test_db, "us", consecutive_losses=3)
        guard = PsychologyGuard(psych_config, db=test_db)
        guard.record_trade_result("us", is_win=True)

        state = queries.get_bot_state(test_db, "us")
        assert state["consecutive_losses"] == 0

    def test_record_loss_increments(self, psych_config, test_db: Database):
        queries.update_bot_state(test_db, "us", consecutive_losses=1)
        guard = PsychologyGuard(psych_config, db=test_db)
        guard.record_trade_result("us", is_win=False)

        state = queries.get_bot_state(test_db, "us")
        assert state["consecutive_losses"] == 2


# ---------------------------------------------------------------------------
# Agent Nodes
# ---------------------------------------------------------------------------

class TestAnalyzeMarketNode:
    def test_with_signals(self):
        df = make_test_df(100)
        state: AgentState = {
            "market": "us",
            "assets": ["AAPL"],
            "price_data": {"AAPL": df},
            "sentiment_scores": {},
            "messages": [],
        }
        result = analyze_market(state, signal_generator=SignalGenerator())
        assert "signals" in result
        assert "best_signal" in result
        assert isinstance(result["messages"], list)

    def test_empty_assets(self):
        state: AgentState = {
            "market": "us",
            "assets": [],
            "price_data": {},
            "sentiment_scores": {},
            "messages": [],
        }
        result = analyze_market(state)
        assert result["action"] == "skip"
        assert result["best_signal"] is None


class TestCheckRiskNode:
    def test_no_signal_skips(self):
        state: AgentState = {"market": "us", "best_signal": None, "messages": []}
        result = check_risk(state)
        assert result["action"] == "skip"

    def test_with_signal_no_risk_mgr(self):
        state: AgentState = {
            "market": "us",
            "best_signal": {
                "asset": "AAPL", "direction": "long", "confidence": 0.8,
                "strategy": "momentum", "stop_loss": 145.0, "take_profit": 165.0,
                "metadata": {},
            },
            "messages": [],
        }
        result = check_risk(state)
        assert result["risk_approved"] is True


class TestCheckPsychologyNode:
    def test_no_signal_skips(self):
        state: AgentState = {
            "market": "us",
            "best_signal": None,
            "risk_approved": False,
            "messages": [],
        }
        result = check_psychology(state)
        assert result["psych_approved"] is False

    def test_approved_without_guard(self):
        state: AgentState = {
            "market": "us",
            "best_signal": {"asset": "AAPL", "direction": "long", "confidence": 0.8},
            "risk_approved": True,
            "sentiment_scores": {},
            "messages": [],
        }
        result = check_psychology(state)
        assert result["psych_approved"] is True


class TestEdgeFunctions:
    def test_should_trade_yes(self):
        assert should_trade({"action": "trade"}) == "check_risk"

    def test_should_trade_no(self):
        assert should_trade({"action": "skip"}) == "end"

    def test_should_execute_yes(self):
        assert should_execute({"risk_approved": True, "psych_approved": True}) == "execute"

    def test_should_execute_no(self):
        assert should_execute({"risk_approved": False, "psych_approved": True}) == "end"
        assert should_execute({"risk_approved": True, "psych_approved": False}) == "end"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

class TestTradingGraph:
    def test_build_graph(self):
        graph = build_trading_graph()
        assert graph is not None

    def test_create_agent(self):
        agent = create_trading_agent()
        assert agent is not None

    def test_invoke_agent_skip(self):
        agent = create_trading_agent()
        result = agent.invoke({
            "market": "us",
            "assets": [],
            "price_data": {},
            "sentiment_scores": {},
            "messages": [],
        })
        assert result["action"] == "skip"

    def test_invoke_agent_with_data(self):
        df = make_test_df(100)
        agent = create_trading_agent(deps={
            "signal_generator": SignalGenerator(),
        })
        result = agent.invoke({
            "market": "us",
            "assets": ["AAPL"],
            "price_data": {"AAPL": df},
            "sentiment_scores": {},
            "messages": [],
        })
        assert "signals" in result
        assert isinstance(result["messages"], list)
