"""
LangGraph trading agent graph definition.

Defines the state machine that drives the autonomous trading loop:
  analyze_market → should_trade? → check_risk → check_psychology → execute_trade
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agent.state import AgentState
from agent.nodes import (
    analyze_market,
    check_risk,
    check_psychology,
    execute_trade,
    should_trade,
    should_execute,
)


def build_trading_graph(deps: dict[str, Any] | None = None) -> StateGraph:
    """Build the trading agent graph.

    Args:
        deps: Dependency injection dict with:
            - signal_generator: SignalGenerator
            - risk_manager: RiskManager
            - psychology_guard: PsychologyGuard
            - broker: BaseBroker
            - db: Database
            - capital: float

    Returns:
        Compiled StateGraph ready to invoke.
    """
    deps = deps or {}

    # Bind dependencies to nodes
    def _analyze(state: AgentState) -> dict:
        return analyze_market(state, **deps)

    def _check_risk(state: AgentState) -> dict:
        return check_risk(state, **deps)

    def _check_psych(state: AgentState) -> dict:
        return check_psychology(state, **deps)

    def _execute(state: AgentState) -> dict:
        return execute_trade(state, **deps)

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze_market", _analyze)
    graph.add_node("check_risk", _check_risk)
    graph.add_node("check_psychology", _check_psych)
    graph.add_node("execute_trade", _execute)

    # Set entry point
    graph.set_entry_point("analyze_market")

    # Add conditional edges
    graph.add_conditional_edges(
        "analyze_market",
        should_trade,
        {
            "check_risk": "check_risk",
            "end": END,
        },
    )

    graph.add_edge("check_risk", "check_psychology")

    graph.add_conditional_edges(
        "check_psychology",
        should_execute,
        {
            "execute": "execute_trade",
            "end": END,
        },
    )

    graph.add_edge("execute_trade", END)

    return graph


def create_trading_agent(deps: dict[str, Any] | None = None):
    """Create a compiled trading agent ready for invocation.

    Usage:
        agent = create_trading_agent(deps)
        result = agent.invoke({
            "market": "us",
            "assets": ["AAPL", "MSFT"],
            "price_data": {...},
            "sentiment_scores": {...},
        })
    """
    graph = build_trading_graph(deps)
    return graph.compile()
