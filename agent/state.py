"""
LangGraph agent state schema.

Defines the typed state that flows through the trading agent graph.
Each node reads from and writes to this state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from strategy.base import Signal


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph trading agent."""

    # Market context
    market: str                          # "us" or "india"
    asset: str                           # current ticker being analyzed
    assets: list[str]                    # full watchlist

    # Data
    price_data: dict[str, Any]           # OHLCV DataFrames keyed by asset
    sentiment_scores: dict[str, float]   # sentiment per asset
    news_items: dict[str, list[dict]]    # news per asset

    # Signals
    signals: list[dict[str, Any]]        # raw signal dicts
    best_signal: dict[str, Any] | None   # chosen signal to act on

    # Risk check
    risk_approved: bool
    risk_reason: str
    position_size: float
    stop_loss: float | None
    take_profit: float | None

    # Psychology check
    psych_approved: bool
    psych_reason: str

    # Execution
    order_result: dict[str, Any] | None
    trade_id: int | None

    # Control flow
    action: str                          # "trade", "skip", "halt", "error"
    error: str
    iteration: int                       # loop counter
    messages: list[str]                  # audit trail
