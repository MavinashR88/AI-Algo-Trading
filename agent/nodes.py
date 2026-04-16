"""
LangGraph agent nodes.

Each function is a node in the trading agent graph. Nodes read from
AgentState, perform work, and return state updates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from agent.state import AgentState
from strategy.base import Signal
from strategy.signal_generator import SignalGenerator
from risk.risk_manager import RiskManager, RiskAssessment
from agent.psychology_guard import PsychologyGuard, PsychologyCheck

logger = structlog.get_logger(__name__)


def analyze_market(state: AgentState, **deps: Any) -> dict[str, Any]:
    """Node: Fetch data and generate signals for all assets.

    Dependencies (passed via deps):
        - signal_generator: SignalGenerator
        - data_fetcher: MarketDataFetcher
        - sentiment_analyzer: SentimentAnalyzer
    """
    market = state["market"]
    assets = state.get("assets", [])
    price_data = state.get("price_data", {})
    sentiment_scores = state.get("sentiment_scores", {})

    signal_gen: SignalGenerator = deps.get("signal_generator", SignalGenerator())

    all_signals: list[dict[str, Any]] = []
    best_signal: dict[str, Any] | None = None
    best_confidence = 0.0

    for asset in assets:
        df = price_data.get(asset)
        if df is None or (hasattr(df, "empty") and df.empty):
            continue

        sentiment = sentiment_scores.get(asset)
        sig = signal_gen.get_best_signal(df, asset, market, sentiment)

        if sig and sig.is_actionable:
            sig_dict = {
                "asset": sig.asset,
                "direction": sig.direction,
                "confidence": sig.confidence,
                "strategy": sig.strategy,
                "reasoning": sig.reasoning,
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
                "timeframe": sig.timeframe,
                "metadata": sig.metadata,
            }
            all_signals.append(sig_dict)

            if sig.confidence > best_confidence:
                best_confidence = sig.confidence
                best_signal = sig_dict

    logger.info(
        "agent.analyze_complete",
        market=market,
        total_signals=len(all_signals),
        best_asset=best_signal["asset"] if best_signal else None,
    )

    return {
        "signals": all_signals,
        "best_signal": best_signal,
        "action": "trade" if best_signal else "skip",
        "messages": state.get("messages", []) + [
            f"[{datetime.utcnow().isoformat()}] Analyzed {len(assets)} assets, "
            f"found {len(all_signals)} signals."
        ],
    }


def check_risk(state: AgentState, **deps: Any) -> dict[str, Any]:
    """Node: Run risk management checks on the best signal."""
    best_signal = state.get("best_signal")
    if not best_signal:
        return {"action": "skip", "risk_approved": False, "risk_reason": "No signal"}

    market = state["market"]
    risk_mgr: RiskManager = deps.get("risk_manager")

    if not risk_mgr:
        return {
            "risk_approved": True,
            "position_size": 1,
            "stop_loss": best_signal.get("stop_loss"),
            "take_profit": best_signal.get("take_profit"),
        }

    # Get current capital from state or default
    capital = deps.get("capital", 100_000.0)
    current_atr = best_signal.get("metadata", {}).get("atr", 2.0)

    assessment = risk_mgr.assess_trade(
        market=market,
        asset=best_signal["asset"],
        direction=best_signal["direction"],
        entry_price=best_signal.get("metadata", {}).get("entry_price", 0),
        current_atr=current_atr,
        capital=capital,
        signal_stop_loss=best_signal.get("stop_loss"),
        signal_take_profit=best_signal.get("take_profit"),
    )

    return {
        "risk_approved": assessment.approved,
        "risk_reason": assessment.reason,
        "position_size": assessment.position_size,
        "stop_loss": assessment.stop_loss or best_signal.get("stop_loss"),
        "take_profit": assessment.take_profit or best_signal.get("take_profit"),
        "action": "trade" if assessment.approved else "skip",
        "messages": state.get("messages", []) + [
            f"[{datetime.utcnow().isoformat()}] Risk check: "
            f"{'APPROVED' if assessment.approved else 'REJECTED'} — {assessment.reason}"
        ],
    }


def check_psychology(state: AgentState, **deps: Any) -> dict[str, Any]:
    """Node: Run psychology guard checks."""
    best_signal = state.get("best_signal")
    if not best_signal or not state.get("risk_approved", False):
        return {"psych_approved": False, "action": "skip"}

    psych_guard: PsychologyGuard = deps.get("psychology_guard")
    if not psych_guard:
        return {"psych_approved": True}

    market = state["market"]
    sentiment = state.get("sentiment_scores", {}).get(best_signal["asset"])

    check = psych_guard.check(
        market=market,
        asset=best_signal["asset"],
        direction=best_signal["direction"],
        confidence=best_signal["confidence"],
        sentiment_score=sentiment,
    )

    return {
        "psych_approved": check.approved,
        "psych_reason": check.reason,
        "action": "trade" if check.approved else "skip",
        "messages": state.get("messages", []) + [
            f"[{datetime.utcnow().isoformat()}] Psychology check: "
            f"{'APPROVED' if check.approved else 'REJECTED'} — {check.reason}"
        ],
    }


def execute_trade(state: AgentState, **deps: Any) -> dict[str, Any]:
    """Node: Execute the trade via broker."""
    best_signal = state.get("best_signal")
    if not best_signal:
        return {"action": "skip"}

    if not state.get("risk_approved") or not state.get("psych_approved"):
        return {"action": "skip"}

    broker = deps.get("broker")
    db = deps.get("db")

    # Record the signal in DB
    trade_id = None
    if db:
        from db import queries as q
        signal_id = q.insert_signal(
            db,
            market=state["market"],
            asset=best_signal["asset"],
            direction=best_signal["direction"],
            confidence=best_signal["confidence"],
            timeframe=best_signal.get("timeframe", "15m"),
            strategy=best_signal["strategy"],
            sentiment_score=state.get("sentiment_scores", {}).get(best_signal["asset"]),
            reasoning=best_signal.get("reasoning", ""),
        )
        q.mark_signal_acted(db, signal_id)

    order_result = None
    if broker:
        import asyncio
        from brokers.base import OrderSide

        side = OrderSide.BUY if best_signal["direction"] == "long" else OrderSide.SELL
        try:
            order = asyncio.get_event_loop().run_until_complete(
                broker.place_order(
                    asset=best_signal["asset"],
                    side=side,
                    quantity=state.get("position_size", 1),
                    limit_price=best_signal.get("metadata", {}).get("entry_price"),
                )
            )
            order_result = {
                "order_id": order.order_id,
                "status": order.status.value,
                "filled_qty": order.filled_quantity,
                "filled_price": order.filled_avg_price,
            }

            # Record trade in DB
            if db and order.filled_quantity > 0:
                trade_id = q.insert_trade(
                    db,
                    market=state["market"],
                    asset=best_signal["asset"],
                    direction=best_signal["direction"],
                    entry_price=order.filled_avg_price,
                    quantity=order.filled_quantity,
                    entry_time=datetime.utcnow().isoformat(),
                    strategy=best_signal["strategy"],
                    stop_loss=state.get("stop_loss"),
                    take_profit=state.get("take_profit"),
                    signal_id=signal_id,
                    confidence=best_signal["confidence"],
                    order_id=order.order_id,
                )
        except Exception as e:
            logger.error("agent.execute_error", error=str(e))
            return {
                "action": "error",
                "error": str(e),
                "messages": state.get("messages", []) + [
                    f"[{datetime.utcnow().isoformat()}] Execution error: {e}"
                ],
            }

    return {
        "order_result": order_result,
        "trade_id": trade_id,
        "action": "trade",
        "messages": state.get("messages", []) + [
            f"[{datetime.utcnow().isoformat()}] Executed {best_signal['direction']} "
            f"{best_signal['asset']} — qty={state.get('position_size', 0)}"
        ],
    }


def should_trade(state: AgentState) -> str:
    """Edge: Decide whether to proceed to execution or skip."""
    action = state.get("action", "skip")
    if action == "trade":
        return "check_risk"
    return "end"


def should_execute(state: AgentState) -> str:
    """Edge: After risk+psychology, decide whether to execute."""
    if state.get("risk_approved") and state.get("psych_approved"):
        return "execute"
    return "end"
