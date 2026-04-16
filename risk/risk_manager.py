"""
Risk management engine.

Enforces all risk rules before a trade is executed:
- Max risk per trade (1% of capital)
- Max open positions per market
- Max daily trades
- Daily/weekly drawdown limits
- ATR-based position sizing
- Stop loss and take profit calculation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import structlog

from db.database import Database
from db import queries

logger = structlog.get_logger(__name__)


@dataclass
class RiskAssessment:
    """Result of a risk check."""
    approved: bool
    reason: str = ""
    position_size: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_amount: float = 0.0
    risk_reward_ratio: float = 0.0


class RiskManager:
    """Enforces risk management rules."""

    def __init__(self, config: dict[str, Any], db: Database | None = None):
        self.config = config
        self.db = db

        # Risk parameters from config
        self.max_risk_per_trade_pct = config.get("max_risk_per_trade_pct", 0.01)
        self.max_open_positions = config.get("max_open_positions_per_market", 3)
        self.max_daily_trades = config.get("max_trades_per_market_per_day", 10)
        self.max_daily_drawdown_pct = config.get("max_daily_drawdown_pct", 0.03)
        self.max_weekly_drawdown_pct = config.get("max_weekly_drawdown_pct", 0.08)
        self.sl_atr_mult = config.get("stop_loss_atr_multiplier", 2.0)
        self.tp_rr_ratio = config.get("take_profit_rr_ratio", 2.0)
        self.trailing_activation_rr = config.get("trailing_stop_activation_rr", 1.5)
        self.trailing_atr_mult = config.get("trailing_stop_atr_multiplier", 1.5)

    def assess_trade(
        self,
        market: str,
        asset: str,
        direction: str,
        entry_price: float,
        current_atr: float,
        capital: float,
        signal_stop_loss: float | None = None,
        signal_take_profit: float | None = None,
    ) -> RiskAssessment:
        """Run all risk checks and return assessment.

        Args:
            market: "us" or "india".
            asset: Ticker symbol.
            direction: "long" or "short".
            entry_price: Expected entry price.
            current_atr: Current ATR value for the asset.
            capital: Current available capital.
            signal_stop_loss: Strategy-suggested stop loss.
            signal_take_profit: Strategy-suggested take profit.

        Returns:
            RiskAssessment with approval/rejection and sizing details.
        """
        # 1. Check if market is halted
        if self.db:
            state = queries.get_bot_state(self.db, market)
            if state and state["halted"]:
                return RiskAssessment(
                    approved=False,
                    reason=f"Market {market} is halted: {state.get('halt_reason', 'unknown')}",
                )

        # 2. Check max open positions
        if self.db:
            open_trades = queries.get_open_trades(self.db, market)
            if len(open_trades) >= self.max_open_positions:
                return RiskAssessment(
                    approved=False,
                    reason=f"Max open positions reached ({self.max_open_positions})",
                )

            # Check if already in this asset
            if any(t["asset"] == asset for t in open_trades):
                return RiskAssessment(
                    approved=False,
                    reason=f"Already have open position in {asset}",
                )

        # 3. Check max daily trades
        if self.db:
            daily_count = queries.count_trades_today(self.db, market)
            if daily_count >= self.max_daily_trades:
                return RiskAssessment(
                    approved=False,
                    reason=f"Max daily trades reached ({self.max_daily_trades})",
                )

        # 4. Check drawdown limits
        if self.db:
            state = queries.get_bot_state(self.db, market)
            if state:
                if state["current_drawdown_pct"] >= self.max_daily_drawdown_pct:
                    return RiskAssessment(
                        approved=False,
                        reason=f"Daily drawdown limit hit ({state['current_drawdown_pct']:.2%})",
                    )
                if state["weekly_drawdown_pct"] >= self.max_weekly_drawdown_pct:
                    return RiskAssessment(
                        approved=False,
                        reason=f"Weekly drawdown limit hit ({state['weekly_drawdown_pct']:.2%})",
                    )

        # 5. Calculate position sizing
        stop_loss = signal_stop_loss or self._calculate_stop_loss(
            entry_price, current_atr, direction,
        )
        take_profit = signal_take_profit or self._calculate_take_profit(
            entry_price, stop_loss, direction,
        )

        risk_per_share = abs(entry_price - stop_loss) if stop_loss else entry_price * 0.02
        risk_amount = capital * self.max_risk_per_trade_pct

        if risk_per_share > 0:
            position_size = math.floor(risk_amount / risk_per_share)
        else:
            position_size = 0

        # Cap at 20% of capital
        max_size = math.floor((capital * 0.20) / entry_price) if entry_price > 0 else 0
        position_size = min(position_size, max_size)

        if position_size <= 0:
            return RiskAssessment(
                approved=False,
                reason="Position size too small or capital insufficient",
            )

        # 6. Risk-reward ratio check
        reward = abs(take_profit - entry_price) if take_profit else risk_per_share * self.tp_rr_ratio
        rr_ratio = reward / risk_per_share if risk_per_share > 0 else 0

        if rr_ratio < 1.0:
            return RiskAssessment(
                approved=False,
                reason=f"Risk-reward ratio too low ({rr_ratio:.2f})",
            )

        return RiskAssessment(
            approved=True,
            position_size=position_size,
            stop_loss=round(stop_loss, 2) if stop_loss else None,
            take_profit=round(take_profit, 2) if take_profit else None,
            risk_amount=round(risk_amount, 2),
            risk_reward_ratio=round(rr_ratio, 2),
        )

    def _calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str,
    ) -> float:
        """Calculate ATR-based stop loss."""
        distance = atr * self.sl_atr_mult
        if direction == "long":
            return entry_price - distance
        else:
            return entry_price + distance

    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
    ) -> float:
        """Calculate take profit based on risk-reward ratio."""
        risk = abs(entry_price - stop_loss)
        reward = risk * self.tp_rr_ratio
        if direction == "long":
            return entry_price + reward
        else:
            return entry_price - reward

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_atr: float,
        direction: str,
    ) -> float | None:
        """Calculate trailing stop if activated.

        Trailing stop activates when profit reaches trailing_activation_rr * risk.
        """
        risk = abs(entry_price * self.max_risk_per_trade_pct)
        activation = risk * self.trailing_activation_rr

        if direction == "long":
            profit = current_price - entry_price
            if profit >= activation:
                return current_price - current_atr * self.trailing_atr_mult
        else:
            profit = entry_price - current_price
            if profit >= activation:
                return current_price + current_atr * self.trailing_atr_mult

        return None

    def update_drawdown(self, market: str, capital: float) -> None:
        """Update drawdown tracking in bot_state."""
        if self.db is None:
            return

        state = queries.get_bot_state(self.db, market)
        if not state:
            return

        peak = max(state["peak_capital"], capital)
        drawdown = (peak - capital) / peak if peak > 0 else 0

        queries.update_bot_state(
            self.db, market,
            capital=capital,
            peak_capital=peak,
            current_drawdown_pct=round(drawdown, 4),
        )

        if drawdown >= self.max_daily_drawdown_pct:
            queries.update_bot_state(
                self.db, market,
                halted=1,
                halt_reason=f"Daily drawdown limit: {drawdown:.2%}",
            )
            logger.warning("risk.daily_drawdown_halt", market=market, drawdown=drawdown)
