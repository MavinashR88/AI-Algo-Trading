"""
Backtesting engine.

Replays historical data through a strategy to measure performance.
Tracks trades, P&L, win rate, Sharpe ratio, max drawdown, and more.
Results are stored in the backtest_results table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from db.database import Database
from db import queries
from strategy.base import BaseStrategy, Signal

logger = structlog.get_logger(__name__)


@dataclass
class BacktestTrade:
    """Record of a simulated trade."""
    asset: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    net_pnl: float
    strategy: str


@dataclass
class BacktestResult:
    """Complete backtest performance report."""
    market: str
    strategy: str
    start_date: str
    end_date: str
    trades: list[BacktestTrade] = field(default_factory=list)

    # Computed metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    avg_trade_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float | None = None
    profit_factor: float | None = None
    max_consecutive_losses: int = 0
    avg_trade_duration: str = ""

    @property
    def is_profitable(self) -> bool:
        return self.net_pnl > 0

    @property
    def meets_graduation(self) -> bool:
        """Check if backtest meets minimum graduation criteria."""
        return (
            self.total_trades >= 100
            and self.win_rate >= 0.60
            and (self.sharpe_ratio or 0) >= 1.0
            and self.max_drawdown_pct <= 0.20
        )


class Backtester:
    """Backtesting engine for strategy evaluation."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_per_trade_pct: float = 0.01,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001,
        db: Database | None = None,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.db = db

    def run(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        warmup_bars: int = 50,
    ) -> BacktestResult:
        """Run a backtest on historical data.

        Args:
            strategy: Strategy to test.
            df: Full OHLCV DataFrame.
            asset: Ticker symbol.
            market: "us" or "india".
            warmup_bars: Number of initial bars to skip (indicator warmup).

        Returns:
            BacktestResult with all metrics.
        """
        if df.empty or len(df) < warmup_bars + 10:
            return BacktestResult(
                market=market,
                strategy=strategy.name,
                start_date="",
                end_date="",
            )

        capital = self.initial_capital
        peak_capital = capital
        trades: list[BacktestTrade] = []
        equity_curve: list[float] = []
        position: dict[str, Any] | None = None

        start_date = str(df.index[warmup_bars])
        end_date = str(df.index[-1])

        for i in range(warmup_bars, len(df)):
            # Slice data up to current bar
            current_df = df.iloc[:i + 1]
            current_bar = df.iloc[i]
            current_time = str(df.index[i])

            # Check if we should exit existing position
            if position is not None:
                exit_price = self._check_exit(position, current_bar)
                if exit_price is not None:
                    trade = self._close_position(position, exit_price, current_time)
                    trades.append(trade)
                    capital += trade.net_pnl
                    position = None

            # Generate signal if no position
            if position is None:
                try:
                    signal = strategy.analyze(current_df, asset, market)
                except Exception:
                    continue

                if signal.is_actionable and signal.confidence >= 0.5:
                    entry_price = current_bar["Close"]
                    # Apply slippage
                    if signal.direction == "long":
                        entry_price *= (1 + self.slippage_pct)
                    else:
                        entry_price *= (1 - self.slippage_pct)

                    # Position sizing based on risk
                    quantity = self._calculate_quantity(
                        capital, entry_price, signal.stop_loss, signal.direction,
                    )

                    if quantity > 0:
                        position = {
                            "asset": asset,
                            "direction": signal.direction,
                            "entry_price": entry_price,
                            "entry_time": current_time,
                            "quantity": quantity,
                            "stop_loss": signal.stop_loss,
                            "take_profit": signal.take_profit,
                            "strategy": signal.strategy,
                        }

            # Track equity
            equity = capital
            if position is not None:
                unrealized = self._unrealized_pnl(position, current_bar["Close"])
                equity += unrealized
            equity_curve.append(equity)
            peak_capital = max(peak_capital, equity)

        # Close any remaining position at last price
        if position is not None:
            exit_price = df.iloc[-1]["Close"]
            trade = self._close_position(position, exit_price, end_date)
            trades.append(trade)
            capital += trade.net_pnl

        # Calculate metrics
        result = self._compute_metrics(trades, equity_curve, market, strategy.name, start_date, end_date)

        # Store in DB
        if self.db is not None and result.total_trades > 0:
            self._store_result(result)

        return result

    def _calculate_quantity(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float | None,
        direction: str,
    ) -> float:
        """Calculate position size based on risk management."""
        risk_amount = capital * self.risk_per_trade_pct

        if stop_loss and entry_price > 0:
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                quantity = risk_amount / risk_per_share
            else:
                quantity = risk_amount / (entry_price * 0.02)  # Default 2% stop
        else:
            quantity = risk_amount / (entry_price * 0.02)

        # Don't use more than 20% of capital on a single trade
        max_quantity = (capital * 0.20) / entry_price
        quantity = min(quantity, max_quantity)

        return max(0, math.floor(quantity))

    def _check_exit(self, position: dict, bar: pd.Series) -> float | None:
        """Check if stop loss or take profit is hit."""
        high = bar["High"]
        low = bar["Low"]
        sl = position.get("stop_loss")
        tp = position.get("take_profit")

        if position["direction"] == "long":
            if sl and low <= sl:
                return sl  # Stop loss hit
            if tp and high >= tp:
                return tp  # Take profit hit
        else:
            if sl and high >= sl:
                return sl
            if tp and low <= tp:
                return tp

        return None

    def _close_position(
        self,
        position: dict,
        exit_price: float,
        exit_time: str,
    ) -> BacktestTrade:
        """Close a position and calculate P&L."""
        entry_price = position["entry_price"]
        quantity = position["quantity"]

        # Apply slippage on exit
        if position["direction"] == "long":
            exit_price *= (1 - self.slippage_pct)
        else:
            exit_price *= (1 + self.slippage_pct)

        if position["direction"] == "long":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        pnl_pct = pnl / (entry_price * quantity) if entry_price * quantity > 0 else 0
        fees = (entry_price + exit_price) * quantity * self.commission_pct
        net_pnl = pnl - fees

        return BacktestTrade(
            asset=position["asset"],
            direction=position["direction"],
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            entry_time=position["entry_time"],
            exit_time=exit_time,
            quantity=quantity,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            fees=round(fees, 2),
            net_pnl=round(net_pnl, 2),
            strategy=position["strategy"],
        )

    def _unrealized_pnl(self, position: dict, current_price: float) -> float:
        """Calculate unrealized P&L for an open position."""
        if position["direction"] == "long":
            return (current_price - position["entry_price"]) * position["quantity"]
        else:
            return (position["entry_price"] - current_price) * position["quantity"]

    def _compute_metrics(
        self,
        trades: list[BacktestTrade],
        equity_curve: list[float],
        market: str,
        strategy_name: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Compute all performance metrics."""
        result = BacktestResult(
            market=market,
            strategy=strategy_name,
            start_date=start_date,
            end_date=end_date,
            trades=trades,
        )

        if not trades:
            return result

        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.net_pnl > 0)
        result.losing_trades = sum(1 for t in trades if t.net_pnl <= 0)
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        result.total_pnl = sum(t.pnl for t in trades)
        result.total_fees = sum(t.fees for t in trades)
        result.net_pnl = sum(t.net_pnl for t in trades)
        result.avg_trade_pnl = result.net_pnl / result.total_trades

        # Max consecutive losses
        streak = 0
        max_streak = 0
        for t in trades:
            if t.net_pnl <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        result.max_consecutive_losses = max_streak

        # Max drawdown from equity curve
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0.0
            for val in equity_curve:
                peak = max(peak, val)
                dd = (peak - val) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            result.max_drawdown_pct = round(max_dd, 4)

        # Sharpe ratio (annualized, assuming daily returns)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                result.sharpe_ratio = round(
                    float(returns.mean() / returns.std() * np.sqrt(252)), 2
                )

        # Profit factor
        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl <= 0))
        if gross_loss > 0:
            result.profit_factor = round(gross_profit / gross_loss, 2)

        return result

    def _store_result(self, result: BacktestResult) -> None:
        """Store backtest results in database."""
        if self.db is None:
            return
        try:
            queries.insert_backtest_result(
                self.db,
                market=result.market,
                strategy=result.strategy,
                start_date=result.start_date,
                end_date=result.end_date,
                total_trades=result.total_trades,
                winning_trades=result.winning_trades,
                losing_trades=result.losing_trades,
                win_rate=result.win_rate,
                sharpe_ratio=result.sharpe_ratio,
                max_drawdown_pct=result.max_drawdown_pct,
                profit_factor=result.profit_factor,
                avg_trade_duration=result.avg_trade_duration,
                max_consecutive_losses=result.max_consecutive_losses,
                total_pnl=result.total_pnl,
                total_fees=result.total_fees,
                net_pnl=result.net_pnl,
            )
            logger.info(
                "backtester.result_stored",
                strategy=result.strategy,
                trades=result.total_trades,
                win_rate=result.win_rate,
            )
        except Exception as e:
            logger.error("backtester.store_error", error=str(e))
