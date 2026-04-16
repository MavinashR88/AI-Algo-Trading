"""
Momentum strategy.

Uses MACD crossover and RSI to detect momentum shifts.
Long when MACD crosses above signal with RSI not overbought.
Short when MACD crosses below signal with RSI not oversold.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy.base import BaseStrategy, Signal
from strategy import indicators as ind


class MomentumStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "momentum"

    def analyze(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> Signal:
        fast = self.config.get("fast_period", 12)
        slow = self.config.get("slow_period", 26)
        signal_period = self.config.get("signal_period", 9)
        lookback = self.config.get("lookback_days", 20)

        if not self._validate_df(df, min_rows=slow + signal_period):
            return self._hold_signal(asset)

        close = df["Close"]
        macd_line, signal_line, histogram = ind.macd(close, fast, slow, signal_period)
        rsi_values = ind.rsi(close, 14)
        atr_values = ind.atr(df["High"], df["Low"], close, 14)

        current_macd = macd_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        current_signal = signal_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        current_rsi = rsi_values.iloc[-1]
        current_atr = atr_values.iloc[-1]
        current_price = close.iloc[-1]

        # Check for NaN
        if any(pd.isna(v) for v in [current_macd, prev_macd, current_signal, prev_signal, current_rsi]):
            return self._hold_signal(asset, "Indicator values not ready")

        # MACD bullish crossover
        bullish_cross = prev_macd <= prev_signal and current_macd > current_signal
        # MACD bearish crossover
        bearish_cross = prev_macd >= prev_signal and current_macd < current_signal

        # ROC for momentum confirmation
        roc = ind.price_rate_of_change(close, lookback)
        current_roc = roc.iloc[-1] if not pd.isna(roc.iloc[-1]) else 0

        if bullish_cross and current_rsi < 70:
            confidence = min(0.5 + abs(current_roc) / 100 + (70 - current_rsi) / 200, 0.95)
            stop = current_price - 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_price + 4 * current_atr if not pd.isna(current_atr) else None
            return Signal(
                asset=asset,
                direction="long",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"MACD bullish crossover, RSI={current_rsi:.0f}, ROC={current_roc:.1f}%",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"rsi": current_rsi, "macd_hist": float(histogram.iloc[-1])},
            )

        if bearish_cross and current_rsi > 30:
            confidence = min(0.5 + abs(current_roc) / 100 + (current_rsi - 30) / 200, 0.95)
            stop = current_price + 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_price - 4 * current_atr if not pd.isna(current_atr) else None
            return Signal(
                asset=asset,
                direction="short",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"MACD bearish crossover, RSI={current_rsi:.0f}, ROC={current_roc:.1f}%",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"rsi": current_rsi, "macd_hist": float(histogram.iloc[-1])},
            )

        return self._hold_signal(asset, f"No momentum signal. RSI={current_rsi:.0f}")
