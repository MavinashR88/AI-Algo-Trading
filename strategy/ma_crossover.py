"""
Moving Average Crossover strategy.

Uses fast/slow EMA (or SMA) crossover with a trend filter MA.
Long when fast crosses above slow and price above trend MA.
Short when fast crosses below slow and price below trend MA.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy.base import BaseStrategy, Signal
from strategy import indicators as ind


class MACrossoverStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "ma_crossover"

    def analyze(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> Signal:
        fast_period = self.config.get("fast_ma", 9)
        slow_period = self.config.get("slow_ma", 21)
        signal_period = self.config.get("signal_ma", 50)
        ma_type = self.config.get("ma_type", "EMA").upper()

        min_rows = signal_period + 5
        if not self._validate_df(df, min_rows=min_rows):
            return self._hold_signal(asset)

        close = df["Close"]
        ma_func = ind.ema if ma_type == "EMA" else ind.sma

        fast_ma = ma_func(close, fast_period)
        slow_ma = ma_func(close, slow_period)
        trend_ma = ma_func(close, signal_period)
        atr_values = ind.atr(df["High"], df["Low"], close, 14)

        current_fast = fast_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        current_slow = slow_ma.iloc[-1]
        prev_slow = slow_ma.iloc[-2]
        current_trend = trend_ma.iloc[-1]
        current_price = close.iloc[-1]
        current_atr = atr_values.iloc[-1]

        if any(pd.isna(v) for v in [current_fast, prev_fast, current_slow, prev_slow, current_trend]):
            return self._hold_signal(asset, "MA values not ready")

        bullish_cross = prev_fast <= prev_slow and current_fast > current_slow
        bearish_cross = prev_fast >= prev_slow and current_fast < current_slow
        above_trend = current_price > current_trend
        below_trend = current_price < current_trend

        # Measure separation as confidence factor
        separation = abs(current_fast - current_slow) / current_price * 100

        if bullish_cross and above_trend:
            confidence = min(0.55 + separation * 0.1, 0.90)
            stop = current_price - 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_price + 4 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="long",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"{ma_type} {fast_period}/{slow_period} bullish cross, above {signal_period} trend.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"fast_ma": current_fast, "slow_ma": current_slow, "trend_ma": current_trend},
            )

        if bearish_cross and below_trend:
            confidence = min(0.55 + separation * 0.1, 0.90)
            stop = current_price + 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_price - 4 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="short",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"{ma_type} {fast_period}/{slow_period} bearish cross, below {signal_period} trend.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"fast_ma": current_fast, "slow_ma": current_slow, "trend_ma": current_trend},
            )

        return self._hold_signal(asset, f"No MA crossover signal.")
