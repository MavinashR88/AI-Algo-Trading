"""
Mean reversion strategy.

Uses Bollinger Bands and RSI to identify overbought/oversold conditions.
Long when price touches lower band with RSI oversold.
Short when price touches upper band with RSI overbought.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy.base import BaseStrategy, Signal
from strategy import indicators as ind


class MeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "mean_reversion"

    def analyze(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> Signal:
        bb_period = self.config.get("bollinger_period", 20)
        bb_std = self.config.get("bollinger_std", 2.0)
        rsi_period = self.config.get("rsi_period", 14)
        rsi_oversold = self.config.get("rsi_oversold", 30)
        rsi_overbought = self.config.get("rsi_overbought", 70)

        min_rows = max(bb_period, rsi_period) + 5
        if not self._validate_df(df, min_rows=min_rows):
            return self._hold_signal(asset)

        close = df["Close"]
        upper, middle, lower = ind.bollinger_bands(close, bb_period, bb_std)
        rsi_values = ind.rsi(close, rsi_period)
        atr_values = ind.atr(df["High"], df["Low"], close, 14)

        current_price = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_middle = middle.iloc[-1]
        current_rsi = rsi_values.iloc[-1]
        current_atr = atr_values.iloc[-1]

        if any(pd.isna(v) for v in [current_upper, current_lower, current_rsi]):
            return self._hold_signal(asset, "Indicator values not ready")

        # Price at/below lower band + RSI oversold → long
        if current_price <= current_lower and current_rsi <= rsi_oversold:
            # Confidence based on how far below band and how oversold
            bb_pct = (current_lower - current_price) / (current_upper - current_lower) if (current_upper - current_lower) > 0 else 0
            rsi_pct = (rsi_oversold - current_rsi) / rsi_oversold
            confidence = min(0.5 + bb_pct * 0.3 + rsi_pct * 0.2, 0.95)

            stop = current_price - 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_middle  # Target: middle band

            return Signal(
                asset=asset,
                direction="long",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Price at lower BB, RSI={current_rsi:.0f} (oversold). Target: middle band.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"rsi": current_rsi, "bb_position": "lower"},
            )

        # Price at/above upper band + RSI overbought → short
        if current_price >= current_upper and current_rsi >= rsi_overbought:
            bb_pct = (current_price - current_upper) / (current_upper - current_lower) if (current_upper - current_lower) > 0 else 0
            rsi_pct = (current_rsi - rsi_overbought) / (100 - rsi_overbought) if rsi_overbought < 100 else 0
            confidence = min(0.5 + bb_pct * 0.3 + rsi_pct * 0.2, 0.95)

            stop = current_price + 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_middle

            return Signal(
                asset=asset,
                direction="short",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Price at upper BB, RSI={current_rsi:.0f} (overbought). Target: middle band.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"rsi": current_rsi, "bb_position": "upper"},
            )

        return self._hold_signal(
            asset,
            f"No mean reversion signal. RSI={current_rsi:.0f}, price within bands.",
        )
