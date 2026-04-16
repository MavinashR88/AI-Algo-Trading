"""
RSI Divergence strategy.

Detects bullish/bearish divergences between price and RSI:
- Bullish divergence: price makes lower low, RSI makes higher low → long
- Bearish divergence: price makes higher high, RSI makes lower high → short
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from strategy.base import BaseStrategy, Signal
from strategy import indicators as ind


class RSIDivergenceStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "rsi_divergence"

    def analyze(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> Signal:
        rsi_period = self.config.get("rsi_period", 14)
        lookback = self.config.get("divergence_lookback", 10)
        min_strength = self.config.get("min_divergence_strength", 0.3)

        min_rows = rsi_period + lookback + 5
        if not self._validate_df(df, min_rows=min_rows):
            return self._hold_signal(asset)

        close = df["Close"]
        rsi_values = ind.rsi(close, rsi_period)
        atr_values = ind.atr(df["High"], df["Low"], close, 14)

        if rsi_values.isna().sum() > len(rsi_values) * 0.3:
            return self._hold_signal(asset, "RSI not ready")

        current_atr = atr_values.iloc[-1]
        current_price = close.iloc[-1]

        # Find recent swing lows and highs in price and RSI
        recent_close = close.iloc[-lookback:].values
        recent_rsi = rsi_values.iloc[-lookback:].values

        # Remove NaN
        valid_mask = ~np.isnan(recent_rsi)
        if valid_mask.sum() < 4:
            return self._hold_signal(asset, "Not enough valid RSI values")

        recent_close = recent_close[valid_mask]
        recent_rsi = recent_rsi[valid_mask]

        # Bullish divergence: price trending down, RSI trending up
        price_slope = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
        rsi_slope = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]

        # Normalize slopes
        price_slope_norm = price_slope / recent_close.mean() * 100 if recent_close.mean() != 0 else 0
        rsi_slope_norm = rsi_slope / 50  # normalize to RSI midpoint

        divergence_strength = abs(price_slope_norm - rsi_slope_norm)

        if price_slope_norm < -min_strength and rsi_slope_norm > min_strength:
            # Bullish divergence
            confidence = min(0.5 + divergence_strength * 0.15, 0.90)
            stop = current_price - 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_price + 3 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="long",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Bullish RSI divergence. Price slope: {price_slope_norm:.2f}, RSI slope: {rsi_slope_norm:.2f}.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"divergence_strength": divergence_strength, "rsi": float(recent_rsi[-1])},
            )

        if price_slope_norm > min_strength and rsi_slope_norm < -min_strength:
            # Bearish divergence
            confidence = min(0.5 + divergence_strength * 0.15, 0.90)
            stop = current_price + 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_price - 3 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="short",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Bearish RSI divergence. Price slope: {price_slope_norm:.2f}, RSI slope: {rsi_slope_norm:.2f}.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"divergence_strength": divergence_strength, "rsi": float(recent_rsi[-1])},
            )

        return self._hold_signal(asset, f"No RSI divergence detected.")
