"""
Breakout strategy.

Detects price breakouts above resistance or below support with
volume confirmation. Requires N-period high/low break with
above-average volume.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy.base import BaseStrategy, Signal
from strategy import indicators as ind


class BreakoutStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "breakout"

    def analyze(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> Signal:
        lookback = self.config.get("lookback_period", 20)
        vol_threshold = self.config.get("volume_threshold", 1.5)
        confirm_candles = self.config.get("confirmation_candles", 2)

        min_rows = lookback + confirm_candles + 5
        if not self._validate_df(df, min_rows=min_rows):
            return self._hold_signal(asset)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Current values
        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]

        # Resistance and support from lookback (excluding last N candles)
        lookback_slice = df.iloc[-(lookback + confirm_candles):-confirm_candles]
        resistance = lookback_slice["High"].max()
        support = lookback_slice["Low"].min()

        # Volume analysis
        vol_ma = ind.volume_ma(volume, lookback)
        current_vol_ma = vol_ma.iloc[-1]
        vol_ratio = current_volume / current_vol_ma if current_vol_ma > 0 and not pd.isna(current_vol_ma) else 0

        # ATR for stop/target
        atr_values = ind.atr(high, low, close, 14)
        current_atr = atr_values.iloc[-1]

        # Check for breakout above resistance
        recent_closes = close.iloc[-confirm_candles:]
        all_above = all(c > resistance for c in recent_closes)
        all_below = all(c < support for c in recent_closes)

        if all_above and vol_ratio >= vol_threshold:
            breakout_pct = (current_price - resistance) / resistance
            confidence = min(0.5 + breakout_pct * 5 + (vol_ratio - 1) * 0.1, 0.95)

            stop = resistance - 0.5 * current_atr if not pd.isna(current_atr) else None
            tp = current_price + 3 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="long",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Breakout above {resistance:.2f} resistance, vol {vol_ratio:.1f}x avg.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"resistance": resistance, "vol_ratio": vol_ratio},
            )

        if all_below and vol_ratio >= vol_threshold:
            breakout_pct = (support - current_price) / support
            confidence = min(0.5 + breakout_pct * 5 + (vol_ratio - 1) * 0.1, 0.95)

            stop = support + 0.5 * current_atr if not pd.isna(current_atr) else None
            tp = current_price - 3 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="short",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Breakdown below {support:.2f} support, vol {vol_ratio:.1f}x avg.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"support": support, "vol_ratio": vol_ratio},
            )

        return self._hold_signal(
            asset,
            f"No breakout. Price between {support:.2f}-{resistance:.2f}, vol {vol_ratio:.1f}x.",
        )
