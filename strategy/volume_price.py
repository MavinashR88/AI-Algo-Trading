"""
Volume-Price analysis strategy.

Uses VWAP and volume patterns to confirm price direction.
Long when price crosses above VWAP with increasing volume.
Short when price crosses below VWAP with increasing volume.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy.base import BaseStrategy, Signal
from strategy import indicators as ind


class VolumePriceStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "volume_price"

    def analyze(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> Signal:
        vwap_period = self.config.get("vwap_period", 20)
        vol_ma_period = self.config.get("volume_ma_period", 20)
        pv_threshold = self.config.get("price_volume_threshold", 1.3)

        min_rows = max(vwap_period, vol_ma_period) + 5
        if not self._validate_df(df, min_rows=min_rows):
            return self._hold_signal(asset)

        close = df["Close"]
        volume = df["Volume"]
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]

        # VWAP
        vwap_values = ind.vwap(df, vwap_period)
        current_vwap = vwap_values.iloc[-1]
        prev_vwap = vwap_values.iloc[-2] if len(vwap_values) > 1 else current_vwap

        # Volume analysis
        vol_avg = ind.volume_ma(volume, vol_ma_period)
        current_vol = volume.iloc[-1]
        current_vol_avg = vol_avg.iloc[-1]
        vol_ratio = current_vol / current_vol_avg if current_vol_avg > 0 and not pd.isna(current_vol_avg) else 0

        # OBV trend
        obv_values = ind.obv(close, volume)
        obv_slope = (obv_values.iloc[-1] - obv_values.iloc[-5]) if len(obv_values) > 5 else 0

        # ATR for stop/target
        atr_values = ind.atr(df["High"], df["Low"], close, 14)
        current_atr = atr_values.iloc[-1]

        if any(pd.isna(v) for v in [current_vwap, current_vol_avg]):
            return self._hold_signal(asset, "VWAP/Volume indicators not ready")

        # Bullish: price crosses above VWAP with strong volume
        crosses_above = prev_price <= prev_vwap and current_price > current_vwap
        crosses_below = prev_price >= prev_vwap and current_price < current_vwap

        if crosses_above and vol_ratio >= pv_threshold and obv_slope > 0:
            confidence = min(0.5 + (vol_ratio - 1) * 0.15, 0.90)
            stop = current_vwap - current_atr if not pd.isna(current_atr) else None
            tp = current_price + 3 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="long",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Price crossed above VWAP with {vol_ratio:.1f}x volume. OBV rising.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"vwap": current_vwap, "vol_ratio": vol_ratio},
            )

        if crosses_below and vol_ratio >= pv_threshold and obv_slope < 0:
            confidence = min(0.5 + (vol_ratio - 1) * 0.15, 0.90)
            stop = current_vwap + current_atr if not pd.isna(current_atr) else None
            tp = current_price - 3 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="short",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Price crossed below VWAP with {vol_ratio:.1f}x volume. OBV falling.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"vwap": current_vwap, "vol_ratio": vol_ratio},
            )

        return self._hold_signal(asset, f"No volume-price signal. Vol ratio: {vol_ratio:.1f}x.")
