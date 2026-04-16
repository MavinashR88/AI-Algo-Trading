"""
Sentiment-based strategy.

Uses news sentiment scores to generate signals.
Long when sentiment is strongly positive, short when strongly negative.
Combines with basic price trend confirmation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy.base import BaseStrategy, Signal
from strategy import indicators as ind


class SentimentStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "sentiment"

    def analyze(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> Signal:
        min_long = self.config.get("min_score_for_long", 0.3)
        max_short = self.config.get("max_score_for_short", -0.3)

        if sentiment_score is None:
            return self._hold_signal(asset, "No sentiment data available")

        if not self._validate_df(df, min_rows=20):
            return self._hold_signal(asset)

        close = df["Close"]
        current_price = close.iloc[-1]
        atr_values = ind.atr(df["High"], df["Low"], close, 14)
        current_atr = atr_values.iloc[-1]

        # Basic trend confirmation via 20-period EMA
        ema_20 = ind.ema(close, 20)
        current_ema = ema_20.iloc[-1]
        above_ema = current_price > current_ema if not pd.isna(current_ema) else True

        # RSI for overbought/oversold filter
        rsi_values = ind.rsi(close, 14)
        current_rsi = rsi_values.iloc[-1] if not pd.isna(rsi_values.iloc[-1]) else 50

        if sentiment_score >= min_long and above_ema and current_rsi < 75:
            # Bullish sentiment + uptrend
            confidence = min(0.4 + sentiment_score * 0.4 + (0.1 if above_ema else 0), 0.90)
            stop = current_price - 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_price + 3 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="long",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Bullish sentiment ({sentiment_score:.2f}), price above EMA20, RSI={current_rsi:.0f}.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"sentiment": sentiment_score, "rsi": current_rsi},
            )

        if sentiment_score <= max_short and not above_ema and current_rsi > 25:
            confidence = min(0.4 + abs(sentiment_score) * 0.4 + (0.1 if not above_ema else 0), 0.90)
            stop = current_price + 2 * current_atr if not pd.isna(current_atr) else None
            tp = current_price - 3 * current_atr if not pd.isna(current_atr) else None

            return Signal(
                asset=asset,
                direction="short",
                confidence=round(confidence, 3),
                strategy=self.name,
                reasoning=f"Bearish sentiment ({sentiment_score:.2f}), price below EMA20, RSI={current_rsi:.0f}.",
                stop_loss=round(stop, 2) if stop else None,
                take_profit=round(tp, 2) if tp else None,
                metadata={"sentiment": sentiment_score, "rsi": current_rsi},
            )

        return self._hold_signal(
            asset,
            f"Sentiment={sentiment_score:.2f}, no clear signal with trend confirmation.",
        )
