"""
Abstract strategy interface.

All strategies inherit from BaseStrategy. Each produces a Signal
with direction, confidence, and reasoning — the trading agent
decides whether to act on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Signal:
    """Trading signal produced by a strategy."""
    asset: str
    direction: str          # "long", "short", or "hold"
    confidence: float       # 0.0 to 1.0
    strategy: str           # strategy name that generated it
    timeframe: str = "15m"
    reasoning: str = ""
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Signal is actionable if direction is long or short with confidence > 0."""
        return self.direction in ("long", "short") and self.confidence > 0


class BaseStrategy(ABC):
    """Abstract strategy that all trading strategies must implement."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'momentum', 'breakout')."""
        ...

    @abstractmethod
    def analyze(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> Signal:
        """Analyze price data and produce a signal.

        Args:
            df: OHLCV DataFrame (must have Open, High, Low, Close, Volume columns).
            asset: Ticker symbol.
            market: "us" or "india".
            sentiment_score: Optional sentiment score [-1, 1] for sentiment-aware strategies.

        Returns:
            Signal with direction, confidence, and reasoning.
        """
        ...

    def _validate_df(self, df: pd.DataFrame, min_rows: int = 20) -> bool:
        """Check that DataFrame has enough data for analysis."""
        if df is None or df.empty or len(df) < min_rows:
            return False
        required = {"Open", "High", "Low", "Close", "Volume"}
        return required.issubset(set(df.columns))

    def _hold_signal(self, asset: str, reason: str = "Insufficient data") -> Signal:
        """Return a neutral hold signal."""
        return Signal(
            asset=asset,
            direction="hold",
            confidence=0.0,
            strategy=self.name,
            reasoning=reason,
        )
