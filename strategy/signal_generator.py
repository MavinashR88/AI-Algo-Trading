"""
Signal generator — orchestrates all strategies for a market.

Runs each active strategy against current data and returns the best
signal (or ensemble signal from the hybrid strategy).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from strategy.base import BaseStrategy, Signal
from strategy.momentum import MomentumStrategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.breakout import BreakoutStrategy
from strategy.ma_crossover import MACrossoverStrategy
from strategy.rsi_divergence import RSIDivergenceStrategy
from strategy.volume_price import VolumePriceStrategy
from strategy.sentiment_strategy import SentimentStrategy
from strategy.hybrid import HybridStrategy

logger = structlog.get_logger(__name__)

# Registry of all available strategies
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "ma_crossover": MACrossoverStrategy,
    "rsi_divergence": RSIDivergenceStrategy,
    "volume_price": VolumePriceStrategy,
    "sentiment": SentimentStrategy,
}


class SignalGenerator:
    """Generates trading signals using configured strategies."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.strategies: dict[str, BaseStrategy] = {}
        self.hybrid: HybridStrategy | None = None
        self._init_strategies()

    def _init_strategies(self) -> None:
        """Initialize active strategies from config."""
        active = self.config.get("active_strategies", list(STRATEGY_REGISTRY.keys()))
        weights = self.config.get("weights", {})

        for name in active:
            if name == "hybrid":
                continue
            if name in STRATEGY_REGISTRY:
                strategy_config = self.config.get(name, {})
                self.strategies[name] = STRATEGY_REGISTRY[name](config=strategy_config)
            else:
                logger.warning("signal_generator.unknown_strategy", name=name)

        # Initialize hybrid with all other strategies
        if "hybrid" in active or not active:
            self.hybrid = HybridStrategy(
                strategies=list(self.strategies.values()),
                weights=weights,
                config=self.config.get("hybrid", {}),
            )

    def generate(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> list[Signal]:
        """Generate signals from all active strategies.

        Returns all individual signals plus the hybrid ensemble signal.
        """
        signals: list[Signal] = []

        for name, strategy in self.strategies.items():
            try:
                signal = strategy.analyze(df, asset, market, sentiment_score)
                signals.append(signal)
                logger.debug(
                    "signal_generator.strategy_result",
                    strategy=name,
                    asset=asset,
                    direction=signal.direction,
                    confidence=signal.confidence,
                )
            except Exception as e:
                logger.error(
                    "signal_generator.strategy_error",
                    strategy=name,
                    asset=asset,
                    error=str(e),
                )

        # Run hybrid ensemble
        if self.hybrid:
            try:
                hybrid_signal = self.hybrid.analyze(df, asset, market, sentiment_score)
                signals.append(hybrid_signal)
            except Exception as e:
                logger.error("signal_generator.hybrid_error", asset=asset, error=str(e))

        return signals

    def get_best_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
        prefer_hybrid: bool = True,
    ) -> Signal | None:
        """Get the single best signal for an asset.

        If prefer_hybrid is True, returns the hybrid ensemble signal.
        Otherwise returns the individual signal with highest confidence.
        """
        signals = self.generate(df, asset, market, sentiment_score)

        if not signals:
            return None

        # Filter to actionable signals
        actionable = [s for s in signals if s.is_actionable]

        if not actionable:
            return None

        if prefer_hybrid:
            hybrid_signals = [s for s in actionable if s.strategy == "hybrid"]
            if hybrid_signals:
                return hybrid_signals[0]

        # Return highest confidence actionable signal
        return max(actionable, key=lambda s: s.confidence)

    def get_strategy(self, name: str) -> BaseStrategy | None:
        """Get a specific strategy by name."""
        if name == "hybrid":
            return self.hybrid
        return self.strategies.get(name)
