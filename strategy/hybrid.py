"""
Hybrid/Ensemble strategy.

Aggregates signals from all other strategies using weighted voting.
Only produces a signal when multiple strategies agree, reducing
false positives.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategy.base import BaseStrategy, Signal


class HybridStrategy(BaseStrategy):
    """Ensemble strategy that combines signals from multiple sub-strategies."""

    def __init__(
        self,
        strategies: list[BaseStrategy] | None = None,
        weights: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.strategies = strategies or []
        self.weights = weights or {}

    @property
    def name(self) -> str:
        return "hybrid"

    def analyze(
        self,
        df: pd.DataFrame,
        asset: str,
        market: str = "us",
        sentiment_score: float | None = None,
    ) -> Signal:
        if not self.strategies:
            return self._hold_signal(asset, "No sub-strategies configured")

        # Collect signals from all strategies
        signals: list[Signal] = []
        for strat in self.strategies:
            try:
                sig = strat.analyze(df, asset, market, sentiment_score)
                signals.append(sig)
            except Exception:
                continue

        if not signals:
            return self._hold_signal(asset, "All sub-strategies failed")

        return self._aggregate(signals, asset)

    def _aggregate(self, signals: list[Signal], asset: str) -> Signal:
        """Weighted vote across sub-strategy signals."""
        long_score = 0.0
        short_score = 0.0
        hold_count = 0
        total_weight = 0.0
        reasons = []

        for sig in signals:
            weight = self.weights.get(sig.strategy, 1.0 / len(signals))
            total_weight += weight

            if sig.direction == "long":
                long_score += weight * sig.confidence
                reasons.append(f"{sig.strategy}: LONG ({sig.confidence:.2f})")
            elif sig.direction == "short":
                short_score += weight * sig.confidence
                reasons.append(f"{sig.strategy}: SHORT ({sig.confidence:.2f})")
            else:
                hold_count += 1
                reasons.append(f"{sig.strategy}: HOLD")

        # Normalize
        if total_weight > 0:
            long_score /= total_weight
            short_score /= total_weight

        # Need minimum agreement threshold
        min_agreement = self.config.get("min_agreement", 0.3)

        # Find best stop/target from agreeing signals
        long_signals = [s for s in signals if s.direction == "long"]
        short_signals = [s for s in signals if s.direction == "short"]

        if long_score > short_score and long_score >= min_agreement:
            # Use the most confident signal's stop/target
            best = max(long_signals, key=lambda s: s.confidence) if long_signals else None
            return Signal(
                asset=asset,
                direction="long",
                confidence=round(long_score, 3),
                strategy=self.name,
                reasoning=f"Ensemble LONG ({long_score:.2f} vs short {short_score:.2f}). {'; '.join(reasons)}",
                stop_loss=best.stop_loss if best else None,
                take_profit=best.take_profit if best else None,
                metadata={
                    "long_score": long_score,
                    "short_score": short_score,
                    "agreeing_strategies": len(long_signals),
                    "total_strategies": len(signals),
                },
            )

        if short_score > long_score and short_score >= min_agreement:
            best = max(short_signals, key=lambda s: s.confidence) if short_signals else None
            return Signal(
                asset=asset,
                direction="short",
                confidence=round(short_score, 3),
                strategy=self.name,
                reasoning=f"Ensemble SHORT ({short_score:.2f} vs long {long_score:.2f}). {'; '.join(reasons)}",
                stop_loss=best.stop_loss if best else None,
                take_profit=best.take_profit if best else None,
                metadata={
                    "long_score": long_score,
                    "short_score": short_score,
                    "agreeing_strategies": len(short_signals),
                    "total_strategies": len(signals),
                },
            )

        return self._hold_signal(
            asset,
            f"No consensus. Long={long_score:.2f}, Short={short_score:.2f}, Hold={hold_count}.",
        )
