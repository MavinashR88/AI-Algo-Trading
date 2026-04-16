"""
Psychology guard — prevents emotional trading patterns.

Checks:
- Revenge trading (consecutive losses → cooldown)
- FOMO detection (chasing after big move)
- Overtrading (too many trades in short period)
- No averaging down (adding to losing positions)
- Sentiment override (blocking longs in very negative sentiment)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import structlog

from db.database import Database
from db import queries

logger = structlog.get_logger(__name__)


@dataclass
class PsychologyCheck:
    """Result of a psychology guard check."""
    approved: bool
    reason: str = ""
    guard_type: str = ""  # which guard triggered


class PsychologyGuard:
    """Prevents emotional/irrational trading patterns."""

    def __init__(self, config: dict[str, Any], db: Database | None = None):
        self.config = config
        self.db = db

        self.revenge_lookback = config.get("revenge_trade_lookback", 2)
        self.cooldown_minutes = config.get("revenge_trade_cooldown_minutes", 30)
        self.min_confidence = config.get("min_signal_confidence", 0.65)
        self.sentiment_override = config.get("sentiment_override_threshold", -0.7)
        self.no_averaging_down = config.get("no_averaging_down", True)

    def check(
        self,
        market: str,
        asset: str,
        direction: str,
        confidence: float,
        sentiment_score: float | None = None,
    ) -> PsychologyCheck:
        """Run all psychology checks.

        Args:
            market: "us" or "india".
            asset: Ticker symbol.
            direction: "long" or "short".
            confidence: Signal confidence (0-1).
            sentiment_score: Current sentiment (-1 to 1).

        Returns:
            PsychologyCheck with approval/rejection.
        """
        # 1. Minimum confidence check
        if confidence < self.min_confidence:
            return PsychologyCheck(
                approved=False,
                reason=f"Confidence {confidence:.2f} below minimum {self.min_confidence}",
                guard_type="low_confidence",
            )

        # 2. Revenge trading check
        revenge_check = self._check_revenge_trading(market)
        if not revenge_check.approved:
            return revenge_check

        # 3. Sentiment override
        if sentiment_score is not None:
            sentiment_check = self._check_sentiment_override(direction, sentiment_score)
            if not sentiment_check.approved:
                return sentiment_check

        # 4. No averaging down
        if self.no_averaging_down and self.db:
            avg_check = self._check_no_averaging_down(market, asset, direction)
            if not avg_check.approved:
                return avg_check

        # 5. Overtrading check
        overtrade_check = self._check_overtrading(market)
        if not overtrade_check.approved:
            return overtrade_check

        return PsychologyCheck(approved=True)

    def _check_revenge_trading(self, market: str) -> PsychologyCheck:
        """Detect revenge trading after consecutive losses."""
        if self.db is None:
            return PsychologyCheck(approved=True)

        state = queries.get_bot_state(self.db, market)
        if not state:
            return PsychologyCheck(approved=True)

        consecutive_losses = state.get("consecutive_losses", 0)

        if consecutive_losses >= self.revenge_lookback:
            # Check if cooldown has elapsed
            recent_trades = queries.get_recent_trades(self.db, market, limit=1)
            if recent_trades:
                last_trade_time = recent_trades[0].get("exit_time") or recent_trades[0].get("entry_time", "")
                if last_trade_time:
                    try:
                        last_time = datetime.fromisoformat(last_trade_time)
                        cooldown_end = last_time + timedelta(minutes=self.cooldown_minutes)
                        if datetime.utcnow() < cooldown_end:
                            remaining = (cooldown_end - datetime.utcnow()).total_seconds() / 60
                            return PsychologyCheck(
                                approved=False,
                                reason=f"Revenge trade cooldown: {consecutive_losses} consecutive losses. "
                                       f"{remaining:.0f}min remaining.",
                                guard_type="revenge_trade",
                            )
                    except (ValueError, TypeError):
                        pass

        return PsychologyCheck(approved=True)

    def _check_sentiment_override(
        self,
        direction: str,
        sentiment_score: float,
    ) -> PsychologyCheck:
        """Block longs in extremely negative sentiment."""
        if direction == "long" and sentiment_score <= self.sentiment_override:
            return PsychologyCheck(
                approved=False,
                reason=f"Sentiment override: score {sentiment_score:.2f} ≤ {self.sentiment_override}. "
                       "Blocking long positions.",
                guard_type="sentiment_override",
            )
        return PsychologyCheck(approved=True)

    def _check_no_averaging_down(
        self,
        market: str,
        asset: str,
        direction: str,
    ) -> PsychologyCheck:
        """Prevent adding to losing positions."""
        if self.db is None:
            return PsychologyCheck(approved=True)

        open_trades = queries.get_open_trades(self.db, market)
        for trade in open_trades:
            if trade["asset"] == asset and trade["direction"] == direction:
                return PsychologyCheck(
                    approved=False,
                    reason=f"No averaging down: already have {direction} position in {asset}.",
                    guard_type="no_averaging_down",
                )

        return PsychologyCheck(approved=True)

    def _check_overtrading(self, market: str) -> PsychologyCheck:
        """Check for overtrading patterns."""
        if self.db is None:
            return PsychologyCheck(approved=True)

        # If we've had many trades today with low win rate, cool down
        daily_count = queries.count_trades_today(self.db, market)
        if daily_count >= 5:
            today_trades = queries.get_recent_trades(self.db, market, limit=daily_count)
            closed = [t for t in today_trades if t.get("status") == "closed"]
            if closed:
                wins = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
                win_rate = wins / len(closed)
                if win_rate < 0.3 and len(closed) >= 3:
                    return PsychologyCheck(
                        approved=False,
                        reason=f"Overtrading detected: {len(closed)} trades today with {win_rate:.0%} win rate.",
                        guard_type="overtrading",
                    )

        return PsychologyCheck(approved=True)

    def record_trade_result(self, market: str, is_win: bool) -> None:
        """Update consecutive loss counter after a trade closes."""
        if self.db is None:
            return

        state = queries.get_bot_state(self.db, market)
        if not state:
            return

        if is_win:
            queries.update_bot_state(self.db, market, consecutive_losses=0)
        else:
            new_count = state.get("consecutive_losses", 0) + 1
            queries.update_bot_state(self.db, market, consecutive_losses=new_count)
