"""
Security hardening utilities.

- API key validation and sanitization
- Rate limiting for external API calls
- Input validation for trading parameters
- Audit logging for security events
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

VALID_MARKETS = {"us", "india"}
VALID_DIRECTIONS = {"long", "short"}
VALID_ORDER_TYPES = {"market", "limit", "stop", "stop_limit"}
VALID_MODES = {"paper", "live"}

# Ticker symbol pattern: 1-20 uppercase letters/digits, optional .NS suffix
TICKER_PATTERN = re.compile(r"^[A-Z0-9]{1,20}(\.[A-Z]{1,5})?$")


def validate_market(market: str) -> bool:
    """Validate market identifier."""
    return market in VALID_MARKETS


def validate_direction(direction: str) -> bool:
    """Validate trade direction."""
    return direction in VALID_DIRECTIONS


def validate_ticker(ticker: str) -> bool:
    """Validate ticker symbol format."""
    return bool(TICKER_PATTERN.match(ticker))


def validate_quantity(quantity: float) -> bool:
    """Validate trade quantity."""
    return isinstance(quantity, (int, float)) and quantity > 0 and quantity < 1_000_000


def validate_price(price: float) -> bool:
    """Validate price is reasonable."""
    return isinstance(price, (int, float)) and price > 0 and price < 10_000_000


def validate_trade_params(
    market: str,
    asset: str,
    direction: str,
    quantity: float,
    price: float,
) -> tuple[bool, str]:
    """Validate all trade parameters at once.

    Returns:
        (is_valid, error_message)
    """
    if not validate_market(market):
        return False, f"Invalid market: {market}"
    if not validate_ticker(asset):
        return False, f"Invalid ticker: {asset}"
    if not validate_direction(direction):
        return False, f"Invalid direction: {direction}"
    if not validate_quantity(quantity):
        return False, f"Invalid quantity: {quantity}"
    if not validate_price(price):
        return False, f"Invalid price: {price}"
    return True, ""


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

@dataclass
class RateLimit:
    """Rate limit configuration."""
    max_calls: int
    window_seconds: float


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self):
        self._calls: dict[str, list[float]] = defaultdict(list)
        self._limits: dict[str, RateLimit] = {
            "newsapi": RateLimit(max_calls=95, window_seconds=86400),   # 95/day (buffer from 100)
            "anthropic": RateLimit(max_calls=50, window_seconds=60),     # 50/min
            "yfinance": RateLimit(max_calls=100, window_seconds=60),     # 100/min
            "broker": RateLimit(max_calls=200, window_seconds=60),       # 200/min
        }

    def configure(self, service: str, max_calls: int, window_seconds: float) -> None:
        """Configure rate limit for a service."""
        self._limits[service] = RateLimit(max_calls=max_calls, window_seconds=window_seconds)

    def check(self, service: str) -> bool:
        """Check if a call is allowed under rate limits.

        Returns True if allowed, False if rate limited.
        """
        limit = self._limits.get(service)
        if not limit:
            return True

        now = time.monotonic()
        calls = self._calls[service]

        # Remove expired calls
        cutoff = now - limit.window_seconds
        self._calls[service] = [t for t in calls if t > cutoff]
        calls = self._calls[service]

        if len(calls) >= limit.max_calls:
            logger.warning(
                "security.rate_limited",
                service=service,
                calls=len(calls),
                limit=limit.max_calls,
            )
            return False

        return True

    def record(self, service: str) -> None:
        """Record a call to a service."""
        self._calls[service].append(time.monotonic())

    def call_if_allowed(self, service: str) -> bool:
        """Check and record in one step. Returns True if call was allowed."""
        if self.check(service):
            self.record(service)
            return True
        return False

    def remaining(self, service: str) -> int:
        """Get remaining calls for a service in current window."""
        limit = self._limits.get(service)
        if not limit:
            return 999

        now = time.monotonic()
        cutoff = now - limit.window_seconds
        active_calls = sum(1 for t in self._calls.get(service, []) if t > cutoff)
        return max(0, limit.max_calls - active_calls)


# ---------------------------------------------------------------------------
# API key sanitization
# ---------------------------------------------------------------------------

def sanitize_key(key: str) -> str:
    """Sanitize an API key for logging (show only last 4 chars)."""
    if not key or len(key) < 8:
        return "***"
    return f"***{key[-4:]}"


def hash_key(key: str) -> str:
    """Hash an API key for comparison/logging without exposing it."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def validate_api_key_format(key: str, service: str) -> bool:
    """Validate API key format for known services."""
    if not key:
        return False

    patterns = {
        "alpaca": r"^[A-Z0-9]{16,24}$",
        "anthropic": r"^sk-ant-",
        "newsapi": r"^[a-f0-9]{32}$",
        "twilio_sid": r"^AC[a-f0-9]{32}$",
    }

    pattern = patterns.get(service)
    if pattern:
        return bool(re.match(pattern, key))
    return len(key) >= 8  # Generic minimum length


# ---------------------------------------------------------------------------
# Singleton rate limiter
# ---------------------------------------------------------------------------

_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
