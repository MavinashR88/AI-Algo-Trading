"""
Abstract broker interface.

All broker adapters inherit from BaseBroker and implement the same methods.
This allows the trading agent to be broker-agnostic — it calls the same
interface whether trading US (Alpaca) or India (Zerodha).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Normalized order representation across all brokers."""
    order_id: str
    asset: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None  # limit price, None for market
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_avg_price: float = 0.0
    broker_data: dict[str, Any] | None = None  # raw broker response


@dataclass
class Position:
    """Normalized position representation."""
    asset: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str  # 'long' or 'short'


@dataclass
class AccountInfo:
    """Normalized account info."""
    buying_power: float
    cash: float
    portfolio_value: float
    currency: str


class BaseBroker(ABC):
    """Abstract interface that all broker adapters must implement."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the broker."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly disconnect from the broker."""
        ...

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account balance and buying power."""
        ...

    @abstractmethod
    async def place_order(
        self,
        asset: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        """Place an order. Returns the Order with its broker-assigned ID."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancellation succeeded."""
        ...

    @abstractmethod
    async def get_order(self, order_id: str) -> Order:
        """Get the current status of an order."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    @abstractmethod
    async def get_position(self, asset: str) -> Position | None:
        """Get position for a specific asset, or None if no position."""
        ...

    @abstractmethod
    async def close_position(self, asset: str) -> Order:
        """Close entire position for an asset via market order."""
        ...

    @abstractmethod
    async def get_quote(self, asset: str) -> dict[str, float]:
        """Get current bid/ask/last for an asset."""
        ...

    @abstractmethod
    async def is_market_open(self) -> bool:
        """Check if the market is currently open for trading."""
        ...

    @property
    @abstractmethod
    def market_name(self) -> str:
        """Return the market identifier ('us' or 'india')."""
        ...

    @property
    @abstractmethod
    def currency(self) -> str:
        """Return the currency code ('USD' or 'INR')."""
        ...
