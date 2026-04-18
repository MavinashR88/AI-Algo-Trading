"""
Alpaca broker adapter for US markets (NYSE/NASDAQ).

Uses alpaca-py SDK for REST + WebSocket.
Supports both paper and live trading via base URL switch.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from brokers.base import (
    AccountInfo,
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = structlog.get_logger(__name__)


# Mapping from Alpaca SDK status strings to our normalized enum
_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "replaced": OrderStatus.SUBMITTED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.PENDING,
}


class AlpacaBroker(BaseBroker):
    """Alpaca adapter using alpaca-py SDK."""

    def __init__(self, paper: bool = True):
        self._paper = paper
        self._trading_client = None
        self._data_client = None
        self._connected = False

    async def connect(self) -> None:
        """Initialize Alpaca SDK clients."""
        # Import here to avoid import errors when alpaca-py not installed
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient

        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        self._trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=self._paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        self._connected = True
        logger.info("alpaca.connected", paper=self._paper)

    async def disconnect(self) -> None:
        self._trading_client = None
        self._data_client = None
        self._connected = False
        logger.info("alpaca.disconnected")

    async def get_account(self) -> AccountInfo:
        account = self._trading_client.get_account()
        return AccountInfo(
            buying_power=float(account.buying_power),
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
            currency="USD",
        )

    async def place_order(
        self,
        asset: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
            StopOrderRequest,
            StopLimitOrderRequest,
        )
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

        alpaca_side = AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL

        if order_type == OrderType.MARKET:
            request = MarketOrderRequest(
                symbol=asset,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )
        elif order_type == OrderType.LIMIT:
            request = LimitOrderRequest(
                symbol=asset,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
        elif order_type == OrderType.STOP:
            request = StopOrderRequest(
                symbol=asset,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                stop_price=stop_price,
            )
        elif order_type == OrderType.STOP_LIMIT:
            request = StopLimitOrderRequest(
                symbol=asset,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                stop_price=stop_price,
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        result = self._trading_client.submit_order(request)
        logger.info(
            "alpaca.order_placed",
            asset=asset, side=side.value, qty=quantity,
            order_type=order_type.value, order_id=str(result.id),
        )

        return self._normalize_order(result)

    async def cancel_order(self, order_id: str) -> bool:
        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info("alpaca.order_cancelled", order_id=order_id)
            return True
        except Exception as e:
            logger.error("alpaca.cancel_failed", order_id=order_id, error=str(e))
            return False

    async def get_order(self, order_id: str) -> Order:
        result = self._trading_client.get_order_by_id(order_id)
        return self._normalize_order(result)

    async def get_positions(self) -> list[Position]:
        positions = self._trading_client.get_all_positions()
        return [self._normalize_position(p) for p in positions]

    async def get_position(self, asset: str) -> Position | None:
        try:
            pos = self._trading_client.get_open_position(asset)
            return self._normalize_position(pos)
        except Exception:
            return None

    async def close_position(self, asset: str) -> Order:
        result = self._trading_client.close_position(asset)
        logger.info("alpaca.position_closed", asset=asset)
        return self._normalize_order(result)

    async def get_quote(self, asset: str) -> dict[str, float]:
        from alpaca.data.requests import StockLatestQuoteRequest
        request = StockLatestQuoteRequest(symbol_or_symbols=asset)
        quotes = self._data_client.get_stock_latest_quote(request)
        q = quotes[asset]
        return {
            "bid": float(q.bid_price),
            "ask": float(q.ask_price),
            "last": float((q.bid_price + q.ask_price) / 2),
        }

    async def is_market_open(self) -> bool:
        clock = self._trading_client.get_clock()
        return clock.is_open

    @property
    def market_name(self) -> str:
        return "us"

    @property
    def currency(self) -> str:
        return "USD"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_order(self, raw: Any) -> Order:
        status_str = str(raw.status.value) if hasattr(raw.status, 'value') else str(raw.status)
        return Order(
            order_id=str(raw.id),
            asset=raw.symbol,
            side=OrderSide.BUY if str(raw.side.value) == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET,  # simplified
            quantity=float(raw.qty) if raw.qty else 0.0,
            price=float(raw.limit_price) if raw.limit_price else None,
            status=_STATUS_MAP.get(status_str, OrderStatus.PENDING),
            filled_quantity=float(raw.filled_qty) if raw.filled_qty else 0.0,
            filled_avg_price=float(raw.filled_avg_price) if raw.filled_avg_price else 0.0,
        )

    def _normalize_position(self, raw: Any) -> Position:
        qty = float(raw.qty)
        return Position(
            asset=raw.symbol,
            quantity=abs(qty),
            avg_entry_price=float(raw.avg_entry_price),
            current_price=float(raw.current_price),
            market_value=float(raw.market_value),
            unrealized_pnl=float(raw.unrealized_pl),
            unrealized_pnl_pct=float(raw.unrealized_plpc),
            side="long" if qty > 0 else "short",
        )
