"""
Zerodha Kite Connect broker adapter for Indian markets (NSE/BSE).

In paper mode: simulates orders locally (no Kite API calls).
In live mode: uses kiteconnect SDK for real order execution.

Paper simulation tracks positions, fills market orders at last price,
and maintains a virtual order book. This lets us paper trade India
without a Kite Connect subscription.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
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


class ZerodhaBroker(BaseBroker):
    """Zerodha Kite Connect adapter with built-in paper simulation."""

    def __init__(self, paper: bool = True):
        self._paper = paper
        self._kite = None
        self._connected = False

        # Paper trading state
        self._paper_capital: float = 1_000_000.0  # 10 lakh INR default
        self._paper_positions: dict[str, dict] = {}
        self._paper_orders: dict[str, Order] = {}

    async def connect(self) -> None:
        if self._paper:
            logger.info("zerodha.paper_mode", capital=self._paper_capital)
            self._connected = True
            return

        # Live mode — requires kiteconnect SDK
        from kiteconnect import KiteConnect

        api_key = os.environ.get("ZERODHA_API_KEY", "")
        access_token = os.environ.get("ZERODHA_ACCESS_TOKEN", "")

        if not api_key or not access_token:
            raise ValueError(
                "ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN must be set for live trading"
            )

        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)
        self._connected = True
        logger.info("zerodha.connected", mode="live")

    async def disconnect(self) -> None:
        self._kite = None
        self._connected = False
        logger.info("zerodha.disconnected")

    async def get_account(self) -> AccountInfo:
        if self._paper:
            # Calculate portfolio value from positions
            portfolio = self._paper_capital
            for pos in self._paper_positions.values():
                portfolio += pos["quantity"] * pos["current_price"]
            return AccountInfo(
                buying_power=self._paper_capital,
                cash=self._paper_capital,
                portfolio_value=portfolio,
                currency="INR",
            )

        margins = self._kite.margins(segment="equity")
        return AccountInfo(
            buying_power=float(margins["available"]["live_balance"]),
            cash=float(margins["available"]["cash"]),
            portfolio_value=float(margins["net"]),
            currency="INR",
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
        if self._paper:
            return await self._paper_place_order(
                asset, side, quantity, order_type, limit_price
            )

        # Live order via Kite
        kite_side = "BUY" if side == OrderSide.BUY else "SELL"
        kite_order_type = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "SL-M",
            OrderType.STOP_LIMIT: "SL",
        }[order_type]

        params: dict[str, Any] = {
            "tradingsymbol": asset,
            "exchange": "NSE",
            "transaction_type": kite_side,
            "quantity": int(quantity),
            "order_type": kite_order_type,
            "product": "MIS",  # intraday
            "validity": "DAY",
        }
        if limit_price is not None:
            params["price"] = limit_price
        if stop_price is not None:
            params["trigger_price"] = stop_price

        order_id = self._kite.place_order(variety="regular", **params)
        logger.info(
            "zerodha.order_placed",
            asset=asset, side=side.value, qty=quantity, order_id=order_id,
        )

        return Order(
            order_id=str(order_id),
            asset=asset,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=limit_price,
            status=OrderStatus.SUBMITTED,
        )

    async def cancel_order(self, order_id: str) -> bool:
        if self._paper:
            if order_id in self._paper_orders:
                self._paper_orders[order_id] = Order(
                    **{**self._paper_orders[order_id].__dict__,
                       "status": OrderStatus.CANCELLED}
                )
                return True
            return False

        try:
            self._kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except Exception as e:
            logger.error("zerodha.cancel_failed", order_id=order_id, error=str(e))
            return False

    async def get_order(self, order_id: str) -> Order:
        if self._paper:
            if order_id in self._paper_orders:
                return self._paper_orders[order_id]
            raise ValueError(f"Order {order_id} not found")

        history = self._kite.order_history(order_id)
        latest = history[-1]
        return Order(
            order_id=str(latest["order_id"]),
            asset=latest["tradingsymbol"],
            side=OrderSide.BUY if latest["transaction_type"] == "BUY" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=float(latest["quantity"]),
            price=float(latest["price"]) if latest["price"] else None,
            status=OrderStatus.FILLED if latest["status"] == "COMPLETE" else OrderStatus.PENDING,
            filled_quantity=float(latest["filled_quantity"]),
            filled_avg_price=float(latest["average_price"]) if latest["average_price"] else 0.0,
        )

    async def get_positions(self) -> list[Position]:
        if self._paper:
            return [
                Position(
                    asset=asset,
                    quantity=pos["quantity"],
                    avg_entry_price=pos["avg_entry_price"],
                    current_price=pos["current_price"],
                    market_value=pos["quantity"] * pos["current_price"],
                    unrealized_pnl=(pos["current_price"] - pos["avg_entry_price"]) * pos["quantity"],
                    unrealized_pnl_pct=(
                        (pos["current_price"] - pos["avg_entry_price"]) / pos["avg_entry_price"]
                        if pos["avg_entry_price"] > 0 else 0.0
                    ),
                    side="long" if pos["quantity"] > 0 else "short",
                )
                for asset, pos in self._paper_positions.items()
                if pos["quantity"] != 0
            ]

        net = self._kite.positions()["net"]
        return [
            Position(
                asset=p["tradingsymbol"],
                quantity=abs(float(p["quantity"])),
                avg_entry_price=float(p["average_price"]),
                current_price=float(p["last_price"]),
                market_value=abs(float(p["quantity"])) * float(p["last_price"]),
                unrealized_pnl=float(p["pnl"]),
                unrealized_pnl_pct=(
                    float(p["pnl"]) / (abs(float(p["quantity"])) * float(p["average_price"]))
                    if float(p["average_price"]) > 0 else 0.0
                ),
                side="long" if float(p["quantity"]) > 0 else "short",
            )
            for p in net
            if float(p["quantity"]) != 0
        ]

    async def get_position(self, asset: str) -> Position | None:
        positions = await self.get_positions()
        for p in positions:
            if p.asset == asset:
                return p
        return None

    async def close_position(self, asset: str) -> Order:
        pos = await self.get_position(asset)
        if not pos:
            raise ValueError(f"No position found for {asset}")

        side = OrderSide.SELL if pos.side == "long" else OrderSide.BUY
        return await self.place_order(asset, side, pos.quantity)

    async def get_quote(self, asset: str) -> dict[str, float]:
        if self._paper:
            # In paper mode, quote comes from data module (yfinance).
            # Return a placeholder — real data fetcher will override.
            price = self._paper_positions.get(asset, {}).get("current_price", 0.0)
            return {"bid": price, "ask": price, "last": price}

        quote = self._kite.quote(f"NSE:{asset}")
        data = quote[f"NSE:{asset}"]
        return {
            "bid": float(data["depth"]["buy"][0]["price"]),
            "ask": float(data["depth"]["sell"][0]["price"]),
            "last": float(data["last_price"]),
        }

    async def is_market_open(self) -> bool:
        # NSE hours: 9:15 AM to 3:30 PM IST
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    @property
    def market_name(self) -> str:
        return "india"

    @property
    def currency(self) -> str:
        return "INR"

    # ------------------------------------------------------------------
    # Paper trading simulation
    # ------------------------------------------------------------------

    async def _paper_place_order(
        self,
        asset: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        limit_price: float | None,
    ) -> Order:
        """Simulate order fill at current price (market orders fill immediately)."""
        order_id = f"paper-{uuid.uuid4().hex[:12]}"

        # For paper trading, assume fill at limit price or a simulated price
        fill_price = limit_price or self._paper_positions.get(
            asset, {}
        ).get("current_price", 100.0)

        cost = fill_price * quantity
        if side == OrderSide.BUY:
            if cost > self._paper_capital:
                logger.warning("zerodha.paper.insufficient_capital", cost=cost)
                return Order(
                    order_id=order_id, asset=asset, side=side,
                    order_type=order_type, quantity=quantity,
                    price=fill_price, status=OrderStatus.REJECTED,
                )
            self._paper_capital -= cost
            existing = self._paper_positions.get(asset, {"quantity": 0, "avg_entry_price": 0, "current_price": fill_price})
            new_qty = existing["quantity"] + quantity
            if new_qty > 0:
                new_avg = (
                    (existing["avg_entry_price"] * existing["quantity"] + fill_price * quantity) / new_qty
                )
            else:
                new_avg = fill_price
            self._paper_positions[asset] = {
                "quantity": new_qty,
                "avg_entry_price": new_avg,
                "current_price": fill_price,
            }
        else:
            # Sell
            self._paper_capital += cost
            if asset in self._paper_positions:
                self._paper_positions[asset]["quantity"] -= quantity
                if self._paper_positions[asset]["quantity"] <= 0:
                    del self._paper_positions[asset]

        order = Order(
            order_id=order_id,
            asset=asset,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=fill_price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_avg_price=fill_price,
        )
        self._paper_orders[order_id] = order

        logger.info(
            "zerodha.paper.order_filled",
            asset=asset, side=side.value, qty=quantity, price=fill_price,
        )
        return order

    def set_paper_capital(self, amount: float) -> None:
        """Set initial paper trading capital."""
        self._paper_capital = amount

    def update_paper_price(self, asset: str, price: float) -> None:
        """Update current price for paper position tracking."""
        if asset in self._paper_positions:
            self._paper_positions[asset]["current_price"] = price
