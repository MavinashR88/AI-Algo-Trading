"""Tests for broker adapters."""

from __future__ import annotations

import pytest

from brokers.base import (
    AccountInfo,
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from brokers.zerodha_broker import ZerodhaBroker


class TestBaseBrokerInterface:
    """Verify the abstract interface has all required methods."""

    def test_cannot_instantiate_abstract(self):
        """BaseBroker is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseBroker()

    def test_order_dataclass(self):
        """Order dataclass holds all fields."""
        order = Order(
            order_id="test-123",
            asset="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            price=None,
            status=OrderStatus.FILLED,
            filled_quantity=10,
            filled_avg_price=150.0,
        )
        assert order.order_id == "test-123"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.FILLED

    def test_position_dataclass(self):
        """Position dataclass computes correctly."""
        pos = Position(
            asset="AAPL",
            quantity=10,
            avg_entry_price=150.0,
            current_price=160.0,
            market_value=1600.0,
            unrealized_pnl=100.0,
            unrealized_pnl_pct=0.0667,
            side="long",
        )
        assert pos.unrealized_pnl == 100.0


class TestZerodhaPaperTrading:
    """Test Zerodha paper trading simulator."""

    @pytest.fixture
    def broker(self):
        b = ZerodhaBroker(paper=True)
        b.set_paper_capital(1_000_000.0)
        return b

    @pytest.mark.asyncio
    async def test_connect_paper_mode(self, broker):
        """Paper mode connects without credentials."""
        await broker.connect()
        assert broker._connected is True

    @pytest.mark.asyncio
    async def test_get_account_paper(self, broker):
        """Paper account returns initial capital."""
        await broker.connect()
        account = await broker.get_account()
        assert account.cash == 1_000_000.0
        assert account.currency == "INR"

    @pytest.mark.asyncio
    async def test_place_buy_order_paper(self, broker):
        """Paper buy order fills immediately."""
        await broker.connect()
        order = await broker.place_order(
            asset="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            limit_price=2500.0,
        )
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10
        assert order.filled_avg_price == 2500.0

    @pytest.mark.asyncio
    async def test_capital_deducted_on_buy(self, broker):
        """Paper buy deducts capital correctly."""
        await broker.connect()
        await broker.place_order(
            asset="TCS", side=OrderSide.BUY, quantity=5, limit_price=3000.0,
        )
        account = await broker.get_account()
        assert account.cash == 1_000_000.0 - (5 * 3000.0)

    @pytest.mark.asyncio
    async def test_position_tracked_after_buy(self, broker):
        """Paper position appears after buy."""
        await broker.connect()
        await broker.place_order(
            asset="INFY", side=OrderSide.BUY, quantity=20, limit_price=1500.0,
        )
        positions = await broker.get_positions()
        assert len(positions) == 1
        assert positions[0].asset == "INFY"
        assert positions[0].quantity == 20

    @pytest.mark.asyncio
    async def test_sell_closes_position(self, broker):
        """Paper sell removes position."""
        await broker.connect()
        await broker.place_order(
            asset="SBIN", side=OrderSide.BUY, quantity=10, limit_price=500.0,
        )
        await broker.place_order(
            asset="SBIN", side=OrderSide.SELL, quantity=10, limit_price=520.0,
        )
        positions = await broker.get_positions()
        assert len(positions) == 0

    @pytest.mark.asyncio
    async def test_insufficient_capital_rejected(self, broker):
        """Order rejected when insufficient capital."""
        await broker.connect()
        broker.set_paper_capital(100.0)  # very low capital
        order = await broker.place_order(
            asset="RELIANCE", side=OrderSide.BUY, quantity=1000, limit_price=2500.0,
        )
        assert order.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_market_properties(self, broker):
        """Broker reports correct market and currency."""
        assert broker.market_name == "india"
        assert broker.currency == "INR"

    @pytest.mark.asyncio
    async def test_close_position(self, broker):
        """Close position via helper method."""
        await broker.connect()
        await broker.place_order(
            asset="ITC", side=OrderSide.BUY, quantity=100, limit_price=400.0,
        )
        order = await broker.close_position("ITC")
        assert order.status == OrderStatus.FILLED

        positions = await broker.get_positions()
        assert len(positions) == 0

    @pytest.mark.asyncio
    async def test_cancel_order(self, broker):
        """Cancel a paper order."""
        await broker.connect()
        order = await broker.place_order(
            asset="LT", side=OrderSide.BUY, quantity=5, limit_price=2000.0,
        )
        # Paper orders fill immediately, but cancel should still work
        result = await broker.cancel_order(order.order_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, broker):
        """Cancel nonexistent order returns False."""
        await broker.connect()
        result = await broker.cancel_order("fake-order-id")
        assert result is False
