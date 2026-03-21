"""Tests for src/execution/base.py."""

from datetime import datetime

import pytest

from src.execution.base import Broker, Fill, Order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _market_order(**kwargs) -> Order:
    defaults = {"symbol": "SPY", "side": "buy", "quantity": 10.0, "order_type": "market"}
    defaults.update(kwargs)
    return Order(**defaults)


def _limit_order(**kwargs) -> Order:
    defaults = {
        "symbol": "SPY",
        "side": "sell",
        "quantity": 5.0,
        "order_type": "limit",
        "limit_price": 450.0,
    }
    defaults.update(kwargs)
    return Order(**defaults)


def _fill(order: Order | None = None, **kwargs) -> Fill:
    if order is None:
        order = _market_order()
    defaults = {
        "order": order,
        "fill_price": 450.0,
        "fill_quantity": order.quantity,
        "slippage": 0.05,
        "timestamp": datetime(2024, 6, 15, 10, 30),
    }
    defaults.update(kwargs)
    return Fill(**defaults)


# ---------------------------------------------------------------------------
# Order – happy path
# ---------------------------------------------------------------------------

class TestOrderValid:
    def test_market_buy(self) -> None:
        order = _market_order()
        assert order.side == "buy"
        assert order.order_type == "market"
        assert order.limit_price is None

    def test_market_sell(self) -> None:
        order = _market_order(side="sell")
        assert order.side == "sell"

    def test_limit_order(self) -> None:
        order = _limit_order()
        assert order.order_type == "limit"
        assert order.limit_price == 450.0

    def test_fractional_quantity(self) -> None:
        order = _market_order(quantity=0.5)
        assert order.quantity == 0.5


# ---------------------------------------------------------------------------
# Order – validation errors
# ---------------------------------------------------------------------------

class TestOrderValidation:
    def test_rejects_invalid_side(self) -> None:
        with pytest.raises(ValueError, match="side must be one of"):
            _market_order(side="short")

    def test_rejects_invalid_order_type(self) -> None:
        with pytest.raises(ValueError, match="order_type must be one of"):
            Order(symbol="SPY", side="buy", quantity=10.0, order_type="stop")

    def test_rejects_zero_quantity(self) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            _market_order(quantity=0.0)

    def test_rejects_negative_quantity(self) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            _market_order(quantity=-5.0)

    def test_rejects_limit_order_without_price(self) -> None:
        with pytest.raises(ValueError, match="limit_price is required"):
            Order(symbol="SPY", side="buy", quantity=10.0, order_type="limit")

    def test_rejects_limit_order_with_negative_price(self) -> None:
        with pytest.raises(ValueError, match="limit_price must be positive"):
            _limit_order(limit_price=-100.0)

    def test_rejects_market_order_with_limit_price(self) -> None:
        with pytest.raises(ValueError, match="limit_price must be None"):
            _market_order(limit_price=450.0)


# ---------------------------------------------------------------------------
# Fill – happy path
# ---------------------------------------------------------------------------

class TestFillValid:
    def test_full_fill(self) -> None:
        order = _market_order(quantity=10.0)
        fill = _fill(order, fill_quantity=10.0)
        assert fill.fill_quantity == order.quantity

    def test_partial_fill(self) -> None:
        order = _market_order(quantity=10.0)
        fill = _fill(order, fill_quantity=3.0)
        assert fill.fill_quantity < order.quantity

    def test_negative_slippage(self) -> None:
        fill = _fill(slippage=-0.02)
        assert fill.slippage == -0.02

    def test_zero_slippage(self) -> None:
        fill = _fill(slippage=0.0)
        assert fill.slippage == 0.0


# ---------------------------------------------------------------------------
# Fill – validation errors
# ---------------------------------------------------------------------------

class TestFillValidation:
    def test_rejects_non_positive_fill_price(self) -> None:
        with pytest.raises(ValueError, match="fill_price must be positive"):
            _fill(fill_price=0.0)

    def test_rejects_negative_fill_price(self) -> None:
        with pytest.raises(ValueError, match="fill_price must be positive"):
            _fill(fill_price=-10.0)

    def test_rejects_zero_fill_quantity(self) -> None:
        with pytest.raises(ValueError, match="fill_quantity must be positive"):
            _fill(fill_quantity=0.0)

    def test_rejects_negative_fill_quantity(self) -> None:
        with pytest.raises(ValueError, match="fill_quantity must be positive"):
            _fill(fill_quantity=-1.0)

    def test_rejects_fill_quantity_exceeding_order(self) -> None:
        order = _market_order(quantity=10.0)
        with pytest.raises(ValueError, match="exceeds order quantity"):
            _fill(order, fill_quantity=15.0)


# ---------------------------------------------------------------------------
# Broker – abstract enforcement
# ---------------------------------------------------------------------------

class TestBrokerAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Broker()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_instantiate(self) -> None:
        class Partial(Broker):
            def submit_order(self, order: Order) -> str:
                return "id-1"

        with pytest.raises(TypeError, match="abstract"):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_can_instantiate(self) -> None:
        class Concrete(Broker):
            def submit_order(self, order: Order) -> str:
                return "id-1"

            def get_positions(self) -> dict[str, float]:
                return {}

            def get_portfolio_value(self) -> float:
                return 0.0

        broker = Concrete()
        assert isinstance(broker, Broker)
