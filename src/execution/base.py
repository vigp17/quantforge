"""Base interfaces for the execution layer.

Defines the Order and Fill containers and the abstract Broker interface
that all execution backends (paper, Alpaca, etc.) must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

_VALID_SIDES: set[str] = {"buy", "sell"}
_VALID_ORDER_TYPES: set[str] = {"market", "limit"}


@dataclass
class Order:
    """A trade order to be submitted to a broker.

    Args:
        symbol: Ticker symbol (e.g. "SPY").
        side: Order direction — "buy" or "sell".
        quantity: Number of shares/units. Must be positive.
        order_type: Execution type — "market" or "limit".
        limit_price: Required for limit orders, must be None for market orders.

    Raises:
        ValueError: If any field violates its constraints.
    """

    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: Literal["market", "limit"]
    limit_price: float | None = None

    def __post_init__(self) -> None:
        """Validate order fields."""
        if self.side not in _VALID_SIDES:
            raise ValueError(f"side must be one of {_VALID_SIDES}, got '{self.side}'")
        if self.order_type not in _VALID_ORDER_TYPES:
            raise ValueError(
                f"order_type must be one of {_VALID_ORDER_TYPES}, got '{self.order_type}'"
            )
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders")
        if self.order_type == "limit" and self.limit_price is not None and self.limit_price <= 0:
            raise ValueError(f"limit_price must be positive, got {self.limit_price}")
        if self.order_type == "market" and self.limit_price is not None:
            raise ValueError("limit_price must be None for market orders")


@dataclass
class Fill:
    """A completed (or partial) fill for an order.

    Args:
        order: The original order that was filled.
        fill_price: Actual execution price.
        fill_quantity: Number of shares/units filled. Must be positive and
            not exceed the order quantity.
        slippage: Price slippage from the expected price (can be negative).
        timestamp: When the fill occurred.

    Raises:
        ValueError: If any field violates its constraints.
    """

    order: Order
    fill_price: float
    fill_quantity: float
    slippage: float
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate fill fields."""
        if self.fill_price <= 0:
            raise ValueError(f"fill_price must be positive, got {self.fill_price}")
        if self.fill_quantity <= 0:
            raise ValueError(f"fill_quantity must be positive, got {self.fill_quantity}")
        if self.fill_quantity > self.order.quantity:
            raise ValueError(
                f"fill_quantity ({self.fill_quantity}) exceeds "
                f"order quantity ({self.order.quantity})"
            )


class Broker(ABC):
    """Abstract interface for order execution backends.

    All brokers (paper trading, Alpaca live, etc.) must implement this
    interface.
    """

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution.

        Args:
            order: The order to submit.

        Returns:
            A unique order ID string.
        """
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Get current positions.

        Returns:
            Mapping of symbol to quantity held.
        """
        ...

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Get the total portfolio value.

        Returns:
            Current portfolio value in base currency.
        """
        ...
