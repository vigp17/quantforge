"""Base interfaces for the data layer.

Defines the AssetData container and the abstract DataProvider interface
that all data sources (Yahoo, FRED, Alpaca, etc.) must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

REQUIRED_OHLCV_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


@dataclass
class AssetData:
    """Container for a single asset's market data.

    Args:
        symbol: Ticker symbol (e.g. "SPY").
        ohlcv: DataFrame with columns open, high, low, close, volume
            and a DatetimeIndex.
        metadata: Arbitrary metadata (exchange, sector, etc.).

    Raises:
        ValueError: If ohlcv is missing required columns or lacks a
            DatetimeIndex.
    """

    symbol: str
    ohlcv: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ohlcv schema after initialization."""
        missing = REQUIRED_OHLCV_COLUMNS - set(self.ohlcv.columns)
        if missing:
            raise ValueError(f"ohlcv DataFrame missing required columns: {sorted(missing)}")
        if not isinstance(self.ohlcv.index, pd.DatetimeIndex):
            raise ValueError("ohlcv DataFrame must have a DatetimeIndex")


class DataProvider(ABC):
    """Abstract interface for market data sources.

    All data providers (Yahoo Finance, FRED, Alpaca, etc.) must implement
    this interface so that downstream consumers can swap providers without
    code changes.
    """

    @abstractmethod
    def fetch_historical(self, symbol: str, start: str, end: str) -> AssetData:
        """Fetch historical OHLCV data for a single asset.

        Args:
            symbol: Ticker symbol (e.g. "SPY").
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.

        Returns:
            AssetData with the requested date range.
        """
        ...

    @abstractmethod
    def fetch_realtime(self, symbol: str) -> AssetData:
        """Fetch the latest available data for a single asset.

        Args:
            symbol: Ticker symbol (e.g. "SPY").

        Returns:
            AssetData with the most recent bar(s).
        """
        ...

    @abstractmethod
    def fetch_universe(self, symbols: list[str], start: str, end: str) -> dict[str, AssetData]:
        """Fetch historical data for multiple assets.

        Args:
            symbols: List of ticker symbols.
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.

        Returns:
            Mapping of symbol to AssetData.
        """
        ...
