"""Yahoo Finance data provider.

Uses the yfinance library to fetch historical and recent OHLCV data.
"""

import logging
import time

import pandas as pd
import yfinance as yf

from src.data.base import AssetData, DataProvider

logger = logging.getLogger(__name__)

# Minimum seconds between consecutive yfinance downloads.
_MIN_REQUEST_INTERVAL = 0.25


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flatten a yfinance DataFrame into standard OHLCV columns.

    yfinance returns a MultiIndex with (Price, Ticker) or (Ticker, Price)
    depending on the call.  This helper always returns a simple DataFrame
    with lowercase columns: open, high, low, close, volume.

    Args:
        df: Raw DataFrame from ``yf.download``.
        symbol: Ticker used to slice when columns are MultiIndex.

    Returns:
        Flattened DataFrame with lowercase column names.

    Raises:
        ValueError: If the resulting DataFrame is empty.
    """
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    # Handle MultiIndex columns produced by yfinance.
    if isinstance(df.columns, pd.MultiIndex):
        # Try (Ticker, Price) first — produced by group_by='ticker'
        top_level = df.columns.get_level_values(0)
        if symbol in top_level:
            df = df[symbol].copy()
        else:
            # (Price, Ticker) layout — produced by single-symbol download
            df = df.droplevel(level=1, axis=1).copy()

    df.columns = [c.lower() for c in df.columns]
    return df


class YahooFinanceProvider(DataProvider):
    """Data provider backed by Yahoo Finance via *yfinance*.

    Args:
        rate_limit: Minimum seconds between API calls.
    """

    def __init__(self, rate_limit: float = _MIN_REQUEST_INTERVAL) -> None:
        self._rate_limit = rate_limit
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep if needed to respect the rate limit."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def fetch_historical(self, symbol: str, start: str, end: str) -> AssetData:
        """Fetch historical OHLCV data for a single asset.

        Args:
            symbol: Ticker symbol (e.g. "SPY").
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format (exclusive in yfinance).

        Returns:
            AssetData with the requested date range.

        Raises:
            ValueError: If the symbol is invalid or no data is returned.
            ConnectionError: If the network request fails.
        """
        logger.info("Fetching historical data for %s from %s to %s", symbol, start, end)
        self._throttle()

        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            raise ConnectionError(f"Failed to download data for {symbol}: {exc}") from exc

        df = _normalize_ohlcv(df, symbol)
        logger.info("Fetched %d rows for %s", len(df), symbol)
        return AssetData(symbol=symbol, ohlcv=df, metadata={"source": "yahoo"})

    def fetch_realtime(self, symbol: str) -> AssetData:
        """Fetch the most recent trading day's data.

        Args:
            symbol: Ticker symbol (e.g. "SPY").

        Returns:
            AssetData with the latest available bar.

        Raises:
            ValueError: If the symbol is invalid or no data is returned.
            ConnectionError: If the network request fails.
        """
        logger.info("Fetching realtime data for %s", symbol)
        self._throttle()

        try:
            df = yf.download(
                symbol,
                period="5d",
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            raise ConnectionError(f"Failed to download realtime data for {symbol}: {exc}") from exc

        df = _normalize_ohlcv(df, symbol)
        # Keep only the last row.
        df = df.iloc[[-1]]
        logger.info("Fetched realtime bar for %s at %s", symbol, df.index[-1])
        return AssetData(symbol=symbol, ohlcv=df, metadata={"source": "yahoo"})

    def fetch_universe(self, symbols: list[str], start: str, end: str) -> dict[str, AssetData]:
        """Fetch historical data for multiple assets in one call.

        Uses ``yf.download`` with ``group_by='ticker'`` for efficiency.

        Args:
            symbols: List of ticker symbols.
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format (exclusive in yfinance).

        Returns:
            Mapping of symbol to AssetData.  Symbols that fail to
            download are logged and omitted from the result.

        Raises:
            ConnectionError: If the bulk download itself fails.
        """
        if not symbols:
            return {}

        logger.info("Fetching universe of %d symbols from %s to %s", len(symbols), start, end)
        self._throttle()

        try:
            raw = yf.download(
                symbols,
                start=start,
                end=end,
                group_by="ticker",
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            raise ConnectionError(f"Failed to download universe data: {exc}") from exc

        results: dict[str, AssetData] = {}
        for symbol in symbols:
            try:
                df = _normalize_ohlcv(raw, symbol)
                # Drop rows where all values are NaN (symbol had no trading).
                df = df.dropna(how="all")
                if df.empty:
                    logger.warning("No data returned for %s, skipping", symbol)
                    continue
                results[symbol] = AssetData(symbol=symbol, ohlcv=df, metadata={"source": "yahoo"})
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping %s: %s", symbol, exc)

        logger.info("Fetched data for %d / %d symbols", len(results), len(symbols))
        return results
