"""SQLite caching layer for data providers.

Wraps any DataProvider with transparent on-disk caching so repeated
requests for the same date ranges avoid redundant API calls.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.data.base import AssetData, DataProvider

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "data/cache/market_data.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL NOT NULL,
    high   REAL NOT NULL,
    low    REAL NOT NULL,
    close  REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, date)
)
"""

_UPSERT_SQL = """
INSERT OR REPLACE INTO ohlcv (symbol, date, open, high, low, close, volume)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


class SQLiteCacheProvider(DataProvider):
    """Caching decorator that fronts any DataProvider with a SQLite store.

    Args:
        provider: The upstream DataProvider to cache.
        db_path: Path to the SQLite database file.  Use ``":memory:"``
            for an in-memory database (useful in tests).
    """

    def __init__(
        self,
        provider: DataProvider,
        db_path: str = _DEFAULT_DB_PATH,
    ) -> None:
        self._provider = provider
        self._db_path = db_path

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_cached(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Read cached rows for *symbol* in [start, end).

        Returns:
            DataFrame with DatetimeIndex and lowercase ohlcv columns,
            or an empty DataFrame if nothing is cached.
        """
        query = (
            "SELECT date, open, high, low, close, volume "
            "FROM ohlcv WHERE symbol = ? AND date >= ? AND date < ? "
            "ORDER BY date"
        )
        df = pd.read_sql_query(query, self._conn, params=(symbol, start, end))
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index.name = None
        return df

    def _write_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Insert or replace rows into the cache."""
        rows = [
            (
                symbol,
                idx.strftime("%Y-%m-%d"),
                r["open"],
                r["high"],
                r["low"],
                r["close"],
                r["volume"],
            )
            for idx, r in df.iterrows()
        ]
        self._conn.executemany(_UPSERT_SQL, rows)
        self._conn.commit()
        logger.debug("Cached %d rows for %s", len(rows), symbol)

    @staticmethod
    def _missing_date_ranges(cached: pd.DataFrame, start: str, end: str) -> list[tuple[str, str]]:
        """Identify date ranges not yet covered by the cache.

        Uses the first and last dates in *cached* to determine whether
        the leading or trailing portion of ``[start, end)`` is missing.
        This avoids false gaps from market holidays that ``bdate_range``
        would incorrectly flag.

        Returns a list of ``(gap_start, gap_end)`` strings where
        *gap_end* is exclusive (suitable for passing to a provider).
        """
        if cached.empty:
            return [(start, end)]

        req_start = pd.Timestamp(start)
        req_end = pd.Timestamp(end)
        cache_start = cached.index.min()
        cache_end = cached.index.max()

        gaps: list[tuple[str, str]] = []

        # Leading gap: requested start is before the first cached date.
        if req_start < cache_start:
            gaps.append(
                (
                    req_start.strftime("%Y-%m-%d"),
                    cache_start.strftime("%Y-%m-%d"),
                )
            )

        # Trailing gap: requested end is after the last cached date.
        # Add one business day to cache_end so we don't re-fetch it.
        day_after_cache = cache_end + pd.tseries.offsets.BDay(1)
        if day_after_cache < req_end:
            gaps.append(
                (
                    day_after_cache.strftime("%Y-%m-%d"),
                    req_end.strftime("%Y-%m-%d"),
                )
            )

        return gaps

    # ------------------------------------------------------------------
    # DataProvider interface
    # ------------------------------------------------------------------

    def fetch_historical(self, symbol: str, start: str, end: str) -> AssetData:
        """Fetch historical data, serving from cache when possible.

        Args:
            symbol: Ticker symbol.
            start: Start date (YYYY-MM-DD, inclusive).
            end: End date (YYYY-MM-DD, exclusive).

        Returns:
            AssetData covering the requested range.
        """
        cached = self._read_cached(symbol, start, end)

        gaps = self._missing_date_ranges(cached, start, end)

        if gaps:
            logger.info(
                "Cache miss/partial for %s [%s, %s): %d gap(s)",
                symbol,
                start,
                end,
                len(gaps),
            )
            fetched_frames: list[pd.DataFrame] = []
            for gap_start, gap_end in gaps:
                asset = self._provider.fetch_historical(symbol, gap_start, gap_end)
                self._write_cache(symbol, asset.ohlcv)
                fetched_frames.append(asset.ohlcv)

            if cached.empty:
                combined = pd.concat(fetched_frames).sort_index()
            else:
                combined = pd.concat([cached, *fetched_frames]).sort_index()
            # Remove any duplicates from overlapping fetches.
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            logger.info("Full cache hit for %s [%s, %s)", symbol, start, end)
            combined = cached

        return AssetData(
            symbol=symbol,
            ohlcv=combined,
            metadata={"source": "cache"},
        )

    def fetch_realtime(self, symbol: str) -> AssetData:
        """Fetch realtime data (always delegates to the upstream provider).

        Realtime data is never cached because it changes intraday.

        Args:
            symbol: Ticker symbol.

        Returns:
            AssetData with the latest bar.
        """
        return self._provider.fetch_realtime(symbol)

    def fetch_universe(self, symbols: list[str], start: str, end: str) -> dict[str, AssetData]:
        """Fetch data for multiple symbols, using cache per-symbol.

        Args:
            symbols: List of ticker symbols.
            start: Start date (YYYY-MM-DD, inclusive).
            end: End date (YYYY-MM-DD, exclusive).

        Returns:
            Mapping of symbol to AssetData.
        """
        results: dict[str, AssetData] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_historical(symbol, start, end)
            except Exception:
                logger.warning("Failed to fetch %s, skipping", symbol, exc_info=True)
        return results

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self, symbol: str | None = None) -> None:
        """Delete cached data.

        Args:
            symbol: If provided, only clear data for this symbol.
                Otherwise clear the entire cache.
        """
        if symbol is None:
            self._conn.execute("DELETE FROM ohlcv")
            logger.info("Cleared entire cache")
        else:
            self._conn.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
            logger.info("Cleared cache for %s", symbol)
        self._conn.commit()

    def cache_stats(self) -> dict[str, int]:
        """Return the number of cached rows per symbol.

        Returns:
            Mapping of symbol to row count.
        """
        cursor = self._conn.execute(
            "SELECT symbol, COUNT(*) FROM ohlcv GROUP BY symbol ORDER BY symbol"
        )
        return dict(cursor.fetchall())
