"""Tests for src/data/cache.py."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.base import AssetData, DataProvider
from src.data.cache import SQLiteCacheProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(
    symbol: str,
    start: str,
    periods: int,
    seed: int = 42,
) -> AssetData:
    """Build an AssetData with deterministic OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    data = {
        "open": rng.random(periods) * 100 + 400,
        "high": rng.random(periods) * 100 + 410,
        "low": rng.random(periods) * 100 + 390,
        "close": rng.random(periods) * 100 + 400,
        "volume": rng.integers(1_000_000, 10_000_000, size=periods).astype(float),
    }
    df = pd.DataFrame(data, index=dates)
    return AssetData(symbol=symbol, ohlcv=df, metadata={"source": "mock"})


def _mock_provider(assets: dict[str, AssetData] | None = None) -> MagicMock:
    """Create a mock DataProvider whose fetch_historical returns from *assets*."""
    provider = MagicMock(spec=DataProvider)

    def _fetch_historical(symbol: str, start: str, end: str) -> AssetData:
        if assets and symbol in assets:
            asset = assets[symbol]
            mask = (asset.ohlcv.index >= start) & (asset.ohlcv.index < end)
            filtered = asset.ohlcv.loc[mask]
            return AssetData(symbol=symbol, ohlcv=filtered, metadata=asset.metadata)
        raise ValueError(f"No data for {symbol}")

    provider.fetch_historical.side_effect = _fetch_historical
    return provider


# ---------------------------------------------------------------------------
# Cache miss – upstream is called
# ---------------------------------------------------------------------------

class TestCacheMiss:
    def test_fetches_from_provider_on_first_call(self) -> None:
        asset = _make_asset("SPY", "2024-01-02", periods=5)
        upstream = _mock_provider({"SPY": asset})
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        result = cache.fetch_historical("SPY", "2024-01-02", "2024-01-09")

        assert isinstance(result, AssetData)
        assert result.symbol == "SPY"
        assert len(result.ohlcv) > 0
        upstream.fetch_historical.assert_called_once()

    def test_data_is_stored_after_miss(self) -> None:
        asset = _make_asset("SPY", "2024-01-02", periods=5)
        upstream = _mock_provider({"SPY": asset})
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        cache.fetch_historical("SPY", "2024-01-02", "2024-01-09")
        stats = cache.cache_stats()

        assert "SPY" in stats
        assert stats["SPY"] == 5


# ---------------------------------------------------------------------------
# Cache hit – upstream is NOT called
# ---------------------------------------------------------------------------

class TestCacheHit:
    def test_serves_from_cache_on_second_call(self) -> None:
        asset = _make_asset("SPY", "2024-01-02", periods=5)
        upstream = _mock_provider({"SPY": asset})
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        # First call populates cache.
        first = cache.fetch_historical("SPY", "2024-01-02", "2024-01-09")
        upstream.fetch_historical.reset_mock()

        # Second call should be served entirely from cache.
        second = cache.fetch_historical("SPY", "2024-01-02", "2024-01-09")

        upstream.fetch_historical.assert_not_called()
        pd.testing.assert_frame_equal(
            first.ohlcv.reset_index(drop=True),
            second.ohlcv.reset_index(drop=True),
        )
        # Dates should match even if freq metadata differs.
        assert list(first.ohlcv.index) == list(second.ohlcv.index)

    def test_cached_data_has_correct_columns(self) -> None:
        asset = _make_asset("SPY", "2024-01-02", periods=3)
        upstream = _mock_provider({"SPY": asset})
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        cache.fetch_historical("SPY", "2024-01-02", "2024-01-05")
        upstream.fetch_historical.reset_mock()

        result = cache.fetch_historical("SPY", "2024-01-02", "2024-01-05")

        assert set(result.ohlcv.columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(result.ohlcv.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Partial cache – only the gap is fetched
# ---------------------------------------------------------------------------

class TestPartialCache:
    def test_fetches_only_missing_dates(self) -> None:
        # Full data covers 10 business days starting 2024-01-02.
        full = _make_asset("SPY", "2024-01-02", periods=10)
        upstream = _mock_provider({"SPY": full})
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        # Seed the cache with the first 5 days.
        cache.fetch_historical("SPY", "2024-01-02", "2024-01-09")
        upstream.fetch_historical.reset_mock()

        # Now request a wider range that extends beyond the cached window.
        result = cache.fetch_historical("SPY", "2024-01-02", "2024-01-16")

        # Upstream should be called for the gap only.
        upstream.fetch_historical.assert_called()
        # The result should cover the full range that has data.
        assert len(result.ohlcv) > 5


# ---------------------------------------------------------------------------
# fetch_realtime – always delegates
# ---------------------------------------------------------------------------

class TestRealtimePassthrough:
    def test_realtime_delegates_to_provider(self) -> None:
        asset = _make_asset("SPY", "2024-06-14", periods=1)
        upstream = _mock_provider()
        upstream.fetch_realtime.return_value = asset
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        result = cache.fetch_realtime("SPY")

        upstream.fetch_realtime.assert_called_once_with("SPY")
        assert result.symbol == "SPY"


# ---------------------------------------------------------------------------
# fetch_universe – per-symbol caching
# ---------------------------------------------------------------------------

class TestFetchUniverse:
    def test_returns_dict_of_assets(self) -> None:
        assets = {
            "SPY": _make_asset("SPY", "2024-01-02", periods=5),
            "QQQ": _make_asset("QQQ", "2024-01-02", periods=5, seed=99),
        }
        upstream = _mock_provider(assets)
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        result = cache.fetch_universe(["SPY", "QQQ"], "2024-01-02", "2024-01-09")

        assert "SPY" in result
        assert "QQQ" in result
        stats = cache.cache_stats()
        assert stats["SPY"] == 5
        assert stats["QQQ"] == 5

    def test_skips_failing_symbols(self) -> None:
        assets = {"SPY": _make_asset("SPY", "2024-01-02", periods=5)}
        upstream = _mock_provider(assets)
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        result = cache.fetch_universe(["SPY", "BAD"], "2024-01-02", "2024-01-09")

        assert "SPY" in result
        assert "BAD" not in result


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

class TestCacheManagement:
    def test_clear_cache_single_symbol(self) -> None:
        assets = {
            "SPY": _make_asset("SPY", "2024-01-02", periods=3),
            "QQQ": _make_asset("QQQ", "2024-01-02", periods=3, seed=99),
        }
        upstream = _mock_provider(assets)
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        cache.fetch_historical("SPY", "2024-01-02", "2024-01-05")
        cache.fetch_historical("QQQ", "2024-01-02", "2024-01-05")
        cache.clear_cache(symbol="SPY")

        stats = cache.cache_stats()
        assert "SPY" not in stats
        assert stats["QQQ"] == 3

    def test_clear_cache_all(self) -> None:
        assets = {"SPY": _make_asset("SPY", "2024-01-02", periods=3)}
        upstream = _mock_provider(assets)
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        cache.fetch_historical("SPY", "2024-01-02", "2024-01-05")
        cache.clear_cache()

        assert cache.cache_stats() == {}

    def test_cache_stats_empty(self) -> None:
        upstream = _mock_provider()
        cache = SQLiteCacheProvider(upstream, db_path=":memory:")

        assert cache.cache_stats() == {}
