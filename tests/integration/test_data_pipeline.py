"""Integration test: YahooFinanceProvider → SQLiteCacheProvider → FeatureEngineer.

Exercises the full data pipeline against live Yahoo Finance data.
Run with:  pytest tests/integration/ -v -m network
"""

import time

import pandas as pd
import pytest

from src.data.base import REQUIRED_OHLCV_COLUMNS
from src.data.cache import SQLiteCacheProvider
from src.data.features import ALL_FEATURES, FeatureEngineer
from src.data.yahoo import YahooFinanceProvider

# ~18 months of data to satisfy the 252-row momentum warmup.
_START = "2024-01-02"
_END = "2025-06-30"
_SYMBOLS = ["SPY", "QQQ"]


@pytest.mark.network
class TestDataPipeline:
    """End-to-end pipeline: fetch → cache → features."""

    def setup_method(self) -> None:
        yahoo = YahooFinanceProvider(rate_limit=0.1)
        self.cache = SQLiteCacheProvider(yahoo, db_path=":memory:")
        self.fe = FeatureEngineer(nan_handling="trim", normalize=True)

    # ------------------------------------------------------------------
    # Stage 1: Fetch + cache
    # ------------------------------------------------------------------

    def test_fetch_universe(self) -> None:
        """Fetching SPY and QQQ returns valid AssetData for both."""
        universe = self.cache.fetch_universe(_SYMBOLS, _START, _END)

        assert set(universe.keys()) == set(_SYMBOLS)
        for symbol, asset in universe.items():
            assert asset.symbol == symbol
            assert isinstance(asset.ohlcv.index, pd.DatetimeIndex)
            assert set(asset.ohlcv.columns) >= REQUIRED_OHLCV_COLUMNS
            assert len(asset.ohlcv) > 250, f"{symbol} has only {len(asset.ohlcv)} rows"

    def test_cache_hit_is_faster(self) -> None:
        """Second fetch should be served from SQLite and be significantly faster."""
        # First fetch — hits Yahoo.
        t0 = time.perf_counter()
        self.cache.fetch_universe(_SYMBOLS, _START, _END)
        first_elapsed = time.perf_counter() - t0

        # Second fetch — should be a full cache hit (no network).
        t0 = time.perf_counter()
        self.cache.fetch_universe(_SYMBOLS, _START, _END)
        second_elapsed = time.perf_counter() - t0

        # Cache hit should be at least 2× faster.
        assert second_elapsed < first_elapsed / 2, (
            f"Cache hit ({second_elapsed:.3f}s) was not significantly faster "
            f"than first fetch ({first_elapsed:.3f}s)"
        )

    def test_cache_stats_populated(self) -> None:
        """After fetching, cache_stats reports rows for each symbol."""
        self.cache.fetch_universe(_SYMBOLS, _START, _END)
        stats = self.cache.cache_stats()

        for symbol in _SYMBOLS:
            assert symbol in stats
            assert stats[symbol] > 250

    # ------------------------------------------------------------------
    # Stage 2: Feature engineering
    # ------------------------------------------------------------------

    def test_features_shape_and_columns(self) -> None:
        """Feature DataFrame has correct MultiIndex columns and no NaNs."""
        universe = self.cache.fetch_universe(_SYMBOLS, _START, _END)
        features = self.fe.compute(universe)

        assert not features.empty
        assert features.isna().sum().sum() == 0

        symbols = features.columns.get_level_values("symbol").unique()
        feat_names = features.columns.get_level_values("feature").unique()
        assert set(symbols) == set(_SYMBOLS)
        assert set(feat_names) == set(ALL_FEATURES)

    def test_features_are_normalised(self) -> None:
        """Z-scored features should have mean ~0 and std ~1."""
        universe = self.cache.fetch_universe(_SYMBOLS, _START, _END)
        features = self.fe.compute(universe)

        for col in features.columns:
            assert abs(features[col].mean()) < 1e-8, f"{col} mean != 0"
            assert abs(features[col].std() - 1.0) < 1e-8, f"{col} std != 1"

    def test_features_row_count_after_trim(self) -> None:
        """Trimmed rows should equal input rows minus warmup."""
        universe = self.cache.fetch_universe(_SYMBOLS, _START, _END)
        input_len = min(len(a.ohlcv) for a in universe.values())
        features = self.fe.compute(universe)

        expected = input_len - self.fe.warmup_rows()
        assert len(features) == expected

    # ------------------------------------------------------------------
    # Full round-trip
    # ------------------------------------------------------------------

    def test_full_round_trip(self) -> None:
        """Fetch → cache → re-fetch (cache hit) → features all succeed."""
        # Fetch from Yahoo and cache.
        first = self.cache.fetch_universe(_SYMBOLS, _START, _END)

        # Re-fetch from cache.
        second = self.cache.fetch_universe(_SYMBOLS, _START, _END)
        for symbol in _SYMBOLS:
            assert list(first[symbol].ohlcv.index) == list(second[symbol].ohlcv.index)

        # Compute features on cached data.
        features = self.fe.compute(second)
        assert not features.empty
        assert features.isna().sum().sum() == 0
