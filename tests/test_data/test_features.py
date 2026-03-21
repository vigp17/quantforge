"""Tests for src/data/features.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.base import AssetData
from src.data.features import ALL_FEATURES, FeatureEngineer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(
    symbol: str = "SPY",
    periods: int = 300,
    seed: int = 42,
) -> AssetData:
    """Build an AssetData with realistic-ish random OHLCV."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=periods)
    close = 400 + np.cumsum(rng.standard_normal(periods) * 2)
    df = pd.DataFrame(
        {
            "open": close + rng.standard_normal(periods) * 0.5,
            "high": close + np.abs(rng.standard_normal(periods)) * 1.5,
            "low": close - np.abs(rng.standard_normal(periods)) * 1.5,
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, size=periods).astype(float),
        },
        index=dates,
    )
    return AssetData(symbol=symbol, ohlcv=df)


def _make_monotonic_asset(
    symbol: str = "UP",
    periods: int = 60,
    start_price: float = 100.0,
    step: float = 1.0,
) -> AssetData:
    """Build an AssetData with strictly increasing close prices."""
    dates = pd.bdate_range("2024-01-02", periods=periods)
    close = np.arange(start_price, start_price + periods * step, step)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(periods, 5_000_000.0),
        },
        index=dates,
    )
    return AssetData(symbol=symbol, ohlcv=df)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_trim_mode_drops_warmup_rows(self) -> None:
        data = {"SPY": _make_asset(periods=300)}
        fe = FeatureEngineer(nan_handling="trim", normalize=False)
        result = fe.compute(data)

        warmup = fe.warmup_rows()
        # After trimming NaNs, rows should be original minus warmup.
        assert len(result) == 300 - warmup

    def test_fill_mode_preserves_all_rows(self) -> None:
        data = {"SPY": _make_asset(periods=300)}
        fe = FeatureEngineer(nan_handling="fill", normalize=False)
        result = fe.compute(data)

        assert len(result) == 300
        assert result.isna().sum().sum() == 0

    def test_columns_match_features_and_symbols(self) -> None:
        data = {
            "SPY": _make_asset("SPY", periods=300),
            "QQQ": _make_asset("QQQ", periods=300, seed=99),
        }
        fe = FeatureEngineer(nan_handling="trim", normalize=False)
        result = fe.compute(data)

        assert result.columns.names == ["symbol", "feature"]
        symbols = result.columns.get_level_values("symbol").unique()
        features = result.columns.get_level_values("feature").unique()
        assert set(symbols) == {"SPY", "QQQ"}
        assert set(features) == set(ALL_FEATURES)

    def test_subset_features(self) -> None:
        data = {"SPY": _make_asset(periods=50)}
        fe = FeatureEngineer(nan_handling="trim", normalize=False)
        result = fe.compute(data, features_list=["log_returns", "rsi"])

        features = result.columns.get_level_values("feature").unique()
        assert set(features) == {"log_returns", "rsi"}

    def test_unknown_feature_raises(self) -> None:
        data = {"SPY": _make_asset(periods=50)}
        fe = FeatureEngineer()
        with pytest.raises(ValueError, match="Unknown features"):
            fe.compute(data, features_list=["log_returns", "magic_indicator"])

    def test_empty_data_returns_empty_frame(self) -> None:
        fe = FeatureEngineer()
        result = fe.compute({})
        assert result.empty


# ---------------------------------------------------------------------------
# Known-value checks
# ---------------------------------------------------------------------------

class TestKnownValues:
    def test_rsi_of_strictly_increasing_series_near_100(self) -> None:
        """RSI of a monotonically rising price should approach 100."""
        data = {"UP": _make_monotonic_asset(periods=60)}
        fe = FeatureEngineer(nan_handling="trim", normalize=False)
        result = fe.compute(data, features_list=["rsi"])

        rsi_values = result[("UP", "rsi")]
        # After warmup, every RSI reading should be very close to 100.
        assert (rsi_values > 95).all(), f"RSI min was {rsi_values.min():.2f}"

    def test_log_returns_of_constant_price_is_zero(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1_000_000.0,
            },
            index=dates,
        )
        data = {"FLAT": AssetData(symbol="FLAT", ohlcv=df)}
        fe = FeatureEngineer(nan_handling="trim", normalize=False)
        result = fe.compute(data, features_list=["log_returns"])

        np.testing.assert_allclose(result[("FLAT", "log_returns")].values, 0.0, atol=1e-12)

    def test_volatility_of_constant_price_is_zero(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1_000_000.0,
            },
            index=dates,
        )
        data = {"FLAT": AssetData(symbol="FLAT", ohlcv=df)}
        fe = FeatureEngineer(nan_handling="trim", normalize=False)
        result = fe.compute(data, features_list=["volatility"])

        np.testing.assert_allclose(result[("FLAT", "volatility")].values, 0.0, atol=1e-12)

    def test_macd_histogram_sign(self) -> None:
        """For a strictly rising price, fast EMA > slow EMA → positive MACD."""
        data = {"UP": _make_monotonic_asset(periods=60)}
        fe = FeatureEngineer(nan_handling="trim", normalize=False)
        result = fe.compute(data, features_list=["macd"])

        # After enough warmup the MACD histogram should be non-negative.
        macd_vals = result[("UP", "macd")]
        assert (macd_vals.iloc[-10:] >= -1e-9).all()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_zscore_normalised_has_zero_mean_unit_std(self) -> None:
        data = {"SPY": _make_asset(periods=300)}
        fe = FeatureEngineer(nan_handling="trim", normalize=True)
        result = fe.compute(data)

        for col in result.columns:
            np.testing.assert_allclose(result[col].mean(), 0.0, atol=1e-10)
            np.testing.assert_allclose(result[col].std(), 1.0, atol=1e-10)

    def test_raw_mode_skips_normalisation(self) -> None:
        data = {"SPY": _make_asset(periods=300)}
        fe_raw = FeatureEngineer(nan_handling="trim", normalize=False)
        fe_norm = FeatureEngineer(nan_handling="trim", normalize=True)

        raw = fe_raw.compute(data)
        normed = fe_norm.compute(data)

        # Raw and normalised should have the same shape but different values.
        assert raw.shape == normed.shape
        assert not np.allclose(raw.values, normed.values)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNanHandling:
    def test_trim_has_no_nans(self) -> None:
        data = {"SPY": _make_asset(periods=300)}
        fe = FeatureEngineer(nan_handling="trim", normalize=False)
        result = fe.compute(data)

        assert result.isna().sum().sum() == 0

    def test_fill_has_no_nans(self) -> None:
        data = {"SPY": _make_asset(periods=300)}
        fe = FeatureEngineer(nan_handling="fill", normalize=False)
        result = fe.compute(data)

        assert result.isna().sum().sum() == 0

    def test_fill_returns_more_rows_than_trim(self) -> None:
        data = {"SPY": _make_asset(periods=300)}
        trim = FeatureEngineer(nan_handling="trim", normalize=False).compute(data)
        fill = FeatureEngineer(nan_handling="fill", normalize=False).compute(data)

        assert len(fill) > len(trim)
