"""Tests for src/data/base.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.base import AssetData, DataProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(
    columns: list[str] | None = None,
    index: pd.Index | None = None,
    rows: int = 5,
) -> pd.DataFrame:
    """Helper to build an OHLCV-like DataFrame."""
    if columns is None:
        columns = ["open", "high", "low", "close", "volume"]
    if index is None:
        index = pd.date_range("2024-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(42)
    data = rng.random((rows, len(columns)))
    return pd.DataFrame(data, columns=columns, index=index)


# ---------------------------------------------------------------------------
# AssetData – happy path
# ---------------------------------------------------------------------------

class TestAssetDataValid:
    def test_creates_with_valid_ohlcv(self) -> None:
        df = _make_ohlcv()
        asset = AssetData(symbol="SPY", ohlcv=df)

        assert asset.symbol == "SPY"
        assert list(asset.ohlcv.columns) == ["open", "high", "low", "close", "volume"]
        assert asset.metadata == {}

    def test_accepts_extra_columns(self) -> None:
        df = _make_ohlcv(columns=["open", "high", "low", "close", "volume", "vwap"])
        asset = AssetData(symbol="QQQ", ohlcv=df, metadata={"exchange": "NASDAQ"})

        assert "vwap" in asset.ohlcv.columns
        assert asset.metadata == {"exchange": "NASDAQ"}

    def test_accepts_custom_metadata(self) -> None:
        df = _make_ohlcv()
        meta = {"sector": "Technology", "exchange": "NYSE"}
        asset = AssetData(symbol="AAPL", ohlcv=df, metadata=meta)

        assert asset.metadata["sector"] == "Technology"


# ---------------------------------------------------------------------------
# AssetData – validation errors
# ---------------------------------------------------------------------------

class TestAssetDataValidation:
    def test_rejects_missing_single_column(self) -> None:
        df = _make_ohlcv(columns=["open", "high", "low", "close"])
        with pytest.raises(ValueError, match="volume"):
            AssetData(symbol="SPY", ohlcv=df)

    def test_rejects_missing_multiple_columns(self) -> None:
        df = _make_ohlcv(columns=["open", "close"])
        with pytest.raises(ValueError, match="missing required columns"):
            AssetData(symbol="SPY", ohlcv=df)

    def test_rejects_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="missing required columns"):
            AssetData(symbol="SPY", ohlcv=df)

    def test_rejects_integer_index(self) -> None:
        df = _make_ohlcv(index=pd.RangeIndex(5))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            AssetData(symbol="SPY", ohlcv=df)

    def test_rejects_string_index(self) -> None:
        df = _make_ohlcv(index=pd.Index(["a", "b", "c", "d", "e"]))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            AssetData(symbol="SPY", ohlcv=df)


# ---------------------------------------------------------------------------
# DataProvider – abstract enforcement
# ---------------------------------------------------------------------------

class TestDataProviderAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            DataProvider()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_instantiate(self) -> None:
        class Partial(DataProvider):
            def fetch_historical(self, symbol: str, start: str, end: str) -> AssetData:
                ...

        with pytest.raises(TypeError, match="abstract"):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_can_instantiate(self) -> None:
        class Concrete(DataProvider):
            def fetch_historical(self, symbol: str, start: str, end: str) -> AssetData:
                ...

            def fetch_realtime(self, symbol: str) -> AssetData:
                ...

            def fetch_universe(
                self, symbols: list[str], start: str, end: str
            ) -> dict[str, AssetData]:
                ...

        provider = Concrete()
        assert isinstance(provider, DataProvider)
