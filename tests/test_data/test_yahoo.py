"""Tests for src/data/yahoo.py."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.base import AssetData, REQUIRED_OHLCV_COLUMNS
from src.data.yahoo import YahooFinanceProvider, _normalize_ohlcv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yf_single_ticker_df(symbol: str = "SPY", rows: int = 5) -> pd.DataFrame:
    """Build a DataFrame mimicking yfinance single-ticker output.

    yfinance returns MultiIndex columns: (Price, Ticker).
    """
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(42)
    data = rng.random((rows, 5)) * 100
    columns = pd.MultiIndex.from_tuples(
        [("Close", symbol), ("High", symbol), ("Low", symbol),
         ("Open", symbol), ("Volume", symbol)],
        names=["Price", "Ticker"],
    )
    return pd.DataFrame(data, index=dates, columns=columns)


def _make_yf_multi_ticker_df(
    symbols: list[str], rows: int = 5,
) -> pd.DataFrame:
    """Build a DataFrame mimicking yfinance group_by='ticker' output.

    Columns are MultiIndex: (Ticker, Price).
    """
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(42)
    tuples = []
    for sym in symbols:
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            tuples.append((sym, col))
    columns = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Price"])
    data = rng.random((rows, len(tuples))) * 100
    return pd.DataFrame(data, index=dates, columns=columns)


# ---------------------------------------------------------------------------
# _normalize_ohlcv – unit tests (no network)
# ---------------------------------------------------------------------------

class TestNormalizeOhlcv:
    def test_flattens_single_ticker_multiindex(self) -> None:
        raw = _make_yf_single_ticker_df("SPY")
        df = _normalize_ohlcv(raw, "SPY")
        assert set(df.columns) == {"close", "high", "low", "open", "volume"}
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_flattens_multi_ticker_multiindex(self) -> None:
        raw = _make_yf_multi_ticker_df(["SPY", "QQQ"])
        df = _normalize_ohlcv(raw, "SPY")
        assert set(df.columns) == {"close", "high", "low", "open", "volume"}

    def test_raises_on_empty_dataframe(self) -> None:
        with pytest.raises(ValueError, match="No data returned"):
            _normalize_ohlcv(pd.DataFrame(), "SPY")


# ---------------------------------------------------------------------------
# YahooFinanceProvider – mocked tests (no network)
# ---------------------------------------------------------------------------

class TestYahooProviderMocked:
    def setup_method(self) -> None:
        self.provider = YahooFinanceProvider(rate_limit=0.0)

    @patch("src.data.yahoo.yf.download")
    def test_fetch_historical_returns_asset_data(self, mock_dl) -> None:
        mock_dl.return_value = _make_yf_single_ticker_df("SPY", rows=10)
        asset = self.provider.fetch_historical("SPY", "2024-01-01", "2024-01-15")

        assert isinstance(asset, AssetData)
        assert asset.symbol == "SPY"
        assert set(asset.ohlcv.columns) >= REQUIRED_OHLCV_COLUMNS
        assert isinstance(asset.ohlcv.index, pd.DatetimeIndex)
        assert len(asset.ohlcv) == 10
        assert asset.metadata["source"] == "yahoo"

    @patch("src.data.yahoo.yf.download")
    def test_fetch_historical_raises_on_empty(self, mock_dl) -> None:
        mock_dl.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No data returned"):
            self.provider.fetch_historical("INVALID999", "2024-01-01", "2024-01-15")

    @patch("src.data.yahoo.yf.download")
    def test_fetch_historical_wraps_network_error(self, mock_dl) -> None:
        mock_dl.side_effect = Exception("connection timeout")
        with pytest.raises(ConnectionError, match="connection timeout"):
            self.provider.fetch_historical("SPY", "2024-01-01", "2024-01-15")

    @patch("src.data.yahoo.yf.download")
    def test_fetch_realtime_returns_single_row(self, mock_dl) -> None:
        mock_dl.return_value = _make_yf_single_ticker_df("SPY", rows=5)
        asset = self.provider.fetch_realtime("SPY")

        assert len(asset.ohlcv) == 1
        assert asset.symbol == "SPY"

    @patch("src.data.yahoo.yf.download")
    def test_fetch_universe_returns_dict(self, mock_dl) -> None:
        mock_dl.return_value = _make_yf_multi_ticker_df(["SPY", "QQQ"])
        result = self.provider.fetch_universe(["SPY", "QQQ"], "2024-01-01", "2024-01-15")

        assert isinstance(result, dict)
        assert "SPY" in result
        assert "QQQ" in result
        for asset in result.values():
            assert isinstance(asset, AssetData)
            assert set(asset.ohlcv.columns) >= REQUIRED_OHLCV_COLUMNS

    def test_fetch_universe_empty_list(self) -> None:
        result = self.provider.fetch_universe([], "2024-01-01", "2024-01-15")
        assert result == {}

    @patch("src.data.yahoo.yf.download")
    def test_fetch_universe_skips_bad_symbols(self, mock_dl) -> None:
        raw = _make_yf_multi_ticker_df(["SPY"])
        mock_dl.return_value = raw
        # "BAD" isn't in the downloaded data so it should be skipped.
        result = self.provider.fetch_universe(["SPY", "BAD"], "2024-01-01", "2024-01-15")

        assert "SPY" in result
        assert "BAD" not in result

    @patch("src.data.yahoo.yf.download")
    def test_fetch_universe_wraps_network_error(self, mock_dl) -> None:
        mock_dl.side_effect = Exception("network down")
        with pytest.raises(ConnectionError, match="network down"):
            self.provider.fetch_universe(["SPY"], "2024-01-01", "2024-01-15")


# ---------------------------------------------------------------------------
# YahooFinanceProvider – live network tests
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestYahooProviderNetwork:
    """Integration tests that hit the real Yahoo Finance API.

    Skip with: pytest -m "not network"
    """

    def setup_method(self) -> None:
        self.provider = YahooFinanceProvider()

    def test_fetch_historical_spy(self) -> None:
        asset = self.provider.fetch_historical("SPY", "2024-01-02", "2024-01-10")

        assert asset.symbol == "SPY"
        assert isinstance(asset.ohlcv.index, pd.DatetimeIndex)
        assert set(asset.ohlcv.columns) >= REQUIRED_OHLCV_COLUMNS
        assert len(asset.ohlcv) > 0

    def test_fetch_historical_invalid_symbol(self) -> None:
        with pytest.raises((ValueError, ConnectionError)):
            self.provider.fetch_historical("ZZZZZ999XYZ", "2024-01-02", "2024-01-10")

    def test_fetch_realtime_spy(self) -> None:
        asset = self.provider.fetch_realtime("SPY")

        assert asset.symbol == "SPY"
        assert len(asset.ohlcv) == 1
        assert set(asset.ohlcv.columns) >= REQUIRED_OHLCV_COLUMNS

    def test_fetch_universe_multiple(self) -> None:
        result = self.provider.fetch_universe(
            ["SPY", "QQQ"], "2024-01-02", "2024-01-10"
        )
        assert "SPY" in result
        assert "QQQ" in result
        for asset in result.values():
            assert len(asset.ohlcv) > 0
            assert isinstance(asset.ohlcv.index, pd.DatetimeIndex)
