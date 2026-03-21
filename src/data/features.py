"""Feature engineering pipeline.

Computes a standard set of technical features from OHLCV data.
Each feature is implemented as a private method so the set is easy to extend.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.data.base import AssetData

logger = logging.getLogger(__name__)

# All available feature names, in default computation order.
ALL_FEATURES: list[str] = [
    "log_returns",
    "volatility",
    "momentum",
    "rsi",
    "macd",
    "bollinger_width",
    "volume_zscore",
]

# Warmup period (rows) required by each feature.
_WARMUP: dict[str, int] = {
    "log_returns": 1,
    "volatility": 20,
    "momentum": 252,
    "rsi": 14,
    "macd": 33,  # 26-day slow EMA + 9-day signal → ~33 rows
    "bollinger_width": 20,
    "volume_zscore": 20,
}


class FeatureEngineer:
    """Compute technical features from OHLCV data.

    Args:
        nan_handling: How to treat NaN rows from rolling warm-up periods.
            ``"trim"`` drops leading NaN rows (default).
            ``"fill"`` forward-fills then back-fills remaining NaNs.
        normalize: If ``True`` (default), z-score every feature column.
    """

    def __init__(
        self,
        nan_handling: Literal["trim", "fill"] = "trim",
        normalize: bool = True,
    ) -> None:
        self._nan_handling = nan_handling
        self._normalize = normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        data: dict[str, AssetData],
        features_list: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute features for every symbol in *data*.

        Args:
            data: Mapping of symbol to AssetData.
            features_list: Subset of feature names to compute.  When
                ``None``, all features in ``ALL_FEATURES`` are computed.

        Returns:
            DataFrame with a DatetimeIndex and one column per
            ``(symbol, feature)`` pair, using a MultiIndex on columns.

        Raises:
            ValueError: If *features_list* contains unknown names.
        """
        if features_list is None:
            features_list = list(ALL_FEATURES)

        unknown = set(features_list) - set(ALL_FEATURES)
        if unknown:
            raise ValueError(f"Unknown features: {sorted(unknown)}")

        frames: dict[tuple[str, str], pd.Series] = {}
        for symbol, asset in data.items():
            ohlcv = asset.ohlcv.copy()
            for feat_name in features_list:
                method = getattr(self, f"_{feat_name}")
                series = method(ohlcv)
                series.name = feat_name
                frames[(symbol, feat_name)] = series

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames)
        result.columns = pd.MultiIndex.from_tuples(result.columns, names=["symbol", "feature"])

        # NaN handling
        if self._nan_handling == "trim":
            result = result.dropna()
        else:
            result = result.ffill().bfill()

        # Normalisation
        if self._normalize:
            mean = result.mean()
            std = result.std().replace(0, 1)
            result = (result - mean) / std

        return result

    def warmup_rows(self, features_list: list[str] | None = None) -> int:
        """Return the number of leading rows consumed by warm-up.

        Args:
            features_list: Feature subset.  ``None`` means all features.

        Returns:
            Maximum warm-up period across the requested features.
        """
        if features_list is None:
            features_list = list(ALL_FEATURES)
        return max(_WARMUP.get(f, 0) for f in features_list)

    # ------------------------------------------------------------------
    # Individual feature methods
    # ------------------------------------------------------------------

    @staticmethod
    def _log_returns(ohlcv: pd.DataFrame) -> pd.Series:
        """Daily log returns of the close price."""
        return np.log(ohlcv["close"] / ohlcv["close"].shift(1))

    @staticmethod
    def _volatility(ohlcv: pd.DataFrame, window: int = 20) -> pd.Series:
        """Rolling annualised volatility of log returns."""
        log_ret = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        return log_ret.rolling(window).std() * np.sqrt(252)

    @staticmethod
    def _momentum(ohlcv: pd.DataFrame) -> pd.Series:
        """12-month minus 1-month momentum (skip recent month)."""
        ret_12m = ohlcv["close"].pct_change(252)
        ret_1m = ohlcv["close"].pct_change(21)
        return ret_12m - ret_1m

    @staticmethod
    def _rsi(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index (Wilder smoothing)."""
        delta = ohlcv["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # When avg_loss is 0 (all gains), RSI = 100.
        rsi = rsi.fillna(100.0)
        return rsi

    @staticmethod
    def _macd(
        ohlcv: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.Series:
        """MACD histogram (MACD line minus signal line)."""
        ema_fast = ohlcv["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = ohlcv["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    @staticmethod
    def _bollinger_width(ohlcv: pd.DataFrame, window: int = 20) -> pd.Series:
        """Bollinger Band width (upper - lower) / middle."""
        middle = ohlcv["close"].rolling(window).mean()
        std = ohlcv["close"].rolling(window).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        return (upper - lower) / middle

    @staticmethod
    def _volume_zscore(ohlcv: pd.DataFrame, window: int = 20) -> pd.Series:
        """Z-score of volume relative to its rolling window."""
        roll_mean = ohlcv["volume"].rolling(window).mean()
        roll_std = ohlcv["volume"].rolling(window).std().replace(0, 1)
        return (ohlcv["volume"] - roll_mean) / roll_std
