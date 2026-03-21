"""Multi-factor momentum signal generator.

Computes three momentum factors per asset and combines them into a
composite z-scored signal:

  1. **mom_12_1** — 12-month return minus 1-month return (Jegadeesh &
     Titman cross-sectional momentum, skips the most recent month to
     avoid short-term reversal).
  2. **high_52w** — proximity to the 52-week high (current price /
     52-week high).  Stocks near their high tend to continue rising.
  3. **mom_1m** — 1-month return (short-term reversal indicator; sign
     is *not* flipped, so positive = recent strength).

The composite score is the equal-weighted average of cross-sectionally
z-scored factors.  Confidence reflects how well the three factors agree
on direction.
"""

import logging

import numpy as np
import pandas as pd

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator

logger = logging.getLogger(__name__)

# Minimum trading days required (12 months ≈ 252 days).
_MIN_OBS = 252


class MomentumSignal(SignalGenerator):
    """Multi-factor momentum signal implementing SignalGenerator.

    Args:
        lookback_12m: Trading days for the 12-month window (default 252).
        lookback_1m: Trading days for the 1-month window (default 21).
        lookback_52w: Trading days for the 52-week high (default 252).
    """

    def __init__(
        self,
        lookback_12m: int = 252,
        lookback_1m: int = 21,
        lookback_52w: int = 252,
    ) -> None:
        self._lookback_12m = lookback_12m
        self._lookback_1m = lookback_1m
        self._lookback_52w = lookback_52w
        self._last_factors: dict[str, dict[str, float]] | None = None

    # ------------------------------------------------------------------
    # SignalGenerator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "momentum"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        """Compute momentum factors and return a composite signal.

        Assets with fewer than ``_MIN_OBS`` rows are silently skipped
        with a warning.

        Args:
            data: Mapping of symbol to AssetData.

        Returns:
            Signal with one composite momentum score per eligible asset.

        Raises:
            ValueError: If no assets have sufficient data.
        """
        factors: dict[str, dict[str, float]] = {}

        for symbol, asset in data.items():
            close = asset.ohlcv["close"]
            if len(close) < _MIN_OBS:
                logger.warning(
                    "Skipping %s: only %d observations (need %d)",
                    symbol,
                    len(close),
                    _MIN_OBS,
                )
                continue

            factors[symbol] = self._compute_factors(close)

        if not factors:
            raise ValueError(
                f"No assets have sufficient data (minimum {_MIN_OBS} trading days required)"
            )

        self._last_factors = factors
        return self._build_signal(factors)

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        """Recompute momentum from the latest data slice.

        Momentum factors are cheap to compute, so ``update`` simply
        delegates to ``generate``.

        Args:
            new_data: Mapping of symbol to AssetData.

        Returns:
            Updated Signal.
        """
        return self.generate(new_data)

    # ------------------------------------------------------------------
    # Internal: factor computation
    # ------------------------------------------------------------------

    def _compute_factors(self, close: pd.Series) -> dict[str, float]:
        """Compute the three raw momentum factors for a single asset.

        Args:
            close: Close price series with DatetimeIndex.

        Returns:
            Dict with keys ``mom_12_1``, ``high_52w``, ``mom_1m``.
        """
        # 12-month return minus 1-month return.
        ret_12m = close.iloc[-1] / close.iloc[-self._lookback_12m] - 1.0
        ret_1m = close.iloc[-1] / close.iloc[-self._lookback_1m] - 1.0
        mom_12_1 = ret_12m - ret_1m

        # Proximity to 52-week high.
        high_52w_price = close.iloc[-self._lookback_52w :].max()
        high_52w = close.iloc[-1] / high_52w_price

        # 1-month return.
        mom_1m = ret_1m

        return {"mom_12_1": float(mom_12_1), "high_52w": float(high_52w), "mom_1m": float(mom_1m)}

    # ------------------------------------------------------------------
    # Internal: cross-sectional z-scoring and signal construction
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore_factors(
        factors: dict[str, dict[str, float]],
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Cross-sectionally z-score each factor.

        When there is only one asset the raw values are used (z-score
        is undefined for n=1).

        Args:
            factors: ``{symbol: {factor_name: raw_value}}``.

        Returns:
            Tuple of (zscored, factor_names) where ``zscored`` maps
            symbol to an array of z-scored factor values.
        """
        symbols = list(factors.keys())
        factor_names = list(next(iter(factors.values())).keys())
        n_assets = len(symbols)

        # Build matrix: rows = assets, cols = factors.
        raw = np.array([[factors[s][f] for f in factor_names] for s in symbols])

        if n_assets > 1:
            mean = raw.mean(axis=0)
            std = raw.std(axis=0, ddof=1)
            std[std == 0] = 1.0
            zscored_matrix = (raw - mean) / std
        else:
            # Single asset — normalise each factor to [-1, 1] by its
            # absolute value so the composite is still interpretable.
            max_abs = np.abs(raw).clip(min=1e-10)
            zscored_matrix = raw / max_abs

        result: dict[str, np.ndarray] = {}
        for i, sym in enumerate(symbols):
            result[sym] = zscored_matrix[i]

        return result, factor_names

    def _build_signal(self, factors: dict[str, dict[str, float]]) -> Signal:
        """Build a Signal from the computed factors.

        Args:
            factors: ``{symbol: {factor_name: raw_value}}``.

        Returns:
            Signal with composite scores, confidence, and per-factor
            metadata.
        """
        zscored, factor_names = self._zscore_factors(factors)
        symbols = list(factors.keys())

        composite = np.array([zscored[s].mean() for s in symbols])
        confidences = np.array([self._factor_agreement(zscored[s]) for s in symbols])

        logger.info(
            "Momentum signal: %d assets, composite range [%.3f, %.3f]",
            len(symbols),
            composite.min(),
            composite.max(),
        )

        return Signal(
            name=self.name,
            values=composite,
            confidence=confidences,
            regime=None,
            metadata={
                "symbols": symbols,
                "factor_names": factor_names,
                "raw_factors": factors,
                "zscored_factors": {s: zscored[s].tolist() for s in symbols},
            },
        )

    @staticmethod
    def _factor_agreement(zscored_factors: np.ndarray) -> float:
        """Compute confidence from factor directional agreement.

        When all factors point in the same direction the confidence is
        high.  When they disagree the confidence drops toward 0.

        Args:
            zscored_factors: Array of z-scored factor values for one asset.

        Returns:
            Confidence value in [0, 1].
        """
        signs = np.sign(zscored_factors)
        # Fraction of factors that agree with the majority sign.
        if len(signs) == 0:
            return 0.0
        dominant_sign = np.sign(signs.sum())
        if dominant_sign == 0:
            # Perfect split — low confidence.
            return 0.0
        agreement = np.mean(signs == dominant_sign)
        return float(agreement)
