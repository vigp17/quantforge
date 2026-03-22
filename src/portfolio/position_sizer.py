"""Position sizing algorithms for the portfolio engine.

Provides three sizing methods (Kelly, volatility-scaled, equal-weight) all
governed by a ``max_position`` ceiling so no single name can dominate.

Signal values are assumed to correspond to ``returns_df.columns`` in order.
Positive values indicate a long bias, negative a short bias.  Confidence
values modulate the final weight multiplicatively.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.signals.base import Signal

logger = logging.getLogger(__name__)

_ANNUALISATION_FACTOR = 252  # trading days per year
_MIN_OBS = 2  # minimum rows needed for any volatility estimate


class PositionSizer:
    """Converts signals and return history into target portfolio weights.

    Args:
        max_position: Hard ceiling on any single asset weight (absolute
            value).  Defaults to 0.30.
    """

    def __init__(self, max_position: float = 0.30) -> None:
        if not 0.0 < max_position <= 1.0:
            raise ValueError(f"max_position must be in (0, 1], got {max_position}")
        self.max_position = max_position

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def size(
        self,
        signal: Signal,
        returns_df: pd.DataFrame,
        method: str = "volatility_scaled",
    ) -> dict[str, float]:
        """Compute target weights for every asset in *returns_df*.

        Signal values select direction (positive → long, negative → short or
        zero depending on the method) and signal confidence scales the raw
        weight from the chosen algorithm.

        Args:
            signal: Current signal snapshot.  ``signal.values`` and
                ``signal.confidence`` must have length equal to
                ``len(returns_df.columns)``.
            returns_df: Daily return series, one column per asset.
            method: One of ``"kelly"``, ``"volatility_scaled"``,
                ``"equal_weight"``.

        Returns:
            Symbol-to-weight mapping.  Absolute weights sum to at most 1.0
            and no single weight exceeds ``max_position``.

        Raises:
            ValueError: If *method* is unknown or array lengths mismatch.
        """
        symbols = list(returns_df.columns)
        n = len(symbols)

        if len(signal.values) != n:
            raise ValueError(
                f"signal.values length {len(signal.values)} does not match returns_df columns {n}"
            )

        if method == "kelly":
            raw_weights = self._kelly_weights(symbols, returns_df)
        elif method == "volatility_scaled":
            raw_weights = self.target_vol_weights(returns_df)
        elif method == "equal_weight":
            raw_weights = {sym: 1.0 / n if n > 0 else 0.0 for sym in symbols}
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: "
                "'kelly', 'volatility_scaled', 'equal_weight'."
            )

        # Apply signal direction and confidence.
        weights: dict[str, float] = {}
        for i, sym in enumerate(symbols):
            direction = float(np.sign(signal.values[i])) if signal.values[i] != 0 else 0.0
            conf = float(np.clip(signal.confidence[i], 0.0, 1.0))
            weights[sym] = direction * raw_weights.get(sym, 0.0) * conf

        # Clip and normalise.
        weights = self._clip_and_normalise(weights)
        return weights

    def kelly_fraction(
        self,
        returns: pd.Series,
        half_kelly: float = 0.5,
    ) -> float:
        """Compute the (fractional) Kelly criterion for a single asset.

        Uses the discrete-outcome approximation:
        ``f* = (p * b - q) / b`` where *p* is the win rate, *q = 1 - p*,
        and *b* is the average win / average loss ratio.

        Args:
            returns: Return series for a single asset.
            half_kelly: Fraction of full Kelly to apply (0, 1].
                Defaults to 0.5 (half-Kelly).

        Returns:
            Kelly fraction in [0, max_position], floored at 0 (no shorts
            from Kelly — a negative Kelly signals an unfavourable bet).
        """
        clean = returns.dropna()
        if len(clean) < _MIN_OBS:
            logger.warning("Insufficient returns data for Kelly calculation; returning 0.")
            return 0.0

        wins = clean[clean > 0]
        losses = clean[clean < 0]

        if len(wins) == 0:
            return 0.0

        p = len(wins) / len(clean)
        q = 1.0 - p
        avg_win = float(wins.mean())
        avg_loss = float(losses.abs().mean()) if len(losses) > 0 else avg_win

        if avg_loss == 0:
            logger.warning("Zero average loss in Kelly; returning 0.")
            return 0.0

        b = avg_win / avg_loss
        full_kelly = (p * b - q) / b
        fraction = half_kelly * full_kelly
        result = float(np.clip(fraction, 0.0, self.max_position))
        return result

    def target_vol_weights(
        self,
        returns_df: pd.DataFrame,
        target_vol: float = 0.15,
    ) -> dict[str, float]:
        """Compute inverse-volatility weights scaled to a target portfolio vol.

        Each asset receives a weight proportional to ``1 / sigma`` where
        *sigma* is its annualised daily-return standard deviation.  Weights
        are then scaled so that the equal-correlation-zero portfolio
        approximation hits *target_vol*.

        Args:
            returns_df: Daily return series, one column per asset.
            target_vol: Desired annualised portfolio volatility (0, 1].
                Defaults to 0.15 (15 %).

        Returns:
            Symbol-to-weight mapping with non-negative weights summing to
            at most 1.0, each clipped to ``max_position``.
        """
        symbols = list(returns_df.columns)
        if returns_df.empty or len(returns_df) < _MIN_OBS:
            return {sym: 0.0 for sym in symbols}

        ann_vols: dict[str, float] = {}
        for sym in symbols:
            daily_std = float(returns_df[sym].dropna().std())
            if daily_std <= 0 or np.isnan(daily_std):
                ann_vols[sym] = np.inf
            else:
                ann_vols[sym] = daily_std * np.sqrt(_ANNUALISATION_FACTOR)

        inv_vols = {sym: (1.0 / v if v < np.inf else 0.0) for sym, v in ann_vols.items()}
        total_inv = sum(inv_vols.values())

        if total_inv == 0:
            logger.warning("All assets have zero or infinite volatility; returning zero weights.")
            return {sym: 0.0 for sym in symbols}

        # Normalised inverse-vol weights.
        norm_weights = {sym: iv / total_inv for sym, iv in inv_vols.items()}

        # Implied portfolio vol under zero-correlation assumption:
        # port_var = sum((w_i * sigma_i)^2)
        port_var = sum(
            (norm_weights[sym] * ann_vols[sym]) ** 2 for sym in symbols if ann_vols[sym] < np.inf
        )
        port_vol = np.sqrt(port_var)

        if port_vol <= 0:
            return {sym: 0.0 for sym in symbols}

        scale = target_vol / port_vol
        scaled = {sym: norm_weights[sym] * scale for sym in symbols}
        return self._clip_and_normalise(scaled)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _kelly_weights(
        self,
        symbols: list[str],
        returns_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute per-asset Kelly fractions (half-Kelly by default)."""
        raw: dict[str, float] = {}
        for sym in symbols:
            raw[sym] = self.kelly_fraction(returns_df[sym])

        total = sum(raw.values())
        if total == 0:
            return {sym: 0.0 for sym in symbols}

        # Normalise so weights sum to 1 before signal modulation.
        return {sym: v / total for sym, v in raw.items()}

    def _clip_and_normalise(self, weights: dict[str, float]) -> dict[str, float]:
        """Clip each weight to max_position and scale down if sum > 1.0."""
        clipped = {
            sym: float(np.sign(w)) * min(abs(w), self.max_position) for sym, w in weights.items()
        }
        total_abs = sum(abs(w) for w in clipped.values())
        if total_abs > 1.0:
            scale = 1.0 / total_abs
            clipped = {sym: w * scale for sym, w in clipped.items()}
        return clipped
