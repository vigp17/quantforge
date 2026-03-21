"""Signal ensemble that combines multiple SignalGenerators.

Collects signals from an arbitrary number of child generators and
fuses them into a single consensus Signal.  Three combination strategies
are supported:

  1. **equal_weight** — simple arithmetic mean of signal values.
  2. **confidence_weighted** — weighted average using each signal's
     per-asset confidence as weights (higher-confidence signals
     contribute more).
  3. **majority_vote** — directional consensus; the combined value
     is +1 / -1 / 0 based on the sign-agreement of children, and
     confidence is the fraction of children that agree.

If a child generator fails, its contribution is silently dropped
(with a warning) so that the ensemble degrades gracefully.
"""

import logging
from typing import Literal

import numpy as np

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator

logger = logging.getLogger(__name__)

CombineMethod = Literal["equal_weight", "confidence_weighted", "majority_vote"]


class SignalEnsemble(SignalGenerator):
    """Combines multiple SignalGenerators into a consensus signal.

    Args:
        generators: Child signal generators to ensemble.
        method: Combination strategy.
    """

    def __init__(
        self,
        generators: list[SignalGenerator],
        method: CombineMethod = "equal_weight",
    ) -> None:
        if not generators:
            raise ValueError("SignalEnsemble requires at least one child generator")
        self._generators = generators
        self._method = method

    # ------------------------------------------------------------------
    # SignalGenerator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ensemble"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        """Run all children and combine their signals.

        Args:
            data: Mapping of symbol to AssetData.

        Returns:
            Combined consensus Signal.

        Raises:
            RuntimeError: If all child generators fail.
        """
        signals = self._collect_signals(data, method="generate")
        return self._combine(signals)

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        """Incrementally update all children and recombine.

        Args:
            new_data: Mapping of symbol to AssetData.

        Returns:
            Updated consensus Signal.

        Raises:
            RuntimeError: If all child generators fail.
        """
        signals = self._collect_signals(new_data, method="update")
        return self._combine(signals)

    # ------------------------------------------------------------------
    # Internal: signal collection
    # ------------------------------------------------------------------

    def _collect_signals(
        self,
        data: dict[str, AssetData],
        method: str,
    ) -> list[Signal]:
        """Call each child generator and collect successful results.

        Args:
            data: Asset data to pass to each child.
            method: ``"generate"`` or ``"update"``.

        Returns:
            List of Signal objects from children that succeeded.

        Raises:
            RuntimeError: If every child fails.
        """
        signals: list[Signal] = []
        for gen in self._generators:
            try:
                fn = getattr(gen, method)
                sig = fn(data)
                signals.append(sig)
                logger.info(
                    "Ensemble: %s produced signal with %d values",
                    gen.name,
                    sig.values.size,
                )
            except Exception:
                logger.warning(
                    "Ensemble: %s failed, skipping",
                    gen.name,
                    exc_info=True,
                )

        if not signals:
            raise RuntimeError("All child signal generators failed")

        return signals

    # ------------------------------------------------------------------
    # Internal: combination
    # ------------------------------------------------------------------

    def _combine(self, signals: list[Signal]) -> Signal:
        """Combine collected signals using the configured method.

        All child signals must have the same ``values`` shape.  If
        shapes differ, signals are aligned by truncating to the
        minimum length.

        Args:
            signals: Non-empty list of child Signal objects.

        Returns:
            Combined Signal.
        """
        # Align to common length (in case children cover different
        # asset subsets).
        min_len = min(s.values.size for s in signals)
        values_stack = np.stack([s.values[:min_len] for s in signals])
        conf_stack = np.stack([s.confidence[:min_len] for s in signals])

        if self._method == "equal_weight":
            combined_values, combined_conf = self._equal_weight(values_stack, conf_stack)
        elif self._method == "confidence_weighted":
            combined_values, combined_conf = self._confidence_weighted(values_stack, conf_stack)
        elif self._method == "majority_vote":
            combined_values, combined_conf = self._majority_vote(values_stack)
        else:
            raise ValueError(f"Unknown combination method: {self._method}")

        # Regime: adopt from the first child that reports one (typically HMM).
        regime = None
        for sig in signals:
            if sig.regime is not None:
                regime = sig.regime
                break

        # Build attribution metadata.
        contributions: dict[str, dict] = {}
        for sig in signals:
            contributions[sig.name] = {
                "values": sig.values[:min_len].tolist(),
                "confidence": sig.confidence[:min_len].tolist(),
                "regime": sig.regime,
            }

        logger.info(
            "Ensemble combined %d signals via %s: values range [%.4f, %.4f]",
            len(signals),
            self._method,
            combined_values.min(),
            combined_values.max(),
        )

        return Signal(
            name=self.name,
            values=combined_values,
            confidence=combined_conf,
            regime=regime,
            metadata={
                "method": self._method,
                "n_signals": len(signals),
                "contributors": [s.name for s in signals],
                "contributions": contributions,
            },
        )

    # ------------------------------------------------------------------
    # Combination strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _equal_weight(values: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simple arithmetic mean.

        Args:
            values: (n_signals, n_assets).
            conf: (n_signals, n_assets).

        Returns:
            Tuple of (combined_values, combined_confidence).
        """
        combined_values = values.mean(axis=0)
        combined_conf = np.clip(conf.mean(axis=0), 0.0, 1.0)
        return combined_values, combined_conf

    @staticmethod
    def _confidence_weighted(values: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted average using per-asset confidence as weights.

        If all confidences for an asset are zero, falls back to equal
        weighting for that asset.

        Args:
            values: (n_signals, n_assets).
            conf: (n_signals, n_assets).

        Returns:
            Tuple of (combined_values, combined_confidence).
        """
        weight_sum = conf.sum(axis=0)  # (n_assets,)
        # Avoid division by zero — fall back to equal weight.
        safe_sum = np.where(weight_sum > 0, weight_sum, 1.0)
        combined_values = (values * conf).sum(axis=0) / safe_sum
        combined_conf = np.clip(conf.mean(axis=0), 0.0, 1.0)
        return combined_values, combined_conf

    @staticmethod
    def _majority_vote(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Directional consensus (sign agreement).

        Args:
            values: (n_signals, n_assets).

        Returns:
            Tuple of (combined_values, agreement_ratio) where
            combined_values is +1/-1/0 and agreement_ratio is
            the fraction of signals agreeing with the majority.
        """
        signs = np.sign(values)  # (n_signals, n_assets)

        # Sum of signs per asset.
        sign_sum = signs.sum(axis=0)  # (n_assets,)
        combined_values = np.sign(sign_sum).astype(np.float64)

        # Agreement ratio: fraction of children matching the majority.
        n_assets = values.shape[1]
        agreement = np.zeros(n_assets, dtype=np.float64)
        for j in range(n_assets):
            if combined_values[j] == 0:
                # Perfect split — zero agreement.
                agreement[j] = 0.0
            else:
                agreement[j] = np.mean(signs[:, j] == combined_values[j])
        agreement = np.clip(agreement, 0.0, 1.0)

        return combined_values, agreement
