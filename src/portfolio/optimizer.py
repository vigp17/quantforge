"""Mean-variance portfolio optimizer implementing the PortfolioAgent interface.

Two modes are available:
- ``"mean_variance"`` (default) — classic Markowitz optimisation via SLSQP.
- ``"risk_parity"`` — inverse-volatility weighting that approximates equal
  risk contribution under the zero-correlation assumption.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.signals.base import Signal

logger = logging.getLogger(__name__)

_EPS = 1e-8  # numerical floor for variances


class MeanVarianceOptimizer(PortfolioAgent):
    """Markowitz mean-variance optimiser with an optional risk-parity mode.

    Args:
        returns_df: Historical daily return series, one column per asset.
            Used for covariance estimation.
        risk_aversion: Lambda in ``max w'μ − (λ/2)·w'Σw``.
            Higher values penalise variance more, shifting weight toward
            lower-volatility assets.  Defaults to 1.0.
        lookback: Number of most-recent rows used for covariance estimation.
            Defaults to 60.
        max_position: Upper bound on any single asset weight.  Defaults to 0.30.
        mode: ``"mean_variance"`` or ``"risk_parity"``.  Defaults to
            ``"mean_variance"``.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        risk_aversion: float = 1.0,
        lookback: int = 60,
        max_position: float = 0.30,
        mode: str = "mean_variance",
    ) -> None:
        if mode not in ("mean_variance", "risk_parity"):
            raise ValueError(f"mode must be 'mean_variance' or 'risk_parity', got '{mode}'")
        if risk_aversion <= 0:
            raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
        if not 0.0 < max_position <= 1.0:
            raise ValueError(f"max_position must be in (0, 1], got {max_position}")

        self._returns_df = returns_df
        self.risk_aversion = risk_aversion
        self.lookback = lookback
        self.max_position = max_position
        self.mode = mode

    # ------------------------------------------------------------------
    # PortfolioAgent interface
    # ------------------------------------------------------------------

    def decide(
        self,
        signals: list[Signal],
        current_portfolio: dict[str, Any],
    ) -> PortfolioAction:
        """Produce target allocation weights.

        Expected returns are estimated by aggregating signal values weighted
        by their confidence scores.  Covariance is estimated from the most
        recent ``lookback`` rows of ``returns_df``.

        Args:
            signals: List of signals from the signal engine.  Each signal's
                ``values`` array must have length equal to the number of
                columns in ``returns_df``.  An empty list causes all
                expected returns to default to zero.
            current_portfolio: Current holdings (unused, available for
                subclass extension).

        Returns:
            PortfolioAction with optimised weights.
        """
        symbols = list(self._returns_df.columns)
        n = len(symbols)

        mu = self._aggregate_mu(signals, n)
        sigma = self._estimate_covariance(n)
        regime = self._extract_regime(signals)
        confidence = self._aggregate_confidence(signals, n)

        if self.mode == "risk_parity":
            weights_arr = self._risk_parity(sigma, n)
        else:
            weights_arr = self._mv_optimize(mu, sigma, n)

        weights = {sym: float(w) for sym, w in zip(symbols, weights_arr)}
        logger.debug("MeanVarianceOptimizer (%s) decided weights: %s", self.mode, weights)
        return PortfolioAction(
            weights=weights,
            confidence=confidence,
            regime_context=regime,
            risk_metrics={"risk_aversion": self.risk_aversion, "mode": self.mode},
        )

    def train(
        self,
        historical_signals: list[Signal],
        returns: pd.DataFrame,
    ) -> dict[str, Any]:
        """No-op — mean-variance optimisation requires no training phase.

        Args:
            historical_signals: Unused.
            returns: Unused.

        Returns:
            Empty dict.
        """
        return {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_covariance(self, n: int) -> np.ndarray:
        """Return an (n, n) sample covariance matrix from recent history."""
        recent = self._returns_df.tail(self.lookback).dropna()
        if len(recent) < 2:
            logger.warning(
                "Too few observations (%d) for covariance estimation; using identity.",
                len(recent),
            )
            return np.eye(n) * _EPS

        cov = recent.cov().values
        # Regularise: add small diagonal to guarantee positive-definiteness.
        cov += np.eye(n) * _EPS
        return cov

    def _aggregate_mu(self, signals: list[Signal], n: int) -> np.ndarray:
        """Compute per-asset expected returns as confidence-weighted mean of signals.

        Falls back to zeros when no signals are provided or all confidences are zero.
        """
        if not signals:
            return np.zeros(n)

        total_weight = np.zeros(n)
        weighted_sum = np.zeros(n)

        for sig in signals:
            if len(sig.values) != n:
                logger.warning(
                    "Signal '%s' has length %d, expected %d; skipping.",
                    sig.name,
                    len(sig.values),
                    n,
                )
                continue
            conf = np.clip(sig.confidence, 0.0, 1.0)
            weighted_sum += sig.values * conf
            total_weight += conf

        mask = total_weight > 0
        mu = np.where(mask, weighted_sum / np.where(mask, total_weight, 1.0), 0.0)
        return mu

    def _aggregate_confidence(self, signals: list[Signal], n: int) -> float:
        """Return mean confidence across all signals and assets."""
        if not signals:
            return 0.0
        confs = [float(np.mean(sig.confidence)) for sig in signals if len(sig.confidence) == n]
        return float(np.mean(confs)) if confs else 0.0

    def _extract_regime(self, signals: list[Signal]) -> str:
        """Return the regime string from the first signal that carries one."""
        for sig in signals:
            if sig.regime:
                return sig.regime
        return "unknown"

    def _mv_optimize(self, mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
        """Solve the mean-variance problem via SLSQP.

        Maximises ``w'μ − (λ/2)·w'Σw`` subject to:
          - ``sum(w) <= 1``
          - ``0 <= w_i <= max_position`` for all i
        """
        lam = self.risk_aversion

        def objective(w: np.ndarray) -> float:
            return -(w @ mu - (lam / 2.0) * w @ sigma @ w)

        def grad(w: np.ndarray) -> np.ndarray:
            return -(mu - lam * sigma @ w)

        w0 = np.ones(n) / n * min(1.0, self.max_position * n)
        bounds = [(0.0, self.max_position)] * n
        constraints = [{"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)}]

        result = minimize(
            objective,
            w0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if not result.success:
            logger.warning(
                "MV optimisation did not converge: %s. Returning equal-weight fallback.",
                result.message,
            )
            fallback = np.ones(n) / n
            return np.clip(fallback, 0.0, self.max_position)

        return np.clip(result.x, 0.0, self.max_position)

    def _risk_parity(self, sigma: np.ndarray, n: int) -> np.ndarray:
        """Inverse-volatility weights approximating equal risk contribution.

        Under the zero-correlation assumption each asset's marginal risk
        contribution is proportional to its weight times its variance, so
        setting ``w_i ∝ 1/σ_i`` equalises per-asset risk contributions.
        """
        vols = np.sqrt(np.diag(sigma))
        vols = np.where(vols > _EPS, vols, _EPS)
        inv_vols = 1.0 / vols
        raw = inv_vols / inv_vols.sum()
        clipped = np.clip(raw, 0.0, self.max_position)
        total = clipped.sum()
        if total > 1.0:
            clipped /= total
        return clipped
