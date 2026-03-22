"""FiLM (Feature-wise Linear Modulation) conditioning for the portfolio engine.

Conditions a feature vector on a regime posterior via a learned scale-and-shift:

    output = γ(regime) ⊙ features + β(regime)

Two public classes are provided:

FiLMConditioner
    Pure PyTorch module — use as a building block inside larger networks.

FiLMConditionedAgent
    PortfolioAgent wrapper that applies FiLM to the IQNAgent's state vector
    before the quantile network sees it.  Regime posteriors are sourced from
    signal metadata (key ``"regime_posterior"``) or inferred from the signal's
    ``regime`` string as a one-hot vector, falling back to a uniform prior.

Reference: Perez et al. (2018) "FiLM: Visual Reasoning with a General
           Conditioning Layer", AAAI 2018.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.portfolio.iqn_agent import IQNAgent
from src.signals.base import Signal

logger = logging.getLogger(__name__)

# Canonical ordering used when converting regime label strings to one-hot.
_REGIME_ORDER: tuple[str, ...] = ("bull", "neutral", "bear")


# ---------------------------------------------------------------------------
# FiLM module
# ---------------------------------------------------------------------------


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation.

    Learns per-feature scale (γ) and shift (β) as linear projections of the
    regime posterior:

        γ = gamma_net(regime)   # (batch, feature_dim)
        β = beta_net(regime)    # (batch, feature_dim)
        output = γ ⊙ features + β

    Initialised to the identity transform (γ ≡ 1, β ≡ 0), so at the start
    of training the module is a no-op and regime influence is learned gradually.

    Args:
        n_regimes: Dimensionality of the regime posterior input vector.
        feature_dim: Dimensionality of the feature vector to modulate.
    """

    def __init__(self, n_regimes: int, feature_dim: int) -> None:
        super().__init__()
        self.n_regimes = n_regimes
        self.feature_dim = feature_dim

        self.gamma_net = nn.Linear(n_regimes, feature_dim)
        self.beta_net = nn.Linear(n_regimes, feature_dim)

        # Identity init: γ(regime) ≡ 1, β(regime) ≡ 0 for any input.
        # Weight = 0 means the regime posterior has no effect initially;
        # bias = 1 (gamma) / 0 (beta) makes the transform a no-op.
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(
        self,
        features: torch.Tensor,  # (batch, feature_dim)
        regime: torch.Tensor,  # (batch, n_regimes)
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            features: Feature vectors to condition, shape
                ``(batch, feature_dim)``.
            regime: Regime posterior probabilities, shape
                ``(batch, n_regimes)``.  Typically sums to 1 per row.

        Returns:
            Conditioned feature tensor, same shape as *features*.
        """
        gamma = self.gamma_net(regime)  # (batch, feature_dim)
        beta = self.beta_net(regime)  # (batch, feature_dim)
        return gamma * features + beta


# ---------------------------------------------------------------------------
# FiLM-conditioned portfolio agent
# ---------------------------------------------------------------------------


class FiLMConditionedAgent(PortfolioAgent):
    """IQNAgent wrapped with FiLM regime conditioning.

    Intercepts the IQN's state vector at decide-time, modulates it via
    :class:`FiLMConditioner`, then feeds the conditioned state to the
    quantile network:

        state_raw        = [sig_values | sig_conf | current_weights]
        state_conditioned = FiLM(state_raw, regime_posterior)
        weights           = softmax(mean_τ IQN(state_conditioned, τ))

    Regime posteriors are resolved (in priority order) from:

    1. ``signal.metadata["regime_posterior"]`` — explicit float array of
       length ``n_regimes`` (e.g. from an HMM).
    2. ``signal.regime`` string — converted to a one-hot vector using the
       ordering ``("bull", "neutral", "bear")``.
    3. Uniform distribution ``[1/K, …, 1/K]`` — fallback when no regime
       information is present in any signal.

    Because FiLMConditioner is initialised to identity (γ=1, β=0), the
    fallback and the non-fallback paths produce identical output before
    any training, matching the unconditioned IQN baseline.

    Args:
        iqn_agent: The wrapped :class:`IQNAgent`.
        n_regimes: Number of regime classes.  Defaults to 3.
        feature_dim: State vector width passed through FiLM.  Defaults to
            ``iqn_agent.state_dim``.
    """

    def __init__(
        self,
        iqn_agent: IQNAgent,
        n_regimes: int = 3,
        feature_dim: int | None = None,
    ) -> None:
        self._iqn = iqn_agent
        self.n_regimes = n_regimes
        fd = feature_dim if feature_dim is not None else iqn_agent.state_dim
        self.film = FiLMConditioner(n_regimes=n_regimes, feature_dim=fd)

    # ------------------------------------------------------------------
    # PortfolioAgent interface
    # ------------------------------------------------------------------

    def decide(
        self,
        signals: list[Signal],
        current_portfolio: dict[str, Any],
    ) -> PortfolioAction:
        """Produce FiLM-conditioned portfolio weights.

        Args:
            signals: Current signal snapshots.  Regime posterior is read from
                signal metadata or inferred from the regime label string.
            current_portfolio: Current holdings ``{symbol: weight}``.

        Returns:
            PortfolioAction with softmax-normalised weights and confidence
            derived from quantile spread.
        """
        iqn = self._iqn
        symbols = iqn._infer_symbols(current_portfolio)
        state = iqn._build_state(signals, current_portfolio, symbols)

        regime_posterior = self._extract_regime_posterior(signals)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        regime_t = torch.tensor(regime_posterior, dtype=torch.float32).unsqueeze(0)

        self.film.eval()
        iqn._network.eval()
        with torch.no_grad():
            conditioned = self.film(state_t, regime_t)  # (1, state_dim)
            tau = torch.rand(1, iqn._n_quantiles)
            q_vals, _ = iqn._network(conditioned, tau)  # (1, n_q, n_assets)

        q_vals = q_vals.squeeze(0)  # (n_q, n_assets)
        mean_q = q_vals.mean(dim=0)  # (n_assets,)
        weights_t = F.softmax(mean_q, dim=-1)

        std_q = float(q_vals.std(dim=0).mean().item())
        confidence = float(np.clip(1.0 - std_q, 0.0, 1.0))

        # Renormalise in float64 to avoid float32 precision exceeding 1.0.
        weights_arr = weights_t.numpy().astype(np.float64)
        weights_arr /= weights_arr.sum()
        weights = {sym: float(weights_arr[i]) for i, sym in enumerate(symbols)}

        regime_str = "unknown"
        for sig in signals:
            if sig.regime:
                regime_str = sig.regime
                break

        return PortfolioAction(
            weights=weights,
            confidence=confidence,
            regime_context=regime_str,
        )

    def train(
        self,
        historical_signals: list[Signal],
        returns: pd.DataFrame,
    ) -> dict[str, Any]:
        """Delegate training to the wrapped IQNAgent.

        FiLM parameters are trained jointly with the IQN in the full UARC
        loop; this method exposes the PortfolioAgent contract for offline
        back-testing pipelines.

        Args:
            historical_signals: Historical signal snapshots.
            returns: Asset return series aligned with *historical_signals*.

        Returns:
            Metrics dict forwarded from the inner IQNAgent.
        """
        return self._iqn.train(historical_signals, returns)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_regime_posterior(self, signals: list[Signal]) -> np.ndarray:
        """Extract or infer a normalised regime posterior from signals.

        Args:
            signals: Signal list from the current timestep.

        Returns:
            Float32 array of shape ``(n_regimes,)`` summing to 1.
        """
        n_reg = self.n_regimes

        # Priority 1: explicit posterior stored in signal metadata.
        for sig in signals:
            posterior = sig.metadata.get("regime_posterior")
            if posterior is not None:
                arr = np.asarray(posterior, dtype=np.float32)
                if arr.shape == (n_reg,) and arr.sum() > 0:
                    return arr / arr.sum()

        # Priority 2: regime string → one-hot.
        for sig in signals:
            if sig.regime:
                return self._regime_str_to_onehot(sig.regime)

        # Priority 3: uniform fallback.
        logger.debug("No regime signal found; using uniform posterior.")
        return np.full(n_reg, 1.0 / n_reg, dtype=np.float32)

    def _regime_str_to_onehot(self, regime: str) -> np.ndarray:
        """Convert a regime label to a one-hot vector.

        Args:
            regime: Label such as ``"bull"``, ``"neutral"``, or ``"bear"``.
                Unknown labels fall back to a uniform distribution.

        Returns:
            Float32 array of shape ``(n_regimes,)``.
        """
        n_reg = self.n_regimes
        order = _REGIME_ORDER[:n_reg]
        if regime in order:
            one_hot = np.zeros(n_reg, dtype=np.float32)
            one_hot[order.index(regime)] = 1.0
            return one_hot
        logger.debug("Unknown regime '%s'; using uniform posterior.", regime)
        return np.full(n_reg, 1.0 / n_reg, dtype=np.float32)
