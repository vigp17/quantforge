"""Implicit Quantile Network (IQN) portfolio agent.

Implements the PortfolioAgent interface using distributional reinforcement
learning.  The agent learns a mapping from market state to a distribution
over per-asset Q-values, represented by its quantile function.

Architecture
-----------
State encoder     : Linear(state_dim, hidden_dim) → ReLU
Quantile embedding: cos(π·i·τ) for i=1..embedding_dim → Linear → ReLU
Interaction       : state_feat ⊙ tau_emb
Output head       : Linear(hidden_dim) → ReLU → Linear(n_assets)

The untrained agent produces roughly uniform weights because softmax over
near-zero logits (random network outputs centred at zero) ≈ 1/N.

State vector (default, length = 3 × n_assets)
--------------------------------------------
[signal_values | signal_confidence | current_weights]
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.signals.base import Signal

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64  # default mini-batch size for gradient steps


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer.

    Args:
        capacity: Maximum number of transitions to store.
        state_dim: Flat state vector length.
        n_assets: Action dimensionality (portfolio weight vector length).
    """

    def __init__(self, capacity: int, state_dim: int, n_assets: int) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.n_assets = n_assets
        self._pos: int = 0
        self._size: int = 0

        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, n_assets), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer (overwrites oldest when full)."""
        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._next_states[self._pos] = next_state
        self._dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to draw.

        Returns:
            Dict with keys ``states``, ``actions``, ``rewards``,
            ``next_states``, ``dones``.

        Raises:
            ValueError: If fewer than *batch_size* experiences are stored.
        """
        if self._size < batch_size:
            raise ValueError(f"Buffer has {self._size} experiences; {batch_size} requested.")
        idx = np.random.choice(self._size, batch_size, replace=False)
        return {
            "states": self._states[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_states": self._next_states[idx],
            "dones": self._dones[idx],
        }

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------


class _QuantileEmbedding(nn.Module):
    """Cosine basis quantile embedding.

    Maps τ ∈ [0, 1] → ``hidden_dim``-dimensional vector via
    ``φ(τ) = ReLU(Linear([cos(π·i·τ)]_{i=1}^{embedding_dim}))``.

    Args:
        embedding_dim: Number of cosine basis functions.
        hidden_dim: Output dimension (must match state encoder output).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        i_vals = torch.arange(1, embedding_dim + 1, dtype=torch.float32) * math.pi
        self.register_buffer("i_values", i_vals)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Embed quantile levels.

        Args:
            tau: Sampled quantile levels, shape ``(batch, n_quantiles)``.

        Returns:
            Quantile embeddings, shape ``(batch, n_quantiles, hidden_dim)``.
        """
        # (B, n_q, 1) * (emb_dim,) → (B, n_q, emb_dim)
        cos_feat = torch.cos(tau.unsqueeze(-1) * self.i_values)
        return F.relu(self.linear(cos_feat))  # (B, n_q, hidden_dim)


class _IQNNetwork(nn.Module):
    """IQN forward model.

    Args:
        state_dim: Flat input state dimensionality.
        n_assets: Number of assets (output width per quantile).
        embedding_dim: Cosine basis size for quantile embedding.
        hidden_dim: Hidden width throughout.
    """

    def __init__(
        self,
        state_dim: int,
        n_assets: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.quantile_embedding = _QuantileEmbedding(embedding_dim, hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
        )

    def forward(
        self,
        state: torch.Tensor,  # (batch, state_dim)
        tau: torch.Tensor,  # (batch, n_quantiles)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-value quantiles.

        Returns:
            Tuple of:
              - quantile_values ``(batch, n_quantiles, n_assets)``
              - tau (echoed for convenience when computing loss)
        """
        state_feat = self.state_encoder(state)  # (B, hidden)
        tau_emb = self.quantile_embedding(tau)  # (B, n_q, hidden)
        combined = state_feat.unsqueeze(1) * tau_emb  # (B, n_q, hidden)
        q_values = self.output_head(combined)  # (B, n_q, n_assets)
        return q_values, tau


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def _huber_quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Huber quantile regression loss (IQN / QR-DQN variant).

    Computes ρ_τ(u) = |τ − I(u < 0)| · huber_κ(u) averaged over all
    (prediction quantile, target quantile) cross-pairs.

    Args:
        pred: ``(batch, n_tau_pred, n_assets)`` — predicted quantile values.
        target: ``(batch, n_tau_target, n_assets)`` — target quantile values.
        tau: ``(batch, n_tau_pred)`` — quantile levels corresponding to *pred*.
        kappa: Huber threshold δ.  Defaults to 1.0.

    Returns:
        Scalar loss.
    """
    # Residuals over cross-product of pred and target quantile indices.
    # u[b, p, t, a] = target[b, t, a] − pred[b, p, a]
    u = target.unsqueeze(1) - pred.unsqueeze(2)  # (B, n_p, n_t, n_a)

    huber = torch.where(
        u.abs() <= kappa,
        0.5 * u.pow(2),
        kappa * (u.abs() - 0.5 * kappa),
    )

    tau_exp = tau.unsqueeze(2).unsqueeze(3)  # (B, n_p, 1, 1)
    indicator = (u.detach() < 0).float()
    weight = (tau_exp - indicator).abs()

    return (weight * huber).mean(dim=2).mean()


# ---------------------------------------------------------------------------
# IQN Agent
# ---------------------------------------------------------------------------


class IQNAgent(PortfolioAgent):
    """Distributional RL portfolio agent backed by an Implicit Quantile Network.

    Learns a full return distribution per asset via quantile regression,
    enabling uncertainty-aware allocation.  A freshly initialised (untrained)
    agent returns roughly uniform weights because softmax over near-zero
    logits ≈ 1/N.

    Args:
        n_assets: Number of assets in the portfolio.
        state_dim: Flat state vector length fed to the network.  Defaults to
            ``3 * n_assets`` (signal values | signal confidence | weights).
        embedding_dim: Cosine basis functions for quantile embedding (64).
        hidden_dim: Hidden width for all layers (128).
        n_quantiles: Quantile samples per forward pass (32).
        lr: Adam learning rate (1e-4).
        gamma: Temporal discount factor (0.99).
        tau_target: Polyak coefficient for soft target-network update (0.005).
        buffer_capacity: Maximum replay buffer capacity (10 000).
    """

    def __init__(
        self,
        n_assets: int,
        state_dim: int | None = None,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_quantiles: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau_target: float = 0.005,
        buffer_capacity: int = 10_000,
    ) -> None:
        if n_assets < 1:
            raise ValueError(f"n_assets must be >= 1, got {n_assets}")

        self.n_assets = n_assets
        self.state_dim = state_dim if state_dim is not None else 3 * n_assets
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self._n_quantiles = n_quantiles
        self.gamma = gamma
        self._tau_target = tau_target

        self._network = _IQNNetwork(self.state_dim, n_assets, embedding_dim, hidden_dim)
        self._target = _IQNNetwork(self.state_dim, n_assets, embedding_dim, hidden_dim)
        self._target.load_state_dict(self._network.state_dict())
        for p in self._target.parameters():
            p.requires_grad_(False)

        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        self._buffer = ReplayBuffer(buffer_capacity, self.state_dim, n_assets)

    # ------------------------------------------------------------------
    # PortfolioAgent interface
    # ------------------------------------------------------------------

    def decide(
        self,
        signals: list[Signal],
        current_portfolio: dict[str, Any],
    ) -> PortfolioAction:
        """Produce target weights by averaging Q-value quantiles.

        Expected returns are estimated by taking the mean across quantile
        samples and passing through softmax to produce a valid simplex weight
        vector.  Confidence is derived from the inverse of quantile spread.

        Args:
            signals: Current signal snapshots.  Values must have length
                ``n_assets``; mismatched signals are silently skipped.
            current_portfolio: Current holdings ``{symbol: weight}``.
                Symbols are inferred from this dict; generic names
                (``"asset_i"``) are used when it is empty.

        Returns:
            PortfolioAction with softmax-normalised weights.
        """
        symbols = self._infer_symbols(current_portfolio)
        state = self._build_state(signals, current_portfolio, symbols)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, S)

        self._network.eval()
        with torch.no_grad():
            tau = torch.rand(1, self._n_quantiles)
            q_vals, _ = self._network(state_t, tau)  # (1, n_q, n_assets)

        q_vals = q_vals.squeeze(0)  # (n_q, n_assets)
        mean_q = q_vals.mean(dim=0)  # (n_assets,)
        weights_t = F.softmax(mean_q, dim=-1)  # (n_assets,) sums to 1

        # Confidence = 1 − mean std across quantiles, clamped to [0, 1].
        std_q = float(q_vals.std(dim=0).mean().item())
        confidence = float(np.clip(1.0 - std_q, 0.0, 1.0))

        # Renormalize in float64 to avoid float32 rounding pushing the sum
        # fractionally above 1.0 and tripping PortfolioAction validation.
        weights_arr = weights_t.numpy().astype(np.float64)
        weights_arr /= weights_arr.sum()
        weights = {sym: float(weights_arr[i]) for i, sym in enumerate(symbols)}

        regime = "unknown"
        for sig in signals:
            if sig.regime:
                regime = sig.regime
                break

        return PortfolioAction(
            weights=weights,
            confidence=confidence,
            regime_context=regime,
        )

    def train(
        self,
        historical_signals: list[Signal],
        returns: pd.DataFrame,
    ) -> dict[str, Any]:
        """Offline RL training from historical signals and returns.

        Simulates through the signal history collecting transitions, then
        runs Huber quantile regression gradient steps on sampled mini-batches.

        Args:
            historical_signals: Sequence of historical Signal snapshots.
            returns: Daily return series aligned with *historical_signals*.
                Columns correspond to assets (in order).

        Returns:
            Dict with keys:

            - ``loss`` — mean Huber quantile loss over gradient steps.
            - ``steps`` — number of gradient steps taken.
            - ``buffer_size`` — current replay buffer occupancy.
        """
        if not historical_signals or returns.empty:
            return {"loss": 0.0, "steps": 0, "buffer_size": len(self._buffer)}

        n = self.n_assets
        symbols = list(returns.columns)[:n]
        horizon = min(len(historical_signals), len(returns))

        prev_weights = np.ones(n, dtype=np.float32) / n

        # Collect transitions by simulating through history.
        for t in range(horizon):
            sig = historical_signals[t]
            port = {sym: float(prev_weights[i]) for i, sym in enumerate(symbols)}
            state = self._build_state([sig], port, symbols)

            with torch.no_grad():
                state_t = torch.tensor(state).unsqueeze(0)
                tau = torch.rand(1, self._n_quantiles)
                q_vals, _ = self._target(state_t, tau)
                weights_t = F.softmax(q_vals.squeeze(0).mean(0), dim=-1).numpy()

            ret_row = returns.iloc[t].values[:n].astype(np.float32)
            reward = float(np.dot(weights_t, ret_row))

            if t + 1 < horizon:
                next_port = {sym: float(weights_t[i]) for i, sym in enumerate(symbols)}
                next_state = self._build_state([historical_signals[t + 1]], next_port, symbols)
            else:
                next_state = state.copy()

            self._buffer.push(state, weights_t, reward, next_state, t == horizon - 1)
            prev_weights = weights_t

        # Gradient steps — roughly one per 10 observations.
        batch_size = min(_BATCH_SIZE, len(self._buffer))
        n_steps = max(1, horizon // 10)
        total_loss = 0.0
        steps = 0

        if len(self._buffer) >= batch_size:
            for _ in range(n_steps):
                total_loss += self._gradient_step(batch_size)
                steps += 1
                self._soft_update()

        return {
            "loss": float(total_loss / max(steps, 1)),
            "steps": steps,
            "buffer_size": len(self._buffer),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | pathlib.Path) -> None:
        """Persist network weights, target weights, and optimiser state.

        Args:
            path: Destination ``.pt`` file path.
        """
        path = pathlib.Path(path)
        torch.save(
            {
                "network": self._network.state_dict(),
                "target": self._target.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "n_assets": self.n_assets,
                "state_dim": self.state_dim,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "n_quantiles": self._n_quantiles,
                "gamma": self.gamma,
                "tau_target": self._tau_target,
            },
            path,
        )
        logger.info("IQNAgent checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | pathlib.Path) -> None:
        """Restore network weights from a checkpoint file.

        Args:
            path: A ``.pt`` file previously created by :meth:`save_checkpoint`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu")
        self._network.load_state_dict(ckpt["network"])
        self._target.load_state_dict(ckpt["target"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        logger.info("IQNAgent checkpoint loaded from %s", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gradient_step(self, batch_size: int) -> float:
        """One mini-batch Huber quantile regression gradient step."""
        batch = self._buffer.sample(batch_size)

        states = torch.tensor(batch["states"])
        rewards = torch.tensor(batch["rewards"])
        next_states = torch.tensor(batch["next_states"])
        dones = torch.tensor(batch["dones"])

        n_q = self._n_quantiles

        self._network.train()
        tau = torch.rand(batch_size, n_q)
        q_vals, tau = self._network(states, tau)  # (B, n_q, n_assets)

        with torch.no_grad():
            tau_next = torch.rand(batch_size, n_q)
            q_next, _ = self._target(next_states, tau_next)  # (B, n_q, n_assets)
            targets = rewards.view(-1, 1, 1) + self.gamma * q_next * (1.0 - dones).view(
                -1, 1, 1
            )  # (B, n_q, n_assets)

        loss = _huber_quantile_loss(q_vals, targets, tau)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
        self._optimizer.step()

        return float(loss.item())

    def _soft_update(self) -> None:
        """Polyak-average online → target network: θ′ ← τθ + (1−τ)θ′."""
        tau = self._tau_target
        for p, tp in zip(self._network.parameters(), self._target.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def _infer_symbols(self, current_portfolio: dict[str, Any]) -> list[str]:
        """Derive ordered symbol list from the portfolio dict."""
        syms = list(current_portfolio.keys())[: self.n_assets] if current_portfolio else []
        while len(syms) < self.n_assets:
            syms.append(f"asset_{len(syms)}")
        return syms

    def _build_state(
        self,
        signals: list[Signal],
        current_portfolio: dict[str, float],
        symbols: list[str],
    ) -> np.ndarray:
        """Construct the flat state vector ``[sig_values | sig_conf | weights]``.

        Signals with mismatched length are silently skipped.  Falls back to
        zeros for signal features and uniform weights when inputs are empty.
        """
        n = self.n_assets
        sig_values = np.zeros(n, dtype=np.float32)
        total_conf = np.zeros(n, dtype=np.float32)
        n_valid = 0

        for sig in signals:
            if len(sig.values) != n:
                continue
            c = np.clip(sig.confidence, 0.0, 1.0).astype(np.float32)
            sig_values += sig.values.astype(np.float32) * c
            total_conf += c
            n_valid += 1

        mask = total_conf > 0
        sig_values = np.where(mask, sig_values / np.where(mask, total_conf, 1.0), 0.0)
        sig_conf = total_conf / max(n_valid, 1)

        curr_weights = np.array(
            [float(current_portfolio.get(sym, 1.0 / n)) for sym in symbols],
            dtype=np.float32,
        )

        return np.concatenate([sig_values, sig_conf, curr_weights])
