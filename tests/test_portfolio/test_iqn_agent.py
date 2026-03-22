"""Tests for src/portfolio/iqn_agent.py."""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.portfolio.iqn_agent import (
    IQNAgent,
    ReplayBuffer,
    _IQNNetwork,
    _QuantileEmbedding,
    _huber_quantile_loss,
)
from src.signals.base import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(
    values: list[float],
    confidence: list[float] | None = None,
    regime: str | None = None,
) -> Signal:
    vals = np.array(values, dtype=float)
    conf = np.array(confidence if confidence is not None else [1.0] * len(values))
    return Signal(name="test", values=vals, confidence=conf, regime=regime)


def _returns_df(n_assets: int = 3, n_rows: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(rng.normal(0.001, 0.01, (n_rows, n_assets)), columns=symbols)


def _agent(n_assets: int = 3, **kwargs) -> IQNAgent:
    return IQNAgent(n_assets=n_assets, **kwargs)


def _portfolio(n_assets: int = 3) -> dict[str, float]:
    w = 1.0 / n_assets
    return {f"asset_{i}": w for i in range(n_assets)}


# ---------------------------------------------------------------------------
# Quantile embedding
# ---------------------------------------------------------------------------


class TestQuantileEmbedding:
    def test_output_shape(self) -> None:
        emb = _QuantileEmbedding(embedding_dim=64, hidden_dim=128)
        tau = torch.rand(4, 32)
        out = emb(tau)
        assert out.shape == (4, 32, 128)

    def test_different_tau_different_output(self) -> None:
        """Different quantile levels must produce different embeddings."""
        emb = _QuantileEmbedding(embedding_dim=64, hidden_dim=128)
        tau1 = torch.zeros(1, 4)
        tau2 = torch.full((1, 4), 0.5)
        with torch.no_grad():
            out1 = emb(tau1)
            out2 = emb(tau2)
        assert not torch.allclose(out1, out2)

    def test_same_tau_same_output(self) -> None:
        emb = _QuantileEmbedding(embedding_dim=64, hidden_dim=128)
        tau = torch.rand(2, 8)
        with torch.no_grad():
            assert torch.allclose(emb(tau), emb(tau))

    def test_output_non_negative(self) -> None:
        """ReLU output must be ≥ 0."""
        emb = _QuantileEmbedding(embedding_dim=32, hidden_dim=64)
        tau = torch.rand(3, 16)
        with torch.no_grad():
            out = emb(tau)
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# IQN network — forward pass shapes
# ---------------------------------------------------------------------------


class TestIQNNetwork:
    def test_forward_shape(self) -> None:
        net = _IQNNetwork(state_dim=9, n_assets=3, embedding_dim=64, hidden_dim=128)
        state = torch.rand(4, 9)
        tau = torch.rand(4, 32)
        q_vals, tau_out = net(state, tau)
        assert q_vals.shape == (4, 32, 3)
        assert tau_out.shape == (4, 32)

    def test_batch_size_one(self) -> None:
        net = _IQNNetwork(state_dim=6, n_assets=2, embedding_dim=64, hidden_dim=128)
        q_vals, _ = net(torch.rand(1, 6), torch.rand(1, 16))
        assert q_vals.shape == (1, 16, 2)

    def test_different_quantiles_differ(self) -> None:
        """Low and high quantile levels must produce different Q-values."""
        net = _IQNNetwork(state_dim=6, n_assets=2, embedding_dim=64, hidden_dim=128)
        state = torch.rand(1, 6)
        tau = torch.linspace(0.05, 0.95, 9).unsqueeze(0)  # (1, 9)
        with torch.no_grad():
            q_vals, _ = net(state, tau)
        assert not torch.allclose(q_vals[:, 0, :], q_vals[:, -1, :])

    def test_tau_echoed(self) -> None:
        net = _IQNNetwork(state_dim=6, n_assets=2, embedding_dim=64, hidden_dim=128)
        tau = torch.rand(2, 8)
        _, tau_out = net(torch.rand(2, 6), tau)
        assert torch.equal(tau, tau_out)


# ---------------------------------------------------------------------------
# Huber quantile loss
# ---------------------------------------------------------------------------


class TestHuberQuantileLoss:
    def test_zero_when_pred_equals_target_single_quantile(self) -> None:
        """With n_tau=1, the only cross-pair is (pred[0], target[0]), so u=0."""
        x = torch.rand(4, 1, 3)
        tau = torch.rand(4, 1)
        loss = _huber_quantile_loss(x, x.clone(), tau)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_scalar_output(self) -> None:
        pred = torch.rand(4, 8, 3)
        target = torch.rand(4, 8, 3)
        tau = torch.rand(4, 8)
        loss = _huber_quantile_loss(pred, target, tau)
        assert loss.shape == ()

    def test_non_negative(self) -> None:
        pred = torch.rand(4, 8, 3)
        target = torch.rand(4, 8, 3)
        tau = torch.rand(4, 8)
        assert float(_huber_quantile_loss(pred, target, tau)) >= 0.0


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_empty_initially(self) -> None:
        buf = ReplayBuffer(capacity=100, state_dim=9, n_assets=3)
        assert len(buf) == 0

    def test_push_increments_size(self) -> None:
        buf = ReplayBuffer(capacity=100, state_dim=9, n_assets=3)
        state = np.zeros(9, dtype=np.float32)
        action = np.ones(3, dtype=np.float32) / 3
        for _ in range(5):
            buf.push(state, action, 0.0, state, False)
        assert len(buf) == 5

    def test_capacity_cap(self) -> None:
        buf = ReplayBuffer(capacity=10, state_dim=9, n_assets=3)
        state = np.zeros(9, dtype=np.float32)
        action = np.ones(3, dtype=np.float32) / 3
        for _ in range(25):
            buf.push(state, action, 0.0, state, False)
        assert len(buf) == 10

    def test_sample_shapes(self) -> None:
        buf = ReplayBuffer(capacity=100, state_dim=9, n_assets=3)
        state = np.random.rand(9).astype(np.float32)
        action = np.array([0.4, 0.3, 0.3], dtype=np.float32)
        for _ in range(20):
            buf.push(state, action, 0.5, state, False)
        batch = buf.sample(8)
        assert batch["states"].shape == (8, 9)
        assert batch["actions"].shape == (8, 3)
        assert batch["rewards"].shape == (8,)
        assert batch["next_states"].shape == (8, 9)
        assert batch["dones"].shape == (8,)

    def test_sample_raises_when_insufficient(self) -> None:
        buf = ReplayBuffer(capacity=100, state_dim=9, n_assets=3)
        with pytest.raises(ValueError, match="Buffer has"):
            buf.sample(10)

    def test_data_stored_correctly(self) -> None:
        buf = ReplayBuffer(capacity=10, state_dim=3, n_assets=2)
        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action = np.array([0.6, 0.4], dtype=np.float32)
        buf.push(state, action, 0.123, state * 2, True)
        batch = buf.sample(1)
        np.testing.assert_allclose(batch["states"][0], state)
        np.testing.assert_allclose(batch["actions"][0], action)
        assert abs(batch["rewards"][0] - 0.123) < 1e-6
        assert batch["dones"][0] == 1.0

    def test_sample_without_replacement(self) -> None:
        buf = ReplayBuffer(capacity=20, state_dim=3, n_assets=2)
        for i in range(20):
            buf.push(
                np.array([float(i), 0.0, 0.0]),
                np.array([0.5, 0.5]),
                0.0,
                np.zeros(3),
                False,
            )
        batch = buf.sample(10)
        # All sampled states should be distinct (index field)
        assert len(set(batch["states"][:, 0].tolist())) == 10


# ---------------------------------------------------------------------------
# decide()
# ---------------------------------------------------------------------------


class TestDecide:
    def test_returns_portfolio_action(self) -> None:
        action = _agent(3).decide([_signal([0.1, -0.05, 0.0])], _portfolio(3))
        assert isinstance(action, PortfolioAction)

    def test_weights_sum_to_one(self) -> None:
        action = _agent(3).decide([_signal([0.1, 0.2, 0.3])], _portfolio(3))
        assert abs(sum(action.weights.values()) - 1.0) < 1e-5

    def test_weights_non_negative(self) -> None:
        """Softmax ensures all weights ≥ 0."""
        action = _agent(3).decide([_signal([0.1, 0.2, 0.3])], _portfolio(3))
        for w in action.weights.values():
            assert w >= 0.0

    def test_confidence_in_unit_interval(self) -> None:
        action = _agent(3).decide([_signal([0.1, 0.2, 0.3])], _portfolio(3))
        assert 0.0 <= action.confidence <= 1.0

    def test_correct_symbols_from_portfolio(self) -> None:
        port = {"SPY": 0.4, "QQQ": 0.3, "TLT": 0.3}
        action = _agent(3).decide([_signal([0.1, 0.2, 0.3])], port)
        assert set(action.weights.keys()) == {"SPY", "QQQ", "TLT"}

    def test_regime_propagated_from_signal(self) -> None:
        sig = _signal([0.1, 0.2], regime="bull")
        action = _agent(2).decide([sig], _portfolio(2))
        assert action.regime_context == "bull"

    def test_no_regime_defaults_unknown(self) -> None:
        action = _agent(2).decide([_signal([0.1, 0.2])], _portfolio(2))
        assert action.regime_context == "unknown"

    def test_empty_signals_still_valid(self) -> None:
        action = _agent(3).decide([], _portfolio(3))
        assert isinstance(action, PortfolioAction)
        assert abs(sum(action.weights.values()) - 1.0) < 1e-5

    def test_empty_portfolio_uses_generic_names(self) -> None:
        action = _agent(2).decide([_signal([0.1, 0.2])], {})
        assert len(action.weights) == 2
        assert "asset_0" in action.weights
        assert "asset_1" in action.weights

    def test_mismatched_signal_length_skipped(self) -> None:
        """A signal whose length != n_assets must be silently ignored."""
        bad_sig = _signal([0.1])            # length 1, agent has 3 assets
        good_sig = _signal([0.1, 0.2, 0.3])
        action = _agent(3).decide([bad_sig, good_sig], _portfolio(3))
        assert isinstance(action, PortfolioAction)

    def test_single_asset_weight_is_one(self) -> None:
        """softmax over a single logit = 1.0."""
        action = _agent(1).decide([_signal([0.5])], {"SPY": 1.0})
        assert abs(action.weights["SPY"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Untrained agent → roughly uniform weights
# ---------------------------------------------------------------------------


class TestUntrainedUniform:
    def test_single_call_near_uniform(self) -> None:
        """Fresh random network → softmax(≈0) ≈ 1/N per weight."""
        torch.manual_seed(0)
        n = 4
        agent = _agent(n)
        sig = _signal([0.0] * n)
        port = {f"asset_{i}": 1.0 / n for i in range(n)}
        action = agent.decide([sig], port)
        expected = 1.0 / n
        for w in action.weights.values():
            assert abs(w - expected) < 0.15

    def test_average_over_trials_near_uniform(self) -> None:
        """Mean across many tau draws should converge close to 1/N."""
        n = 3
        agent = _agent(n)
        sig = _signal([0.0] * n)
        port = {f"asset_{i}": 1.0 / n for i in range(n)}
        weight_sums = np.zeros(n)
        n_trials = 30
        for _ in range(n_trials):
            action = agent.decide([sig], port)
            weight_sums += np.array(list(action.weights.values()))
        avg = weight_sums / n_trials
        for w in avg:
            assert abs(w - 1.0 / n) < 0.06


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


class TestTrain:
    def test_returns_metrics_dict(self) -> None:
        agent = _agent(3)
        signals = [_signal([0.01, 0.02, -0.01]) for _ in range(20)]
        metrics = agent.train(signals, _returns_df(3, 20))
        assert isinstance(metrics, dict)
        assert {"loss", "steps", "buffer_size"} <= metrics.keys()

    def test_loss_is_finite(self) -> None:
        agent = _agent(3)
        signals = [_signal([0.01, 0.02, -0.01]) for _ in range(30)]
        metrics = agent.train(signals, _returns_df(3, 30))
        assert np.isfinite(metrics["loss"])

    def test_loss_is_non_negative(self) -> None:
        agent = _agent(3)
        signals = [_signal([0.01, 0.02, -0.01]) for _ in range(30)]
        metrics = agent.train(signals, _returns_df(3, 30))
        assert metrics["loss"] >= 0.0

    def test_buffer_grows_after_train(self) -> None:
        agent = _agent(3)
        signals = [_signal([0.01, 0.02, -0.01]) for _ in range(20)]
        agent.train(signals, _returns_df(3, 20))
        assert len(agent._buffer) > 0

    def test_buffer_size_reported_correctly(self) -> None:
        agent = _agent(3)
        signals = [_signal([0.0, 0.0, 0.0]) for _ in range(15)]
        metrics = agent.train(signals, _returns_df(3, 15))
        assert metrics["buffer_size"] == len(agent._buffer)

    def test_empty_inputs_return_zero_loss(self) -> None:
        metrics = _agent(3).train([], pd.DataFrame())
        assert metrics["loss"] == 0.0
        assert metrics["steps"] == 0

    def test_train_twice_accumulates_buffer(self) -> None:
        agent = _agent(3)
        signals = [_signal([0.01, 0.02, -0.01]) for _ in range(15)]
        ret = _returns_df(3, 15)
        agent.train(signals, ret)
        size_after_first = len(agent._buffer)
        agent.train(signals, ret)
        assert len(agent._buffer) >= size_after_first

    def test_gradient_steps_reported(self) -> None:
        agent = _agent(3)
        signals = [_signal([0.0, 0.0, 0.0]) for _ in range(30)]
        metrics = agent.train(signals, _returns_df(3, 30))
        assert metrics["steps"] > 0


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "iqn.pt"
            _agent(3).save_checkpoint(path)
            assert path.exists()

    def test_save_load_roundtrip_weights(self) -> None:
        """Loaded agent must produce identical decide() output for same tau."""
        agent = _agent(3)
        sig = _signal([0.1, 0.2, 0.3])
        port = _portfolio(3)

        torch.manual_seed(42)
        action_before = agent.decide([sig], port)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "iqn.pt"
            agent.save_checkpoint(path)

            agent2 = _agent(3)
            agent2.load_checkpoint(path)

        torch.manual_seed(42)
        action_after = agent2.decide([sig], port)

        for sym in action_before.weights:
            assert abs(action_before.weights[sym] - action_after.weights[sym]) < 1e-6

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _agent(3).load_checkpoint("/nonexistent/path.pt")

    def test_checkpoint_contains_hyperparams(self) -> None:
        agent = _agent(3, n_quantiles=16, hidden_dim=64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "iqn.pt"
            agent.save_checkpoint(path)
            ckpt = torch.load(path, map_location="cpu")
        assert ckpt["n_assets"] == 3
        assert ckpt["n_quantiles"] == 16
        assert ckpt["hidden_dim"] == 64
        assert ckpt["gamma"] == agent.gamma

    def test_target_network_persisted(self) -> None:
        """Target network weights must survive a save/load cycle."""
        agent = _agent(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "iqn.pt"
            agent.save_checkpoint(path)
            ckpt = torch.load(path, map_location="cpu")
        assert "target" in ckpt


# ---------------------------------------------------------------------------
# PortfolioAgent interface compliance
# ---------------------------------------------------------------------------


class TestInterfaceCompliance:
    def test_is_portfolio_agent(self) -> None:
        assert isinstance(_agent(3), PortfolioAgent)

    def test_decide_signature(self) -> None:
        agent = _agent(3)
        action = agent.decide([_signal([0.1, 0.2, 0.3])], _portfolio(3))
        assert isinstance(action, PortfolioAction)

    def test_train_signature(self) -> None:
        result = _agent(3).train([], pd.DataFrame())
        assert isinstance(result, dict)

    def test_action_passes_internal_validation(self) -> None:
        """PortfolioAction.__post_init__ must not raise for any decide() output."""
        agent = _agent(4)
        sig = _signal([0.1, 0.2, 0.3, 0.4])
        action = agent.decide([sig], _portfolio(4))
        # Reaching here without ValueError means validation passed.
        assert isinstance(action, PortfolioAction)

    def test_many_assets(self) -> None:
        n = 10
        agent = _agent(n)
        sig = _signal([0.01] * n)
        port = {f"asset_{i}": 1.0 / n for i in range(n)}
        action = agent.decide([sig], port)
        assert len(action.weights) == n
        assert abs(sum(action.weights.values()) - 1.0) < 1e-5

    def test_invalid_n_assets_raises(self) -> None:
        with pytest.raises(ValueError, match="n_assets"):
            IQNAgent(n_assets=0)
