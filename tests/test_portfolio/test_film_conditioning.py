"""Tests for src/portfolio/film_conditioning.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.portfolio.film_conditioning import FiLMConditionedAgent, FiLMConditioner
from src.portfolio.iqn_agent import IQNAgent
from src.signals.base import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signal(
    values: list[float],
    confidence: list[float] | None = None,
    regime: str | None = None,
    regime_posterior: list[float] | None = None,
) -> Signal:
    vals = np.array(values, dtype=float)
    conf = np.array(confidence if confidence is not None else [1.0] * len(values))
    metadata: dict = {}
    if regime_posterior is not None:
        metadata["regime_posterior"] = np.array(regime_posterior, dtype=np.float32)
    return Signal(name="test", values=vals, confidence=conf, regime=regime, metadata=metadata)


def _iqn(n_assets: int = 3, **kwargs) -> IQNAgent:
    return IQNAgent(n_assets=n_assets, **kwargs)


def _agent(n_assets: int = 3, n_regimes: int = 3, **kwargs) -> FiLMConditionedAgent:
    return FiLMConditionedAgent(_iqn(n_assets), n_regimes=n_regimes, **kwargs)


def _portfolio(n_assets: int = 3) -> dict[str, float]:
    return {f"asset_{i}": 1.0 / n_assets for i in range(n_assets)}


# ---------------------------------------------------------------------------
# FiLMConditioner: shape
# ---------------------------------------------------------------------------


class TestFiLMConditionerShape:
    def test_output_shape_matches_features(self) -> None:
        film = FiLMConditioner(n_regimes=3, feature_dim=12)
        features = torch.rand(4, 12)
        regime = torch.rand(4, 3)
        out = film(features, regime)
        assert out.shape == (4, 12)

    def test_batch_size_one(self) -> None:
        film = FiLMConditioner(n_regimes=3, feature_dim=9)
        out = film(torch.rand(1, 9), torch.rand(1, 3))
        assert out.shape == (1, 9)

    def test_arbitrary_dims(self) -> None:
        film = FiLMConditioner(n_regimes=5, feature_dim=32)
        out = film(torch.rand(8, 32), torch.rand(8, 5))
        assert out.shape == (8, 32)


# ---------------------------------------------------------------------------
# FiLMConditioner: identity initialisation
# ---------------------------------------------------------------------------


class TestFiLMIdentityInit:
    def test_gamma_bias_is_ones(self) -> None:
        film = FiLMConditioner(n_regimes=3, feature_dim=8)
        assert torch.allclose(film.gamma_net.bias, torch.ones(8))

    def test_beta_bias_is_zeros(self) -> None:
        film = FiLMConditioner(n_regimes=3, feature_dim=8)
        assert torch.allclose(film.beta_net.bias, torch.zeros(8))

    def test_gamma_weight_is_zeros(self) -> None:
        film = FiLMConditioner(n_regimes=3, feature_dim=8)
        assert torch.allclose(film.gamma_net.weight, torch.zeros(8, 3))

    def test_beta_weight_is_zeros(self) -> None:
        film = FiLMConditioner(n_regimes=3, feature_dim=8)
        assert torch.allclose(film.beta_net.weight, torch.zeros(8, 3))

    def test_output_equals_input_at_init(self) -> None:
        """With identity init, FiLM(features, any_regime) == features."""
        film = FiLMConditioner(n_regimes=3, feature_dim=6)
        features = torch.rand(4, 6)
        # Try several regime inputs — should all produce the same output = features.
        for regime in [
            torch.tensor([[1.0, 0.0, 0.0]] * 4),
            torch.tensor([[0.0, 0.5, 0.5]] * 4),
            torch.rand(4, 3),
        ]:
            with torch.no_grad():
                out = film(features, regime)
            assert torch.allclose(out, features, atol=1e-6), (
                f"Identity init failed for regime {regime[0].tolist()}"
            )

    def test_output_equals_input_single(self) -> None:
        film = FiLMConditioner(n_regimes=3, feature_dim=9)
        x = torch.tensor([[1.0, -2.0, 0.5, 3.0, 0.0, 1.5, -1.0, 0.25, 0.75]])
        regime = torch.tensor([[0.8, 0.15, 0.05]])
        with torch.no_grad():
            out = film(x, regime)
        assert torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# FiLMConditioner: different regimes produce different outputs
# ---------------------------------------------------------------------------


class TestFiLMRegimeDifference:
    def _non_identity_film(self, n_regimes: int = 3, feature_dim: int = 6) -> FiLMConditioner:
        """Return a FiLMConditioner with non-zero, non-uniform weights."""
        film = FiLMConditioner(n_regimes=n_regimes, feature_dim=feature_dim)
        with torch.no_grad():
            torch.manual_seed(42)
            nn.init.kaiming_uniform_(film.gamma_net.weight)
            nn.init.kaiming_uniform_(film.beta_net.weight)
        return film

    def test_different_one_hot_regimes_differ(self) -> None:
        film = self._non_identity_film()
        features = torch.ones(1, 6)
        bull = torch.tensor([[1.0, 0.0, 0.0]])
        bear = torch.tensor([[0.0, 0.0, 1.0]])
        with torch.no_grad():
            out_bull = film(features, bull)
            out_bear = film(features, bear)
        assert not torch.allclose(out_bull, out_bear)

    def test_different_soft_posteriors_differ(self) -> None:
        film = self._non_identity_film()
        features = torch.rand(2, 6)
        r1 = torch.tensor([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
        r2 = torch.tensor([[0.05, 0.05, 0.9], [0.05, 0.05, 0.9]])
        with torch.no_grad():
            assert not torch.allclose(film(features, r1), film(features, r2))

    def test_same_regime_same_output(self) -> None:
        film = self._non_identity_film()
        features = torch.rand(3, 6)
        regime = torch.rand(3, 3)
        with torch.no_grad():
            assert torch.allclose(film(features, regime), film(features, regime))

    def test_linear_interpolation_of_regimes(self) -> None:
        """Interpolating between two regimes should interpolate outputs."""
        film = self._non_identity_film(feature_dim=4)
        features = torch.ones(1, 4)
        r1 = torch.tensor([[1.0, 0.0, 0.0]])
        r2 = torch.tensor([[0.0, 0.0, 1.0]])
        r_mid = 0.5 * r1 + 0.5 * r2
        with torch.no_grad():
            out1 = film(features, r1)
            out2 = film(features, r2)
            out_mid = film(features, r_mid)
        expected_mid = 0.5 * out1 + 0.5 * out2
        assert torch.allclose(out_mid, expected_mid, atol=1e-5)


# ---------------------------------------------------------------------------
# FiLMConditionedAgent: decide()
# ---------------------------------------------------------------------------


class TestFiLMConditionedAgentDecide:
    def test_returns_portfolio_action(self) -> None:
        agent = _agent(3)
        sig = _signal([0.1, 0.2, 0.3], regime="bull")
        action = agent.decide([sig], _portfolio(3))
        assert isinstance(action, PortfolioAction)

    def test_weights_sum_to_one(self) -> None:
        agent = _agent(3)
        sig = _signal([0.1, 0.2, 0.3], regime="neutral")
        action = agent.decide([sig], _portfolio(3))
        assert abs(sum(action.weights.values()) - 1.0) < 1e-5

    def test_weights_non_negative(self) -> None:
        agent = _agent(3)
        sig = _signal([0.1, 0.2, 0.3], regime="bear")
        action = agent.decide([sig], _portfolio(3))
        for w in action.weights.values():
            assert w >= 0.0

    def test_confidence_in_unit_interval(self) -> None:
        agent = _agent(3)
        sig = _signal([0.1, 0.2, 0.3], regime="bull")
        action = agent.decide([sig], _portfolio(3))
        assert 0.0 <= action.confidence <= 1.0

    def test_correct_symbols_from_portfolio(self) -> None:
        agent = _agent(3)
        port = {"SPY": 0.4, "QQQ": 0.3, "TLT": 0.3}
        sig = _signal([0.1, 0.2, 0.3])
        action = agent.decide([sig], port)
        assert set(action.weights.keys()) == {"SPY", "QQQ", "TLT"}

    def test_regime_string_propagated(self) -> None:
        agent = _agent(2)
        sig = _signal([0.1, 0.2], regime="bear")
        action = agent.decide([sig], _portfolio(2))
        assert action.regime_context == "bear"

    def test_explicit_posterior_in_metadata(self) -> None:
        """Signal with regime_posterior in metadata should be consumed."""
        agent = _agent(3)
        sig = _signal([0.1, 0.2, 0.3], regime_posterior=[0.8, 0.15, 0.05])
        action = agent.decide([sig], _portfolio(3))
        assert isinstance(action, PortfolioAction)

    def test_posterior_takes_priority_over_string(self) -> None:
        """Explicit posterior metadata beats the regime string."""
        agent = _agent(3)
        # Include both — metadata should win; regime string still used for label.
        sig = _signal(
            [0.1, 0.2, 0.3],
            regime="bull",
            regime_posterior=[0.1, 0.1, 0.8],
        )
        action = agent.decide([sig], _portfolio(3))
        # We can't inspect FiLM directly, but output must be a valid action.
        assert isinstance(action, PortfolioAction)
        assert action.regime_context == "bull"  # label still comes from regime str

    def test_action_passes_validation(self) -> None:
        """PortfolioAction.__post_init__ must not raise."""
        agent = _agent(4)
        sig = _signal([0.1, 0.2, 0.3, 0.4], regime="neutral")
        action = agent.decide([sig], _portfolio(4))
        assert isinstance(action, PortfolioAction)

    def test_single_asset(self) -> None:
        agent = _agent(1, n_regimes=3)
        sig = _signal([0.5], regime="bull")
        action = agent.decide([sig], {"SPY": 1.0})
        assert abs(action.weights["SPY"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Fallback behaviour when no regime signal
# ---------------------------------------------------------------------------


class TestFallback:
    def test_no_regime_in_signal(self) -> None:
        """No regime string and no metadata → uniform fallback, still valid."""
        agent = _agent(3)
        sig = _signal([0.1, 0.2, 0.3])   # regime=None, no metadata
        action = agent.decide([sig], _portfolio(3))
        assert isinstance(action, PortfolioAction)
        assert action.regime_context == "unknown"

    def test_empty_signals(self) -> None:
        agent = _agent(3)
        action = agent.decide([], _portfolio(3))
        assert isinstance(action, PortfolioAction)
        assert abs(sum(action.weights.values()) - 1.0) < 1e-5

    def test_uniform_and_one_hot_same_at_identity_init(self) -> None:
        """Before training, FiLM is identity → uniform posterior == one-hot."""
        torch.manual_seed(0)
        iqn = _iqn(3)
        agent_uniform = FiLMConditionedAgent(iqn, n_regimes=3)
        agent_onehot = FiLMConditionedAgent(iqn, n_regimes=3)

        # Both share the same IQN and have identity FiLM — same tau seed.
        sig_no_regime = _signal([0.1, 0.2, 0.3])
        sig_bull = _signal([0.1, 0.2, 0.3], regime="bull")
        port = _portfolio(3)

        torch.manual_seed(7)
        action_uniform = agent_uniform.decide([sig_no_regime], port)
        torch.manual_seed(7)
        action_onehot = agent_onehot.decide([sig_bull], port)

        for sym in port:
            assert abs(action_uniform.weights[sym] - action_onehot.weights[sym]) < 1e-5

    def test_unknown_regime_string_uniform_fallback(self) -> None:
        """An unrecognised regime string falls back to uniform."""
        agent = _agent(3)
        sig = _signal([0.1, 0.2, 0.3], regime="sideways")  # not in _REGIME_ORDER
        action = agent.decide([sig], _portfolio(3))
        assert isinstance(action, PortfolioAction)

    def test_empty_portfolio(self) -> None:
        agent = _agent(2)
        sig = _signal([0.1, 0.2])
        action = agent.decide([sig], {})
        assert len(action.weights) == 2


# ---------------------------------------------------------------------------
# Regime posterior extraction
# ---------------------------------------------------------------------------


class TestRegimePosteriorExtraction:
    def test_metadata_posterior_normalised(self) -> None:
        """Unnormalised posterior in metadata must be normalised."""
        agent = _agent(3)
        sig = _signal([0.0, 0.0, 0.0], regime_posterior=[4.0, 2.0, 2.0])
        posterior = agent._extract_regime_posterior([sig])
        assert abs(posterior.sum() - 1.0) < 1e-6
        np.testing.assert_allclose(posterior, [0.5, 0.25, 0.25], atol=1e-6)

    def test_bull_onehot(self) -> None:
        agent = _agent(3)
        sig = _signal([0.0, 0.0, 0.0], regime="bull")
        posterior = agent._extract_regime_posterior([sig])
        np.testing.assert_array_equal(posterior, [1.0, 0.0, 0.0])

    def test_neutral_onehot(self) -> None:
        agent = _agent(3)
        posterior = agent._extract_regime_posterior([_signal([0.0], regime="neutral")])
        np.testing.assert_array_equal(posterior, [0.0, 1.0, 0.0])

    def test_bear_onehot(self) -> None:
        agent = _agent(3)
        posterior = agent._extract_regime_posterior([_signal([0.0], regime="bear")])
        np.testing.assert_array_equal(posterior, [0.0, 0.0, 1.0])

    def test_no_regime_uniform(self) -> None:
        agent = _agent(3)
        posterior = agent._extract_regime_posterior([_signal([0.0, 0.0, 0.0])])
        np.testing.assert_allclose(posterior, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_unknown_string_uniform(self) -> None:
        agent = _agent(3)
        posterior = agent._extract_regime_posterior([_signal([0.0], regime="crash")])
        np.testing.assert_allclose(posterior, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_metadata_takes_priority(self) -> None:
        """metadata posterior beats the regime string."""
        agent = _agent(3)
        sig = _signal([0.0], regime="bull", regime_posterior=[0.1, 0.2, 0.7])
        posterior = agent._extract_regime_posterior([sig])
        np.testing.assert_allclose(posterior, [0.1, 0.2, 0.7], atol=1e-5)

    def test_two_regime_setup(self) -> None:
        """n_regimes=2 should still work with bull/neutral labels."""
        agent = _agent(2, n_regimes=2)
        sig = _signal([0.0, 0.0], regime="neutral")
        posterior = agent._extract_regime_posterior([sig])
        assert posterior.shape == (2,)
        assert abs(posterior.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# PortfolioAgent interface compliance
# ---------------------------------------------------------------------------


class TestInterfaceCompliance:
    def test_is_portfolio_agent(self) -> None:
        assert isinstance(_agent(3), PortfolioAgent)

    def test_decide_returns_portfolio_action(self) -> None:
        action = _agent(3).decide([_signal([0.1, 0.2, 0.3])], _portfolio(3))
        assert isinstance(action, PortfolioAction)

    def test_train_returns_dict(self) -> None:
        result = _agent(3).train([], pd.DataFrame())
        assert isinstance(result, dict)

    def test_train_delegates_to_iqn(self) -> None:
        """train() must return a dict with the keys IQNAgent.train() produces."""
        agent = _agent(3)
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(
            rng.normal(0.001, 0.01, (20, 3)),
            columns=["asset_0", "asset_1", "asset_2"],
        )
        signals = [_signal([0.01, 0.02, -0.01]) for _ in range(20)]
        metrics = agent.train(signals, returns)
        assert {"loss", "steps", "buffer_size"} <= metrics.keys()

    def test_film_is_nn_module(self) -> None:
        assert isinstance(_agent(3).film, torch.nn.Module)

    def test_many_assets(self) -> None:
        n = 8
        agent = _agent(n)
        sig = _signal([0.01] * n, regime="bull")
        port = {f"asset_{i}": 1.0 / n for i in range(n)}
        action = agent.decide([sig], port)
        assert len(action.weights) == n
        assert abs(sum(action.weights.values()) - 1.0) < 1e-5
