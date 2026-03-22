"""Tests for src/portfolio/optimizer.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.portfolio.optimizer import MeanVarianceOptimizer
from src.signals.base import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _returns(
    symbols: list[str],
    n: int = 120,
    means: list[float] | None = None,
    vols: list[float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    means = means or [0.0] * len(symbols)
    vols = vols or [0.01] * len(symbols)
    data = {
        sym: rng.normal(m, v, n)
        for sym, m, v in zip(symbols, means, vols)
    }
    return pd.DataFrame(data)


def _signal(
    values: list[float],
    confidence: list[float] | None = None,
    regime: str | None = None,
) -> Signal:
    vals = np.array(values, dtype=float)
    conf = np.array(confidence if confidence is not None else [1.0] * len(values))
    return Signal(name="test", values=vals, confidence=conf, regime=regime)


def _uniform_signal(n: int, value: float = 0.001, conf: float = 1.0) -> Signal:
    return _signal([value] * n, [conf] * n)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_asset_df() -> pd.DataFrame:
    """A: high return / high vol, B: low return / low vol."""
    return _returns(["A", "B"], n=120, means=[0.002, 0.0005], vols=[0.02, 0.005])


@pytest.fixture()
def four_asset_df() -> pd.DataFrame:
    return _returns(["A", "B", "C", "D"], n=120, vols=[0.01, 0.02, 0.015, 0.008])


# ---------------------------------------------------------------------------
# PortfolioAgent interface compliance
# ---------------------------------------------------------------------------

class TestInterfaceCompliance:
    def test_is_portfolio_agent_subclass(self, two_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(two_asset_df)
        assert isinstance(opt, PortfolioAgent)

    def test_decide_returns_portfolio_action(self, two_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(two_asset_df)
        sig = _uniform_signal(2)
        action = opt.decide([sig], {})
        assert isinstance(action, PortfolioAction)

    def test_train_returns_empty_dict(self, two_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(two_asset_df)
        result = opt.train([], pd.DataFrame())
        assert result == {}

    def test_decide_with_empty_signals(self, two_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(two_asset_df)
        action = opt.decide([], {})
        assert isinstance(action, PortfolioAction)
        # With zero mu, SLSQP will allocate near-zero (all-cash is valid)
        assert sum(abs(w) for w in action.weights.values()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Weight constraints
# ---------------------------------------------------------------------------

class TestWeightConstraints:
    def test_weights_sum_leq_one(self, two_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(two_asset_df)
        sig = _uniform_signal(2)
        action = opt.decide([sig], {})
        assert sum(action.weights.values()) <= 1.0 + 1e-6

    def test_no_weight_exceeds_max_position(self, four_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(four_asset_df, max_position=0.30)
        sig = _uniform_signal(4)
        action = opt.decide([sig], {})
        for w in action.weights.values():
            assert w <= 0.30 + 1e-6

    def test_weights_non_negative(self, two_asset_df: pd.DataFrame) -> None:
        """SLSQP bounds enforce w >= 0 (long-only)."""
        opt = MeanVarianceOptimizer(two_asset_df)
        sig = _uniform_signal(2)
        action = opt.decide([sig], {})
        for w in action.weights.values():
            assert w >= -1e-8

    def test_custom_max_position_respected(self, four_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(four_asset_df, max_position=0.15)
        sig = _uniform_signal(4)
        action = opt.decide([sig], {})
        for w in action.weights.values():
            assert w <= 0.15 + 1e-6


# ---------------------------------------------------------------------------
# Risk-aversion shift
# ---------------------------------------------------------------------------

class TestRiskAversionEffect:
    def test_higher_lambda_reduces_high_vol_weight(self, two_asset_df: pd.DataFrame) -> None:
        """Higher risk aversion should shrink the high-vol asset (A) weight."""
        sig = _uniform_signal(2)

        low_lam = MeanVarianceOptimizer(two_asset_df, risk_aversion=0.5).decide([sig], {})
        high_lam = MeanVarianceOptimizer(two_asset_df, risk_aversion=5.0).decide([sig], {})

        # High lambda penalises variance more → less weight in the risky asset.
        assert high_lam.weights["A"] <= low_lam.weights["A"] + 1e-6

    def test_high_lambda_favours_low_vol_asset(self, two_asset_df: pd.DataFrame) -> None:
        sig = _uniform_signal(2)
        opt = MeanVarianceOptimizer(two_asset_df, risk_aversion=10.0)
        action = opt.decide([sig], {})
        assert action.weights["B"] >= action.weights["A"] - 1e-6


# ---------------------------------------------------------------------------
# Signal as alpha / expected-return input
# ---------------------------------------------------------------------------

class TestSignalAsAlpha:
    def test_higher_signal_favours_that_asset(self, two_asset_df: pd.DataFrame) -> None:
        """Boosting B's signal should push more weight toward B."""
        sig_neutral = _signal([0.001, 0.001])
        sig_b_boosted = _signal([0.001, 0.010])

        opt = MeanVarianceOptimizer(two_asset_df, risk_aversion=1.0)
        neutral = opt.decide([sig_neutral], {})
        boosted = opt.decide([sig_b_boosted], {})

        assert boosted.weights["B"] >= neutral.weights["B"] - 1e-6

    def test_zero_confidence_signal_ignored(self, two_asset_df: pd.DataFrame) -> None:
        sig_zero_conf = _signal([1.0, 1.0], confidence=[0.0, 0.0])
        sig_full_conf = _signal([1.0, 1.0], confidence=[1.0, 1.0])

        opt = MeanVarianceOptimizer(two_asset_df, risk_aversion=1.0)
        zero_action = opt.decide([sig_zero_conf], {})
        full_action = opt.decide([sig_full_conf], {})

        # Both valid; zero-confidence is treated as zero-alpha
        assert isinstance(zero_action, PortfolioAction)
        assert sum(full_action.weights.values()) >= sum(zero_action.weights.values()) - 1e-6

    def test_multiple_signals_aggregated(self, two_asset_df: pd.DataFrame) -> None:
        sig1 = _signal([0.001, 0.001], confidence=[1.0, 1.0])
        sig2 = _signal([0.005, 0.001], confidence=[1.0, 1.0])

        opt = MeanVarianceOptimizer(two_asset_df, risk_aversion=1.0)
        action = opt.decide([sig1, sig2], {})
        assert isinstance(action, PortfolioAction)

    def test_mismatched_signal_length_skipped(self, two_asset_df: pd.DataFrame) -> None:
        bad_sig = _signal([0.001])  # length 1, but 2 assets
        good_sig = _signal([0.001, 0.001])

        opt = MeanVarianceOptimizer(two_asset_df)
        # Bad signal should be skipped; falls back to zero-alpha from it.
        action = opt.decide([bad_sig, good_sig], {})
        assert isinstance(action, PortfolioAction)

    def test_regime_propagated_from_signal(self, two_asset_df: pd.DataFrame) -> None:
        sig = _signal([0.001, 0.001], regime="bear")
        opt = MeanVarianceOptimizer(two_asset_df)
        action = opt.decide([sig], {})
        assert action.regime_context == "bear"

    def test_no_regime_defaults_unknown(self, two_asset_df: pd.DataFrame) -> None:
        sig = _signal([0.001, 0.001])  # regime=None
        opt = MeanVarianceOptimizer(two_asset_df)
        action = opt.decide([sig], {})
        assert action.regime_context == "unknown"

    def test_confidence_sets_action_confidence(self, two_asset_df: pd.DataFrame) -> None:
        sig = _signal([0.001, 0.001], confidence=[0.8, 0.6])
        opt = MeanVarianceOptimizer(two_asset_df)
        action = opt.decide([sig], {})
        assert 0.0 <= action.confidence <= 1.0


# ---------------------------------------------------------------------------
# Two-asset known behaviour
# ---------------------------------------------------------------------------

class TestTwoAssetBehaviour:
    def test_high_return_high_vol_vs_low_return_low_vol(
        self, two_asset_df: pd.DataFrame
    ) -> None:
        """With balanced risk aversion the optimizer should hold both assets."""
        sig = _signal([0.002, 0.0005])  # mirrors the true means
        opt = MeanVarianceOptimizer(two_asset_df, risk_aversion=1.0)
        action = opt.decide([sig], {})
        # Both assets should receive positive weight
        assert action.weights["A"] > 0.0
        assert action.weights["B"] > 0.0

    def test_very_high_signal_for_one_asset(self, two_asset_df: pd.DataFrame) -> None:
        """Dominant alpha drives the favoured asset to max_position."""
        sig = _signal([1.0, 0.0])  # massive signal for A, zero for B
        opt = MeanVarianceOptimizer(two_asset_df, risk_aversion=0.1, max_position=0.30)
        action = opt.decide([sig], {})
        # A should be clipped at the position ceiling.
        assert action.weights["A"] == pytest.approx(0.30, abs=1e-4)


# ---------------------------------------------------------------------------
# Risk-parity mode
# ---------------------------------------------------------------------------

class TestRiskParity:
    def test_low_vol_asset_gets_higher_weight(self, two_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(two_asset_df, mode="risk_parity")
        sig = _uniform_signal(2)
        action = opt.decide([sig], {})
        # B has ~4× lower vol → should have higher weight
        assert action.weights["B"] > action.weights["A"]

    def test_equal_vol_equal_weight(self) -> None:
        df = _returns(["X", "Y"], vols=[0.01, 0.01])
        opt = MeanVarianceOptimizer(df, mode="risk_parity")
        sig = _uniform_signal(2)
        action = opt.decide([sig], {})
        assert action.weights["X"] == pytest.approx(action.weights["Y"], rel=0.05)

    def test_risk_contributions_roughly_equal(self, four_asset_df: pd.DataFrame) -> None:
        """Under zero-correlation assumption risk contributions should be equal."""
        opt = MeanVarianceOptimizer(four_asset_df, mode="risk_parity")
        sig = _uniform_signal(4)
        action = opt.decide([sig], {})

        symbols = list(four_asset_df.columns)
        w = np.array([action.weights[s] for s in symbols])
        vols = four_asset_df.tail(60).std().values

        # Risk contribution = w_i * sigma_i  (under zero-correlation)
        rc = w * vols
        held = rc[rc > 1e-9]
        if len(held) > 1:
            # Coefficient of variation of risk contributions should be small
            assert held.std() / held.mean() < 0.25

    def test_weights_sum_leq_one(self, four_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(four_asset_df, mode="risk_parity")
        sig = _uniform_signal(4)
        action = opt.decide([sig], {})
        assert sum(action.weights.values()) <= 1.0 + 1e-6

    def test_no_weight_exceeds_max_position(self, four_asset_df: pd.DataFrame) -> None:
        opt = MeanVarianceOptimizer(four_asset_df, mode="risk_parity", max_position=0.30)
        sig = _uniform_signal(4)
        action = opt.decide([sig], {})
        for w in action.weights.values():
            assert w <= 0.30 + 1e-6

    def test_single_asset_risk_parity(self) -> None:
        df = _returns(["SPY"])
        opt = MeanVarianceOptimizer(df, mode="risk_parity")
        sig = _uniform_signal(1)
        action = opt.decide([sig], {})
        assert action.weights["SPY"] == pytest.approx(
            min(1.0, opt.max_position), abs=1e-6
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_asset_mv(self) -> None:
        df = _returns(["SPY"])
        opt = MeanVarianceOptimizer(df, risk_aversion=1.0)
        sig = _uniform_signal(1, value=0.001)
        action = opt.decide([sig], {})
        assert 0.0 <= action.weights["SPY"] <= opt.max_position + 1e-6

    def test_all_zero_returns_does_not_raise(self) -> None:
        df = pd.DataFrame({"A": np.zeros(60), "B": np.zeros(60)})
        opt = MeanVarianceOptimizer(df)
        sig = _uniform_signal(2)
        action = opt.decide([sig], {})
        assert isinstance(action, PortfolioAction)

    def test_very_short_history_uses_identity_fallback(self) -> None:
        df = _returns(["A", "B"], n=1)  # only 1 row — too few for cov
        opt = MeanVarianceOptimizer(df)
        sig = _uniform_signal(2)
        action = opt.decide([sig], {})
        assert isinstance(action, PortfolioAction)

    def test_lookback_longer_than_history(self) -> None:
        df = _returns(["A", "B"], n=30)
        opt = MeanVarianceOptimizer(df, lookback=200)
        sig = _uniform_signal(2)
        action = opt.decide([sig], {})
        assert isinstance(action, PortfolioAction)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_invalid_mode_raises(self, two_asset_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            MeanVarianceOptimizer(two_asset_df, mode="unknown")

    def test_non_positive_risk_aversion_raises(self, two_asset_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="risk_aversion"):
            MeanVarianceOptimizer(two_asset_df, risk_aversion=0.0)

    def test_invalid_max_position_raises(self, two_asset_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="max_position"):
            MeanVarianceOptimizer(two_asset_df, max_position=1.5)
