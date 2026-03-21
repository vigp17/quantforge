"""Tests for src/signals/montecarlo.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator
from src.signals.montecarlo import MonteCarloSignal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_asset(
    symbol: str,
    periods: int = 300,
    drift: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
) -> AssetData:
    """Build an AssetData with a controllable drift and volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=periods)
    log_returns = drift + vol * rng.standard_normal(periods)
    price = 100.0 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": np.full(periods, 1_000_000.0),
        },
        index=dates,
    )
    return AssetData(symbol=symbol, ohlcv=df)


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------


class TestInterface:
    def test_is_signal_generator(self) -> None:
        gen = MonteCarloSignal()
        assert isinstance(gen, SignalGenerator)

    def test_name_property(self) -> None:
        gen = MonteCarloSignal()
        assert gen.name == "montecarlo"


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_valid_signal(self) -> None:
        data = {"SPY": _make_asset("SPY", periods=300)}
        gen = MonteCarloSignal(n_simulations=500, random_state=42)
        sig = gen.generate(data)

        assert isinstance(sig, Signal)
        assert sig.name == "montecarlo"
        assert sig.regime is None

    def test_signal_shape_matches_assets(self) -> None:
        data = {
            "SPY": _make_asset("SPY", periods=300, seed=1),
            "QQQ": _make_asset("QQQ", periods=300, seed=2),
            "IWM": _make_asset("IWM", periods=300, seed=3),
        }
        gen = MonteCarloSignal(n_simulations=500, random_state=42)
        sig = gen.generate(data)

        assert sig.values.shape == (3,)
        assert sig.confidence.shape == (3,)

    def test_confidence_in_bounds(self) -> None:
        data = {"SPY": _make_asset("SPY", periods=300)}
        gen = MonteCarloSignal(n_simulations=1000, random_state=42)
        sig = gen.generate(data)

        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)

    def test_metadata_keys(self) -> None:
        data = {"SPY": _make_asset("SPY", periods=300)}
        gen = MonteCarloSignal(n_simulations=500, random_state=42)
        sig = gen.generate(data)

        assert "var_5pct" in sig.metadata
        assert "cvar_5pct" in sig.metadata
        assert "prob_loss" in sig.metadata
        assert "expected_return" in sig.metadata
        assert "symbols" in sig.metadata
        assert "n_simulations" in sig.metadata
        assert "horizon_days" in sig.metadata

    def test_values_are_finite(self) -> None:
        data = {"SPY": _make_asset("SPY", periods=300)}
        gen = MonteCarloSignal(n_simulations=500, random_state=42)
        sig = gen.generate(data)

        assert np.all(np.isfinite(sig.values))
        assert np.all(np.isfinite(sig.confidence))


# ---------------------------------------------------------------------------
# Directional correctness
# ---------------------------------------------------------------------------


class TestDirectional:
    def test_positive_drift_positive_expected_return(self) -> None:
        """An asset with strong positive drift should have positive expected return."""
        data = {"UP": _make_asset("UP", periods=300, drift=0.002, vol=0.005, seed=1)}
        gen = MonteCarloSignal(n_simulations=5000, random_state=42)
        sig = gen.generate(data)

        assert sig.values[0] > 0, f"Expected positive return, got {sig.values[0]}"

    def test_negative_drift_negative_expected_return(self) -> None:
        """An asset with strong negative drift should have negative expected return."""
        data = {"DOWN": _make_asset("DOWN", periods=300, drift=-0.002, vol=0.005, seed=1)}
        gen = MonteCarloSignal(n_simulations=5000, random_state=42)
        sig = gen.generate(data)

        assert sig.values[0] < 0, f"Expected negative return, got {sig.values[0]}"

    def test_zero_drift_expected_return_near_zero(self) -> None:
        """An asset with zero drift should have expected return near zero."""
        data = {"FLAT": _make_asset("FLAT", periods=500, drift=0.0, vol=0.01, seed=1)}
        gen = MonteCarloSignal(n_simulations=10000, random_state=42)
        sig = gen.generate(data)

        assert abs(sig.values[0]) < 0.05, f"Expected near-zero return, got {sig.values[0]}"

    def test_zero_drift_prob_loss_near_half(self) -> None:
        """With zero drift, probability of loss should be around 0.5."""
        data = {"FLAT": _make_asset("FLAT", periods=500, drift=0.0, vol=0.01, seed=1)}
        gen = MonteCarloSignal(n_simulations=10000, random_state=42)
        sig = gen.generate(data)

        prob_loss = sig.metadata["prob_loss"]["FLAT"]
        assert 0.3 < prob_loss < 0.7, f"Expected prob_loss near 0.5, got {prob_loss}"

    def test_high_drift_low_prob_loss(self) -> None:
        """Strong positive drift should mean low probability of loss."""
        data = {"UP": _make_asset("UP", periods=300, drift=0.003, vol=0.005, seed=1)}
        gen = MonteCarloSignal(n_simulations=5000, random_state=42)
        sig = gen.generate(data)

        prob_loss = sig.metadata["prob_loss"]["UP"]
        assert prob_loss < 0.3, f"Expected low prob_loss, got {prob_loss}"


# ---------------------------------------------------------------------------
# VaR and CVaR
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    def test_var_is_negative_for_zero_drift(self) -> None:
        """5% VaR should be negative — it represents worst-case losses."""
        data = {"FLAT": _make_asset("FLAT", periods=300, drift=0.0, vol=0.01, seed=1)}
        gen = MonteCarloSignal(n_simulations=5000, random_state=42)
        sig = gen.generate(data)

        var_5 = sig.metadata["var_5pct"]["FLAT"]
        assert var_5 < 0, f"VaR at 5% should be negative, got {var_5}"

    def test_cvar_worse_than_var(self) -> None:
        """CVaR should be more negative than (or equal to) VaR."""
        data = {"SPY": _make_asset("SPY", periods=300, drift=0.0, vol=0.01, seed=1)}
        gen = MonteCarloSignal(n_simulations=5000, random_state=42)
        sig = gen.generate(data)

        var_5 = sig.metadata["var_5pct"]["SPY"]
        cvar_5 = sig.metadata["cvar_5pct"]["SPY"]
        assert cvar_5 <= var_5, f"CVaR ({cvar_5}) should be <= VaR ({var_5})"

    def test_var_scales_with_volatility(self) -> None:
        """Higher volatility should produce a more negative VaR."""
        data_low = {"LO": _make_asset("LO", periods=300, drift=0.0, vol=0.005, seed=1)}
        data_high = {"HI": _make_asset("HI", periods=300, drift=0.0, vol=0.02, seed=1)}

        gen = MonteCarloSignal(n_simulations=5000, random_state=42)
        sig_low = gen.generate(data_low)
        gen_hi = MonteCarloSignal(n_simulations=5000, random_state=42)
        sig_high = gen_hi.generate(data_high)

        var_low = sig_low.metadata["var_5pct"]["LO"]
        var_high = sig_high.metadata["var_5pct"]["HI"]

        assert var_high < var_low, (
            f"Higher vol VaR ({var_high}) should be more negative than lower vol VaR ({var_low})"
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        """Two runs with the same seed should produce identical results."""
        data = {"SPY": _make_asset("SPY", periods=300)}
        gen1 = MonteCarloSignal(n_simulations=1000, random_state=123)
        gen2 = MonteCarloSignal(n_simulations=1000, random_state=123)

        sig1 = gen1.generate(data)
        sig2 = gen2.generate(data)

        np.testing.assert_array_equal(sig1.values, sig2.values)
        np.testing.assert_array_equal(sig1.confidence, sig2.confidence)

    def test_different_seed_different_result(self) -> None:
        """Different seeds should produce different results."""
        data = {"SPY": _make_asset("SPY", periods=300)}
        gen1 = MonteCarloSignal(n_simulations=1000, random_state=42)
        gen2 = MonteCarloSignal(n_simulations=1000, random_state=99)

        sig1 = gen1.generate(data)
        sig2 = gen2.generate(data)

        # Values almost certainly differ with different seeds.
        assert not np.array_equal(sig1.values, sig2.values)


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_short_asset_skipped(self, caplog) -> None:
        data = {
            "GOOD": _make_asset("GOOD", periods=300),
            "SHORT": _make_asset("SHORT", periods=30),
        }
        gen = MonteCarloSignal(n_simulations=500, random_state=42)
        sig = gen.generate(data)

        assert sig.values.shape == (1,)
        assert sig.metadata["symbols"] == ["GOOD"]
        assert "Skipping SHORT" in caplog.text

    def test_all_insufficient_raises(self) -> None:
        data = {"A": _make_asset("A", periods=30)}
        gen = MonteCarloSignal()

        with pytest.raises(ValueError, match="No assets have sufficient data"):
            gen.generate(data)

    def test_empty_data_raises(self) -> None:
        gen = MonteCarloSignal()

        with pytest.raises(ValueError, match="No assets have sufficient data"):
            gen.generate({})


# ---------------------------------------------------------------------------
# update() delegates to generate()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_returns_signal(self) -> None:
        data = {"SPY": _make_asset("SPY", periods=300)}
        gen = MonteCarloSignal(n_simulations=500, random_state=42)
        sig = gen.update(data)

        assert isinstance(sig, Signal)
        assert sig.name == "montecarlo"

    def test_update_matches_generate(self) -> None:
        data = {"SPY": _make_asset("SPY", periods=300)}
        gen1 = MonteCarloSignal(n_simulations=1000, random_state=42)
        gen2 = MonteCarloSignal(n_simulations=1000, random_state=42)

        sig_gen = gen1.generate(data)
        sig_upd = gen2.update(data)

        np.testing.assert_array_almost_equal(sig_gen.values, sig_upd.values)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_custom_horizon(self) -> None:
        data = {"SPY": _make_asset("SPY", periods=300)}
        gen_short = MonteCarloSignal(n_simulations=5000, horizon_days=5, random_state=42)
        gen_long = MonteCarloSignal(n_simulations=5000, horizon_days=63, random_state=42)

        sig_short = gen_short.generate(data)
        sig_long = gen_long.generate(data)

        # Longer horizon should produce larger absolute expected return
        # (in magnitude), assuming non-zero drift calibrated from data.
        # At minimum they should differ.
        assert sig_short.values[0] != sig_long.values[0]

    def test_calibration_window(self) -> None:
        """Restricting the calibration window should change the estimates."""
        data = {"SPY": _make_asset("SPY", periods=300, drift=0.001)}

        gen_full = MonteCarloSignal(n_simulations=5000, calibration_window=None, random_state=42)
        gen_short = MonteCarloSignal(n_simulations=5000, calibration_window=60, random_state=42)

        sig_full = gen_full.generate(data)
        sig_short = gen_short.generate(data)

        # Different calibration windows should yield different expected returns.
        assert sig_full.values[0] != sig_short.values[0]

    def test_more_simulations_reduces_variance(self) -> None:
        """More simulations should produce more stable estimates."""
        data = {"SPY": _make_asset("SPY", periods=300, drift=0.0, vol=0.01)}

        results = []
        for seed in range(5):
            gen = MonteCarloSignal(n_simulations=10000, random_state=seed)
            sig = gen.generate(data)
            results.append(sig.values[0])

        results_few = []
        for seed in range(5):
            gen = MonteCarloSignal(n_simulations=100, random_state=seed)
            sig = gen.generate(data)
            results_few.append(sig.values[0])

        # Variance across seeds should be lower with more simulations.
        assert np.std(results) < np.std(results_few)
