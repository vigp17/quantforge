"""Tests for src/signals/ensemble.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator
from src.signals.ensemble import SignalEnsemble

# ---------------------------------------------------------------------------
# Stub generators for deterministic testing
# ---------------------------------------------------------------------------


class _StubGenerator(SignalGenerator):
    """A generator that returns a pre-configured Signal."""

    def __init__(
        self,
        gen_name: str,
        values: np.ndarray,
        confidence: np.ndarray,
        regime: str | None = None,
    ) -> None:
        self._name = gen_name
        self._values = values
        self._confidence = confidence
        self._regime = regime

    @property
    def name(self) -> str:
        return self._name

    def generate(self, data: dict[str, AssetData]) -> Signal:
        return Signal(
            name=self._name,
            values=self._values.copy(),
            confidence=self._confidence.copy(),
            regime=self._regime,
        )

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        return self.generate(new_data)


class _FailingGenerator(SignalGenerator):
    """A generator that always raises."""

    @property
    def name(self) -> str:
        return "failing"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        raise RuntimeError("Intentional failure")

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        raise RuntimeError("Intentional failure")


def _dummy_data() -> dict[str, AssetData]:
    """Minimal valid data dict (ensemble doesn't read it directly)."""
    dates = pd.bdate_range("2023-01-02", periods=10)
    df = pd.DataFrame(
        {
            "open": np.ones(10),
            "high": np.ones(10),
            "low": np.ones(10),
            "close": np.ones(10),
            "volume": np.ones(10),
        },
        index=dates,
    )
    return {"DUMMY": AssetData(symbol="DUMMY", ohlcv=df)}


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------


class TestInterface:
    def test_is_signal_generator(self) -> None:
        gen = SignalEnsemble([_StubGenerator("a", np.array([1.0]), np.array([0.5]))])
        assert isinstance(gen, SignalGenerator)

    def test_name_property(self) -> None:
        gen = SignalEnsemble([_StubGenerator("a", np.array([1.0]), np.array([0.5]))])
        assert gen.name == "ensemble"

    def test_empty_generators_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            SignalEnsemble([])


# ---------------------------------------------------------------------------
# Equal weight combination
# ---------------------------------------------------------------------------


class TestEqualWeight:
    def test_average_of_two_signals(self) -> None:
        g1 = _StubGenerator("s1", np.array([2.0, 4.0]), np.array([0.8, 0.6]))
        g2 = _StubGenerator("s2", np.array([6.0, 0.0]), np.array([0.4, 1.0]))
        ens = SignalEnsemble([g1, g2], method="equal_weight")

        sig = ens.generate(_dummy_data())

        np.testing.assert_array_almost_equal(sig.values, [4.0, 2.0])
        np.testing.assert_array_almost_equal(sig.confidence, [0.6, 0.8])

    def test_single_child_passthrough(self) -> None:
        g = _StubGenerator("only", np.array([3.0]), np.array([0.9]))
        ens = SignalEnsemble([g], method="equal_weight")
        sig = ens.generate(_dummy_data())

        assert sig.values[0] == pytest.approx(3.0)
        assert sig.confidence[0] == pytest.approx(0.9)

    def test_three_signals(self) -> None:
        g1 = _StubGenerator("a", np.array([1.0]), np.array([0.6]))
        g2 = _StubGenerator("b", np.array([2.0]), np.array([0.9]))
        g3 = _StubGenerator("c", np.array([3.0]), np.array([0.3]))
        ens = SignalEnsemble([g1, g2, g3], method="equal_weight")
        sig = ens.generate(_dummy_data())

        assert sig.values[0] == pytest.approx(2.0)
        assert sig.confidence[0] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Confidence-weighted combination
# ---------------------------------------------------------------------------


class TestConfidenceWeighted:
    def test_weighted_average(self) -> None:
        # s1 has high confidence → should dominate.
        g1 = _StubGenerator("s1", np.array([10.0]), np.array([0.9]))
        g2 = _StubGenerator("s2", np.array([0.0]), np.array([0.1]))
        ens = SignalEnsemble([g1, g2], method="confidence_weighted")
        sig = ens.generate(_dummy_data())

        # Weighted: (10*0.9 + 0*0.1) / (0.9+0.1) = 9.0
        assert sig.values[0] == pytest.approx(9.0)

    def test_equal_confidence_equals_equal_weight(self) -> None:
        g1 = _StubGenerator("s1", np.array([2.0]), np.array([0.5]))
        g2 = _StubGenerator("s2", np.array([8.0]), np.array([0.5]))
        ens = SignalEnsemble([g1, g2], method="confidence_weighted")
        sig = ens.generate(_dummy_data())

        # (2*0.5 + 8*0.5) / (0.5+0.5) = 5.0
        assert sig.values[0] == pytest.approx(5.0)

    def test_zero_confidence_fallback(self) -> None:
        """If all confidences are zero, should fall back to equal weight."""
        g1 = _StubGenerator("s1", np.array([4.0]), np.array([0.0]))
        g2 = _StubGenerator("s2", np.array([6.0]), np.array([0.0]))
        ens = SignalEnsemble([g1, g2], method="confidence_weighted")
        sig = ens.generate(_dummy_data())

        # (4*0 + 6*0) / 1.0 = 0.0 (safe fallback)
        assert np.isfinite(sig.values[0])

    def test_multi_asset(self) -> None:
        g1 = _StubGenerator("s1", np.array([1.0, 5.0]), np.array([0.8, 0.2]))
        g2 = _StubGenerator("s2", np.array([3.0, 1.0]), np.array([0.2, 0.8]))
        ens = SignalEnsemble([g1, g2], method="confidence_weighted")
        sig = ens.generate(_dummy_data())

        # Asset 0: (1*0.8 + 3*0.2) / (0.8+0.2) = 1.4
        # Asset 1: (5*0.2 + 1*0.8) / (0.2+0.8) = 1.8
        np.testing.assert_array_almost_equal(sig.values, [1.4, 1.8])


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_two_positive_one_negative(self) -> None:
        """2 out of 3 positive → combined should be +1."""
        g1 = _StubGenerator("a", np.array([0.5]), np.array([0.5]))
        g2 = _StubGenerator("b", np.array([1.2]), np.array([0.5]))
        g3 = _StubGenerator("c", np.array([-0.3]), np.array([0.5]))
        ens = SignalEnsemble([g1, g2, g3], method="majority_vote")
        sig = ens.generate(_dummy_data())

        assert sig.values[0] == 1.0
        # 2 out of 3 agree → confidence = 2/3
        assert sig.confidence[0] == pytest.approx(2.0 / 3.0)

    def test_all_agree_positive(self) -> None:
        g1 = _StubGenerator("a", np.array([1.0]), np.array([0.5]))
        g2 = _StubGenerator("b", np.array([2.0]), np.array([0.5]))
        ens = SignalEnsemble([g1, g2], method="majority_vote")
        sig = ens.generate(_dummy_data())

        assert sig.values[0] == 1.0
        assert sig.confidence[0] == pytest.approx(1.0)

    def test_all_agree_negative(self) -> None:
        g1 = _StubGenerator("a", np.array([-1.0]), np.array([0.5]))
        g2 = _StubGenerator("b", np.array([-0.5]), np.array([0.5]))
        g3 = _StubGenerator("c", np.array([-2.0]), np.array([0.5]))
        ens = SignalEnsemble([g1, g2, g3], method="majority_vote")
        sig = ens.generate(_dummy_data())

        assert sig.values[0] == -1.0
        assert sig.confidence[0] == pytest.approx(1.0)

    def test_perfect_split(self) -> None:
        """Equally split → zero value, zero confidence."""
        g1 = _StubGenerator("a", np.array([1.0]), np.array([0.5]))
        g2 = _StubGenerator("b", np.array([-1.0]), np.array([0.5]))
        ens = SignalEnsemble([g1, g2], method="majority_vote")
        sig = ens.generate(_dummy_data())

        assert sig.values[0] == 0.0
        assert sig.confidence[0] == 0.0

    def test_multi_asset_majority(self) -> None:
        # Asset 0: +, +, - → +1, conf 2/3
        # Asset 1: -, -, + → -1, conf 2/3
        g1 = _StubGenerator("a", np.array([1.0, -1.0]), np.array([0.5, 0.5]))
        g2 = _StubGenerator("b", np.array([0.5, -2.0]), np.array([0.5, 0.5]))
        g3 = _StubGenerator("c", np.array([-0.3, 0.1]), np.array([0.5, 0.5]))
        ens = SignalEnsemble([g1, g2, g3], method="majority_vote")
        sig = ens.generate(_dummy_data())

        np.testing.assert_array_almost_equal(sig.values, [1.0, -1.0])
        np.testing.assert_array_almost_equal(sig.confidence, [2.0 / 3.0, 2.0 / 3.0])


# ---------------------------------------------------------------------------
# Graceful failure handling
# ---------------------------------------------------------------------------


class TestGracefulFailure:
    def test_one_child_fails_others_used(self) -> None:
        g_ok = _StubGenerator("ok", np.array([5.0]), np.array([0.8]))
        g_fail = _FailingGenerator()
        ens = SignalEnsemble([g_ok, g_fail], method="equal_weight")
        sig = ens.generate(_dummy_data())

        assert sig.values[0] == pytest.approx(5.0)
        assert sig.metadata["n_signals"] == 1

    def test_two_of_three_fail(self) -> None:
        g_ok = _StubGenerator("ok", np.array([3.0]), np.array([0.7]))
        ens = SignalEnsemble(
            [_FailingGenerator(), g_ok, _FailingGenerator()],
            method="equal_weight",
        )
        sig = ens.generate(_dummy_data())
        assert sig.values[0] == pytest.approx(3.0)

    def test_all_fail_raises(self) -> None:
        ens = SignalEnsemble([_FailingGenerator()], method="equal_weight")
        with pytest.raises(RuntimeError, match="All child signal generators failed"):
            ens.generate(_dummy_data())

    def test_failure_logged(self, caplog) -> None:
        g_ok = _StubGenerator("ok", np.array([1.0]), np.array([0.5]))
        g_fail = _FailingGenerator()
        ens = SignalEnsemble([g_ok, g_fail], method="equal_weight")
        ens.generate(_dummy_data())

        assert "failing failed" in caplog.text


# ---------------------------------------------------------------------------
# Regime passthrough
# ---------------------------------------------------------------------------


class TestRegimePassthrough:
    def test_regime_from_hmm_child(self) -> None:
        g_hmm = _StubGenerator(
            "hmm_regime", np.array([1.0]), np.array([0.9]), regime="bull"
        )
        g_mom = _StubGenerator("momentum", np.array([2.0]), np.array([0.7]))
        ens = SignalEnsemble([g_hmm, g_mom], method="equal_weight")
        sig = ens.generate(_dummy_data())

        assert sig.regime == "bull"

    def test_regime_none_when_no_child_has_one(self) -> None:
        g1 = _StubGenerator("a", np.array([1.0]), np.array([0.5]))
        g2 = _StubGenerator("b", np.array([2.0]), np.array([0.5]))
        ens = SignalEnsemble([g1, g2], method="equal_weight")
        sig = ens.generate(_dummy_data())

        assert sig.regime is None

    def test_first_regime_wins(self) -> None:
        """If multiple children report regimes, the first one wins."""
        g1 = _StubGenerator("a", np.array([1.0]), np.array([0.5]), regime="bear")
        g2 = _StubGenerator("b", np.array([2.0]), np.array([0.5]), regime="bull")
        ens = SignalEnsemble([g1, g2], method="equal_weight")
        sig = ens.generate(_dummy_data())

        assert sig.regime == "bear"


# ---------------------------------------------------------------------------
# Metadata / attribution
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_contains_method(self) -> None:
        g = _StubGenerator("s1", np.array([1.0]), np.array([0.5]))
        ens = SignalEnsemble([g], method="equal_weight")
        sig = ens.generate(_dummy_data())

        assert sig.metadata["method"] == "equal_weight"

    def test_contains_n_signals(self) -> None:
        g1 = _StubGenerator("a", np.array([1.0]), np.array([0.5]))
        g2 = _StubGenerator("b", np.array([2.0]), np.array([0.5]))
        ens = SignalEnsemble([g1, g2], method="equal_weight")
        sig = ens.generate(_dummy_data())

        assert sig.metadata["n_signals"] == 2

    def test_contains_contributors_list(self) -> None:
        g1 = _StubGenerator("momentum", np.array([1.0]), np.array([0.5]))
        g2 = _StubGenerator("montecarlo", np.array([2.0]), np.array([0.5]))
        ens = SignalEnsemble([g1, g2], method="equal_weight")
        sig = ens.generate(_dummy_data())

        assert sig.metadata["contributors"] == ["momentum", "montecarlo"]

    def test_contributions_contain_per_signal_data(self) -> None:
        g1 = _StubGenerator("s1", np.array([1.0, 2.0]), np.array([0.5, 0.6]))
        g2 = _StubGenerator("s2", np.array([3.0, 4.0]), np.array([0.7, 0.8]))
        ens = SignalEnsemble([g1, g2], method="equal_weight")
        sig = ens.generate(_dummy_data())

        contribs = sig.metadata["contributions"]
        assert "s1" in contribs
        assert "s2" in contribs
        assert contribs["s1"]["values"] == [1.0, 2.0]
        assert contribs["s2"]["confidence"] == [0.7, 0.8]


# ---------------------------------------------------------------------------
# update() delegates correctly
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_returns_signal(self) -> None:
        g = _StubGenerator("s1", np.array([1.0]), np.array([0.5]))
        ens = SignalEnsemble([g], method="equal_weight")
        sig = ens.update(_dummy_data())

        assert isinstance(sig, Signal)
        assert sig.name == "ensemble"

    def test_update_calls_child_update(self) -> None:
        """Verify update() propagates to children's update(), not generate()."""

        class _TrackingGenerator(SignalGenerator):
            def __init__(self) -> None:
                self.generate_called = False
                self.update_called = False

            @property
            def name(self) -> str:
                return "tracker"

            def generate(self, data: dict[str, AssetData]) -> Signal:
                self.generate_called = True
                return Signal(name="tracker", values=np.array([1.0]), confidence=np.array([0.5]))

            def update(self, new_data: dict[str, AssetData]) -> Signal:
                self.update_called = True
                return Signal(name="tracker", values=np.array([1.0]), confidence=np.array([0.5]))

        tracker = _TrackingGenerator()
        ens = SignalEnsemble([tracker], method="equal_weight")
        ens.update(_dummy_data())

        assert tracker.update_called
        assert not tracker.generate_called
