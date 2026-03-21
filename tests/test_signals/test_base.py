"""Tests for src/signals/base.py."""

from datetime import datetime

import numpy as np
import pytest

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator


# ---------------------------------------------------------------------------
# Signal – happy path
# ---------------------------------------------------------------------------

class TestSignalValid:
    def test_creates_with_valid_arrays(self) -> None:
        sig = Signal(
            name="test",
            values=np.array([0.5, -0.3, 0.1]),
            confidence=np.array([0.9, 0.8, 0.7]),
        )
        assert sig.name == "test"
        assert sig.regime is None
        assert sig.metadata == {}
        assert isinstance(sig.timestamp, datetime)

    def test_accepts_empty_arrays(self) -> None:
        sig = Signal(
            name="empty",
            values=np.array([]),
            confidence=np.array([]),
        )
        assert sig.values.size == 0

    def test_accepts_regime_and_metadata(self) -> None:
        sig = Signal(
            name="hmm",
            values=np.array([1.0]),
            confidence=np.array([0.95]),
            regime="bull",
            metadata={"n_states": 3},
        )
        assert sig.regime == "bull"
        assert sig.metadata["n_states"] == 3

    def test_confidence_boundary_zero(self) -> None:
        sig = Signal(
            name="edge",
            values=np.array([0.0]),
            confidence=np.array([0.0]),
        )
        assert sig.confidence[0] == 0.0

    def test_confidence_boundary_one(self) -> None:
        sig = Signal(
            name="edge",
            values=np.array([1.0]),
            confidence=np.array([1.0]),
        )
        assert sig.confidence[0] == 1.0


# ---------------------------------------------------------------------------
# Signal – validation errors
# ---------------------------------------------------------------------------

class TestSignalValidation:
    def test_rejects_confidence_above_one(self) -> None:
        with pytest.raises(ValueError, match="confidence values must be in"):
            Signal(
                name="bad",
                values=np.array([0.5]),
                confidence=np.array([1.1]),
            )

    def test_rejects_confidence_below_zero(self) -> None:
        with pytest.raises(ValueError, match="confidence values must be in"):
            Signal(
                name="bad",
                values=np.array([0.5]),
                confidence=np.array([-0.1]),
            )

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            Signal(
                name="bad",
                values=np.array([0.1, 0.2, 0.3]),
                confidence=np.array([0.5, 0.5]),
            )

    def test_rejects_mismatched_2d_shapes(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            Signal(
                name="bad",
                values=np.zeros((3, 2)),
                confidence=np.zeros((2, 3)),
            )

    def test_rejects_mixed_valid_and_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence values must be in"):
            Signal(
                name="bad",
                values=np.array([0.1, 0.2, 0.3]),
                confidence=np.array([0.5, 1.5, 0.8]),
            )


# ---------------------------------------------------------------------------
# SignalGenerator – abstract enforcement
# ---------------------------------------------------------------------------

class TestSignalGeneratorAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            SignalGenerator()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_instantiate(self) -> None:
        class Partial(SignalGenerator):
            @property
            def name(self) -> str:
                return "partial"

            def generate(self, data: dict[str, AssetData]) -> Signal:
                ...

        with pytest.raises(TypeError, match="abstract"):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_can_instantiate(self) -> None:
        class Concrete(SignalGenerator):
            @property
            def name(self) -> str:
                return "concrete"

            def generate(self, data: dict[str, AssetData]) -> Signal:
                ...

            def update(self, new_data: dict[str, AssetData]) -> Signal:
                ...

        gen = Concrete()
        assert gen.name == "concrete"
        assert isinstance(gen, SignalGenerator)
