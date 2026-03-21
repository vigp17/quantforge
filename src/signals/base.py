"""Base interfaces for the signal engine.

Defines the Signal container and the abstract SignalGenerator interface
that all signal producers (HMM, Kalman, iTransformer, etc.) must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.data.base import AssetData


@dataclass
class Signal:
    """Container for a trading signal across one or more assets.

    Args:
        name: Identifier for the signal source (e.g. "hmm_regime").
        values: Signal strength per asset (arbitrary scale).
        confidence: Confidence per signal element, each in [0, 1].
        regime: Current detected market regime, if applicable.
        timestamp: When the signal was generated.
        metadata: Arbitrary extra information.

    Raises:
        ValueError: If confidence values fall outside [0, 1] or if
            values and confidence have mismatched lengths.
    """

    name: str
    values: np.ndarray
    confidence: np.ndarray
    regime: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shapes and confidence bounds."""
        if self.values.shape != self.confidence.shape:
            raise ValueError(
                f"values shape {self.values.shape} does not match "
                f"confidence shape {self.confidence.shape}"
            )
        if self.confidence.size > 0 and (
            np.any(self.confidence < 0) or np.any(self.confidence > 1)
        ):
            raise ValueError("confidence values must be in [0, 1]")


class SignalGenerator(ABC):
    """Abstract interface for signal producers.

    All signal generators (HMM regime, Kalman pairs, momentum, etc.)
    must implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this signal generator."""
        ...

    @abstractmethod
    def generate(self, data: dict[str, AssetData]) -> Signal:
        """Generate signals from a full dataset.

        Args:
            data: Mapping of symbol to AssetData.

        Returns:
            Signal computed over the provided data.
        """
        ...

    @abstractmethod
    def update(self, new_data: dict[str, AssetData]) -> Signal:
        """Incrementally update signals with new data.

        Designed for live trading where recomputing from scratch
        on every tick is too expensive.

        Args:
            new_data: Mapping of symbol to the latest AssetData slice.

        Returns:
            Updated Signal.
        """
        ...
