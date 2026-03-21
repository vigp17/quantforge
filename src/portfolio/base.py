"""Base interfaces for the portfolio engine.

Defines the PortfolioAction container and the abstract PortfolioAgent
interface that all allocation agents (IQN, mean-variance, etc.) must
implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from src.signals.base import Signal

# Tolerance for floating-point weight sum comparison.
_WEIGHT_SUM_TOL = 1e-9


@dataclass
class PortfolioAction:
    """Target portfolio allocation produced by an agent.

    Args:
        weights: Mapping of symbol to target portfolio weight.
            Absolute values must sum to at most 1.0 (cash makes up the
            remainder).
        confidence: Agent's confidence in this allocation, in [0, 1].
        regime_context: Market regime used when making the decision.
        risk_metrics: Associated risk measures (VaR, expected drawdown, etc.).

    Raises:
        ValueError: If absolute weights sum exceeds 1.0 or confidence
            is outside [0, 1].
    """

    weights: dict[str, float]
    confidence: float
    regime_context: str
    risk_metrics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate weight sum and confidence bounds."""
        weight_sum = sum(abs(w) for w in self.weights.values())
        if weight_sum > 1.0 + _WEIGHT_SUM_TOL:
            raise ValueError(f"Absolute weights sum to {weight_sum:.6f}, which exceeds 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


class PortfolioAgent(ABC):
    """Abstract interface for portfolio allocation agents.

    All agents (IQN distributional RL, mean-variance optimizer, etc.)
    must implement this interface.
    """

    @abstractmethod
    def decide(self, signals: list[Signal], current_portfolio: dict) -> PortfolioAction:
        """Decide on a target portfolio allocation.

        Args:
            signals: List of signals from the signal engine.
            current_portfolio: Current holdings as symbol to weight mapping.

        Returns:
            PortfolioAction with target weights and metadata.
        """
        ...

    @abstractmethod
    def train(self, historical_signals: list[Signal], returns: pd.DataFrame) -> dict:
        """Train the agent on historical data.

        Args:
            historical_signals: Historical signal snapshots.
            returns: Asset return series used for training.

        Returns:
            Dictionary of training metrics.
        """
        ...
