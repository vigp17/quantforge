"""Tests for src/portfolio/base.py."""

import pandas as pd
import pytest

from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.signals.base import Signal


# ---------------------------------------------------------------------------
# PortfolioAction – happy path
# ---------------------------------------------------------------------------

class TestPortfolioActionValid:
    def test_creates_with_valid_weights(self) -> None:
        action = PortfolioAction(
            weights={"SPY": 0.4, "TLT": 0.3, "GLD": 0.2},
            confidence=0.85,
            regime_context="bull",
        )
        assert action.weights["SPY"] == 0.4
        assert action.risk_metrics == {}

    def test_weights_summing_to_exactly_one(self) -> None:
        action = PortfolioAction(
            weights={"SPY": 0.5, "TLT": 0.5},
            confidence=0.9,
            regime_context="neutral",
        )
        assert sum(action.weights.values()) == 1.0

    def test_weights_below_one_implies_cash(self) -> None:
        action = PortfolioAction(
            weights={"SPY": 0.3},
            confidence=0.5,
            regime_context="bear",
        )
        assert sum(abs(w) for w in action.weights.values()) < 1.0

    def test_empty_weights_all_cash(self) -> None:
        action = PortfolioAction(
            weights={},
            confidence=0.1,
            regime_context="crisis",
        )
        assert len(action.weights) == 0

    def test_negative_weights_short_positions(self) -> None:
        action = PortfolioAction(
            weights={"SPY": 0.6, "VIX": -0.3},
            confidence=0.7,
            regime_context="bull",
        )
        assert action.weights["VIX"] == -0.3

    def test_confidence_boundary_zero(self) -> None:
        action = PortfolioAction(
            weights={"SPY": 0.5},
            confidence=0.0,
            regime_context="unknown",
        )
        assert action.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        action = PortfolioAction(
            weights={"SPY": 0.5},
            confidence=1.0,
            regime_context="bull",
        )
        assert action.confidence == 1.0

    def test_custom_risk_metrics(self) -> None:
        action = PortfolioAction(
            weights={"SPY": 0.5},
            confidence=0.8,
            regime_context="bull",
            risk_metrics={"var_95": 0.02, "max_drawdown": 0.05},
        )
        assert action.risk_metrics["var_95"] == 0.02


# ---------------------------------------------------------------------------
# PortfolioAction – validation errors
# ---------------------------------------------------------------------------

class TestPortfolioActionValidation:
    def test_rejects_weights_sum_above_one(self) -> None:
        with pytest.raises(ValueError, match="exceeds 1.0"):
            PortfolioAction(
                weights={"SPY": 0.6, "TLT": 0.5},
                confidence=0.5,
                regime_context="bull",
            )

    def test_rejects_abs_weights_sum_above_one_with_shorts(self) -> None:
        with pytest.raises(ValueError, match="exceeds 1.0"):
            PortfolioAction(
                weights={"SPY": 0.7, "VIX": -0.4},
                confidence=0.5,
                regime_context="bull",
            )

    def test_rejects_confidence_above_one(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            PortfolioAction(
                weights={"SPY": 0.5},
                confidence=1.1,
                regime_context="bull",
            )

    def test_rejects_confidence_below_zero(self) -> None:
        with pytest.raises(ValueError, match="confidence must be in"):
            PortfolioAction(
                weights={"SPY": 0.5},
                confidence=-0.1,
                regime_context="bull",
            )


# ---------------------------------------------------------------------------
# PortfolioAgent – abstract enforcement
# ---------------------------------------------------------------------------

class TestPortfolioAgentAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            PortfolioAgent()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_instantiate(self) -> None:
        class Partial(PortfolioAgent):
            def decide(self, signals: list[Signal], current_portfolio: dict) -> PortfolioAction:
                ...

        with pytest.raises(TypeError, match="abstract"):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_can_instantiate(self) -> None:
        class Concrete(PortfolioAgent):
            def decide(self, signals: list[Signal], current_portfolio: dict) -> PortfolioAction:
                ...

            def train(self, historical_signals: list[Signal], returns: pd.DataFrame) -> dict:
                ...

        agent = Concrete()
        assert isinstance(agent, PortfolioAgent)
