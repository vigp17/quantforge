"""Tests for src/portfolio/risk_manager.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.base import PortfolioAction
from src.portfolio.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_action(weights: dict[str, float], confidence: float = 0.8) -> PortfolioAction:
    return PortfolioAction(weights=weights, confidence=confidence, regime_context="bull")


def _make_returns(
    symbols: list[str],
    n: int = 60,
    seed: int = 42,
    corr_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame of synthetic returns.

    If *corr_pairs* is supplied, the second symbol in each pair is derived
    from the first to produce near-perfect positive correlation.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for sym in symbols:
        data[sym] = rng.normal(0.0, 0.01, n)

    if corr_pairs:
        for sym_a, sym_b in corr_pairs:
            noise = rng.normal(0.0, 0.001, n)
            data[sym_b] = data[sym_a] + noise  # ≈ 0.99 correlation

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rm() -> RiskManager:
    return RiskManager()


@pytest.fixture()
def rm_tight() -> RiskManager:
    """Stricter thresholds for edge-case tests."""
    return RiskManager(
        {
            "max_position_pct": 0.20,
            "max_drawdown_pct": 0.10,
            "daily_loss_limit_pct": 0.02,
            "correlation_limit": 0.50,
            "max_leverage": 0.80,
        }
    )


# ---------------------------------------------------------------------------
# validate_action – position clipping
# ---------------------------------------------------------------------------

class TestValidateAction:
    def test_no_clip_when_within_limits(self, rm: RiskManager) -> None:
        action = _make_action({"SPY": 0.25, "TLT": 0.25})
        adjusted, warnings = rm.validate_action(action, {})
        assert adjusted.weights["SPY"] == pytest.approx(0.25)
        assert adjusted.weights["TLT"] == pytest.approx(0.25)
        assert warnings == []

    def test_single_overweight_gets_clipped(self, rm: RiskManager) -> None:
        # SPY at 0.5 exceeds max_position_pct=0.30
        action = _make_action({"SPY": 0.5, "TLT": 0.3})
        # Note: PortfolioAction rejects abs-sum > 1.0, so we construct it
        # manually with a valid sum first, then test within rm.
        # abs(0.5) + abs(0.3) = 0.8 — valid input.
        adjusted, warnings = rm.validate_action(action, {})
        assert adjusted.weights["SPY"] == pytest.approx(0.30)
        assert len(warnings) == 1
        assert "SPY" in warnings[0]

    def test_multiple_overweight_positions_all_clipped(self, rm: RiskManager) -> None:
        action = _make_action({"A": 0.4, "B": 0.4})
        adjusted, warnings = rm.validate_action(action, {})
        assert adjusted.weights["A"] == pytest.approx(0.30)
        assert adjusted.weights["B"] == pytest.approx(0.30)
        assert len(warnings) == 2

    def test_negative_overweight_short_clipped(self, rm: RiskManager) -> None:
        # abs(-0.4) > 0.30 → clip to -0.30
        action = _make_action({"SPY": 0.5, "VIX": -0.4})
        adjusted, warnings = rm.validate_action(action, {})
        assert adjusted.weights["VIX"] == pytest.approx(-0.30)

    def test_leverage_scaled_down(self, rm_tight: RiskManager) -> None:
        # max_leverage=0.80; two positions at 0.35 each = 0.70 abs sum → ok after pos-clip
        # But let's set up positions that after pos-clip still exceed leverage.
        rm = RiskManager({"max_position_pct": 0.30, "max_leverage": 0.50})
        action = _make_action({"A": 0.30, "B": 0.30})  # abs sum = 0.60 > 0.50
        adjusted, warnings = rm.validate_action(action, {})
        total = sum(abs(w) for w in adjusted.weights.values())
        assert total == pytest.approx(0.50, rel=1e-4)
        assert any("leverage" in w.lower() for w in warnings)

    def test_adjusted_action_is_valid_portfolio_action(self, rm: RiskManager) -> None:
        action = _make_action({"SPY": 0.5, "TLT": 0.3})
        adjusted, _ = rm.validate_action(action, {})
        # Must not raise ValueError
        assert isinstance(adjusted, PortfolioAction)

    def test_metadata_preserved(self, rm: RiskManager) -> None:
        action = PortfolioAction(
            weights={"SPY": 0.25},
            confidence=0.9,
            regime_context="bear",
            risk_metrics={"var": 0.02},
        )
        adjusted, _ = rm.validate_action(action, {})
        assert adjusted.confidence == 0.9
        assert adjusted.regime_context == "bear"
        assert adjusted.risk_metrics["var"] == 0.02

    def test_empty_weights_no_warnings(self, rm: RiskManager) -> None:
        action = _make_action({})
        adjusted, warnings = rm.validate_action(action, {})
        assert adjusted.weights == {}
        assert warnings == []


# ---------------------------------------------------------------------------
# check_drawdown
# ---------------------------------------------------------------------------

class TestCheckDrawdown:
    def test_no_drawdown_rising_equity(self, rm: RiskManager) -> None:
        values = [100.0, 105.0, 110.0, 115.0]
        breached, dd = rm.check_drawdown(values)
        assert not breached
        assert dd == pytest.approx(0.0)

    def test_20pct_drop_breaches_15pct_limit(self, rm: RiskManager) -> None:
        # Peak = 100, current = 80 → 20% drawdown
        values = [80.0, 90.0, 100.0, 80.0]
        breached, dd = rm.check_drawdown(values)
        assert breached
        assert dd == pytest.approx(0.20, rel=1e-4)

    def test_14pct_drop_does_not_breach(self, rm: RiskManager) -> None:
        values = [100.0, 86.0]
        breached, dd = rm.check_drawdown(values)
        assert not breached
        assert dd == pytest.approx(0.14, rel=1e-4)

    def test_exactly_at_limit_not_breached(self, rm: RiskManager) -> None:
        # 15% drawdown exactly — not *strictly* greater, so not breached
        values = [100.0, 85.0]
        breached, dd = rm.check_drawdown(values)
        assert not breached
        assert dd == pytest.approx(0.15, rel=1e-4)

    def test_empty_list_returns_no_breach(self, rm: RiskManager) -> None:
        breached, dd = rm.check_drawdown([])
        assert not breached
        assert dd == 0.0

    def test_single_value_no_drawdown(self, rm: RiskManager) -> None:
        breached, dd = rm.check_drawdown([100.0])
        assert not breached
        assert dd == pytest.approx(0.0)

    def test_peak_tracked_across_calls(self) -> None:
        rm = RiskManager()
        rm.check_drawdown([100.0, 120.0])
        # New call: current = 90, but peak from previous = 120 → 25% dd
        breached, dd = rm.check_drawdown([90.0])
        assert breached
        assert dd == pytest.approx(0.25, rel=1e-4)


# ---------------------------------------------------------------------------
# check_daily_loss
# ---------------------------------------------------------------------------

class TestCheckDailyLoss:
    def test_3pt5_pct_loss_breaches(self, rm: RiskManager) -> None:
        yesterday = 100.0
        today = 96.5  # 3.5% loss
        breached, loss = rm.check_daily_loss(today, yesterday)
        assert breached
        assert loss == pytest.approx(0.035, rel=1e-4)

    def test_2pct_loss_does_not_breach(self, rm: RiskManager) -> None:
        breached, loss = rm.check_daily_loss(98.0, 100.0)
        assert not breached
        assert loss == pytest.approx(0.02, rel=1e-4)

    def test_gain_not_breached(self, rm: RiskManager) -> None:
        breached, loss = rm.check_daily_loss(103.0, 100.0)
        assert not breached
        assert loss < 0  # negative = gain

    def test_zero_yesterday_returns_no_breach(self, rm: RiskManager) -> None:
        breached, loss = rm.check_daily_loss(100.0, 0.0)
        assert not breached
        assert loss == 0.0

    def test_exactly_at_limit_not_breached(self, rm: RiskManager) -> None:
        breached, loss = rm.check_daily_loss(97.0, 100.0)
        assert not breached
        assert loss == pytest.approx(0.03, rel=1e-4)


# ---------------------------------------------------------------------------
# check_correlation
# ---------------------------------------------------------------------------

class TestCheckCorrelation:
    def test_highly_correlated_pair_detected(self, rm: RiskManager) -> None:
        returns = _make_returns(["A", "B"], corr_pairs=[("A", "B")])
        weights = {"A": 0.25, "B": 0.25}
        breached, violations = rm.check_correlation(weights, returns)
        assert breached
        assert "A/B" in violations
        assert abs(violations["A/B"]) > 0.70

    def test_uncorrelated_assets_no_violation(self, rm: RiskManager) -> None:
        returns = _make_returns(["X", "Y", "Z"], seed=99)
        weights = {"X": 0.20, "Y": 0.20, "Z": 0.20}
        breached, violations = rm.check_correlation(weights, returns)
        # Should not breach; uncorrelated random series rarely exceed 0.70
        assert not breached or violations == {}

    def test_zero_weight_asset_excluded(self, rm: RiskManager) -> None:
        returns = _make_returns(["A", "B"], corr_pairs=[("A", "B")])
        # B has zero weight → excluded from check
        weights = {"A": 0.25, "B": 0.0}
        breached, violations = rm.check_correlation(weights, returns)
        assert not breached

    def test_single_asset_no_violation(self, rm: RiskManager) -> None:
        returns = _make_returns(["SPY"])
        weights = {"SPY": 0.25}
        breached, violations = rm.check_correlation(weights, returns)
        assert not breached
        assert violations == {}

    def test_empty_weights_no_violation(self, rm: RiskManager) -> None:
        returns = _make_returns(["SPY", "TLT"])
        breached, violations = rm.check_correlation({}, returns)
        assert not breached
        assert violations == {}

    def test_asset_not_in_returns_df_excluded(self, rm: RiskManager) -> None:
        returns = _make_returns(["SPY"])
        weights = {"SPY": 0.25, "GLD": 0.25}  # GLD missing from returns
        # Should not raise; GLD simply skipped
        breached, violations = rm.check_correlation(weights, returns)
        assert not breached


# ---------------------------------------------------------------------------
# should_flatten
# ---------------------------------------------------------------------------

class TestShouldFlatten:
    def test_no_breach_no_flatten(self, rm: RiskManager) -> None:
        values = [100.0, 102.0, 104.0]
        flatten, reasons = rm.should_flatten(values, 104.0, 102.0)
        assert not flatten
        assert reasons == []

    def test_drawdown_breach_triggers_flatten(self, rm: RiskManager) -> None:
        values = [100.0, 80.0]  # 20% drawdown
        flatten, reasons = rm.should_flatten(values, 80.0, 80.0)
        assert flatten
        assert any("drawdown" in r.lower() for r in reasons)

    def test_daily_loss_breach_triggers_flatten(self, rm: RiskManager) -> None:
        values = [100.0, 96.0]  # 4% daily loss
        flatten, reasons = rm.should_flatten(values, 96.0, 100.0)
        assert flatten
        assert any("daily loss" in r.lower() for r in reasons)

    def test_both_breaches_included_in_reasons(self) -> None:
        rm = RiskManager({"max_drawdown_pct": 0.05, "daily_loss_limit_pct": 0.01})
        values = [100.0, 80.0]
        flatten, reasons = rm.should_flatten(values, 80.0, 100.0)
        assert flatten
        assert len(reasons) == 2

    def test_empty_portfolio_no_flatten(self, rm: RiskManager) -> None:
        flatten, reasons = rm.should_flatten([], 0.0, 0.0)
        assert not flatten
        assert reasons == []


# ---------------------------------------------------------------------------
# generate_risk_report
# ---------------------------------------------------------------------------

class TestGenerateRiskReport:
    _EXPECTED_KEYS = {
        "current_value",
        "peak_value",
        "drawdown_pct",
        "drawdown_breached",
        "daily_loss_pct",
        "daily_loss_breached",
        "correlation_breached",
        "correlation_violations",
        "total_leverage",
        "leverage_breached",
        "num_positions",
        "position_limit_violations",
    }

    def test_report_contains_all_expected_keys(self, rm: RiskManager) -> None:
        values = [100.0, 102.0]
        weights = {"SPY": 0.25, "TLT": 0.25}
        returns = _make_returns(["SPY", "TLT"])
        report = rm.generate_risk_report(values, weights, returns)
        assert self._EXPECTED_KEYS.issubset(report.keys())

    def test_report_reflects_drawdown_breach(self) -> None:
        rm = RiskManager({"max_drawdown_pct": 0.10})
        values = [100.0, 85.0]  # 15% dd > 10% limit
        report = rm.generate_risk_report(values, {}, pd.DataFrame())
        assert report["drawdown_breached"]
        assert report["drawdown_pct"] == pytest.approx(0.15, rel=1e-4)

    def test_report_leverage_calculation(self, rm: RiskManager) -> None:
        values = [100.0]
        weights = {"A": 0.30, "B": 0.20}
        returns = _make_returns(["A", "B"])
        report = rm.generate_risk_report(values, weights, returns)
        assert report["total_leverage"] == pytest.approx(0.50, rel=1e-4)
        assert not report["leverage_breached"]

    def test_report_num_positions_ignores_zero_weights(self, rm: RiskManager) -> None:
        values = [100.0]
        weights = {"A": 0.25, "B": 0.0, "C": 0.10}
        returns = _make_returns(["A", "B", "C"])
        report = rm.generate_risk_report(values, weights, returns)
        assert report["num_positions"] == 2

    def test_report_position_limit_violations_listed(self, rm: RiskManager) -> None:
        values = [100.0]
        weights = {"A": 0.40}  # > 0.30 limit, but PortfolioAction won't be used here
        returns = _make_returns(["A"])
        report = rm.generate_risk_report(values, weights, returns)
        assert "A" in report["position_limit_violations"]

    def test_report_empty_portfolio(self, rm: RiskManager) -> None:
        report = rm.generate_risk_report([], {}, pd.DataFrame())
        assert report["current_value"] == 0.0
        assert report["num_positions"] == 0
        assert report["total_leverage"] == 0.0

    def test_report_single_value(self, rm: RiskManager) -> None:
        report = rm.generate_risk_report([100.0], {"SPY": 0.30}, _make_returns(["SPY"]))
        assert report["current_value"] == 100.0
        assert report["daily_loss_pct"] == 0.0  # today == yesterday
