"""Tests for src/portfolio/rebalancer.py."""

from __future__ import annotations

import pytest

from src.execution.base import Order
from src.portfolio.rebalancer import Rebalancer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r() -> Rebalancer:
    return Rebalancer()


def _orders_by_symbol(orders: list[Order]) -> dict[str, Order]:
    return {o.symbol: o for o in orders}


# ---------------------------------------------------------------------------
# compute_trades — basic rebalance
# ---------------------------------------------------------------------------


class TestComputeTradesBasic:
    def test_50_50_to_70_30(self) -> None:
        """Classic two-asset rebalance: A grows, B shrinks."""
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.70, "B": 0.30}
        orders = r.compute_trades(current, target, portfolio_value=10_000)

        by_sym = _orders_by_symbol(orders)
        assert "A" in by_sym and "B" in by_sym
        assert by_sym["A"].side == "buy"
        assert by_sym["B"].side == "sell"
        assert by_sym["A"].quantity == pytest.approx(2_000.0, rel=1e-5)
        assert by_sym["B"].quantity == pytest.approx(2_000.0, rel=1e-5)

    def test_equal_weights_no_trades(self) -> None:
        r = _r()
        w = {"A": 0.50, "B": 0.50}
        orders = r.compute_trades(w, w.copy(), portfolio_value=10_000)
        assert orders == []

    def test_returns_list_of_orders(self) -> None:
        r = _r()
        orders = r.compute_trades(
            {"A": 0.40, "B": 0.60},
            {"A": 0.60, "B": 0.40},
            portfolio_value=5_000,
        )
        assert isinstance(orders, list)
        assert all(isinstance(o, Order) for o in orders)

    def test_all_orders_are_market(self) -> None:
        r = _r()
        orders = r.compute_trades(
            {"A": 0.30, "B": 0.70},
            {"A": 0.70, "B": 0.30},
            portfolio_value=1_000,
        )
        assert all(o.order_type == "market" for o in orders)

    def test_notional_scales_with_portfolio_value(self) -> None:
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.70, "B": 0.30}
        small = r.compute_trades(current, target, portfolio_value=1_000)
        large = r.compute_trades(current, target, portfolio_value=100_000)
        by_small = _orders_by_symbol(small)
        by_large = _orders_by_symbol(large)
        assert by_large["A"].quantity == pytest.approx(100 * by_small["A"].quantity, rel=1e-5)

    def test_three_assets(self) -> None:
        r = _r()
        current = {"A": 0.33, "B": 0.33, "C": 0.34}
        target = {"A": 0.50, "B": 0.20, "C": 0.30}
        orders = r.compute_trades(current, target, portfolio_value=10_000)
        by_sym = _orders_by_symbol(orders)
        assert by_sym["A"].side == "buy"
        assert by_sym["B"].side == "sell"

    def test_sells_before_buys(self) -> None:
        """Sell orders must appear before buy orders in the returned list."""
        r = _r()
        orders = r.compute_trades(
            {"A": 0.20, "B": 0.80},
            {"A": 0.80, "B": 0.20},
            portfolio_value=10_000,
        )
        sides = [o.side for o in orders]
        sells = [i for i, s in enumerate(sides) if s == "sell"]
        buys = [i for i, s in enumerate(sides) if s == "buy"]
        if sells and buys:
            assert max(sells) < min(buys)


# ---------------------------------------------------------------------------
# compute_trades — min_trade_pct filter
# ---------------------------------------------------------------------------


class TestMinTradePct:
    def test_tiny_drift_filtered(self) -> None:
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.505, "B": 0.495}   # 0.5 % drift, below 1 % default
        orders = r.compute_trades(current, target, portfolio_value=10_000)
        assert orders == []

    def test_exactly_at_threshold_not_filtered(self) -> None:
        """A drift exactly equal to min_trade_pct must pass the filter."""
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.51, "B": 0.49}     # exactly 0.01 on each
        orders = r.compute_trades(current, target, portfolio_value=10_000, min_trade_pct=0.01)
        # 0.01 is NOT < 0.01, so orders should be generated
        assert len(orders) == 2

    def test_custom_min_trade_pct(self) -> None:
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.53, "B": 0.47}     # 3 % drift
        # With min 5 %, should be filtered; with min 1 %, should not.
        assert r.compute_trades(current, target, 10_000, min_trade_pct=0.05) == []
        assert r.compute_trades(current, target, 10_000, min_trade_pct=0.01) != []

    def test_one_asset_filtered_one_not(self) -> None:
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.60, "B": 0.45}     # A: +10 % (passes), B: -5 % (passes at 0.01)
        orders = r.compute_trades(current, target, 10_000, min_trade_pct=0.08)
        # Only A exceeds 8 %
        syms = {o.symbol for o in orders}
        assert "A" in syms
        assert "B" not in syms


# ---------------------------------------------------------------------------
# compute_trades — new entry and full exit
# ---------------------------------------------------------------------------


class TestEntryAndExit:
    def test_new_position_entry(self) -> None:
        """Symbol absent from current_weights should generate a buy."""
        r = _r()
        current = {"A": 1.0}
        target = {"A": 0.70, "C": 0.30}
        orders = r.compute_trades(current, target, portfolio_value=10_000)
        by_sym = _orders_by_symbol(orders)
        assert by_sym["C"].side == "buy"
        assert by_sym["C"].quantity == pytest.approx(3_000.0, rel=1e-5)
        assert by_sym["A"].side == "sell"

    def test_full_position_exit(self) -> None:
        """Symbol absent from target_weights should generate a full sell."""
        r = _r()
        current = {"A": 0.60, "B": 0.40}
        target = {"A": 1.0}
        orders = r.compute_trades(current, target, portfolio_value=10_000)
        by_sym = _orders_by_symbol(orders)
        assert "B" in by_sym
        assert by_sym["B"].side == "sell"
        assert by_sym["B"].quantity == pytest.approx(4_000.0, rel=1e-5)

    def test_empty_current_to_target(self) -> None:
        """Starting from an all-cash portfolio should generate only buys."""
        r = _r()
        orders = r.compute_trades(
            {},
            {"A": 0.60, "B": 0.40},
            portfolio_value=10_000,
        )
        assert all(o.side == "buy" for o in orders)
        by_sym = _orders_by_symbol(orders)
        assert by_sym["A"].quantity == pytest.approx(6_000.0, rel=1e-5)
        assert by_sym["B"].quantity == pytest.approx(4_000.0, rel=1e-5)

    def test_full_liquidation(self) -> None:
        """Target of empty dict should sell everything."""
        r = _r()
        orders = r.compute_trades(
            {"A": 0.50, "B": 0.50},
            {},
            portfolio_value=10_000,
        )
        assert all(o.side == "sell" for o in orders)

    def test_zero_portfolio_value_returns_empty(self) -> None:
        r = _r()
        orders = r.compute_trades({"A": 0.50}, {"A": 1.0}, portfolio_value=0)
        assert orders == []

    def test_negative_portfolio_value_raises(self) -> None:
        r = _r()
        with pytest.raises(ValueError, match="portfolio_value"):
            r.compute_trades({"A": 0.50}, {"A": 1.0}, portfolio_value=-100)


# ---------------------------------------------------------------------------
# should_rebalance
# ---------------------------------------------------------------------------


class TestShouldRebalance:
    def test_no_drift_no_rebalance(self) -> None:
        r = _r()
        w = {"A": 0.50, "B": 0.50}
        assert r.should_rebalance(w, w.copy()) is False

    def test_drift_above_threshold_triggers(self) -> None:
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.60, "B": 0.40}    # 10 pp drift, threshold 5 pp
        assert r.should_rebalance(current, target, threshold=0.05) is True

    def test_drift_below_threshold_no_trigger(self) -> None:
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.53, "B": 0.47}    # 3 pp drift, threshold 5 pp
        assert r.should_rebalance(current, target, threshold=0.05) is False

    def test_just_below_threshold_no_trigger(self) -> None:
        """Drift clearly below threshold must not trigger."""
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.54, "B": 0.46}    # 4 pp drift, well below 5 pp threshold
        assert r.should_rebalance(current, target, threshold=0.05) is False

    def test_new_position_triggers(self) -> None:
        """New symbol in target counts as full drift."""
        r = _r()
        assert r.should_rebalance({"A": 1.0}, {"A": 0.70, "B": 0.30}, threshold=0.05) is True

    def test_full_exit_triggers(self) -> None:
        r = _r()
        assert r.should_rebalance({"A": 0.50, "B": 0.50}, {"A": 1.0}, threshold=0.05) is True

    def test_equal_weights_no_trigger(self) -> None:
        r = _r()
        w = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        assert r.should_rebalance(w, w.copy(), threshold=0.05) is False

    def test_custom_threshold(self) -> None:
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.52, "B": 0.48}   # 2 pp drift
        assert r.should_rebalance(current, target, threshold=0.01) is True
        assert r.should_rebalance(current, target, threshold=0.05) is False


# ---------------------------------------------------------------------------
# apply_turnover_constraint
# ---------------------------------------------------------------------------


class TestTurnoverConstraint:
    def test_within_budget_unchanged(self) -> None:
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.60, "B": 0.40}    # turnover = 0.5 * 0.20 = 0.10
        result = r.apply_turnover_constraint(current, target, max_turnover=0.30)
        assert result["A"] == pytest.approx(0.60, rel=1e-5)
        assert result["B"] == pytest.approx(0.40, rel=1e-5)

    def test_large_trade_clipped(self) -> None:
        """Full flip from 100 % A to 100 % B has turnover = 1.0; clip to 0.30."""
        r = _r()
        current = {"A": 1.0, "B": 0.0}
        target = {"A": 0.0, "B": 1.0}
        result = r.apply_turnover_constraint(current, target, max_turnover=0.30)
        # Each delta is ±1.0, scaled by 0.30.
        assert result["A"] == pytest.approx(1.0 - 0.30, rel=1e-5)
        assert result["B"] == pytest.approx(0.0 + 0.30, rel=1e-5)

    def test_50_50_to_70_30_clipped(self) -> None:
        """50/50 → 70/30: unconstrained turnover = 0.20; clip to 0.10."""
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.70, "B": 0.30}
        # Unconstrained turnover = 0.5 * (0.20 + 0.20) = 0.20
        result = r.apply_turnover_constraint(current, target, max_turnover=0.10)
        scale = 0.10 / 0.20
        assert result["A"] == pytest.approx(0.50 + 0.20 * scale, rel=1e-5)
        assert result["B"] == pytest.approx(0.50 - 0.20 * scale, rel=1e-5)

    def test_adjusted_weights_sum_preserved(self) -> None:
        """Uniform scaling of deltas preserves the original weight sum."""
        r = _r()
        current = {"A": 0.30, "B": 0.40, "C": 0.30}
        target = {"A": 0.60, "B": 0.10, "C": 0.30}
        result = r.apply_turnover_constraint(current, target, max_turnover=0.10)
        assert abs(sum(result.values()) - sum(current.values())) < 1e-6

    def test_new_position_entry_clipped(self) -> None:
        r = _r()
        current = {"A": 1.0}
        target = {"A": 0.50, "B": 0.50}
        result = r.apply_turnover_constraint(current, target, max_turnover=0.10)
        # unconstrained turnover = 0.5 * (0.50 + 0.50) = 0.50; scale = 0.10/0.50 = 0.20
        assert result["B"] == pytest.approx(0.50 * 0.20, rel=1e-5)

    def test_full_exit_clipped(self) -> None:
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 1.0, "B": 0.0}
        result = r.apply_turnover_constraint(current, target, max_turnover=0.10)
        # unconstrained turnover = 0.5 * (0.50 + 0.50) = 0.50; scale = 0.20
        assert result["B"] == pytest.approx(0.50 - 0.50 * 0.20, rel=1e-5)

    def test_invalid_max_turnover_raises(self) -> None:
        r = _r()
        with pytest.raises(ValueError, match="max_turnover"):
            r.apply_turnover_constraint({"A": 1.0}, {"A": 0.5}, max_turnover=0.0)
        with pytest.raises(ValueError, match="max_turnover"):
            r.apply_turnover_constraint({"A": 1.0}, {"A": 0.5}, max_turnover=1.5)

    def test_identical_weights_zero_turnover(self) -> None:
        r = _r()
        w = {"A": 0.50, "B": 0.50}
        result = r.apply_turnover_constraint(w, w.copy(), max_turnover=0.30)
        assert result["A"] == pytest.approx(0.50, rel=1e-5)
        assert result["B"] == pytest.approx(0.50, rel=1e-5)


# ---------------------------------------------------------------------------
# Integration: compute_trades after turnover constraint
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_constrained_then_compute_trades(self) -> None:
        """Applying turnover constraint before compute_trades reduces order sizes."""
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        target = {"A": 0.80, "B": 0.20}

        unconstrained_orders = r.compute_trades(current, target, 10_000)
        constrained_target = r.apply_turnover_constraint(current, target, max_turnover=0.10)
        constrained_orders = r.compute_trades(current, constrained_target, 10_000)

        unc_by_sym = _orders_by_symbol(unconstrained_orders)
        con_by_sym = _orders_by_symbol(constrained_orders)
        assert con_by_sym["A"].quantity < unc_by_sym["A"].quantity

    def test_should_rebalance_gates_compute_trades(self) -> None:
        """compute_trades should only be called when should_rebalance is True."""
        r = _r()
        current = {"A": 0.50, "B": 0.50}
        small_target = {"A": 0.52, "B": 0.48}  # 2 pp drift — below 5 pp threshold
        large_target = {"A": 0.60, "B": 0.40}  # 10 pp drift — above 5 pp threshold

        assert r.should_rebalance(current, small_target) is False
        assert r.should_rebalance(current, large_target) is True

        # Only compute trades when rebalance is warranted.
        if r.should_rebalance(current, large_target):
            orders = r.compute_trades(current, large_target, 10_000)
            assert len(orders) > 0
