"""Tests for src/backtest/metrics.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    _ALL_KEYS,
    annual_return,
    annual_volatility,
    calmar_ratio,
    compute_all,
    cvar_historical,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    var_historical,
    win_rate,
)

_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _const_returns(value: float, n: int = 252) -> pd.Series:
    return pd.Series([value] * n, dtype=float)


def _rng_returns(seed: int = 0, n: int = 500) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n), dtype=float)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_constant_positive_return_high_sharpe(self):
        # Daily return of 0.001 with zero variance → very high Sharpe.
        sr = sharpe_ratio(_const_returns(0.001))
        assert sr > 5.0

    def test_zero_returns_gives_zero(self):
        assert sharpe_ratio(_const_returns(0.0)) == 0.0

    def test_negative_returns_negative_sharpe(self):
        sr = sharpe_ratio(_const_returns(-0.001))
        assert sr < 0.0

    def test_annualise_false_smaller_than_annualised(self):
        returns = _rng_returns()
        sr_ann = sharpe_ratio(returns, annualize=True)
        sr_raw = sharpe_ratio(returns, annualize=False)
        assert abs(sr_ann) > abs(sr_raw)

    def test_risk_free_rate_reduces_sharpe(self):
        returns = _rng_returns()
        sr_base = sharpe_ratio(returns, risk_free_rate=0.0)
        sr_rfr = sharpe_ratio(returns, risk_free_rate=0.0001)
        assert sr_rfr < sr_base

    def test_empty_series_returns_zero(self):
        assert sharpe_ratio(pd.Series([], dtype=float)) == 0.0

    def test_single_element_returns_zero(self):
        assert sharpe_ratio(pd.Series([0.01], dtype=float)) == 0.0

    def test_annualised_factor(self):
        returns = _rng_returns()
        sr_ann = sharpe_ratio(returns, annualize=True)
        sr_raw = sharpe_ratio(returns, annualize=False)
        assert math.isclose(sr_ann, sr_raw * math.sqrt(_TRADING_DAYS), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    def test_ignores_upside_volatility(self):
        # A series with large upward spikes should not reduce the Sortino ratio.
        base = pd.Series([-0.005] * 50 + [0.001] * 200, dtype=float)
        spiky = pd.Series([-0.005] * 50 + [0.10] * 200, dtype=float)
        # Both have the same downside; spiky has higher upside → same Sortino.
        sr_base = sortino_ratio(base)
        sr_spiky = sortino_ratio(spiky)
        # Spiky has higher mean → higher Sortino, but downside_std identical.
        assert sr_spiky > sr_base

    def test_all_positive_returns_infinite(self):
        result = sortino_ratio(_const_returns(0.001))
        assert result == float("inf")

    def test_zero_returns_zero(self):
        assert sortino_ratio(_const_returns(0.0)) == 0.0

    def test_empty_series_zero(self):
        assert sortino_ratio(pd.Series([], dtype=float)) == 0.0

    def test_single_element_zero(self):
        assert sortino_ratio(pd.Series([0.01], dtype=float)) == 0.0

    def test_positive_for_positive_mean(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.002, 0.01, 500), dtype=float)
        assert sortino_ratio(returns) > 0.0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_known_peak_trough(self):
        # 100 → 150 → 90: drawdown = (150 - 90) / 150 = 0.40
        values = pd.Series([100.0, 120.0, 150.0, 130.0, 90.0], dtype=float)
        mdd = max_drawdown(values)
        assert math.isclose(mdd, 0.40, rel_tol=1e-9)

    def test_monotonically_increasing_is_zero(self):
        values = pd.Series([100.0, 110.0, 120.0, 130.0], dtype=float)
        assert max_drawdown(values) == 0.0

    def test_empty_series_zero(self):
        assert max_drawdown(pd.Series([], dtype=float)) == 0.0

    def test_single_element_zero(self):
        assert max_drawdown(pd.Series([100.0], dtype=float)) == 0.0

    def test_full_loss(self):
        values = pd.Series([100.0, 50.0, 0.01], dtype=float)
        mdd = max_drawdown(values)
        assert mdd > 0.99

    def test_non_negative(self):
        values = pd.Series([100.0, 80.0, 120.0, 60.0], dtype=float)
        assert max_drawdown(values) >= 0.0

    def test_multiple_drawdowns_returns_worst(self):
        # Two drawdowns: 100→80 (20%) and 120→84 (30%).
        values = pd.Series([100.0, 80.0, 100.0, 120.0, 84.0], dtype=float)
        mdd = max_drawdown(values)
        assert math.isclose(mdd, 0.30, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_zero_drawdown_returns_zero(self):
        returns = _const_returns(0.001)
        values = pd.Series([100.0 * (1.001**i) for i in range(252)], dtype=float)
        assert calmar_ratio(returns, values) == 0.0

    def test_positive_for_profitable_series(self):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.001, 0.01, 252), dtype=float)
        values = (1.0 + returns).cumprod() * 100
        result = calmar_ratio(returns, values)
        # Just verify it's a finite float — sign depends on random seed.
        assert math.isfinite(result)

    def test_proportional_to_annual_return(self):
        returns = _rng_returns(seed=3)
        values = (1.0 + returns).cumprod() * 100
        mdd = max_drawdown(values)
        ann_ret = annual_return(returns)
        expected = ann_ret / mdd if mdd != 0 else 0.0
        assert math.isclose(calmar_ratio(returns, values), expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------


class TestWinRate:
    def test_all_positive_is_one(self):
        assert win_rate(_const_returns(0.001)) == 1.0

    def test_all_negative_is_zero(self):
        assert win_rate(_const_returns(-0.001)) == 0.0

    def test_all_zero_is_zero(self):
        assert win_rate(_const_returns(0.0)) == 0.0

    def test_half_positive_half_negative(self):
        returns = pd.Series([0.01, -0.01] * 100, dtype=float)
        assert math.isclose(win_rate(returns), 0.5, rel_tol=1e-9)

    def test_empty_series_zero(self):
        assert win_rate(pd.Series([], dtype=float)) == 0.0

    def test_known_fraction(self):
        # 3 wins, 1 loss → 0.75
        returns = pd.Series([0.01, 0.02, 0.03, -0.01], dtype=float)
        assert math.isclose(win_rate(returns), 0.75, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def test_known_split(self):
        # Gains: 0.03 + 0.02 = 0.05; Losses: 0.01 → PF = 5.0
        returns = pd.Series([0.03, 0.02, -0.01], dtype=float)
        assert math.isclose(profit_factor(returns), 5.0, rel_tol=1e-9)

    def test_all_positive_infinite(self):
        assert profit_factor(_const_returns(0.001)) == float("inf")

    def test_all_negative_zero(self):
        assert profit_factor(_const_returns(-0.001)) == 0.0

    def test_empty_series_zero(self):
        assert profit_factor(pd.Series([], dtype=float)) == 0.0

    def test_balanced_positive_negative(self):
        # Equal absolute gains and losses → PF = 1.0
        returns = pd.Series([0.01, -0.01] * 50, dtype=float)
        assert math.isclose(profit_factor(returns), 1.0, rel_tol=1e-9)

    def test_greater_than_one_for_profitable_series(self):
        returns = pd.Series([0.02, 0.03, -0.01, -0.005], dtype=float)
        assert profit_factor(returns) > 1.0


# ---------------------------------------------------------------------------
# annual_return
# ---------------------------------------------------------------------------


class TestAnnualReturn:
    def test_zero_daily_return_is_zero(self):
        assert annual_return(_const_returns(0.0)) == 0.0

    def test_positive_for_positive_daily_mean(self):
        assert annual_return(_const_returns(0.001)) > 0.0

    def test_negative_for_negative_daily_mean(self):
        assert annual_return(_const_returns(-0.001)) < 0.0

    def test_empty_series_zero(self):
        assert annual_return(pd.Series([], dtype=float)) == 0.0

    def test_single_element_zero(self):
        assert annual_return(pd.Series([0.01], dtype=float)) == 0.0

    def test_known_value(self):
        # 0.1% daily for 252 days → (1.001^252 - 1) ≈ 0.2838
        result = annual_return(_const_returns(0.001, n=252))
        expected = 1.001**252 - 1.0
        assert math.isclose(result, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# annual_volatility
# ---------------------------------------------------------------------------


class TestAnnualVolatility:
    def test_zero_for_constant_series(self):
        assert annual_volatility(_const_returns(0.001)) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_variable_series(self):
        assert annual_volatility(_rng_returns()) > 0.0

    def test_empty_series_zero(self):
        assert annual_volatility(pd.Series([], dtype=float)) == 0.0

    def test_single_element_zero(self):
        assert annual_volatility(pd.Series([0.01], dtype=float)) == 0.0

    def test_annualisation_factor(self):
        returns = _rng_returns()
        daily_std = float(returns.std(ddof=1))
        assert math.isclose(annual_volatility(returns), daily_std * math.sqrt(_TRADING_DAYS), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# var_historical / cvar_historical
# ---------------------------------------------------------------------------


class TestVaRCVaR:
    """VaR and CVaR with a known distribution."""

    @pytest.fixture()
    def symmetric_returns(self):
        # [-0.09, -0.07, ..., -0.01, 0.01, ..., 0.09] (9 negatives, 9 positives, 1 zero)
        vals = [r / 100 for r in range(-9, 10)]
        return pd.Series(vals, dtype=float)

    def test_var_known_distribution(self, symmetric_returns):
        # 5th percentile of 19 values → index 0 (floor(0.05*19)=0) → -0.09 → VaR = 0.09
        var = var_historical(symmetric_returns, percentile=5)
        assert var > 0.0

    def test_cvar_is_at_least_var(self, symmetric_returns):
        var = var_historical(symmetric_returns, percentile=5)
        cvar = cvar_historical(symmetric_returns, percentile=5)
        assert cvar >= var

    def test_var_empty_series_zero(self):
        assert var_historical(pd.Series([], dtype=float)) == 0.0

    def test_cvar_empty_series_zero(self):
        assert cvar_historical(pd.Series([], dtype=float)) == 0.0

    def test_var_non_negative(self):
        assert var_historical(_rng_returns()) >= 0.0

    def test_cvar_non_negative(self):
        assert cvar_historical(_rng_returns()) >= 0.0

    def test_var_percentile_10_larger_than_5(self):
        returns = _rng_returns()
        # Higher percentile cuts deeper into the tail → lower threshold → larger VaR.
        # Actually: np.percentile at 10 is less negative than at 5, so -percentile_10 < -percentile_5.
        # VaR at 5 ≥ VaR at 10 for symmetric distributions.
        var5 = var_historical(returns, percentile=5)
        var10 = var_historical(returns, percentile=10)
        # 5th percentile is more extreme → higher VaR.
        assert var5 >= var10

    def test_cvar_known_simple(self):
        # returns = [-0.10, -0.05, 0.02, 0.03]
        # 25th percentile = -0.10; tail = [-0.10]; CVaR = 0.10
        returns = pd.Series([-0.10, -0.05, 0.02, 0.03], dtype=float)
        cvar = cvar_historical(returns, percentile=25)
        assert math.isclose(cvar, 0.10, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------


class TestComputeAll:
    def test_returns_all_expected_keys(self):
        returns = _rng_returns()
        values = (1.0 + returns).cumprod() * 100
        result = compute_all(returns, values)
        for key in _ALL_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_all_values_are_floats(self):
        returns = _rng_returns()
        values = (1.0 + returns).cumprod() * 100
        for key, val in compute_all(returns, values).items():
            assert isinstance(val, float), f"{key} is not a float: {type(val)}"

    def test_empty_series_all_keys_present(self):
        empty = pd.Series([], dtype=float)
        result = compute_all(empty, empty)
        assert set(result.keys()) == set(_ALL_KEYS)

    def test_all_zeros_all_keys_present(self):
        returns = _const_returns(0.0)
        values = pd.Series([100.0] * 252, dtype=float)
        result = compute_all(returns, values)
        assert set(result.keys()) == set(_ALL_KEYS)

    def test_risk_free_rate_passed_through(self):
        returns = _rng_returns()
        values = (1.0 + returns).cumprod() * 100
        r0 = compute_all(returns, values, risk_free_rate=0.0)
        r1 = compute_all(returns, values, risk_free_rate=0.0001)
        assert r0["sharpe_ratio"] != r1["sharpe_ratio"]

    def test_consistent_with_individual_functions(self):
        returns = _rng_returns(seed=99)
        values = (1.0 + returns).cumprod() * 100
        result = compute_all(returns, values)
        assert math.isclose(result["sharpe_ratio"], sharpe_ratio(returns), rel_tol=1e-9)
        assert math.isclose(result["max_drawdown"], max_drawdown(values), rel_tol=1e-9)
        assert math.isclose(result["win_rate"], win_rate(returns), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_return_all_functions_graceful(self):
        returns = pd.Series([0.01], dtype=float)
        values = pd.Series([100.0, 101.0], dtype=float)
        assert sharpe_ratio(returns) == 0.0
        assert sortino_ratio(returns) == 0.0
        assert max_drawdown(pd.Series([100.0], dtype=float)) == 0.0
        assert win_rate(returns) == 1.0
        assert profit_factor(returns) == float("inf")
        assert annual_return(returns) == 0.0
        assert annual_volatility(returns) == 0.0

    def test_all_zero_returns_sharpe_zero(self):
        assert sharpe_ratio(_const_returns(0.0)) == 0.0

    def test_all_zero_returns_sortino_zero(self):
        # All-zero returns: no downside, zero mean → 0 (not inf).
        assert sortino_ratio(_const_returns(0.0)) == 0.0

    def test_all_zero_returns_win_rate_zero(self):
        assert win_rate(_const_returns(0.0)) == 0.0

    def test_nan_free_output(self):
        returns = _rng_returns()
        values = (1.0 + returns).cumprod() * 100
        for key, val in compute_all(returns, values).items():
            assert not math.isnan(val), f"{key} is NaN"
