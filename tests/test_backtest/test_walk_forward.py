"""Tests for src/backtest/walk_forward.py."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.walk_forward import WalkForwardResult, WalkForwardValidator
from src.data.base import AssetData
from src.execution.base import Order
from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.portfolio.rebalancer import Rebalancer
from src.portfolio.risk_manager import RiskManager
from src.signals.base import Signal, SignalGenerator

# ---------------------------------------------------------------------------
# Shared helpers — identical style to test_engine.py
# ---------------------------------------------------------------------------

_SYMBOLS = ["SPY", "TLT"]
_TRAIN = 252  # 1 year
_TEST = 63  # 1 quarter


def _make_dates(n: int, start: str = "2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _make_ohlcv(n: int, dates: pd.DatetimeIndex, daily_return: float = 0.0) -> pd.DataFrame:
    prices = [100.0 * (1.0 + daily_return) ** i for i in range(n)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1_000_000] * n,
        },
        index=dates,
    )


def _make_data(
    n_days: int,
    symbols: list[str] = _SYMBOLS,
    daily_return: float = 0.001,
    start: str = "2020-01-02",
) -> dict[str, AssetData]:
    dates = _make_dates(n_days, start)
    return {
        sym: AssetData(symbol=sym, ohlcv=_make_ohlcv(n_days, dates, daily_return))
        for sym in symbols
    }


# ---------------------------------------------------------------------------
# Minimal mock implementations (same pattern as test_engine.py)
# ---------------------------------------------------------------------------


class _TrainCountingAgent(PortfolioAgent):
    """Records how many times train() is called and with what data."""

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols
        self.train_calls: list[dict] = []

    def decide(self, signals: list[Signal], current_portfolio: dict) -> PortfolioAction:
        w = round(1.0 / len(self._symbols), 9)
        weights = {sym: w for sym in self._symbols}
        weights[self._symbols[-1]] = round(1.0 - w * (len(self._symbols) - 1), 9)
        return PortfolioAction(weights=weights, confidence=0.8, regime_context="neutral")

    def train(self, historical_signals: list[Signal], returns: pd.DataFrame) -> dict:
        self.train_calls.append(
            {
                "n_signals": len(historical_signals),
                "returns_shape": returns.shape,
            }
        )
        return {"loss": 0.0}


class _MockSignalGenerator(SignalGenerator):
    def __init__(self, name_: str = "mock") -> None:
        self._name = name_

    @property
    def name(self) -> str:
        return self._name

    def generate(self, data: dict[str, AssetData]) -> Signal:
        return self._build(data)

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        return self._build(new_data)

    def _build(self, data: dict[str, AssetData]) -> Signal:
        n = len(data)
        return Signal(
            name=self._name,
            values=np.ones(n, dtype=float),
            confidence=np.full(n, 0.7, dtype=float),
        )


def _make_engine(
    symbols: list[str] = _SYMBOLS,
    agent: PortfolioAgent | None = None,
    cost_bps: float = 0.0,
    frequency: str = "monthly",
) -> BacktestEngine:
    if agent is None:
        agent = _TrainCountingAgent(symbols)
    return BacktestEngine(
        signals=[_MockSignalGenerator()],
        agent=agent,
        risk_manager=RiskManager({"max_position_pct": 0.60}),
        rebalancer=Rebalancer(),
        initial_capital=100_000.0,
        rebalance_frequency=frequency,
        transaction_cost_bps=cost_bps,
    )


def _make_validator(
    n_days: int,
    train_window: int = _TRAIN,
    test_window: int = _TEST,
    expanding: bool = True,
    cost_bps: float = 0.0,
) -> tuple[WalkForwardValidator, dict[str, AssetData]]:
    data = _make_data(n_days)
    engine = _make_engine(cost_bps=cost_bps)
    validator = WalkForwardValidator(
        engine=engine,
        train_window=train_window,
        test_window=test_window,
        expanding=expanding,
    )
    return validator, data


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_zero_train_window_raises(self):
        with pytest.raises(ValueError, match="train_window"):
            WalkForwardValidator(_make_engine(), train_window=0, test_window=63)

    def test_negative_test_window_raises(self):
        with pytest.raises(ValueError, match="test_window"):
            WalkForwardValidator(_make_engine(), train_window=252, test_window=-1)

    def test_zero_test_window_raises(self):
        with pytest.raises(ValueError, match="test_window"):
            WalkForwardValidator(_make_engine(), train_window=252, test_window=0)

    def test_valid_construction(self):
        v = WalkForwardValidator(_make_engine(), train_window=100, test_window=50)
        assert v is not None


# ---------------------------------------------------------------------------
# Fold count (2 years of data, 252 train + 63 test)
# ---------------------------------------------------------------------------


class TestFoldCount:
    def test_two_years_expanding_correct_folds(self):
        # 2 years ≈ 504 trading days.
        # Folds: train=252,test=63 → folds fit while train+test ≤ 504.
        # fold 0: train[0:252]   test[252:315]
        # fold 1: train[0:315]   test[315:378]
        # fold 2: train[0:378]   test[378:441]
        # fold 3: train[0:441]   test[441:504]
        # fold 4: train[0:504]   test[504:567] → 567>504 → stops at fold 3+1=4 folds
        n = 504
        validator, data = _make_validator(n, _TRAIN, _TEST, expanding=True)
        result = validator.run(data)
        # (504 - 252) // 63 = 4
        assert result.n_folds == 4

    def test_two_years_rolling_correct_folds(self):
        n = 504
        validator, data = _make_validator(n, _TRAIN, _TEST, expanding=False)
        result = validator.run(data)
        assert result.n_folds == 4

    def test_exactly_one_fold(self):
        # Just enough for one fold.
        n = _TRAIN + _TEST
        validator, data = _make_validator(n, _TRAIN, _TEST)
        result = validator.run(data)
        assert result.n_folds == 1

    def test_one_extra_day_does_not_add_fold(self):
        # One day past one fold — not enough for a second test window.
        n = _TRAIN + _TEST + 1
        validator, data = _make_validator(n, _TRAIN, _TEST)
        result = validator.run(data)
        assert result.n_folds == 1

    def test_exactly_two_folds(self):
        n = _TRAIN + 2 * _TEST
        validator, data = _make_validator(n, _TRAIN, _TEST)
        result = validator.run(data)
        assert result.n_folds == 2

    def test_n_folds_matches_folds_list_length(self):
        validator, data = _make_validator(504)
        result = validator.run(data)
        assert result.n_folds == len(result.folds)


# ---------------------------------------------------------------------------
# Data too short
# ---------------------------------------------------------------------------


class TestDataTooShort:
    def test_empty_data_raises(self):
        engine = _make_engine()
        validator = WalkForwardValidator(engine, train_window=252, test_window=63)
        with pytest.raises(ValueError, match="at least one asset"):
            validator.run({})

    def test_too_few_days_raises(self):
        n = _TRAIN + _TEST - 1  # one day short of one fold
        validator, data = _make_validator(n, _TRAIN, _TEST)
        with pytest.raises(ValueError, match="at least"):
            validator.run(data)

    def test_only_train_window_raises(self):
        n = _TRAIN
        validator, data = _make_validator(n, _TRAIN, _TEST)
        with pytest.raises(ValueError):
            validator.run(data)


# ---------------------------------------------------------------------------
# Expanding vs rolling window
# ---------------------------------------------------------------------------


class TestExpandingWindow:
    def test_expanding_train_grows_each_fold(self):
        """Each fold's training data must be larger than the previous fold's."""
        n = _TRAIN + 3 * _TEST
        agent = _TrainCountingAgent(_SYMBOLS)
        engine = _make_engine(agent=agent)
        validator = WalkForwardValidator(engine, train_window=_TRAIN, test_window=_TEST, expanding=True)
        data = _make_data(n)
        result = validator.run(data)

        # agent.train_calls records the returns_shape for each fold.
        shapes = [call["returns_shape"][0] for call in agent.train_calls]
        for i in range(1, len(shapes)):
            assert shapes[i] > shapes[i - 1], (
                f"fold {i} train size {shapes[i]} should exceed fold {i-1} size {shapes[i-1]}"
            )

    def test_expanding_first_fold_size_is_train_window(self):
        n = _TRAIN + _TEST
        agent = _TrainCountingAgent(_SYMBOLS)
        engine = _make_engine(agent=agent)
        validator = WalkForwardValidator(engine, train_window=_TRAIN, test_window=_TEST, expanding=True)
        data = _make_data(n)
        validator.run(data)
        assert agent.train_calls[0]["returns_shape"][0] == _TRAIN


class TestRollingWindow:
    def test_rolling_train_stays_fixed_size(self):
        n = _TRAIN + 3 * _TEST
        agent = _TrainCountingAgent(_SYMBOLS)
        engine = _make_engine(agent=agent)
        validator = WalkForwardValidator(engine, train_window=_TRAIN, test_window=_TEST, expanding=False)
        data = _make_data(n)
        validator.run(data)

        shapes = [call["returns_shape"][0] for call in agent.train_calls]
        assert all(s == _TRAIN for s in shapes), f"Rolling train sizes should all be {_TRAIN}, got {shapes}"

    def test_rolling_vs_expanding_same_fold_count(self):
        n = _TRAIN + 3 * _TEST
        v_exp, data = _make_validator(n, _TRAIN, _TEST, expanding=True)
        v_rol, _ = _make_validator(n, _TRAIN, _TEST, expanding=False)
        assert v_exp.run(data).n_folds == v_rol.run(data).n_folds


# ---------------------------------------------------------------------------
# Combined returns
# ---------------------------------------------------------------------------


class TestCombinedReturns:
    def test_combined_returns_length_equals_sum_of_test_windows(self):
        # Each BacktestResult.returns has length = test_window - 1 (pct_change drops first).
        # Combined = n_folds * (test_window - 1).
        n = _TRAIN + 3 * _TEST
        validator, data = _make_validator(n, _TRAIN, _TEST)
        result = validator.run(data)

        expected_len = sum(len(f.returns) for f in result.folds)
        assert len(result.combined_returns) == expected_len

    def test_combined_returns_is_series(self):
        validator, data = _make_validator(_TRAIN + _TEST, _TRAIN, _TEST)
        result = validator.run(data)
        assert isinstance(result.combined_returns, pd.Series)

    def test_combined_returns_date_indexed(self):
        validator, data = _make_validator(_TRAIN + _TEST, _TRAIN, _TEST)
        result = validator.run(data)
        assert isinstance(result.combined_returns.index, pd.DatetimeIndex)

    def test_combined_returns_monotonically_increasing_index(self):
        n = _TRAIN + 2 * _TEST
        validator, data = _make_validator(n, _TRAIN, _TEST)
        result = validator.run(data)
        assert result.combined_returns.index.is_monotonic_increasing

    def test_no_duplicate_dates_in_combined(self):
        n = _TRAIN + 2 * _TEST
        validator, data = _make_validator(n, _TRAIN, _TEST)
        result = validator.run(data)
        assert result.combined_returns.index.is_unique


# ---------------------------------------------------------------------------
# Fold metrics
# ---------------------------------------------------------------------------


class TestFoldMetrics:
    def test_fold_metrics_length_equals_n_folds(self):
        n = _TRAIN + 3 * _TEST
        validator, data = _make_validator(n, _TRAIN, _TEST)
        result = validator.run(data)
        assert len(result.fold_metrics) == result.n_folds

    def test_each_fold_metric_has_expected_keys(self):
        expected = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "profit_factor", "annual_return", "annual_volatility",
            "var_historical", "cvar_historical",
        }
        validator, data = _make_validator(_TRAIN + _TEST, _TRAIN, _TEST)
        result = validator.run(data)
        for i, fm in enumerate(result.fold_metrics):
            assert expected.issubset(fm.keys()), f"Fold {i} missing keys: {expected - fm.keys()}"

    def test_fold_metrics_match_fold_results(self):
        n = _TRAIN + 2 * _TEST
        validator, data = _make_validator(n, _TRAIN, _TEST)
        result = validator.run(data)
        for i, (fm, fold) in enumerate(zip(result.fold_metrics, result.folds)):
            assert fm == fold.metrics, f"Fold {i} metrics mismatch"


# ---------------------------------------------------------------------------
# Combined metrics
# ---------------------------------------------------------------------------


class TestCombinedMetrics:
    def test_combined_metrics_has_expected_keys(self):
        expected = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "profit_factor", "annual_return", "annual_volatility",
            "var_historical", "cvar_historical",
        }
        validator, data = _make_validator(_TRAIN + _TEST, _TRAIN, _TEST)
        result = validator.run(data)
        assert expected.issubset(result.combined_metrics.keys())

    def test_combined_metrics_is_dict(self):
        validator, data = _make_validator(_TRAIN + _TEST, _TRAIN, _TEST)
        result = validator.run(data)
        assert isinstance(result.combined_metrics, dict)


# ---------------------------------------------------------------------------
# WalkForwardResult fields
# ---------------------------------------------------------------------------


class TestResultFields:
    @pytest.fixture()
    def result(self) -> WalkForwardResult:
        validator, data = _make_validator(_TRAIN + 2 * _TEST)
        return validator.run(data)

    def test_folds_is_list(self, result):
        assert isinstance(result.folds, list)

    def test_combined_returns_present(self, result):
        assert result.combined_returns is not None

    def test_combined_metrics_present(self, result):
        assert result.combined_metrics is not None

    def test_fold_metrics_present(self, result):
        assert isinstance(result.fold_metrics, list)

    def test_n_folds_positive(self, result):
        assert result.n_folds > 0

    def test_each_fold_is_backtest_result(self, result):
        from src.backtest.engine import BacktestResult
        for fold in result.folds:
            assert isinstance(fold, BacktestResult)


# ---------------------------------------------------------------------------
# Agent.train() is called once per fold
# ---------------------------------------------------------------------------


class TestAgentTraining:
    def test_train_called_once_per_fold(self):
        n = _TRAIN + 3 * _TEST
        agent = _TrainCountingAgent(_SYMBOLS)
        engine = _make_engine(agent=agent)
        validator = WalkForwardValidator(engine, train_window=_TRAIN, test_window=_TEST)
        data = _make_data(n)
        result = validator.run(data)
        assert len(agent.train_calls) == result.n_folds

    def test_train_receives_signals(self):
        n = _TRAIN + _TEST
        agent = _TrainCountingAgent(_SYMBOLS)
        engine = _make_engine(agent=agent)
        validator = WalkForwardValidator(engine, train_window=_TRAIN, test_window=_TEST)
        data = _make_data(n)
        validator.run(data)
        # At least one signal per fold (one generator).
        assert agent.train_calls[0]["n_signals"] >= 1
