"""Tests for src/backtest/engine.py."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, BacktestResult, _slice_data
from src.data.base import AssetData
from src.execution.base import Order
from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.portfolio.rebalancer import Rebalancer
from src.portfolio.risk_manager import RiskManager
from src.signals.base import Signal, SignalGenerator

# ---------------------------------------------------------------------------
# Shared test fixtures and helpers
# ---------------------------------------------------------------------------

_SYMBOLS = ["SPY", "TLT"]
_N_DAYS = 60  # ~3 months of daily data
_START = "2023-01-03"
_END = "2023-03-24"


def _make_dates(n: int, start: str = _START) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _make_ohlcv(prices: list[float], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1_000_000] * len(prices),
        },
        index=dates,
    )


def _make_data(
    symbols: list[str] = _SYMBOLS,
    n_days: int = _N_DAYS,
    daily_return: float = 0.001,
    start: str = _START,
) -> dict[str, AssetData]:
    """Synthetic data where each asset has a constant daily return."""
    dates = _make_dates(n_days, start)
    data: dict[str, AssetData] = {}
    for sym in symbols:
        prices = [100.0 * (1.0 + daily_return) ** i for i in range(n_days)]
        data[sym] = AssetData(symbol=sym, ohlcv=_make_ohlcv(prices, dates))
    return data


def _make_flat_data(
    symbols: list[str] = _SYMBOLS,
    n_days: int = _N_DAYS,
    start: str = _START,
) -> dict[str, AssetData]:
    """Synthetic data where all prices are constant (zero returns)."""
    return _make_data(symbols, n_days, daily_return=0.0, start=start)


# ---------------------------------------------------------------------------
# Minimal mock implementations
# ---------------------------------------------------------------------------


class _EqualWeightAgent(PortfolioAgent):
    """Always allocates equally across all symbols seen in signals."""

    def decide(self, signals: list[Signal], current_portfolio: dict) -> PortfolioAction:
        n = len(signals[0].values) if signals else 2
        weight = round(1.0 / n, 9)
        symbols = [f"S{i}" for i in range(n)]  # placeholder — overridden per test
        return PortfolioAction(
            weights={sym: weight for sym in symbols},
            confidence=0.8,
            regime_context="neutral",
        )

    def train(self, historical_signals, returns):
        return {}


class _FixedWeightAgent(PortfolioAgent):
    """Returns a pre-specified weight dict on every call."""

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights

    def decide(self, signals: list[Signal], current_portfolio: dict) -> PortfolioAction:
        return PortfolioAction(
            weights=dict(self._weights),
            confidence=0.9,
            regime_context="neutral",
        )

    def train(self, historical_signals, returns):
        return {}


class _MockSignalGenerator(SignalGenerator):
    """Returns a fixed signal for any data input."""

    def __init__(self, name_: str, n_assets: int = 2) -> None:
        self._name = name_
        self._n = n_assets

    @property
    def name(self) -> str:
        return self._name

    def generate(self, data: dict[str, AssetData]) -> Signal:
        return self._build(data)

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        return self._build(new_data)

    def _build(self, data: dict[str, AssetData]) -> Signal:
        symbols = sorted(data.keys())
        return Signal(
            name=self._name,
            values=np.ones(len(symbols), dtype=float),
            confidence=np.full(len(symbols), 0.7, dtype=float),
            regime="neutral",
        )


class _SymbolAwareAgent(PortfolioAgent):
    """Splits weight equally over known symbols."""

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols

    def decide(self, signals: list[Signal], current_portfolio: dict) -> PortfolioAction:
        w = round(1.0 / len(self._symbols), 9)
        # Ensure sum <= 1 with float precision
        weights = {sym: w for sym in self._symbols}
        weights[self._symbols[-1]] = round(1.0 - w * (len(self._symbols) - 1), 9)
        return PortfolioAction(weights=weights, confidence=0.8, regime_context="neutral")

    def train(self, historical_signals, returns):
        return {}


def _default_engine(
    symbols: list[str] = _SYMBOLS,
    n_days: int = _N_DAYS,
    daily_return: float = 0.001,
    frequency: str = "monthly",
    cost_bps: float = 5.0,
    initial_capital: float = 100_000.0,
    risk_config: dict | None = None,
) -> tuple[BacktestEngine, dict[str, AssetData]]:
    data = _make_data(symbols, n_days, daily_return)
    agent = _SymbolAwareAgent(symbols)
    engine = BacktestEngine(
        signals=[_MockSignalGenerator("mock", len(symbols))],
        agent=agent,
        risk_manager=RiskManager(risk_config),
        rebalancer=Rebalancer(),
        initial_capital=initial_capital,
        rebalance_frequency=frequency,
        transaction_cost_bps=cost_bps,
    )
    return engine, data


# ---------------------------------------------------------------------------
# BacktestEngine constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_invalid_frequency_raises(self):
        with pytest.raises(ValueError, match="rebalance_frequency"):
            BacktestEngine(
                signals=[],
                agent=_SymbolAwareAgent(_SYMBOLS),
                risk_manager=RiskManager(),
                rebalancer=Rebalancer(),
                rebalance_frequency="quarterly",
            )

    def test_non_positive_capital_raises(self):
        with pytest.raises(ValueError, match="initial_capital"):
            BacktestEngine(
                signals=[],
                agent=_SymbolAwareAgent(_SYMBOLS),
                risk_manager=RiskManager(),
                rebalancer=Rebalancer(),
                initial_capital=0.0,
            )

    def test_valid_frequencies_accepted(self):
        for freq in ("daily", "weekly", "monthly"):
            engine = BacktestEngine(
                signals=[],
                agent=_SymbolAwareAgent(_SYMBOLS),
                risk_manager=RiskManager(),
                rebalancer=Rebalancer(),
                rebalance_frequency=freq,
            )
            assert engine is not None


# ---------------------------------------------------------------------------
# run() — basic shape and correctness
# ---------------------------------------------------------------------------


class TestRunBasic:
    def test_portfolio_values_start_at_initial_capital(self):
        # Use zero costs so day-0 rebalancing does not change the recorded NAV.
        engine, data = _default_engine(cost_bps=0.0)
        dates = _make_dates(_N_DAYS)
        result = engine.run(data, start=str(dates[0].date()), end=str(dates[-1].date()))
        assert math.isclose(float(result.portfolio_values.iloc[0]), 100_000.0, rel_tol=1e-6)

    def test_result_is_backtest_result_instance(self):
        engine, data = _default_engine()
        dates = _make_dates(_N_DAYS)
        result = engine.run(data, start=str(dates[0].date()), end=str(dates[-1].date()))
        assert isinstance(result, BacktestResult)

    def test_portfolio_values_length_matches_trading_days(self):
        engine, data = _default_engine()
        dates = _make_dates(_N_DAYS)
        result = engine.run(data, start=str(dates[0].date()), end=str(dates[-1].date()))
        assert len(result.portfolio_values) == _N_DAYS

    def test_returns_length_is_one_less_than_values(self):
        engine, data = _default_engine()
        dates = _make_dates(_N_DAYS)
        result = engine.run(data, start=str(dates[0].date()), end=str(dates[-1].date()))
        assert len(result.returns) == len(result.portfolio_values) - 1

    def test_portfolio_values_date_indexed(self):
        engine, data = _default_engine()
        dates = _make_dates(_N_DAYS)
        result = engine.run(data, start=str(dates[0].date()), end=str(dates[-1].date()))
        assert isinstance(result.portfolio_values.index, pd.DatetimeIndex)

    def test_positive_returns_grow_portfolio(self):
        engine, data = _default_engine(daily_return=0.002)
        dates = _make_dates(_N_DAYS)
        result = engine.run(data, start=str(dates[0].date()), end=str(dates[-1].date()))
        assert float(result.portfolio_values.iloc[-1]) > 100_000.0


# ---------------------------------------------------------------------------
# BacktestResult fields
# ---------------------------------------------------------------------------


class TestBacktestResultFields:
    @pytest.fixture()
    def result(self):
        engine, data = _default_engine()
        dates = _make_dates(_N_DAYS)
        return engine.run(data, start=str(dates[0].date()), end=str(dates[-1].date()))

    def test_portfolio_values_present(self, result):
        assert result.portfolio_values is not None
        assert len(result.portfolio_values) > 0

    def test_returns_present(self, result):
        assert result.returns is not None

    def test_weights_history_present(self, result):
        assert isinstance(result.weights_history, list)
        assert len(result.weights_history) > 0

    def test_trades_history_present(self, result):
        assert isinstance(result.trades_history, list)

    def test_signals_history_present(self, result):
        assert isinstance(result.signals_history, list)

    def test_risk_events_present(self, result):
        assert isinstance(result.risk_events, list)

    def test_metrics_present(self, result):
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

    def test_config_present(self, result):
        assert isinstance(result.config, dict)
        assert "initial_capital" in result.config
        assert "rebalance_frequency" in result.config
        assert "transaction_cost_bps" in result.config

    def test_metrics_contains_expected_keys(self, result):
        expected = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "profit_factor", "annual_return", "annual_volatility",
            "var_historical", "cvar_historical",
        }
        assert expected.issubset(result.metrics.keys())


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    def test_costs_reduce_final_value(self):
        dates = _make_dates(_N_DAYS)
        start, end = str(dates[0].date()), str(dates[-1].date())

        data = _make_flat_data()  # zero returns — any value change is from costs

        engine_free = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=0.0,
        )
        engine_cost = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=50.0,
        )

        result_free = engine_free.run(data, start, end)
        result_cost = engine_cost.run(data, start, end)

        assert float(result_cost.portfolio_values.iloc[-1]) < float(
            result_free.portfolio_values.iloc[-1]
        )

    def test_zero_cost_flat_data_stays_at_capital(self):
        """With zero returns and zero costs, NAV must stay exactly at initial capital."""
        dates = _make_dates(_N_DAYS)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data()

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=0.0,
            initial_capital=100_000.0,
        )
        result = engine.run(data, start, end)
        assert math.isclose(float(result.portfolio_values.iloc[-1]), 100_000.0, rel_tol=1e-6)

    def test_higher_cost_lower_final_value(self):
        dates = _make_dates(_N_DAYS)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data()

        def _run(bps):
            e = BacktestEngine(
                signals=[_MockSignalGenerator("m", 2)],
                agent=_SymbolAwareAgent(_SYMBOLS),
                risk_manager=RiskManager(),
                rebalancer=Rebalancer(),
                rebalance_frequency="daily",
                transaction_cost_bps=bps,
            )
            return float(e.run(data, start, end).portfolio_values.iloc[-1])

        assert _run(1.0) > _run(10.0) > _run(50.0)


# ---------------------------------------------------------------------------
# Rebalance frequency
# ---------------------------------------------------------------------------


class TestRebalanceFrequency:
    def test_daily_has_more_rebalances_than_monthly(self):
        dates = _make_dates(_N_DAYS)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data()

        def _trades_count(freq):
            e = BacktestEngine(
                signals=[_MockSignalGenerator("m", 2)],
                agent=_SymbolAwareAgent(_SYMBOLS),
                risk_manager=RiskManager(),
                rebalancer=Rebalancer(),
                rebalance_frequency=freq,
                transaction_cost_bps=0.0,
            )
            return len(e.run(data, start, end).trades_history)

        assert _trades_count("daily") > _trades_count("weekly") > _trades_count("monthly")

    def test_monthly_rebalances_once_per_month(self):
        # 60 trading days spans ~3 months; expect ≤4 monthly rebalances.
        dates = _make_dates(_N_DAYS)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data()
        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        assert 1 <= len(result.trades_history) <= 4

    def test_weekly_rebalances_once_per_week(self):
        dates = _make_dates(_N_DAYS)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data()
        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="weekly",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        # 60 trading days ≈ 12 weeks
        assert 8 <= len(result.trades_history) <= 14

    def test_daily_rebalance_count_equals_trading_days(self):
        dates = _make_dates(_N_DAYS)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data()
        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="daily",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        assert len(result.trades_history) == _N_DAYS


# ---------------------------------------------------------------------------
# Risk events
# ---------------------------------------------------------------------------


class TestRiskEvents:
    def test_no_risk_events_under_normal_conditions(self):
        # Raise max_position_pct so 50% equal-weight positions do not trigger clipping.
        engine, data = _default_engine(daily_return=0.001, risk_config={"max_position_pct": 0.60})
        dates = _make_dates(_N_DAYS)
        result = engine.run(data, start=str(dates[0].date()), end=str(dates[-1].date()))
        assert result.risk_events == []

    def test_flatten_logged_when_drawdown_breached(self):
        """Falling prices trigger the drawdown limit."""
        # Large daily loss → drawdown breach in early rebalance.
        n = 30
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())

        # Prices drop 5% per day → massive drawdown immediately.
        data: dict[str, AssetData] = {}
        for sym in _SYMBOLS:
            prices = [100.0 * (0.95**i) for i in range(n)]
            data[sym] = AssetData(symbol=sym, ohlcv=_make_ohlcv(prices, dates))

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager({"max_drawdown_pct": 0.10, "daily_loss_limit_pct": 0.03}),
            rebalancer=Rebalancer(),
            rebalance_frequency="daily",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        assert len(result.risk_events) > 0
        assert any("FLATTEN" in ev for ev in result.risk_events)

    def test_reentry_after_flatten_reset(self):
        """After a flatten event, peak is reset so the system re-enters the market."""
        n = 60
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())

        # Prices crash then fully recover: 30 days of 5% drops, 30 days of gains.
        crash = [100.0 * (0.95**i) for i in range(30)]
        recovery = [crash[-1] * (1.10**i) for i in range(30)]
        prices = crash + recovery

        data: dict[str, AssetData] = {}
        for sym in _SYMBOLS:
            data[sym] = AssetData(symbol=sym, ohlcv=_make_ohlcv(prices, dates))

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager({"max_drawdown_pct": 0.10, "daily_loss_limit_pct": 0.10}),
            rebalancer=Rebalancer(),
            rebalance_frequency="daily",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)

        # Must have at least one flatten event during the crash…
        assert any("FLATTEN" in ev for ev in result.risk_events)

        # …and non-zero weights must appear in the recovery period (re-entry happened).
        recovery_weights = [
            entry for entry in result.weights_history
            if entry["date"] >= str(dates[30].date())
        ]
        non_cash = [
            entry for entry in recovery_weights
            if any(entry.get(sym, 0.0) != 0.0 for sym in _SYMBOLS)
        ]
        assert len(non_cash) > 0, "System never re-entered the market after flatten reset"

    def test_position_clipping_logged_as_risk_event(self):
        """Agent proposing 50% positions gets clipped at 30% — warning in risk_events."""
        n = 10
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data(n_days=n)

        class _OverweightAgent(PortfolioAgent):
            def decide(self, signals, current_portfolio):
                # 50% each — exceeds max_position_pct=0.30; sum=1.0 is valid pre-clip.
                return PortfolioAction(
                    weights={"SPY": 0.50, "TLT": 0.50},
                    confidence=0.5,
                    regime_context="bull",
                )

            def train(self, h, r):
                return {}

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_OverweightAgent(),
            risk_manager=RiskManager({"max_position_pct": 0.30}),
            rebalancer=Rebalancer(),
            rebalance_frequency="daily",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        assert len(result.risk_events) > 0


# ---------------------------------------------------------------------------
# Single-asset test
# ---------------------------------------------------------------------------


class TestSingleAsset:
    def test_single_asset_run_completes(self):
        n = 20
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_data(["SPY"], n_days=n, daily_return=0.001)

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 1)],
            agent=_FixedWeightAgent({"SPY": 1.0}),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=5.0,
        )
        result = engine.run(data, start, end)
        assert len(result.portfolio_values) == n
        assert float(result.portfolio_values.iloc[-1]) > 0.0

    def test_single_asset_weights_history_contains_symbol(self):
        n = 10
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_data(["SPY"], n_days=n)

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 1)],
            agent=_FixedWeightAgent({"SPY": 0.9}),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        assert len(result.weights_history) >= 1
        assert "SPY" in result.weights_history[0]


# ---------------------------------------------------------------------------
# Zero-return portfolio
# ---------------------------------------------------------------------------


class TestZeroReturns:
    def test_flat_portfolio_with_costs_decreases(self):
        n = 20
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data(n_days=n)

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="daily",
            transaction_cost_bps=5.0,
        )
        result = engine.run(data, start, end)
        assert float(result.portfolio_values.iloc[-1]) < 100_000.0

    def test_flat_portfolio_no_cost_stays_flat(self):
        n = 20
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data(n_days=n)

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        # Allow tiny float drift
        assert math.isclose(float(result.portfolio_values.iloc[-1]), 100_000.0, rel_tol=1e-6)

    def test_all_returns_near_zero_for_flat_data(self):
        n = 30
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data(n_days=n)

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        # All returns should be 0.0 (or very close) when prices are flat and no costs.
        assert (result.returns.abs() < 1e-9).all()


# ---------------------------------------------------------------------------
# Signals history
# ---------------------------------------------------------------------------


class TestSignalsHistory:
    def test_signals_generated_at_each_rebalance(self):
        n = 20
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data(n_days=n)

        engine = BacktestEngine(
            signals=[_MockSignalGenerator("mock_a", 2), _MockSignalGenerator("mock_b", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        # 2 generators × number of rebalances
        assert len(result.signals_history) > 0
        assert all(isinstance(s, Signal) for s in result.signals_history)

    def test_no_signals_if_no_generators(self):
        n = 10
        dates = _make_dates(n)
        start, end = str(dates[0].date()), str(dates[-1].date())
        data = _make_flat_data(n_days=n)

        engine = BacktestEngine(
            signals=[],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="monthly",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start, end)
        assert result.signals_history == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_data_raises(self):
        engine = BacktestEngine(
            signals=[],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
        )
        with pytest.raises(ValueError, match="at least one asset"):
            engine.run({}, _START, _END)

    def test_no_trading_days_in_range_raises(self):
        engine, data = _default_engine()
        with pytest.raises(ValueError, match="No trading dates"):
            engine.run(data, start="1990-01-01", end="1990-01-05")

    def test_single_day_range(self):
        n = 30
        dates = _make_dates(n)
        # Use the first date as both start and end.
        day = str(dates[0].date())
        data = _make_flat_data(n_days=n)
        engine = BacktestEngine(
            signals=[_MockSignalGenerator("m", 2)],
            agent=_SymbolAwareAgent(_SYMBOLS),
            risk_manager=RiskManager(),
            rebalancer=Rebalancer(),
            rebalance_frequency="daily",
            transaction_cost_bps=0.0,
        )
        result = engine.run(data, start=day, end=day)
        assert len(result.portfolio_values) == 1
        assert math.isclose(float(result.portfolio_values.iloc[0]), 100_000.0, rel_tol=1e-6)

    def test_config_records_start_end(self):
        engine, data = _default_engine()
        dates = _make_dates(_N_DAYS)
        start, end = str(dates[0].date()), str(dates[-1].date())
        result = engine.run(data, start=start, end=end)
        assert result.config["start"] == start
        assert result.config["end"] == end


# ---------------------------------------------------------------------------
# _slice_data helper
# ---------------------------------------------------------------------------


class TestSliceData:
    def test_slice_excludes_future_dates(self):
        dates = _make_dates(10)
        data = _make_flat_data(n_days=10)
        cutoff = dates[4]
        sliced = _slice_data(data, cutoff)
        for sym, asset in sliced.items():
            assert (asset.ohlcv.index <= cutoff).all()

    def test_slice_preserves_all_symbols(self):
        data = _make_flat_data(n_days=10)
        sliced = _slice_data(data, _make_dates(10)[-1])
        assert set(sliced.keys()) == set(data.keys())

    def test_original_data_not_mutated(self):
        data = _make_flat_data(n_days=10)
        original_len = {sym: len(asset.ohlcv) for sym, asset in data.items()}
        _slice_data(data, _make_dates(10)[2])
        for sym, asset in data.items():
            assert len(asset.ohlcv) == original_len[sym]
