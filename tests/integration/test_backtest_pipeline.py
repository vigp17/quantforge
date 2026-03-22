"""Integration test: full backtest pipeline with HMM + Momentum signals,
MeanVarianceOptimizer, RiskManager, Rebalancer, BacktestEngine, and
WalkForwardValidator.

All data is synthetic — no network access required.

Run with:
    pytest tests/integration/test_backtest_pipeline.py -v
    pytest tests/ -v -m "not network"
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.report import ReportGenerator
from src.backtest.walk_forward import WalkForwardResult, WalkForwardValidator
from src.data.base import AssetData
from src.portfolio.optimizer import MeanVarianceOptimizer
from src.portfolio.rebalancer import Rebalancer
from src.portfolio.risk_manager import RiskManager
from src.signals.hmm_regime import HMMRegimeDetector
from src.signals.momentum import MomentumSignal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOLS = ["SPY", "QQQ", "TLT"]
_N_DAYS = 500  # ~2 years of trading days
_SEED = 7
_TRAIN_WINDOW = 252
_TEST_WINDOW = 63
_START = "2020-01-02"


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------


def _make_asset_data(
    symbol: str,
    n_days: int = _N_DAYS,
    daily_return: float = 0.0005,
    daily_vol: float = 0.012,
    seed: int = 0,
) -> AssetData:
    """Synthetic OHLCV data with a geometric random walk."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=_START, periods=n_days)
    log_ret = rng.normal(daily_return, daily_vol, n_days)
    close = 100.0 * np.exp(np.cumsum(log_ret))

    df = pd.DataFrame(
        {
            "open": close * rng.uniform(0.998, 1.000, n_days),
            "high": close * rng.uniform(1.000, 1.005, n_days),
            "low": close * rng.uniform(0.995, 1.000, n_days),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
        },
        index=dates,
    )
    return AssetData(symbol=symbol, ohlcv=df)


def _make_data(
    symbols: list[str] = _SYMBOLS,
    n_days: int = _N_DAYS,
) -> dict[str, AssetData]:
    return {
        sym: _make_asset_data(sym, n_days=n_days, seed=i + _SEED)
        for i, sym in enumerate(symbols)
    }


def _make_returns_df(data: dict[str, AssetData]) -> pd.DataFrame:
    """Build a returns DataFrame aligned to the union of dates in *data*."""
    dates = sorted({d for asset in data.values() for d in asset.ohlcv.index})
    idx = pd.DatetimeIndex(dates)
    df = pd.DataFrame(index=idx)
    for sym, asset in data.items():
        closes = asset.ohlcv["close"].reindex(idx)
        df[sym] = closes.pct_change().fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Module-scoped fixtures — build pipeline objects once for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_data() -> dict[str, AssetData]:
    return _make_data()


@pytest.fixture(scope="module")
def returns_df(pipeline_data) -> pd.DataFrame:
    return _make_returns_df(pipeline_data)


@pytest.fixture(scope="module")
def backtest_result(pipeline_data, returns_df) -> BacktestResult:
    """Run the full BacktestEngine once and cache the result."""
    dates = pd.bdate_range(start=_START, periods=_N_DAYS)
    start, end = str(dates[0].date()), str(dates[-1].date())

    engine = BacktestEngine(
        signals=[
            HMMRegimeDetector(n_states=3, n_iter=30, random_state=_SEED),
            MomentumSignal(),
        ],
        agent=MeanVarianceOptimizer(returns_df, lookback=60, max_position=0.40),
        risk_manager=RiskManager(),
        rebalancer=Rebalancer(),
        initial_capital=100_000.0,
        rebalance_frequency="monthly",
        transaction_cost_bps=5.0,
    )
    return engine.run(pipeline_data, start=start, end=end)


@pytest.fixture(scope="module")
def wf_result(pipeline_data, returns_df) -> WalkForwardResult:
    """Run the WalkForwardValidator once and cache the result."""
    engine = BacktestEngine(
        signals=[
            HMMRegimeDetector(n_states=3, n_iter=30, random_state=_SEED),
            MomentumSignal(),
        ],
        agent=MeanVarianceOptimizer(returns_df, lookback=60, max_position=0.40),
        risk_manager=RiskManager(),
        rebalancer=Rebalancer(),
        initial_capital=100_000.0,
        rebalance_frequency="monthly",
        transaction_cost_bps=5.0,
    )
    validator = WalkForwardValidator(
        engine=engine,
        train_window=_TRAIN_WINDOW,
        test_window=_TEST_WINDOW,
        expanding=True,
    )
    return validator.run(pipeline_data)


@pytest.fixture(scope="module")
def report_path(tmp_path_factory, backtest_result) -> str:
    """Generate the HTML report once and return the path."""
    path = str(tmp_path_factory.mktemp("report") / "integration_report.html")
    ReportGenerator().generate_html(backtest_result, path)
    return path


# ---------------------------------------------------------------------------
# BacktestEngine — shape and structure
# ---------------------------------------------------------------------------


class TestBacktestEngineShape:
    def test_portfolio_values_length_matches_trading_days(self, backtest_result):
        assert len(backtest_result.portfolio_values) == _N_DAYS

    def test_returns_length_is_one_less_than_values(self, backtest_result):
        assert len(backtest_result.returns) == _N_DAYS - 1

    def test_portfolio_values_date_indexed(self, backtest_result):
        assert isinstance(backtest_result.portfolio_values.index, pd.DatetimeIndex)

    def test_weights_history_non_empty(self, backtest_result):
        # Monthly rebalancing over ~500 days → at least 1 rebalance.
        assert len(backtest_result.weights_history) >= 1

    def test_trades_history_non_empty(self, backtest_result):
        assert len(backtest_result.trades_history) >= 1

    def test_config_contains_expected_keys(self, backtest_result):
        for key in ("initial_capital", "rebalance_frequency", "transaction_cost_bps", "agent"):
            assert key in backtest_result.config

    def test_portfolio_values_all_positive(self, backtest_result):
        assert (backtest_result.portfolio_values > 0).all()


# ---------------------------------------------------------------------------
# BacktestEngine — metric validity
# ---------------------------------------------------------------------------


class TestBacktestMetrics:
    def test_sharpe_ratio_is_finite(self, backtest_result):
        sharpe = backtest_result.metrics["sharpe_ratio"]
        assert math.isfinite(sharpe)

    def test_max_drawdown_less_than_one(self, backtest_result):
        """Portfolio must not be wiped out."""
        mdd = backtest_result.metrics["max_drawdown"]
        assert mdd < 1.0

    def test_max_drawdown_non_negative(self, backtest_result):
        assert backtest_result.metrics["max_drawdown"] >= 0.0

    def test_win_rate_in_zero_one(self, backtest_result):
        wr = backtest_result.metrics["win_rate"]
        assert 0.0 <= wr <= 1.0

    def test_annual_volatility_positive(self, backtest_result):
        vol = backtest_result.metrics["annual_volatility"]
        assert vol >= 0.0

    def test_var_non_negative(self, backtest_result):
        assert backtest_result.metrics["var_historical"] >= 0.0

    def test_cvar_gte_var(self, backtest_result):
        assert backtest_result.metrics["cvar_historical"] >= backtest_result.metrics["var_historical"]

    def test_metrics_contains_all_keys(self, backtest_result):
        expected = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "profit_factor", "annual_return", "annual_volatility",
            "var_historical", "cvar_historical",
        }
        assert expected.issubset(backtest_result.metrics.keys())


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


class TestSignals:
    def test_signals_history_non_empty(self, backtest_result):
        # After 252 days (momentum threshold), signals should be generated.
        assert len(backtest_result.signals_history) > 0

    def test_signals_are_signal_instances(self, backtest_result):
        from src.signals.base import Signal as SignalClass
        for sig in backtest_result.signals_history:
            assert isinstance(sig, SignalClass)

    def test_hmm_regime_in_signals(self, backtest_result):
        names = {s.name for s in backtest_result.signals_history}
        assert "hmm_regime" in names

    def test_momentum_in_signals(self, backtest_result):
        """Momentum signal appears once data exceeds its 252-day lookback."""
        names = {s.name for s in backtest_result.signals_history}
        assert "momentum" in names


# ---------------------------------------------------------------------------
# Risk management integration
# ---------------------------------------------------------------------------


class TestRiskIntegration:
    def test_weights_respect_max_position(self, backtest_result):
        """After risk manager clips, no position should exceed 0.30 default."""
        for entry in backtest_result.weights_history:
            for sym in _SYMBOLS:
                w = entry.get(sym, 0.0)
                # MV optimizer uses max_position=0.40 but risk manager clips to 0.30.
                assert abs(w) <= 0.30 + 1e-6, (
                    f"Weight {sym}={w:.4f} exceeds risk limit in entry {entry}"
                )

    def test_risk_events_list_exists(self, backtest_result):
        assert isinstance(backtest_result.risk_events, list)


# ---------------------------------------------------------------------------
# WalkForwardValidator — fold count
# ---------------------------------------------------------------------------


class TestWalkForwardFoldCount:
    def test_correct_number_of_folds(self, wf_result):
        # 500 days, train=252, test=63, expanding.
        # fold 0: test ends at 315; fold 1: 378; fold 2: 441; fold 3: 504 > 500 → 3 folds.
        expected = (_N_DAYS - _TRAIN_WINDOW) // _TEST_WINDOW
        assert wf_result.n_folds == expected

    def test_n_folds_matches_folds_list(self, wf_result):
        assert wf_result.n_folds == len(wf_result.folds)

    def test_n_folds_matches_fold_metrics(self, wf_result):
        assert wf_result.n_folds == len(wf_result.fold_metrics)


# ---------------------------------------------------------------------------
# WalkForwardValidator — out-of-sample returns
# ---------------------------------------------------------------------------


class TestWalkForwardReturns:
    def test_combined_returns_length(self, wf_result):
        expected_len = sum(len(f.returns) for f in wf_result.folds)
        assert len(wf_result.combined_returns) == expected_len

    def test_combined_returns_date_indexed(self, wf_result):
        assert isinstance(wf_result.combined_returns.index, pd.DatetimeIndex)

    def test_combined_returns_no_duplicate_dates(self, wf_result):
        assert wf_result.combined_returns.index.is_unique

    def test_combined_returns_monotonic_dates(self, wf_result):
        assert wf_result.combined_returns.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# WalkForwardValidator — metrics
# ---------------------------------------------------------------------------


class TestWalkForwardMetrics:
    def test_combined_metrics_has_expected_keys(self, wf_result):
        expected = {
            "sharpe_ratio", "max_drawdown", "win_rate",
            "annual_return", "annual_volatility",
        }
        assert expected.issubset(wf_result.combined_metrics.keys())

    def test_fold_metrics_all_have_sharpe(self, wf_result):
        for i, fm in enumerate(wf_result.fold_metrics):
            assert "sharpe_ratio" in fm, f"Fold {i} missing sharpe_ratio"

    def test_fold_metrics_sharpe_finite(self, wf_result):
        for i, fm in enumerate(wf_result.fold_metrics):
            assert math.isfinite(fm["sharpe_ratio"]), f"Fold {i} sharpe_ratio is not finite"

    def test_fold_max_drawdowns_less_than_one(self, wf_result):
        for i, fm in enumerate(wf_result.fold_metrics):
            assert fm["max_drawdown"] < 1.0, f"Fold {i} max_drawdown >= 1.0"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


class TestHtmlReport:
    def test_report_file_exists(self, report_path):
        assert os.path.exists(report_path)

    def test_report_file_non_empty(self, report_path):
        assert os.path.getsize(report_path) > 1000

    def test_report_is_html(self, report_path):
        content = open(report_path, encoding="utf-8").read()
        assert content.strip().startswith("<!DOCTYPE html>")

    def test_report_has_equity_curve(self, report_path):
        content = open(report_path, encoding="utf-8").read()
        assert "Equity Curve" in content

    def test_report_has_no_external_cdn_script(self, report_path):
        import re
        content = open(report_path, encoding="utf-8").read()
        cdn_tags = re.findall(r'<script[^>]+src=["\'][^"\']*cdn\.plot\.ly', content)
        assert len(cdn_tags) == 0


# ---------------------------------------------------------------------------
# End-to-end: compare two strategies (MV vs risk-parity)
# ---------------------------------------------------------------------------


class TestCompareStrategies:
    def test_compare_report_generated(self, pipeline_data, returns_df, tmp_path):
        """Generate a comparison between mean-variance and risk-parity modes."""
        dates = pd.bdate_range(start=_START, periods=_N_DAYS)
        start, end = str(dates[0].date()), str(dates[-1].date())

        def _run(mode: str) -> BacktestResult:
            engine = BacktestEngine(
                signals=[HMMRegimeDetector(n_states=3, n_iter=20, random_state=_SEED)],
                agent=MeanVarianceOptimizer(returns_df, lookback=60, max_position=0.40, mode=mode),
                risk_manager=RiskManager(),
                rebalancer=Rebalancer(),
                initial_capital=100_000.0,
                rebalance_frequency="monthly",
                transaction_cost_bps=5.0,
            )
            return engine.run(pipeline_data, start=start, end=end)

        result_mv = _run("mean_variance")
        result_rp = _run("risk_parity")

        compare_path = str(tmp_path / "compare.html")
        html = ReportGenerator().compare_results(
            [result_mv, result_rp],
            ["Mean-Variance", "Risk-Parity"],
            output_path=compare_path,
        )

        assert os.path.exists(compare_path)
        assert "Mean-Variance" in html
        assert "Risk-Parity" in html
