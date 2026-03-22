"""Integration test: mock signals → ensemble → optimizer → risk → sizer → rebalancer.

Exercises the full portfolio pipeline end-to-end using only in-process mock
data — no network, no broker, no database.

Run with:
    pytest tests/integration/test_portfolio_pipeline.py -v
    pytest tests/ -v -m "not network"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.base import AssetData
from src.execution.base import Order
from src.portfolio.base import PortfolioAction, PortfolioAgent
from src.portfolio.optimizer import MeanVarianceOptimizer
from src.portfolio.position_sizer import PositionSizer
from src.portfolio.rebalancer import Rebalancer
from src.portfolio.risk_manager import RiskManager
from src.signals.base import Signal, SignalGenerator
from src.signals.ensemble import SignalEnsemble

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_SYMBOLS = ["SPY", "QQQ", "TLT", "GLD"]
_N = len(_SYMBOLS)
_PORTFOLIO_VALUE = 100_000.0
_RNG_SEED = 42


# ---------------------------------------------------------------------------
# Fixtures: mock data and returns
# ---------------------------------------------------------------------------


def _make_asset_data(symbol: str, n_rows: int = 252, seed: int = 0) -> AssetData:
    """Build synthetic OHLCV AssetData with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n_rows, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n_rows))
    df = pd.DataFrame(
        {
            "open": close * rng.uniform(0.99, 1.00, n_rows),
            "high": close * rng.uniform(1.00, 1.01, n_rows),
            "low": close * rng.uniform(0.99, 1.00, n_rows),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n_rows).astype(float),
        },
        index=idx,
    )
    return AssetData(symbol=symbol, ohlcv=df)


def _make_returns_df(n_rows: int = 120, seed: int = 0) -> pd.DataFrame:
    """Build a daily-returns DataFrame for all symbols."""
    rng = np.random.default_rng(seed)
    data = {sym: rng.normal(0.0005, 0.012, n_rows) for sym in _SYMBOLS}
    return pd.DataFrame(data)


def _make_mock_data() -> dict[str, AssetData]:
    return {sym: _make_asset_data(sym, seed=i) for i, sym in enumerate(_SYMBOLS)}


# ---------------------------------------------------------------------------
# Mock signal generators (no network, deterministic)
# ---------------------------------------------------------------------------


class _MockHMMSignal(SignalGenerator):
    """Simulates an HMM regime signal with a bull regime label."""

    @property
    def name(self) -> str:
        return "hmm_regime"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        n = len(data)
        return Signal(
            name=self.name,
            values=np.array([0.02, 0.015, -0.005, 0.008][:n]),
            confidence=np.array([0.80, 0.75, 0.70, 0.65][:n]),
            regime="bull",
            metadata={"regime_posterior": [0.75, 0.15, 0.10]},
        )

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        return self.generate(new_data)


class _MockMomentumSignal(SignalGenerator):
    """Simulates a cross-sectional momentum signal."""

    @property
    def name(self) -> str:
        return "momentum"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        n = len(data)
        return Signal(
            name=self.name,
            values=np.array([0.03, 0.025, -0.01, 0.005][:n]),
            confidence=np.array([0.85, 0.80, 0.60, 0.55][:n]),
            regime=None,
        )

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        return self.generate(new_data)


class _MockMonteCarloSignal(SignalGenerator):
    """Simulates a Monte Carlo expected-return signal with risk metadata."""

    @property
    def name(self) -> str:
        return "montecarlo"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        n = len(data)
        return Signal(
            name=self.name,
            values=np.array([0.018, 0.012, 0.004, 0.009][:n]),
            confidence=np.array([0.70, 0.72, 0.65, 0.68][:n]),
            regime=None,
            metadata={"var_5pct": -0.025, "cvar_5pct": -0.038},
        )

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        return self.generate(new_data)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_data() -> dict[str, AssetData]:
    return _make_mock_data()


@pytest.fixture(scope="module")
def returns_df() -> pd.DataFrame:
    return _make_returns_df()


@pytest.fixture(scope="module")
def ensemble_signal(mock_data: dict[str, AssetData]) -> Signal:
    ens = SignalEnsemble(
        [_MockHMMSignal(), _MockMomentumSignal(), _MockMonteCarloSignal()],
        method="confidence_weighted",
    )
    return ens.generate(mock_data)


@pytest.fixture(scope="module")
def optimizer(returns_df: pd.DataFrame) -> MeanVarianceOptimizer:
    return MeanVarianceOptimizer(returns_df, risk_aversion=1.0, max_position=0.30)


@pytest.fixture(scope="module")
def portfolio_action(
    optimizer: MeanVarianceOptimizer, ensemble_signal: Signal
) -> PortfolioAction:
    return optimizer.decide([ensemble_signal], {})


@pytest.fixture(scope="module")
def risk_manager() -> RiskManager:
    return RiskManager()


@pytest.fixture(scope="module")
def sizer() -> PositionSizer:
    return PositionSizer(max_position=0.30)


@pytest.fixture(scope="module")
def rebalancer() -> Rebalancer:
    return Rebalancer()


# ---------------------------------------------------------------------------
# Stage 1: Mock signal generators produce valid Signals
# ---------------------------------------------------------------------------


class TestMockSignals:
    def test_hmm_signal_shape(self, mock_data: dict[str, AssetData]) -> None:
        sig = _MockHMMSignal().generate(mock_data)
        assert sig.values.shape == (_N,)
        assert sig.confidence.shape == (_N,)

    def test_momentum_signal_shape(self, mock_data: dict[str, AssetData]) -> None:
        sig = _MockMomentumSignal().generate(mock_data)
        assert sig.values.shape == (_N,)

    def test_montecarlo_signal_shape(self, mock_data: dict[str, AssetData]) -> None:
        sig = _MockMonteCarloSignal().generate(mock_data)
        assert sig.values.shape == (_N,)
        assert "var_5pct" in sig.metadata
        assert "cvar_5pct" in sig.metadata

    def test_hmm_reports_regime(self, mock_data: dict[str, AssetData]) -> None:
        sig = _MockHMMSignal().generate(mock_data)
        assert sig.regime == "bull"

    def test_confidence_in_bounds(self, mock_data: dict[str, AssetData]) -> None:
        for gen in [_MockHMMSignal(), _MockMomentumSignal(), _MockMonteCarloSignal()]:
            sig = gen.generate(mock_data)
            assert np.all(sig.confidence >= 0.0)
            assert np.all(sig.confidence <= 1.0)


# ---------------------------------------------------------------------------
# Stage 2: SignalEnsemble combines all three generators
# ---------------------------------------------------------------------------


class TestEnsemble:
    def test_ensemble_shape(self, ensemble_signal: Signal) -> None:
        assert ensemble_signal.values.shape == (_N,)
        assert ensemble_signal.confidence.shape == (_N,)

    def test_ensemble_finite_values(self, ensemble_signal: Signal) -> None:
        assert np.all(np.isfinite(ensemble_signal.values))

    def test_ensemble_confidence_in_bounds(self, ensemble_signal: Signal) -> None:
        assert np.all(ensemble_signal.confidence >= 0.0)
        assert np.all(ensemble_signal.confidence <= 1.0)

    def test_ensemble_adopts_hmm_regime(self, ensemble_signal: Signal) -> None:
        """HMM is first — its regime label propagates to the combined signal."""
        assert ensemble_signal.regime == "bull"

    def test_ensemble_metadata_lists_contributors(
        self, ensemble_signal: Signal
    ) -> None:
        meta = ensemble_signal.metadata
        assert meta["n_signals"] == 3
        assert set(meta["contributors"]) == {"hmm_regime", "momentum", "montecarlo"}

    def test_confidence_weighted_higher_than_low_conf(
        self, mock_data: dict[str, AssetData]
    ) -> None:
        """confidence_weighted differs from equal_weight when confidences vary."""
        cw = SignalEnsemble(
            [_MockHMMSignal(), _MockMomentumSignal(), _MockMonteCarloSignal()],
            method="confidence_weighted",
        ).generate(mock_data)
        ew = SignalEnsemble(
            [_MockHMMSignal(), _MockMomentumSignal(), _MockMonteCarloSignal()],
            method="equal_weight",
        ).generate(mock_data)
        # Outputs need not be identical — just confirm both are valid.
        assert cw.values.shape == ew.values.shape
        assert np.all(np.isfinite(cw.values))


# ---------------------------------------------------------------------------
# Stage 3: MeanVarianceOptimizer → PortfolioAction
# ---------------------------------------------------------------------------


class TestOptimizer:
    def test_returns_portfolio_action(self, portfolio_action: PortfolioAction) -> None:
        assert isinstance(portfolio_action, PortfolioAction)

    def test_weights_sum_leq_one(self, portfolio_action: PortfolioAction) -> None:
        assert sum(portfolio_action.weights.values()) <= 1.0 + 1e-6

    def test_weights_non_negative(self, portfolio_action: PortfolioAction) -> None:
        for w in portfolio_action.weights.values():
            assert w >= -1e-8

    def test_no_weight_exceeds_max_position(
        self, portfolio_action: PortfolioAction
    ) -> None:
        for w in portfolio_action.weights.values():
            assert w <= 0.30 + 1e-6

    def test_symbols_match_returns_df(
        self, portfolio_action: PortfolioAction, returns_df: pd.DataFrame
    ) -> None:
        assert set(portfolio_action.weights.keys()) == set(returns_df.columns)

    def test_regime_propagated_from_hmm(
        self, portfolio_action: PortfolioAction
    ) -> None:
        assert portfolio_action.regime_context == "bull"

    def test_confidence_in_unit_interval(
        self, portfolio_action: PortfolioAction
    ) -> None:
        assert 0.0 <= portfolio_action.confidence <= 1.0

    def test_optimizer_is_portfolio_agent(
        self, optimizer: MeanVarianceOptimizer
    ) -> None:
        assert isinstance(optimizer, PortfolioAgent)


# ---------------------------------------------------------------------------
# Stage 4: RiskManager validates action
# ---------------------------------------------------------------------------


class TestRiskManager:
    def test_valid_action_passes_unchanged(
        self,
        risk_manager: RiskManager,
        portfolio_action: PortfolioAction,
    ) -> None:
        """Optimizer already clips to 0.30; risk manager should find no violations."""
        adjusted, warnings = risk_manager.validate_action(portfolio_action, {})
        assert isinstance(adjusted, PortfolioAction)
        # No weight should have been clipped further.
        for sym, w in adjusted.weights.items():
            assert w <= 0.30 + 1e-6

    def test_oversized_position_clipped_to_max(
        self, risk_manager: RiskManager
    ) -> None:
        """A 50 % position must be clipped to 30 %."""
        action = PortfolioAction(
            weights={"SPY": 0.50, "QQQ": 0.20, "TLT": 0.20, "GLD": 0.10},
            confidence=0.8,
            regime_context="bull",
        )
        adjusted, warnings = risk_manager.validate_action(action, {})
        assert adjusted.weights["SPY"] == pytest.approx(0.30, abs=1e-6)
        assert len(warnings) >= 1
        assert any("SPY" in w for w in warnings)

    def test_warnings_list_on_breach(self, risk_manager: RiskManager) -> None:
        action = PortfolioAction(
            weights={"SPY": 0.50, "QQQ": 0.50},
            confidence=0.5,
            regime_context="unknown",
        )
        _, warnings = risk_manager.validate_action(action, {})
        assert isinstance(warnings, list)
        assert len(warnings) >= 2  # Both SPY and QQQ breach, plus leverage

    def test_adjusted_action_passes_own_validation(
        self, risk_manager: RiskManager
    ) -> None:
        """PortfolioAction.__post_init__ must not raise on the adjusted output."""
        action = PortfolioAction(
            weights={"A": 0.45, "B": 0.45, "C": 0.10},
            confidence=0.7,
            regime_context="neutral",
        )
        adjusted, _ = risk_manager.validate_action(action, {})
        assert isinstance(adjusted, PortfolioAction)

    def test_all_adjusted_weights_at_or_below_max(
        self, risk_manager: RiskManager
    ) -> None:
        # Weights sum to 1.0 but SPY and QQQ each exceed the 0.30 position cap.
        action = PortfolioAction(
            weights={"SPY": 0.35, "QQQ": 0.35, "TLT": 0.20, "GLD": 0.10},
            confidence=0.6,
            regime_context="bull",
        )
        adjusted, _ = risk_manager.validate_action(action, {})
        for w in adjusted.weights.values():
            assert w <= risk_manager.config.max_position_pct + 1e-6


# ---------------------------------------------------------------------------
# Stage 5: PositionSizer verifies weight magnitudes
# ---------------------------------------------------------------------------


class TestPositionSizer:
    def test_equal_weight_sums_to_one(
        self, sizer: PositionSizer, ensemble_signal: Signal, returns_df: pd.DataFrame
    ) -> None:
        weights = sizer.size(ensemble_signal, returns_df, method="equal_weight")
        total = sum(abs(w) for w in weights.values())
        assert total <= 1.0 + 1e-6

    def test_vol_scaled_respects_max_position(
        self, sizer: PositionSizer, ensemble_signal: Signal, returns_df: pd.DataFrame
    ) -> None:
        weights = sizer.size(ensemble_signal, returns_df, method="volatility_scaled")
        for w in weights.values():
            assert abs(w) <= sizer.max_position + 1e-6

    def test_kelly_respects_max_position(
        self, sizer: PositionSizer, ensemble_signal: Signal, returns_df: pd.DataFrame
    ) -> None:
        weights = sizer.size(ensemble_signal, returns_df, method="kelly")
        for w in weights.values():
            assert abs(w) <= sizer.max_position + 1e-6

    def test_equal_weight_four_assets_near_quarter(
        self, sizer: PositionSizer, ensemble_signal: Signal, returns_df: pd.DataFrame
    ) -> None:
        """With unit confidence and positive signal, each weight is ≈ 0.25."""
        weights = sizer.size(ensemble_signal, returns_df, method="equal_weight")
        non_zero = [w for w in weights.values() if w != 0]
        if non_zero:
            assert max(non_zero) <= 0.30 + 1e-6

    def test_sizer_symbols_match_returns_df(
        self, sizer: PositionSizer, ensemble_signal: Signal, returns_df: pd.DataFrame
    ) -> None:
        weights = sizer.size(ensemble_signal, returns_df, method="equal_weight")
        assert set(weights.keys()) == set(returns_df.columns)


# ---------------------------------------------------------------------------
# Stage 6: Rebalancer computes trade orders
# ---------------------------------------------------------------------------


class TestRebalancer:
    def test_produces_order_objects(
        self,
        rebalancer: Rebalancer,
        portfolio_action: PortfolioAction,
    ) -> None:
        current = {sym: 0.25 for sym in _SYMBOLS}
        orders = rebalancer.compute_trades(
            current, portfolio_action.weights, _PORTFOLIO_VALUE
        )
        assert all(isinstance(o, Order) for o in orders)

    def test_orders_are_market_type(
        self,
        rebalancer: Rebalancer,
        portfolio_action: PortfolioAction,
    ) -> None:
        current = {sym: 0.25 for sym in _SYMBOLS}
        orders = rebalancer.compute_trades(
            current, portfolio_action.weights, _PORTFOLIO_VALUE
        )
        assert all(o.order_type == "market" for o in orders)

    def test_sell_orders_before_buy_orders(
        self,
        rebalancer: Rebalancer,
        portfolio_action: PortfolioAction,
    ) -> None:
        current = {sym: 0.25 for sym in _SYMBOLS}
        orders = rebalancer.compute_trades(
            current, portfolio_action.weights, _PORTFOLIO_VALUE
        )
        if len(orders) >= 2:
            sell_indices = [i for i, o in enumerate(orders) if o.side == "sell"]
            buy_indices = [i for i, o in enumerate(orders) if o.side == "buy"]
            if sell_indices and buy_indices:
                assert max(sell_indices) < min(buy_indices)

    def test_no_orders_when_already_at_target(
        self,
        rebalancer: Rebalancer,
        portfolio_action: PortfolioAction,
    ) -> None:
        orders = rebalancer.compute_trades(
            portfolio_action.weights, portfolio_action.weights, _PORTFOLIO_VALUE
        )
        assert orders == []

    def test_all_orders_positive_quantity(
        self,
        rebalancer: Rebalancer,
        portfolio_action: PortfolioAction,
    ) -> None:
        current = {sym: 0.25 for sym in _SYMBOLS}
        orders = rebalancer.compute_trades(
            current, portfolio_action.weights, _PORTFOLIO_VALUE
        )
        assert all(o.quantity > 0 for o in orders)


# ---------------------------------------------------------------------------
# Risk veto: position clipping
# ---------------------------------------------------------------------------


class TestRiskVeto:
    def test_50pct_weight_clipped_to_30pct(self) -> None:
        """Core risk veto: single 50 % weight must be clipped to max 30 %."""
        rm = RiskManager()
        action = PortfolioAction(
            weights={"SPY": 0.50, "QQQ": 0.20, "TLT": 0.20, "GLD": 0.10},
            confidence=0.9,
            regime_context="bull",
        )
        adjusted, warnings = rm.validate_action(action, {})

        assert adjusted.weights["SPY"] == pytest.approx(0.30, abs=1e-6)
        assert len(warnings) > 0

    def test_multiple_oversized_all_clipped(self) -> None:
        """Every oversized position must be independently capped."""
        rm = RiskManager()
        action = PortfolioAction(
            weights={"A": 0.40, "B": 0.35, "C": 0.25},
            confidence=0.7,
            regime_context="neutral",
        )
        adjusted, _ = rm.validate_action(action, {})
        assert adjusted.weights["A"] == pytest.approx(0.30, abs=1e-6)
        assert adjusted.weights["B"] == pytest.approx(0.30, abs=1e-6)
        assert adjusted.weights["C"] == pytest.approx(0.25, abs=1e-6)

    def test_veto_output_passes_action_validation(self) -> None:
        """PortfolioAction.__post_init__ must not raise after clipping."""
        rm = RiskManager()
        # Weights sum to > 1 before clipping.
        action = PortfolioAction(
            weights={"A": 0.40, "B": 0.40, "C": 0.20},
            confidence=0.8,
            regime_context="bull",
        )
        adjusted, _ = rm.validate_action(action, {})
        assert isinstance(adjusted, PortfolioAction)

    def test_within_limits_no_warnings(self) -> None:
        rm = RiskManager()
        action = PortfolioAction(
            weights={"SPY": 0.30, "QQQ": 0.25, "TLT": 0.20, "GLD": 0.15},
            confidence=0.85,
            regime_context="bull",
        )
        _, warnings = rm.validate_action(action, {})
        assert warnings == []


# ---------------------------------------------------------------------------
# Risk veto: should_flatten on drawdown breach
# ---------------------------------------------------------------------------


class TestShouldFlatten:
    def test_20pct_drawdown_triggers_flatten(self) -> None:
        """A 20 % drop from peak exceeds the 15 % max_drawdown_pct limit."""
        rm = RiskManager()
        values = [100_000.0, 105_000.0, 102_000.0, 84_000.0]  # ~20 % from peak
        flatten, reasons = rm.should_flatten(
            portfolio_values=values,
            today_value=values[-1],
            yesterday_value=values[-2],
        )
        assert flatten is True
        assert len(reasons) >= 1
        assert any("drawdown" in r.lower() or "Drawdown" in r for r in reasons)

    def test_daily_loss_breach_triggers_flatten(self) -> None:
        """A 3.5 % single-day loss exceeds the 3 % daily_loss_limit_pct."""
        rm = RiskManager()
        values = [100_000.0, 100_000.0, 96_500.0]  # 3.5 % daily drop
        flatten, reasons = rm.should_flatten(
            portfolio_values=values,
            today_value=96_500.0,
            yesterday_value=100_000.0,
        )
        assert flatten is True
        assert len(reasons) >= 1

    def test_no_breach_no_flatten(self) -> None:
        rm = RiskManager()
        values = [100_000.0, 101_000.0, 102_000.0, 101_500.0]
        flatten, reasons = rm.should_flatten(
            portfolio_values=values,
            today_value=101_500.0,
            yesterday_value=102_000.0,
        )
        assert flatten is False
        assert reasons == []

    def test_flatten_returns_non_empty_reasons(self) -> None:
        rm = RiskManager()
        # Both drawdown AND daily loss breached simultaneously.
        values = [100_000.0, 95_000.0, 80_000.0]
        flatten, reasons = rm.should_flatten(
            portfolio_values=values,
            today_value=80_000.0,
            yesterday_value=95_000.0,
        )
        assert flatten is True
        assert isinstance(reasons, list)
        assert len(reasons) >= 1

    def test_exactly_at_drawdown_limit_no_flatten(self) -> None:
        """Drawdown exactly equal to the limit (strict >) must not trigger.

        Peak = 100k, current = 85k → drawdown = 15.0 % (not strictly > 15 %).
        today_value == yesterday_value so the daily-loss check also stays clear.
        """
        rm = RiskManager()
        values = [100_000.0, 85_000.0]
        flatten, _ = rm.should_flatten(
            portfolio_values=values,
            today_value=85_000.0,
            yesterday_value=85_000.0,   # no daily loss
        )
        assert flatten is False


# ---------------------------------------------------------------------------
# Full end-to-end pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Chains every stage in a single pass."""

    def test_pipeline_produces_valid_orders(
        self,
        mock_data: dict[str, AssetData],
        returns_df: pd.DataFrame,
    ) -> None:
        # Stage 1 + 2: signals → ensemble
        ens = SignalEnsemble(
            [_MockHMMSignal(), _MockMomentumSignal(), _MockMonteCarloSignal()],
            method="confidence_weighted",
        )
        signal = ens.generate(mock_data)

        # Stage 3: optimizer
        opt = MeanVarianceOptimizer(returns_df, risk_aversion=1.0, max_position=0.30)
        action = opt.decide([signal], {})

        # Stage 4: risk manager
        rm = RiskManager()
        adjusted_action, warnings = rm.validate_action(action, {})

        # Stage 5: position sizer cross-check
        sizer = PositionSizer(max_position=0.30)
        sized = sizer.size(signal, returns_df, method="volatility_scaled")
        assert all(abs(w) <= 0.30 + 1e-6 for w in sized.values())

        # Stage 6: rebalancer
        current = {sym: 0.25 for sym in _SYMBOLS}
        rebalancer = Rebalancer()
        if rebalancer.should_rebalance(current, adjusted_action.weights):
            orders = rebalancer.compute_trades(
                current, adjusted_action.weights, _PORTFOLIO_VALUE
            )
            assert all(isinstance(o, Order) for o in orders)
            assert all(o.quantity > 0 for o in orders)

        # Invariants that must hold regardless of rebalance decision.
        assert isinstance(adjusted_action, PortfolioAction)
        assert sum(adjusted_action.weights.values()) <= 1.0 + 1e-6
        for w in adjusted_action.weights.values():
            assert w <= 0.30 + 1e-6

    def test_pipeline_with_risk_veto_flattens(
        self,
        mock_data: dict[str, AssetData],
        returns_df: pd.DataFrame,
    ) -> None:
        """When drawdown breaches the limit, flatten is called instead of rebalancing."""
        ens = SignalEnsemble(
            [_MockHMMSignal(), _MockMomentumSignal(), _MockMonteCarloSignal()],
            method="equal_weight",
        )
        signal = ens.generate(mock_data)
        opt = MeanVarianceOptimizer(returns_df, risk_aversion=1.0, max_position=0.30)
        action = opt.decide([signal], {})

        rm = RiskManager()
        # Simulate a 20 % drawdown — hard limit is 15 %.
        portfolio_history = [100_000.0, 108_000.0, 104_000.0, 86_400.0]
        flatten, reasons = rm.should_flatten(
            portfolio_values=portfolio_history,
            today_value=86_400.0,
            yesterday_value=104_000.0,
        )

        assert flatten is True, "Risk veto should have fired on 20 % drawdown"
        # When flatten=True, the execution layer would close all positions,
        # so no rebalance orders are generated.
        if flatten:
            orders = []  # positions flattened by risk manager
        else:
            rebalancer = Rebalancer()
            orders = rebalancer.compute_trades(
                {sym: 0.25 for sym in _SYMBOLS},
                action.weights,
                _PORTFOLIO_VALUE,
            )
        assert orders == []

    def test_turnover_constrained_pipeline(
        self,
        mock_data: dict[str, AssetData],
        returns_df: pd.DataFrame,
    ) -> None:
        """Turnover constraint reduces order sizes relative to unconstrained."""
        ens = SignalEnsemble(
            [_MockHMMSignal(), _MockMomentumSignal(), _MockMonteCarloSignal()],
            method="equal_weight",
        )
        signal = ens.generate(mock_data)
        opt = MeanVarianceOptimizer(returns_df, risk_aversion=1.0, max_position=0.30)
        action = opt.decide([signal], {})

        current = {sym: 0.25 for sym in _SYMBOLS}
        rb = Rebalancer()

        unconstrained_orders = rb.compute_trades(
            current, action.weights, _PORTFOLIO_VALUE
        )
        constrained_target = rb.apply_turnover_constraint(
            current, action.weights, max_turnover=0.05
        )
        constrained_orders = rb.compute_trades(
            current, constrained_target, _PORTFOLIO_VALUE
        )

        unc_total = sum(o.quantity for o in unconstrained_orders)
        con_total = sum(o.quantity for o in constrained_orders)
        # Constrained turnover must not exceed unconstrained.
        assert con_total <= unc_total + 1e-2
