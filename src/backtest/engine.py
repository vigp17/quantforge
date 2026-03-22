"""Backtest engine — event-driven portfolio simulation.

Drives a portfolio through historical data, applying signals, agent decisions,
risk validation, rebalancing, and transaction costs at each rebalance period.

Typical usage::

    engine = BacktestEngine(
        signals=[hmm_gen, momentum_gen],
        agent=optimizer,
        risk_manager=RiskManager(),
        rebalancer=Rebalancer(),
        initial_capital=100_000,
        rebalance_frequency="monthly",
        transaction_cost_bps=5.0,
    )
    result = engine.run(data, start="2022-01-01", end="2023-12-31")
    print(result.metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.backtest.metrics import compute_all
from src.data.base import AssetData
from src.execution.base import Order
from src.portfolio.base import PortfolioAgent
from src.portfolio.rebalancer import Rebalancer
from src.portfolio.risk_manager import RiskManager
from src.signals.base import Signal, SignalGenerator

logger = logging.getLogger(__name__)

_VALID_FREQUENCIES = frozenset({"daily", "weekly", "monthly"})


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Container for a completed backtest run.

    Args:
        portfolio_values: Date-indexed series of portfolio NAV.
        returns: Daily arithmetic return series (aligned with portfolio_values).
        weights_history: List of ``{date: str, symbol: weight, …}`` dicts,
            one entry per rebalance event.
        trades_history: List of order lists, one per rebalance event.
        signals_history: Flat list of all Signal objects generated during
            the run (all generators, all rebalance dates).
        risk_events: Human-readable strings describing any risk violations
            or position adjustments that occurred.
        metrics: Output of :func:`~src.backtest.metrics.compute_all`.
        config: Strategy parameters used for this run (for reproducibility).
    """

    portfolio_values: pd.Series
    returns: pd.Series
    weights_history: list[dict]
    trades_history: list[list[Order]]
    signals_history: list[Signal]
    risk_events: list[str]
    metrics: dict[str, float]
    config: dict[str, Any]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Event-driven backtest engine.

    Args:
        signals: Signal generators called at each rebalance date.
        agent: Portfolio allocation agent.
        risk_manager: Validates and potentially vetoes agent decisions.
        rebalancer: Translates weight changes to trade orders.
        initial_capital: Starting portfolio value in base currency.
        rebalance_frequency: How often to rebalance — ``"daily"``,
            ``"weekly"`` (first trading day of each ISO week), or
            ``"monthly"`` (first trading day of each calendar month).
        transaction_cost_bps: One-way cost per trade in basis points
            (applied to total absolute notional of all orders at each
            rebalance).  Defaults to 5.0 bps.

    Raises:
        ValueError: If ``rebalance_frequency`` is not recognised or
            ``initial_capital`` is non-positive.
    """

    def __init__(
        self,
        signals: list[SignalGenerator],
        agent: PortfolioAgent,
        risk_manager: RiskManager,
        rebalancer: Rebalancer,
        initial_capital: float = 100_000.0,
        rebalance_frequency: str = "monthly",
        transaction_cost_bps: float = 5.0,
    ) -> None:
        if rebalance_frequency not in _VALID_FREQUENCIES:
            raise ValueError(
                f"rebalance_frequency must be one of {sorted(_VALID_FREQUENCIES)}, "
                f"got '{rebalance_frequency}'"
            )
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")

        self._signals = signals
        self._agent = agent
        self._risk_manager = risk_manager
        self._rebalancer = rebalancer
        self._initial_capital = initial_capital
        self._rebalance_frequency = rebalance_frequency
        self._transaction_cost_bps = transaction_cost_bps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: dict[str, AssetData],
        start: str,
        end: str,
    ) -> BacktestResult:
        """Run the backtest over the supplied dataset.

        Args:
            data: Mapping of symbol to :class:`~src.data.base.AssetData`.
                Each asset's ``ohlcv`` DataFrame must have a
                ``DatetimeIndex`` and a ``close`` column.
            start: Inclusive start date in ``YYYY-MM-DD`` format.
            end: Inclusive end date in ``YYYY-MM-DD`` format.

        Returns:
            :class:`BacktestResult` with full history and metrics.

        Raises:
            ValueError: If *data* is empty or the date range yields no
                trading days.
        """
        if not data:
            raise ValueError("data must contain at least one asset")

        # 1. Build sorted trading-date index from all assets, filtered to range.
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        all_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        for asset in data.values():
            all_dates = all_dates.union(asset.ohlcv.index)
        trading_dates = all_dates[(all_dates >= start_ts) & (all_dates <= end_ts)].sort_values()

        if len(trading_dates) == 0:
            raise ValueError(
                f"No trading dates found between {start} and {end} in the supplied data"
            )

        symbols = list(data.keys())

        # 2. Pre-compute daily close-to-close returns matrix (NaN → 0).
        returns_df = pd.DataFrame(index=trading_dates, dtype=float)
        for sym in symbols:
            closes = data[sym].ohlcv["close"].reindex(trading_dates)
            returns_df[sym] = closes.pct_change().fillna(0.0)

        # 3. Determine rebalance dates.
        rebalance_dates = self._get_rebalance_dates(trading_dates)

        # 4. Main simulation loop.
        portfolio_value = self._initial_capital
        current_weights: dict[str, float] = {}

        pv_list: list[tuple[pd.Timestamp, float]] = []
        # Separate NAV window for risk checks — resets after each flatten event
        # so the drawdown clock starts from the post-flatten NAV, not the
        # all-time high that triggered the previous breach.
        pv_for_risk: list[float] = []
        weights_history: list[dict] = []
        trades_history: list[list[Order]] = []
        signals_history: list[Signal] = []
        risk_events: list[str] = []

        for i, date in enumerate(trading_dates):
            # Apply today's market return to the portfolio (skip day 0 — no prior close).
            if i > 0:
                day_portfolio_return = sum(
                    current_weights.get(sym, 0.0) * float(returns_df.loc[date, sym])
                    for sym in symbols
                )
                portfolio_value = portfolio_value * (1.0 + day_portfolio_return)

            pv_list.append((date, portfolio_value))
            pv_for_risk.append(portfolio_value)

            # Rebalance if today is a scheduled rebalance date.
            if date in rebalance_dates:
                sliced = _slice_data(data, date)

                # Generate signals from all generators.
                signals: list[Signal] = []
                for gen in self._signals:
                    try:
                        sig = gen.update(sliced)
                        signals.append(sig)
                    except Exception:
                        logger.exception(
                            "Signal generator '%s' raised an error at %s", gen.name, date
                        )
                signals_history.extend(signals)

                # Agent decides target allocation.
                action = self._agent.decide(signals, dict(current_weights))

                # Risk veto check — use the post-flatten window so the drawdown
                # is measured from the reset peak, not the all-time high.
                yesterday_value = pv_list[-2][1] if len(pv_list) >= 2 else portfolio_value
                flatten, flatten_reasons = self._risk_manager.should_flatten(
                    pv_for_risk, portfolio_value, yesterday_value
                )

                if flatten:
                    for reason in flatten_reasons:
                        msg = f"{date.date()}: FLATTEN — {reason}"
                        risk_events.append(msg)
                        logger.warning(msg)
                    target_weights: dict[str, float] = {}
                    # Reset the risk window and peak so the next rebalance
                    # measures drawdown from the post-flatten NAV, not the
                    # pre-flatten high that triggered this breach.
                    self._risk_manager.reset_peak(portfolio_value)
                    pv_for_risk = [portfolio_value]
                else:
                    adjusted, warnings = self._risk_manager.validate_action(action, {})
                    for w in warnings:
                        risk_events.append(f"{date.date()}: {w}")
                    target_weights = adjusted.weights

                # Compute and record trades.
                orders = self._rebalancer.compute_trades(
                    current_weights, target_weights, portfolio_value
                )
                trades_history.append(orders)

                # Deduct transaction costs (proportional to total notional traded).
                total_notional = sum(o.quantity for o in orders)
                cost = total_notional * self._transaction_cost_bps / 10_000.0
                portfolio_value = max(portfolio_value - cost, 0.0)

                # Patch the latest portfolio value to reflect post-cost NAV.
                pv_list[-1] = (date, portfolio_value)

                current_weights = dict(target_weights)
                weights_history.append({"date": str(date.date()), **current_weights})

                logger.debug(
                    "Rebalanced at %s: weights=%s, orders=%d, cost=%.2f, NAV=%.2f",
                    date.date(),
                    target_weights,
                    len(orders),
                    cost,
                    portfolio_value,
                )

        # 5. Build output series and compute summary metrics.
        pv_series = pd.Series(
            data={ts: val for ts, val in pv_list},
            name="portfolio_value",
            dtype=float,
        )
        returns_series = pv_series.pct_change().dropna()
        metrics = compute_all(returns_series, pv_series)

        return BacktestResult(
            portfolio_values=pv_series,
            returns=returns_series,
            weights_history=weights_history,
            trades_history=trades_history,
            signals_history=signals_history,
            risk_events=risk_events,
            metrics=metrics,
            config=self._config_dict(start, end),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_rebalance_dates(self, trading_dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
        """Return the subset of *trading_dates* on which to rebalance."""
        if self._rebalance_frequency == "daily":
            return set(trading_dates)

        seen: set[tuple] = set()
        rebal: list[pd.Timestamp] = []
        for d in trading_dates:
            if self._rebalance_frequency == "weekly":
                iso = d.isocalendar()
                key = (int(iso.year), int(iso.week))
            else:  # monthly
                key = (d.year, d.month)  # type: ignore[assignment]
            if key not in seen:
                seen.add(key)
                rebal.append(d)

        return set(rebal)

    def _config_dict(self, start: str, end: str) -> dict[str, Any]:
        return {
            "initial_capital": self._initial_capital,
            "rebalance_frequency": self._rebalance_frequency,
            "transaction_cost_bps": self._transaction_cost_bps,
            "n_signal_generators": len(self._signals),
            "agent": type(self._agent).__name__,
            "start": start,
            "end": end,
        }


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _slice_data(
    data: dict[str, AssetData],
    up_to: pd.Timestamp,
) -> dict[str, AssetData]:
    """Return a copy of *data* with each OHLCV DataFrame sliced to ``≤ up_to``.

    Args:
        data: Full dataset.
        up_to: Inclusive upper bound for the slice.

    Returns:
        New ``dict[str, AssetData]`` with truncated OHLCV DataFrames.
    """
    sliced: dict[str, AssetData] = {}
    for sym, asset in data.items():
        ohlcv_slice = asset.ohlcv.loc[asset.ohlcv.index <= up_to]
        sliced[sym] = AssetData(symbol=sym, ohlcv=ohlcv_slice, metadata=dict(asset.metadata))
    return sliced
