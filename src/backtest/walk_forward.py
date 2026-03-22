"""Walk-forward validation for backtest strategies.

Splits historical data into sequential train/test folds, trains the agent on
each train window, runs a backtest on the corresponding test window, and
aggregates the out-of-sample results.

Two window modes are supported:

Expanding (default)
    The train window grows by one test window on each fold; the start date is
    always the beginning of the dataset::

        fold 0:  train=[0, T)            test=[T,   T+W)
        fold 1:  train=[0, T+W)          test=[T+W, T+2W)
        fold 2:  train=[0, T+2W)         test=[T+2W,T+3W)

Rolling
    The train window size is fixed; both start and end advance each fold::

        fold 0:  train=[0,   T)          test=[T,   T+W)
        fold 1:  train=[W,   T+W)        test=[T+W, T+2W)
        fold 2:  train=[2W,  T+2W)       test=[T+2W,T+3W)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import compute_all
from src.data.base import AssetData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Aggregated results from a walk-forward validation run.

    Args:
        folds: :class:`~src.backtest.engine.BacktestResult` for each test fold,
            in chronological order.
        combined_returns: Out-of-sample daily returns concatenated across all
            test folds (no gaps; index is a continuous DatetimeIndex).
        combined_metrics: Performance metrics computed on *combined_returns*
            and the corresponding cumulative portfolio-value series.
        fold_metrics: Per-fold ``metrics`` dicts for stability analysis
            (same keys as :func:`~src.backtest.metrics.compute_all`).
        n_folds: Number of completed folds.
    """

    folds: list[BacktestResult]
    combined_returns: pd.Series
    combined_metrics: dict
    fold_metrics: list[dict]
    n_folds: int


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class WalkForwardValidator:
    """Runs sequential train/test folds over a historical dataset.

    Args:
        engine: Configured :class:`~src.backtest.engine.BacktestEngine`.
            Its agent's ``train()`` method is called on each train window.
        train_window: Number of trading days in the initial training window.
            Defaults to 252 (1 calendar year).
        test_window: Number of trading days in each test (out-of-sample)
            window.  Defaults to 63 (1 quarter).
        expanding: When ``True`` (default) the train window expands fold by
            fold.  When ``False`` the train window rolls forward at a fixed
            size equal to *train_window*.

    Raises:
        ValueError: If *train_window* or *test_window* is not positive.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        train_window: int = 252,
        test_window: int = 63,
        expanding: bool = True,
    ) -> None:
        if train_window <= 0:
            raise ValueError(f"train_window must be positive, got {train_window}")
        if test_window <= 0:
            raise ValueError(f"test_window must be positive, got {test_window}")

        self._engine = engine
        self._train_window = train_window
        self._test_window = test_window
        self._expanding = expanding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data: dict[str, AssetData]) -> WalkForwardResult:
        """Execute the walk-forward validation.

        Args:
            data: Full historical dataset.  Each asset must cover the entire
                date range.  Missing dates for individual assets are forward-
                filled at the close level inside the engine.

        Returns:
            :class:`WalkForwardResult` with per-fold and aggregated metrics.

        Raises:
            ValueError: If *data* is empty or does not contain enough trading
                days to form at least one complete train + test fold.
        """
        if not data:
            raise ValueError("data must contain at least one asset")

        dates = self._sorted_dates(data)
        min_required = self._train_window + self._test_window
        if len(dates) < min_required:
            raise ValueError(
                f"Dataset has {len(dates)} trading days but at least "
                f"{min_required} are required for one fold "
                f"(train_window={self._train_window} + test_window={self._test_window})"
            )

        # Build fold index boundaries.
        folds_indices = self._fold_indices(dates)
        if not folds_indices:
            raise ValueError("No complete folds could be constructed from the available data")

        folds: list[BacktestResult] = []
        fold_metrics: list[dict] = []

        for fold_num, (train_start_i, train_end_i, test_start_i, test_end_i) in enumerate(
            folds_indices
        ):
            train_dates = dates[train_start_i:train_end_i]
            test_dates = dates[test_start_i:test_end_i]

            train_start = str(train_dates[0].date())
            train_end = str(train_dates[-1].date())
            test_start = str(test_dates[0].date())
            test_end = str(test_dates[-1].date())

            logger.info(
                "Walk-forward fold %d/%d: train=[%s, %s] test=[%s, %s]",
                fold_num + 1,
                len(folds_indices),
                train_start,
                train_end,
                test_start,
                test_end,
            )

            # 1. Slice train data and call agent.train().
            train_data = _slice_to_dates(data, train_dates)
            self._train_agent(train_data, train_start, train_end)

            # 2. Run backtest on the test window.
            test_result = self._engine.run(data, start=test_start, end=test_end)

            folds.append(test_result)
            fold_metrics.append(dict(test_result.metrics))

        # 3. Concatenate out-of-sample returns.
        combined_returns = pd.concat([f.returns for f in folds]).sort_index()

        # Build a cumulative value series for the combined-metrics calculation.
        combined_values = (1.0 + combined_returns).cumprod() * self._engine._initial_capital
        combined_metrics = compute_all(combined_returns, combined_values)

        return WalkForwardResult(
            folds=folds,
            combined_returns=combined_returns,
            combined_metrics=combined_metrics,
            fold_metrics=fold_metrics,
            n_folds=len(folds),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sorted_dates(self, data: dict[str, AssetData]) -> pd.DatetimeIndex:
        """Union of all dates across assets, sorted ascending."""
        all_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        for asset in data.values():
            all_dates = all_dates.union(asset.ohlcv.index)
        return all_dates.sort_values()

    def _fold_indices(self, dates: pd.DatetimeIndex) -> list[tuple[int, int, int, int]]:
        """Return ``(train_start, train_end, test_start, test_end)`` index tuples.

        All indices are half-open (Python slice convention): ``[start, end)``.
        """
        folds: list[tuple[int, int, int, int]] = []
        n = len(dates)
        fold = 0

        while True:
            if self._expanding:
                train_start_i = 0
                train_end_i = self._train_window + fold * self._test_window
            else:
                train_start_i = fold * self._test_window
                train_end_i = train_start_i + self._train_window

            test_start_i = train_end_i
            test_end_i = test_start_i + self._test_window

            # Stop if we'd read past the end.
            if test_end_i > n:
                break

            folds.append((train_start_i, train_end_i, test_start_i, test_end_i))
            fold += 1

        return folds

    def _train_agent(
        self,
        train_data: dict[str, AssetData],
        train_start: str,
        train_end: str,
    ) -> None:
        """Generate historical signals and call ``agent.train()``."""
        agent = self._engine._agent
        signals_list = []
        for gen in self._engine._signals:
            try:
                sig = gen.update(train_data)
                signals_list.append(sig)
            except Exception:
                logger.exception("Signal generator '%s' raised during training", gen.name)

        # Build a minimal returns DataFrame from train data close prices.
        train_dates_idx = _union_dates(train_data)
        returns_df = pd.DataFrame(index=train_dates_idx, dtype=float)
        for sym, asset in train_data.items():
            closes = asset.ohlcv["close"].reindex(train_dates_idx)
            returns_df[sym] = closes.pct_change().fillna(0.0)

        try:
            agent.train(signals_list, returns_df)
        except Exception:
            logger.exception("agent.train() raised an error during fold training")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _slice_to_dates(
    data: dict[str, AssetData],
    dates: pd.DatetimeIndex,
) -> dict[str, AssetData]:
    """Return a copy of *data* with each OHLCV DataFrame restricted to *dates*."""
    sliced: dict[str, AssetData] = {}
    for sym, asset in data.items():
        ohlcv_slice = asset.ohlcv.loc[asset.ohlcv.index.isin(dates)]
        sliced[sym] = AssetData(symbol=sym, ohlcv=ohlcv_slice, metadata=dict(asset.metadata))
    return sliced


def _union_dates(data: dict[str, AssetData]) -> pd.DatetimeIndex:
    idx: pd.DatetimeIndex = pd.DatetimeIndex([])
    for asset in data.values():
        idx = idx.union(asset.ohlcv.index)
    return idx.sort_values()
