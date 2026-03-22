"""CLI backtest runner.

Fetches real market data (cached locally in SQLite), runs the full QuantForge
pipeline — HMM regime detection, momentum signals, mean-variance optimisation,
risk management, and walk-forward validation — then writes an HTML report and
prints a summary to the terminal.

Usage
-----
    # Default universe (SPY QQQ TLT GLD SHY), 2020-2025:
    python scripts/run_backtest.py

    # Custom universe and date range:
    python scripts/run_backtest.py --universe SPY QQQ IEF GLD \\
        --start 2018-01-01 --end 2024-12-31

    # Suppress walk-forward (faster):
    python scripts/run_backtest.py --no-walk-forward
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path so the script works from any directory.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.report import ReportGenerator  # noqa: E402
from src.backtest.walk_forward import WalkForwardValidator  # noqa: E402
from src.data.cache import SQLiteCacheProvider  # noqa: E402
from src.data.yahoo import YahooFinanceProvider  # noqa: E402
from src.portfolio.optimizer import MeanVarianceOptimizer  # noqa: E402
from src.portfolio.rebalancer import Rebalancer  # noqa: E402
from src.portfolio.risk_manager import RiskManager  # noqa: E402
from src.signals.hmm_regime import HMMRegimeDetector  # noqa: E402
from src.signals.momentum import MomentumSignal  # noqa: E402

# ---------------------------------------------------------------------------
# Logging — INFO to stderr so stdout stays clean for piping
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # suppress noisy library output
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("run_backtest")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "SHY"]
_DEFAULT_START = "2020-01-01"
_DEFAULT_END = "2025-12-31"
_DEFAULT_REPORT = "data/results/backtest_report.html"
_CACHE_DB = "data/cache/market_data.db"

_DIVIDER = "─" * 60


def _print(msg: str = "") -> None:
    """Print to stdout and flush immediately so progress appears in real time."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QuantForge backtest runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--universe",
        nargs="+",
        default=_DEFAULT_UNIVERSE,
        metavar="SYMBOL",
        help="Space-separated list of ticker symbols to trade.",
    )
    p.add_argument(
        "--start",
        default=_DEFAULT_START,
        metavar="YYYY-MM-DD",
        help="Backtest start date (inclusive).",
    )
    p.add_argument(
        "--end",
        default=_DEFAULT_END,
        metavar="YYYY-MM-DD",
        help="Backtest end date (inclusive).",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        metavar="DOLLARS",
        help="Initial portfolio capital.",
    )
    p.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        metavar="BPS",
        help="Transaction cost in basis points per trade.",
    )
    p.add_argument(
        "--rebalance",
        default="monthly",
        choices=["daily", "weekly", "monthly"],
        help="How often to rebalance the portfolio.",
    )
    p.add_argument(
        "--train-window",
        type=int,
        default=252,
        metavar="DAYS",
        help="Walk-forward training window (trading days).",
    )
    p.add_argument(
        "--test-window",
        type=int,
        default=63,
        metavar="DAYS",
        help="Walk-forward test window (trading days).",
    )
    p.add_argument(
        "--report",
        default=_DEFAULT_REPORT,
        metavar="PATH",
        help="Output path for the HTML report.",
    )
    p.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Skip walk-forward validation (saves time).",
    )
    p.add_argument(
        "--cache-db",
        default=_CACHE_DB,
        metavar="PATH",
        help="SQLite cache database path.",
    )
    return p


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def fetch_data(
    symbols: list[str],
    start: str,
    end: str,
    cache_db: str,
) -> dict:
    """Fetch OHLCV data for *symbols* via Yahoo Finance, caching to SQLite.

    Args:
        symbols: Ticker symbols to fetch.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        cache_db: Path to the SQLite cache database.

    Returns:
        Mapping of symbol → AssetData.
    """
    _print(f"  Connecting to cache at {cache_db} …")
    provider = SQLiteCacheProvider(
        provider=YahooFinanceProvider(),
        db_path=cache_db,
    )

    stats = provider.cache_stats()
    cached_syms = [s for s in symbols if stats.get(s, 0) > 0]
    fresh_syms = [s for s in symbols if s not in cached_syms]

    if cached_syms:
        _print(f"  Cache hit  : {', '.join(cached_syms)}")
    if fresh_syms:
        _print(f"  Fetching   : {', '.join(fresh_syms)} from Yahoo Finance …")

    t0 = time.monotonic()
    data = provider.fetch_universe(symbols, start=start, end=end)
    elapsed = time.monotonic() - t0

    fetched = sorted(data.keys())
    missing = sorted(set(symbols) - set(fetched))

    _print(f"  Fetched {len(fetched)}/{len(symbols)} symbol(s) in {elapsed:.1f}s")
    if missing:
        _print(f"  WARNING — skipped (no data): {', '.join(missing)}")

    for sym, asset in data.items():
        rows = len(asset.ohlcv)
        first = asset.ohlcv.index[0].date()
        last = asset.ohlcv.index[-1].date()
        _print(f"    {sym:6s}  {rows:4d} rows  {first} → {last}")

    return data


# ---------------------------------------------------------------------------
# Returns DataFrame builder
# ---------------------------------------------------------------------------


def build_returns_df(data: dict) -> pd.DataFrame:
    """Build a daily returns DataFrame aligned across all assets."""
    all_dates = pd.DatetimeIndex([])
    for asset in data.values():
        all_dates = all_dates.union(asset.ohlcv.index)
    all_dates = all_dates.sort_values()

    df = pd.DataFrame(index=all_dates)
    for sym, asset in data.items():
        closes = asset.ohlcv["close"].reindex(all_dates)
        df[sym] = closes.pct_change().fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def build_engine(
    data: dict,
    returns_df: pd.DataFrame,
    capital: float,
    rebalance: str,
    cost_bps: float,
) -> BacktestEngine:
    """Assemble BacktestEngine with the standard QuantForge pipeline."""
    return BacktestEngine(
        signals=[
            HMMRegimeDetector(n_states=3, n_iter=100, random_state=42),
            MomentumSignal(),
        ],
        agent=MeanVarianceOptimizer(
            returns_df=returns_df,
            risk_aversion=1.0,
            lookback=60,
            max_position=0.40,  # risk manager will clip to 0.30
        ),
        risk_manager=RiskManager(
            {
                "max_position_pct": 0.30,
                "max_drawdown_pct": 0.15,
                "daily_loss_limit_pct": 0.03,
                "correlation_limit": 0.70,
                "max_leverage": 1.0,
            }
        ),
        rebalancer=Rebalancer(),
        initial_capital=capital,
        rebalance_frequency=rebalance,
        transaction_cost_bps=cost_bps,
    )


# ---------------------------------------------------------------------------
# Terminal metrics printer
# ---------------------------------------------------------------------------


def print_metrics(result, label: str = "Backtest") -> None:
    """Print a formatted metrics summary to stdout."""
    m = result.metrics
    pv = result.portfolio_values
    initial = float(pv.iloc[0])
    final = float(pv.iloc[-1])
    total_return = (final / initial - 1.0) * 100.0

    n_trades = sum(len(orders) for orders in result.trades_history)
    n_rebalances = len(result.trades_history)
    n_risk_events = len(result.risk_events)

    _print()
    _print(_DIVIDER)
    _print(f"  {label}")
    _print(_DIVIDER)
    _print(f"  {'Metric':<28} {'Value':>12}")
    _print(f"  {'─' * 28} {'─' * 12}")
    _print(f"  {'Total Return':<28} {total_return:>+11.2f}%")
    _print(f"  {'Annual Return':<28} {m['annual_return']:>+11.2%}")
    _print(f"  {'Annual Volatility':<28} {m['annual_volatility']:>11.2%}")
    _print(f"  {'Sharpe Ratio':<28} {m['sharpe_ratio']:>12.3f}")
    _print(f"  {'Sortino Ratio':<28} {_fmt_inf(m['sortino_ratio']):>12}")
    _print(f"  {'Max Drawdown':<28} {-m['max_drawdown']:>+11.2%}")
    _print(f"  {'Calmar Ratio':<28} {m['calmar_ratio']:>12.3f}")
    _print(f"  {'Win Rate':<28} {m['win_rate']:>11.1%}")
    _print(f"  {'Profit Factor':<28} {_fmt_inf(m['profit_factor']):>12}")
    _print(f"  {'VaR 5%':<28} {-m['var_historical']:>+11.2%}")
    _print(f"  {'CVaR 5%':<28} {-m['cvar_historical']:>+11.2%}")
    _print(f"  {'─' * 28} {'─' * 12}")
    _print(f"  {'Initial Capital':<28} ${initial:>11,.0f}")
    _print(f"  {'Final Value':<28} ${final:>11,.0f}")
    _print(f"  {'Rebalances':<28} {n_rebalances:>12,}")
    _print(f"  {'Total Orders':<28} {n_trades:>12,}")
    _print(f"  {'Risk Events':<28} {n_risk_events:>12,}")
    if result.risk_events:
        for ev in result.risk_events[:5]:
            _print(f"    ⚠  {ev}")
        if len(result.risk_events) > 5:
            _print(f"    … and {len(result.risk_events) - 5} more")
    _print(_DIVIDER)


def _fmt_inf(value: float) -> str:
    if value == float("inf"):
        return "∞"
    if value == float("-inf"):
        return "-∞"
    return f"{value:.3f}"


def print_walk_forward_summary(wf_result) -> None:
    """Print per-fold and aggregate walk-forward metrics."""
    _print()
    _print(_DIVIDER)
    _print("  Walk-Forward Validation")
    _print(_DIVIDER)
    _print(f"  Folds: {wf_result.n_folds}")
    _print()
    _print(f"  {'Fold':<6} {'Sharpe':>8} {'Max DD':>8} {'Ann Ret':>9} {'Win Rate':>9}")
    _print(f"  {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 9} {'─' * 9}")
    for i, fm in enumerate(wf_result.fold_metrics):
        _print(
            f"  {i + 1:<6} "
            f"{fm['sharpe_ratio']:>8.3f} "
            f"{-fm['max_drawdown']:>+8.2%} "
            f"{fm['annual_return']:>+9.2%} "
            f"{fm['win_rate']:>9.1%}"
        )
    _print(f"  {'─' * 6} {'─' * 8} {'─' * 8} {'─' * 9} {'─' * 9}")
    cm = wf_result.combined_metrics
    _print(
        f"  {'ALL':<6} "
        f"{cm['sharpe_ratio']:>8.3f} "
        f"{-cm['max_drawdown']:>+8.2%} "
        f"{cm['annual_return']:>+9.2%} "
        f"{cm['win_rate']:>9.1%}"
    )
    _print(_DIVIDER)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()

    universe: list[str] = [s.upper() for s in args.universe]
    start: str = args.start
    end: str = args.end

    _print()
    _print("=" * 60)
    _print("  QuantForge Backtest Runner")
    _print("=" * 60)
    _print(f"  Universe  : {', '.join(universe)}")
    _print(f"  Period    : {start}  →  {end}")
    _print(f"  Capital   : ${args.capital:,.0f}")
    _print(f"  Rebalance : {args.rebalance}  |  Cost: {args.cost_bps} bps")
    _print(f"  Report    : {args.report}")
    _print()

    # ------------------------------------------------------------------
    # Step 1 — fetch data
    # ------------------------------------------------------------------
    _print("[1/5]  Fetching market data …")
    data = fetch_data(universe, start, end, args.cache_db)

    if not data:
        _print("ERROR: No data could be fetched. Check your internet connection.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2 — build returns DataFrame for the optimizer
    # ------------------------------------------------------------------
    _print()
    _print("[2/5]  Building returns matrix …")
    returns_df = build_returns_df(data)
    _print(f"  Returns matrix: {len(returns_df)} rows × {len(returns_df.columns)} assets")

    # ------------------------------------------------------------------
    # Step 3 — run backtest
    # ------------------------------------------------------------------
    _print()
    _print("[3/5]  Running backtest …")
    _print("  Signals  : HMMRegimeDetector(n_states=3) + MomentumSignal()")
    _print("  Agent    : MeanVarianceOptimizer(risk_aversion=1.0)")
    _print("  Risk     : max_pos=30%  max_dd=15%  daily_loss=3%")

    engine = build_engine(data, returns_df, args.capital, args.rebalance, args.cost_bps)

    t0 = time.monotonic()
    result = engine.run(data, start=start, end=end)
    elapsed = time.monotonic() - t0
    _print(f"  Completed in {elapsed:.1f}s  ({len(result.portfolio_values)} trading days)")

    print_metrics(result, label="Full-Period Backtest")

    # ------------------------------------------------------------------
    # Step 4 — walk-forward validation
    # ------------------------------------------------------------------
    if not args.no_walk_forward:
        _print()
        _print(
            f"[4/5]  Walk-forward validation "
            f"(train={args.train_window}d, test={args.test_window}d) …"
        )

        # Fresh engine/signal instances for the validator (avoid stale HMM state).
        wf_engine = build_engine(data, returns_df, args.capital, args.rebalance, args.cost_bps)
        validator = WalkForwardValidator(
            engine=wf_engine,
            train_window=args.train_window,
            test_window=args.test_window,
            expanding=True,
        )

        t0 = time.monotonic()
        try:
            wf_result = validator.run(data)
            elapsed = time.monotonic() - t0
            _print(f"  Completed in {elapsed:.1f}s  ({wf_result.n_folds} folds)")
            print_walk_forward_summary(wf_result)
        except ValueError as exc:
            _print(f"  Skipped: {exc}")
    else:
        _print()
        _print("[4/5]  Walk-forward validation skipped (--no-walk-forward).")

    # ------------------------------------------------------------------
    # Step 5 — generate HTML report
    # ------------------------------------------------------------------
    _print()
    _print(f"[5/5]  Generating HTML report → {args.report} …")
    report_dir = os.path.dirname(os.path.abspath(args.report))
    os.makedirs(report_dir, exist_ok=True)

    t0 = time.monotonic()
    ReportGenerator().generate_html(result, args.report)
    elapsed = time.monotonic() - t0
    size_kb = os.path.getsize(args.report) / 1024
    _print(f"  Report written  ({size_kb:.0f} KB)  in {elapsed:.1f}s")

    _print()
    _print("=" * 60)
    _print("  Done.")
    _print("=" * 60)
    _print()


if __name__ == "__main__":
    main()
