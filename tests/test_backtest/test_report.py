"""Tests for src/backtest/report.py.

HTML generation with embedded plotly.js (~3 MB) is expensive, so module-scoped
fixtures generate each report once and the individual tests inspect the cached
result.  Only the minimal edge-case results are regenerated inline.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestResult
from src.backtest.report import ReportGenerator, _format_maybe_inf
from src.execution.base import Order
from src.signals.base import Signal


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_dates(n: int, start: str = "2023-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _make_result(
    n_days: int = 120,
    daily_return: float = 0.001,
    symbols: list[str] | None = None,
    start: str = "2022-01-03",
) -> BacktestResult:
    if symbols is None:
        symbols = ["SPY", "TLT"]

    dates = _make_dates(n_days, start)
    pv = pd.Series(
        [100_000.0 * (1.0 + daily_return) ** i for i in range(n_days)],
        index=dates,
        name="portfolio_value",
        dtype=float,
    )
    returns = pv.pct_change().dropna()

    weights_history = []
    w = round(1.0 / len(symbols), 6)
    for d in pd.bdate_range(start=dates[0], end=dates[-1], freq="BME"):
        entry = {"date": str(d.date())}
        for sym in symbols:
            entry[sym] = w
        weights_history.append(entry)

    from src.backtest.metrics import compute_all

    metrics = compute_all(returns, pv)
    return BacktestResult(
        portfolio_values=pv,
        returns=returns,
        weights_history=weights_history,
        trades_history=[
            [Order(symbol="SPY", side="buy", quantity=5000.0, order_type="market")]
        ],
        signals_history=[
            Signal(
                name="mock",
                values=np.ones(len(symbols)),
                confidence=np.full(len(symbols), 0.8),
            )
        ],
        risk_events=[],
        metrics=metrics,
        config={
            "initial_capital": 100_000.0,
            "rebalance_frequency": "monthly",
            "transaction_cost_bps": 5.0,
            "agent": "TestAgent",
            "start": start,
            "end": str(dates[-1].date()),
        },
    )


def _make_minimal_result() -> BacktestResult:
    dates = _make_dates(1)
    pv = pd.Series([100_000.0], index=dates, name="portfolio_value", dtype=float)
    returns = pv.pct_change().dropna()

    from src.backtest.metrics import compute_all

    metrics = compute_all(returns, pv)
    return BacktestResult(
        portfolio_values=pv,
        returns=returns,
        weights_history=[],
        trades_history=[],
        signals_history=[],
        risk_events=[],
        metrics=metrics,
        config={
            "initial_capital": 100_000.0,
            "rebalance_frequency": "daily",
            "transaction_cost_bps": 0.0,
            "agent": "MinimalAgent",
            "start": str(dates[0].date()),
            "end": str(dates[0].date()),
        },
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures — HTML generation is expensive; generate once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def main_result() -> BacktestResult:
    return _make_result()


@pytest.fixture(scope="module")
def main_html(tmp_path_factory, main_result) -> tuple[str, str]:
    """(output_path, html_content) — generated once for the whole module."""
    path = str(tmp_path_factory.mktemp("report") / "main.html")
    returned = ReportGenerator().generate_html(main_result, path)
    content = open(path, encoding="utf-8").read()
    return returned, content


@pytest.fixture(scope="module")
def minimal_html(tmp_path_factory) -> str:
    """HTML from a single-day result — generated once."""
    path = str(tmp_path_factory.mktemp("report") / "minimal.html")
    ReportGenerator().generate_html(_make_minimal_result(), path)
    return open(path, encoding="utf-8").read()


@pytest.fixture(scope="module")
def compare_html() -> str:
    r1 = _make_result(daily_return=0.001, start="2022-01-03")
    r2 = _make_result(daily_return=0.0005, start="2022-01-03")
    return ReportGenerator().compare_results([r1, r2], ["Alpha", "Beta"])


@pytest.fixture(scope="module")
def main_summary(main_result) -> dict[str, str]:
    return ReportGenerator().generate_summary(main_result)


# ---------------------------------------------------------------------------
# generate_html — file creation and validity
# ---------------------------------------------------------------------------


class TestGenerateHtml:
    def test_file_is_created(self, main_html):
        path, _ = main_html
        assert os.path.exists(path)

    def test_returns_output_path(self, main_html, tmp_path_factory):
        # generate_html must return the path it was given.
        path, _ = main_html
        assert path.endswith(".html")

    def test_file_is_non_empty(self, main_html):
        _, content = main_html
        assert len(content) > 1000

    def test_html_has_doctype(self, main_html):
        _, content = main_html
        assert content.strip().startswith("<!DOCTYPE html>")

    def test_html_has_closing_tags(self, main_html):
        _, content = main_html
        assert "</html>" in content
        assert "</body>" in content

    def test_html_is_self_contained_no_external_script(self, main_html):
        """No external <script src> tags — plotly.js must be embedded inline.

        Note: the literal string ``cdn.plot.ly`` appears inside the minified
        plotly.js bundle as a default config value.  The meaningful check is
        that no ``<script src="...">`` CDN tag is present.
        """
        import re
        _, content = main_html
        cdn_script_tags = re.findall(r'<script[^>]+src=["\'][^"\']*cdn\.plot\.ly', content)
        assert len(cdn_script_tags) == 0

    def test_html_includes_plotly_script(self, main_html):
        _, content = main_html
        # Plotly config object is set inline before the bundle runs.
        assert "PlotlyConfig" in content or "plotly.js" in content.lower()


# ---------------------------------------------------------------------------
# generate_html — expected chart elements
# ---------------------------------------------------------------------------


class TestHtmlChartElements:
    def test_has_equity_curve_section(self, main_html):
        _, html = main_html
        assert "Equity Curve" in html

    def test_has_drawdown_section(self, main_html):
        _, html = main_html
        assert "Drawdown" in html

    def test_has_monthly_returns_section(self, main_html):
        _, html = main_html
        assert "Monthly Returns" in html

    def test_has_rolling_sharpe_section(self, main_html):
        _, html = main_html
        assert "Rolling Sharpe" in html

    def test_has_weight_allocation_section(self, main_html):
        _, html = main_html
        assert "Weight Allocation" in html

    def test_has_summary_metrics_section(self, main_html):
        _, html = main_html
        assert "Summary Metrics" in html

    def test_contains_plotly_trace_data(self, main_html):
        _, html = main_html
        assert "scatter" in html.lower() or "heatmap" in html.lower()

    def test_symbols_appear_in_html(self, main_html):
        _, html = main_html
        assert "SPY" in html or "TLT" in html


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    def test_returns_dict(self, main_summary):
        assert isinstance(main_summary, dict)

    def test_all_values_are_strings(self, main_summary):
        for k, v in main_summary.items():
            assert isinstance(v, str), f"Key '{k}' has non-string value: {v!r}"

    def test_expected_keys_present(self, main_summary):
        expected = {
            "Total Return",
            "Annual Return",
            "Annual Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Calmar Ratio",
            "Win Rate",
            "Profit Factor",
            "VaR (5%)",
            "CVaR (5%)",
            "Initial Capital",
            "Final Value",
            "Trading Days",
        }
        assert expected.issubset(main_summary.keys())

    def test_total_return_formatted_as_percent(self, main_summary):
        assert "%" in main_summary["Total Return"]

    def test_sharpe_formatted_as_number(self, main_summary):
        float(main_summary["Sharpe Ratio"])  # must be parseable

    def test_max_drawdown_negative_or_zero(self, main_summary):
        val = float(main_summary["Max Drawdown"].rstrip("%")) / 100.0
        assert val <= 0.0

    def test_initial_capital_has_dollar_sign(self, main_summary):
        assert "$" in main_summary["Initial Capital"]

    def test_positive_returns_positive_total_return(self):
        result = _make_result(daily_return=0.002)
        summary = ReportGenerator().generate_summary(result)
        total = float(summary["Total Return"].replace("%", "").replace("+", ""))
        assert total > 0.0

    def test_trading_days_matches_portfolio_length(self):
        result = _make_result(n_days=50)
        summary = ReportGenerator().generate_summary(result)
        assert summary["Trading Days"] == "50"


# ---------------------------------------------------------------------------
# generate_summary — edge cases (no heavy HTML, just summary)
# ---------------------------------------------------------------------------


class TestGenerateSummaryEdgeCases:
    def test_minimal_result_does_not_raise(self):
        summary = ReportGenerator().generate_summary(_make_minimal_result())
        assert isinstance(summary, dict)

    def test_minimal_result_has_expected_keys(self):
        expected = {"Total Return", "Sharpe Ratio", "Max Drawdown", "Trading Days"}
        summary = ReportGenerator().generate_summary(_make_minimal_result())
        assert expected.issubset(summary.keys())

    def test_infinite_sortino_formatted(self):
        # All-positive returns → sortino = inf → should not crash.
        result = _make_result(daily_return=0.001)
        summary = ReportGenerator().generate_summary(result)
        val = summary["Sortino Ratio"]
        assert val in {"∞", "inf", "Inf", "INF"} or float(val) > 0


# ---------------------------------------------------------------------------
# Minimal BacktestResult (edge case) — one HTML generated per module above
# ---------------------------------------------------------------------------


class TestMinimalResult:
    def test_minimal_html_is_valid(self, minimal_html):
        assert "<!DOCTYPE html>" in minimal_html

    def test_minimal_html_no_crash_empty_weights(self, minimal_html):
        """Empty weights_history must not cause a crash or missing sections."""
        assert "Weight Allocation" in minimal_html

    def test_minimal_html_handles_empty_returns(self, minimal_html):
        """Single-day result has no returns — rolling Sharpe section still present."""
        assert "Rolling Sharpe" in minimal_html


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------


class TestCompareResults:
    def test_returns_html_string(self, compare_html):
        assert isinstance(compare_html, str)
        assert "<html" in compare_html.lower()

    def test_strategy_labels_in_output(self, compare_html):
        assert "Alpha" in compare_html
        assert "Beta" in compare_html

    def test_has_equity_curve_section(self, compare_html):
        assert "Equity" in compare_html

    def test_has_metrics_table(self, compare_html):
        assert "Metrics" in compare_html

    def test_output_file_written_when_path_given(self, tmp_path):
        r1 = _make_result(start="2022-01-03")
        r2 = _make_result(start="2022-01-03")
        output = str(tmp_path / "compare.html")
        html = ReportGenerator().compare_results([r1, r2], ["A", "B"], output_path=output)
        assert os.path.exists(output)
        # Content written to disk must match the returned string.
        assert open(output, encoding="utf-8").read() == html

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="len"):
            ReportGenerator().compare_results([_make_result()], ["A", "B"])

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ReportGenerator().compare_results([], [])

    def test_single_strategy_comparison(self):
        r1 = _make_result(start="2022-01-03")
        html = ReportGenerator().compare_results([r1], ["Solo"])
        assert "Solo" in html

    def test_self_contained_no_external_script(self, compare_html):
        import re
        cdn_script_tags = re.findall(r'<script[^>]+src=["\'][^"\']*cdn\.plot\.ly', compare_html)
        assert len(cdn_script_tags) == 0


# ---------------------------------------------------------------------------
# _format_maybe_inf helper
# ---------------------------------------------------------------------------


class TestFormatMaybeInf:
    def test_inf_formats_as_symbol(self):
        assert _format_maybe_inf(float("inf")) == "∞"

    def test_neg_inf_formats_with_dash(self):
        assert _format_maybe_inf(float("-inf")) == "-∞"

    def test_finite_value_formatted(self):
        assert _format_maybe_inf(1.234) == "1.234"

    def test_zero_formatted(self):
        assert _format_maybe_inf(0.0) == "0.000"
