"""Backtest report generation with Plotly charts.

Produces self-contained HTML reports (no external network dependencies) with:

* Equity curve
* Drawdown (underwater) chart
* Monthly returns heatmap
* Rolling Sharpe ratio (60-day)
* Weight allocation over time (stacked area)
* Summary metrics table

Public API
----------
ReportGenerator.generate_html   → single-strategy HTML report
ReportGenerator.generate_summary → flat dict of formatted metric strings
ReportGenerator.compare_results → multi-strategy comparison HTML report
"""

from __future__ import annotations

import logging
from math import sqrt

import numpy as np
import plotly.graph_objects as go

from src.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252
_ROLLING_WINDOW = 60

# Colour palette used consistently across charts.
_PALETTE = [
    "#2196F3",
    "#4CAF50",
    "#FF5722",
    "#9C27B0",
    "#FF9800",
    "#00BCD4",
    "#E91E63",
    "#607D8B",
]

_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_HTML_WRAPPER = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body  {{ font-family: Arial, Helvetica, sans-serif; margin: 24px;
             background: #f4f6f9; color: #333; }}
    h1   {{ color: #1a237e; margin-bottom: 4px; }}
    p.sub {{ color: #666; margin-top: 0; margin-bottom: 16px; font-size: 0.9em; }}
    .card {{ background: #fff; border-radius: 6px; padding: 12px 16px;
             margin-bottom: 16px;
             box-shadow: 0 1px 4px rgba(0,0,0,0.10); }}
    h2   {{ color: #283593; margin-top: 0; font-size: 1.1em; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="sub">{subtitle}</p>
  {body}
</body>
</html>
"""


def _fig_div(fig: go.Figure, *, include_plotlyjs: bool | str) -> str:
    """Render a Plotly figure as an HTML ``<div>`` fragment."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displayModeBar": True, "responsive": True},
    )


def _card(title: str, content: str) -> str:
    """Wrap chart HTML in a styled card ``<div>``."""
    return f'<div class="card"><h2>{title}</h2>{content}</div>\n'


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generates HTML performance reports from :class:`~src.backtest.engine.BacktestResult` objects.

    All methods are stateless; no constructor arguments are required.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_html(self, result: BacktestResult, output_path: str) -> str:
        """Build a self-contained HTML report for a single backtest result.

        Args:
            result: Completed backtest result.
            output_path: Filesystem path where the ``.html`` file will be written.
                Parent directory must exist.

        Returns:
            The *output_path* string (for chaining convenience).
        """
        cards: list[str] = []
        first = True  # only the first chart embeds the plotly.js bundle

        def _add(title: str, fig: go.Figure) -> None:
            nonlocal first
            inc = True if first else False
            cards.append(_card(title, _fig_div(fig, include_plotlyjs=inc)))
            first = False

        _add("Equity Curve", _equity_curve(result))
        _add("Drawdown", _drawdown_chart(result))
        _add("Monthly Returns", _monthly_heatmap(result))
        _add("Rolling Sharpe Ratio (60-day)", _rolling_sharpe(result))
        _add("Weight Allocation", _weights_area(result))
        _add("Summary Metrics", _metrics_table(result))

        subtitle = (
            f"Period: {result.config.get('start', '—')} → {result.config.get('end', '—')} · "
            f"Initial capital: ${result.config.get('initial_capital', 0):,.0f} · "
            f"Agent: {result.config.get('agent', '—')} · "
            f"Rebalance: {result.config.get('rebalance_frequency', '—')}"
        )

        html = _HTML_WRAPPER.format(
            title="Backtest Report",
            subtitle=subtitle,
            body="\n".join(cards),
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("Report written to %s (%d bytes)", output_path, len(html))
        return output_path

    def generate_summary(self, result: BacktestResult) -> dict[str, str]:
        """Return key performance metrics as human-readable formatted strings.

        Args:
            result: Completed backtest result.

        Returns:
            Ordered ``dict`` of ``{label: formatted_value}`` suitable for
            tabular display.  All values are strings.
        """
        m = result.metrics
        pv = result.portfolio_values

        initial = float(pv.iloc[0]) if not pv.empty else 0.0
        final = float(pv.iloc[-1]) if not pv.empty else 0.0
        total_return = (final / initial - 1.0) if initial > 0 else 0.0

        return {
            "Total Return": f"{total_return:+.2%}",
            "Annual Return": f"{m.get('annual_return', 0.0):+.2%}",
            "Annual Volatility": f"{m.get('annual_volatility', 0.0):.2%}",
            "Sharpe Ratio": f"{m.get('sharpe_ratio', 0.0):.3f}",
            "Sortino Ratio": _format_maybe_inf(m.get("sortino_ratio", 0.0)),
            "Max Drawdown": f"{-abs(m.get('max_drawdown', 0.0)):.2%}",
            "Calmar Ratio": f"{m.get('calmar_ratio', 0.0):.3f}",
            "Win Rate": f"{m.get('win_rate', 0.0):.1%}",
            "Profit Factor": _format_maybe_inf(m.get("profit_factor", 0.0)),
            "VaR (5%)": f"{-abs(m.get('var_historical', 0.0)):.2%}",
            "CVaR (5%)": f"{-abs(m.get('cvar_historical', 0.0)):.2%}",
            "Initial Capital": f"${initial:,.0f}",
            "Final Value": f"${final:,.0f}",
            "Trading Days": str(len(pv)),
        }

    def compare_results(
        self,
        results: list[BacktestResult],
        labels: list[str],
        output_path: str | None = None,
    ) -> str:
        """Generate a side-by-side HTML comparison of multiple strategies.

        Args:
            results: List of backtest results to compare.
            labels: Human-readable strategy names, one per result.  Must be
                the same length as *results*.
            output_path: Optional file path.  When provided the HTML is also
                written to disk.

        Returns:
            The generated HTML string.  When *output_path* is given, the
            content is additionally written there and the path is logged.

        Raises:
            ValueError: If *results* and *labels* have different lengths, or
                *results* is empty.
        """
        if not results:
            raise ValueError("results must be non-empty")
        if len(results) != len(labels):
            raise ValueError(f"len(results)={len(results)} must equal len(labels)={len(labels)}")

        cards: list[str] = []
        first = True

        def _add(title: str, fig: go.Figure) -> None:
            nonlocal first
            inc = True if first else False
            cards.append(_card(title, _fig_div(fig, include_plotlyjs=inc)))
            first = False

        _add("Equity Curves (normalised to 100)", _compare_equity(results, labels))
        _add("Drawdown Comparison", _compare_drawdown(results, labels))
        _add("Metrics Comparison", _compare_metrics_table(results, labels))

        subtitle = f"Comparing {len(results)} strategies: {', '.join(labels)}"
        html = _HTML_WRAPPER.format(
            title="Strategy Comparison Report",
            subtitle=subtitle,
            body="\n".join(cards),
        )

        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(html)
            logger.info("Comparison report written to %s", output_path)

        return html


# ---------------------------------------------------------------------------
# Single-strategy chart builders
# ---------------------------------------------------------------------------


def _equity_curve(result: BacktestResult) -> go.Figure:
    pv = result.portfolio_values
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pv.index,
            y=pv.values,
            mode="lines",
            name="Portfolio Value",
            line={"color": _PALETTE[0], "width": 2},
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.08)",
        )
    )
    fig.update_layout(
        **_base_layout(),
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        showlegend=False,
    )
    return fig


def _drawdown_chart(result: BacktestResult) -> go.Figure:
    pv = result.portfolio_values
    if pv.empty:
        return go.Figure(layout=_base_layout())

    rolling_max = pv.cummax()
    drawdown = (pv - rolling_max) / rolling_max.replace(0, float("nan"))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=(drawdown * 100).values,
            mode="lines",
            name="Drawdown",
            line={"color": "#EF5350", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.15)",
        )
    )
    fig.update_layout(
        **_base_layout(),
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        yaxis={"ticksuffix": "%"},
        showlegend=False,
    )
    return fig


def _monthly_heatmap(result: BacktestResult) -> go.Figure:
    returns = result.returns
    if returns.empty:
        return go.Figure(layout=_base_layout())

    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    if monthly.empty:
        return go.Figure(layout=_base_layout())

    df = monthly.rename("ret").to_frame()
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Build full grid (NaN for missing months).
    years = sorted(df["year"].unique())
    pivot = df.pivot(index="year", columns="month", values="ret").reindex(
        index=years, columns=range(1, 13)
    )

    z = (pivot.values * 100).tolist()  # percent
    y_labels = [str(y) for y in years]
    x_labels = _MONTH_ABBR

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=[
                [0.0, "#EF5350"],
                [0.5, "#FFFFFF"],
                [1.0, "#66BB6A"],
            ],
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar={"title": "Return %", "ticksuffix": "%"},
            hoverongaps=False,
        )
    )
    fig.update_layout(
        **_base_layout(),
        xaxis_title="Month",
        yaxis_title="Year",
        yaxis={"type": "category"},
    )
    return fig


def _rolling_sharpe(result: BacktestResult) -> go.Figure:
    returns = result.returns
    fig = go.Figure()

    if not returns.empty and len(returns) >= _ROLLING_WINDOW:
        roll_mean = returns.rolling(_ROLLING_WINDOW).mean()
        roll_std = returns.rolling(_ROLLING_WINDOW).std(ddof=1)
        sharpe = (roll_mean / roll_std.replace(0, float("nan"))) * sqrt(_TRADING_DAYS)

        fig.add_trace(
            go.Scatter(
                x=sharpe.index,
                y=sharpe.values,
                mode="lines",
                name=f"Rolling {_ROLLING_WINDOW}-day Sharpe",
                line={"color": _PALETTE[1], "width": 2},
            )
        )
        # Zero reference line.
        fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)

    fig.update_layout(
        **_base_layout(),
        yaxis_title="Sharpe Ratio",
        xaxis_title="Date",
        showlegend=False,
    )
    return fig


def _weights_area(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    wh = result.weights_history
    if not wh:
        fig.update_layout(**_base_layout())
        return fig

    symbols = [k for k in wh[0] if k != "date"]
    dates = [w["date"] for w in wh]

    for i, sym in enumerate(symbols):
        values = [float(w.get(sym, 0.0)) * 100.0 for w in wh]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines",
                name=sym,
                stackgroup="one",
                line={"color": _PALETTE[i % len(_PALETTE)], "width": 0.5},
                fillcolor=_PALETTE[i % len(_PALETTE)],
            )
        )

    fig.update_layout(
        **_base_layout(),
        yaxis_title="Weight (%)",
        xaxis_title="Rebalance Date",
        yaxis={"ticksuffix": "%"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def _metrics_table(result: BacktestResult) -> go.Figure:
    rg = ReportGenerator()
    summary = rg.generate_summary(result)
    headers = list(summary.keys())
    values = list(summary.values())

    fig = go.Figure(
        data=go.Table(
            header=dict(
                values=headers,
                fill_color="#283593",
                font={"color": "white", "size": 12},
                align="center",
                height=28,
            ),
            cells=dict(
                values=[[v] for v in values],
                fill_color=[["#f0f4ff", "#ffffff"] * (len(values) // 2 + 1)][0][: len(values)],
                align="center",
                font={"size": 12},
                height=26,
            ),
        )
    )
    layout = _base_layout()
    layout["height"] = 140
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Multi-strategy chart builders
# ---------------------------------------------------------------------------


def _compare_equity(results: list[BacktestResult], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, (result, label) in enumerate(zip(results, labels)):
        pv = result.portfolio_values
        if pv.empty:
            continue
        normalised = pv / float(pv.iloc[0]) * 100.0
        fig.add_trace(
            go.Scatter(
                x=normalised.index,
                y=normalised.values,
                mode="lines",
                name=label,
                line={"color": _PALETTE[i % len(_PALETTE)], "width": 2},
            )
        )
    fig.add_hline(y=100, line_dash="dot", line_color="grey", line_width=1)
    fig.update_layout(
        **_base_layout(),
        yaxis_title="Indexed Value (start = 100)",
        xaxis_title="Date",
    )
    return fig


def _compare_drawdown(results: list[BacktestResult], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, (result, label) in enumerate(zip(results, labels)):
        pv = result.portfolio_values
        if pv.empty:
            continue
        dd = (pv - pv.cummax()) / pv.cummax().replace(0, float("nan")) * 100.0
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                name=label,
                line={"color": _PALETTE[i % len(_PALETTE)], "width": 1.5},
            )
        )
    fig.update_layout(
        **_base_layout(),
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        yaxis={"ticksuffix": "%"},
    )
    return fig


def _compare_metrics_table(results: list[BacktestResult], labels: list[str]) -> go.Figure:
    rg = ReportGenerator()
    summaries = [rg.generate_summary(r) for r in results]
    metric_names = list(summaries[0].keys()) if summaries else []

    header_vals = ["Metric"] + labels
    rows = [[m] + [s.get(m, "—") for s in summaries] for m in metric_names]

    # Transpose: each column is a list.
    col_values = list(zip(*rows)) if rows else [[""] for _ in header_vals]

    fig = go.Figure(
        data=go.Table(
            header=dict(
                values=header_vals,
                fill_color="#283593",
                font={"color": "white", "size": 12},
                align=["left"] + ["center"] * len(labels),
                height=28,
            ),
            cells=dict(
                values=[list(c) for c in col_values],
                fill_color="white",
                align=["left"] + ["center"] * len(labels),
                font={"size": 11},
                height=24,
            ),
        )
    )
    layout = _base_layout()
    layout["height"] = max(200, len(metric_names) * 26 + 80)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------


def _base_layout() -> dict:
    return {
        "template": "plotly_white",
        "margin": {"l": 50, "r": 20, "t": 20, "b": 40},
        "height": 340,
        "font": {"family": "Arial, sans-serif", "size": 12},
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_maybe_inf(value: float) -> str:
    if value == float("inf"):
        return "∞"
    if value == float("-inf"):
        return "-∞"
    return f"{value:.3f}"
