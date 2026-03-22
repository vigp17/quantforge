"""Backtest performance metrics.

All functions accept a ``pd.Series`` of daily returns or portfolio values
(as indicated per function).  Returns are assumed to be arithmetic
(i.e. ``r_t = (V_t − V_{t-1}) / V_{t-1}``).

Annualisation uses 252 trading days per year throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252

_ALL_KEYS = (
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "annual_return",
    "annual_volatility",
    "var_historical",
    "cvar_historical",
)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """Sharpe ratio of a daily return series.

    Args:
        returns: Daily arithmetic returns.
        risk_free_rate: Daily risk-free rate.  Defaults to 0.
        annualize: Scale to annual units when ``True``.

    Returns:
        Sharpe ratio, or 0.0 when the series is empty or has zero std.
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate
    std = float(excess.std(ddof=1))
    if std == 0.0:
        return 0.0
    sr = float(excess.mean()) / std
    if annualize:
        sr *= np.sqrt(_TRADING_DAYS)
    return float(sr)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """Sortino ratio (penalises only downside volatility).

    Args:
        returns: Daily arithmetic returns.
        risk_free_rate: Daily risk-free rate.  Defaults to 0.
        annualize: Scale to annual units when ``True``.

    Returns:
        Sortino ratio, or 0.0 when the series is empty or has no downside.
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate
    downside = excess[excess < 0]
    if downside.empty:
        # No downside at all: Sortino is infinite for positive mean, zero for zero mean.
        return float("inf") if float(excess.mean()) > 0.0 else 0.0
    downside_std = float(np.sqrt((downside**2).mean()))
    if downside_std == 0.0:
        return 0.0
    ratio = float(excess.mean()) / downside_std
    if annualize:
        ratio *= np.sqrt(_TRADING_DAYS)
    return float(ratio)


def max_drawdown(values: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of a portfolio value series.

    Args:
        values: Portfolio values (prices, not returns).

    Returns:
        Maximum drawdown as a non-negative fraction in [0, 1].
        Returns 0.0 for an empty or single-element series.
    """
    if values.empty or len(values) < 2:
        return 0.0
    rolling_peak = values.cummax()
    drawdowns = (values - rolling_peak) / rolling_peak
    return float(-drawdowns.min())


def calmar_ratio(returns: pd.Series, values: pd.Series) -> float:
    """Calmar ratio: annualised return divided by maximum drawdown.

    Args:
        returns: Daily arithmetic returns.
        values: Portfolio values aligned with *returns*.

    Returns:
        Calmar ratio, or 0.0 when max drawdown is zero or series is empty.
    """
    mdd = max_drawdown(values)
    if mdd == 0.0:
        return 0.0
    ann_ret = annual_return(returns)
    return float(ann_ret / mdd)


def win_rate(returns: pd.Series) -> float:
    """Fraction of returns that are strictly positive.

    Args:
        returns: Daily arithmetic returns.

    Returns:
        Win rate in [0, 1], or 0.0 for an empty series.
    """
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def profit_factor(returns: pd.Series) -> float:
    """Gross profit divided by gross loss.

    Args:
        returns: Daily arithmetic returns.

    Returns:
        Profit factor (≥ 0).  Returns ``inf`` when there are no losing days.
        Returns 0.0 for an empty series or when gross profit is zero with losses.
    """
    if returns.empty:
        return 0.0
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0.0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def annual_return(returns: pd.Series) -> float:
    """Compound annual growth rate from daily returns.

    Args:
        returns: Daily arithmetic returns.

    Returns:
        CAGR as a fraction (e.g. 0.12 = 12 % per year).
        Returns 0.0 for an empty or single-element series.
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    cumulative = float((1.0 + returns).prod())
    n_years = len(returns) / _TRADING_DAYS
    if cumulative <= 0:
        return -1.0
    return float(cumulative ** (1.0 / n_years) - 1.0)


def annual_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns.

    Args:
        returns: Daily arithmetic returns.

    Returns:
        Annualised volatility.  Returns 0.0 for fewer than 2 observations.
    """
    if returns.empty or len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(_TRADING_DAYS))


def var_historical(returns: pd.Series, percentile: float = 5.0) -> float:
    """Historical Value at Risk at the given left-tail percentile.

    Args:
        returns: Daily arithmetic returns.
        percentile: Left-tail percentile in (0, 100).  Defaults to 5.

    Returns:
        VaR as a non-negative loss fraction (e.g. 0.02 = 2 % potential loss).
        Returns 0.0 for an empty series.
    """
    if returns.empty:
        return 0.0
    return float(-np.percentile(returns, percentile))


def cvar_historical(returns: pd.Series, percentile: float = 5.0) -> float:
    """Conditional Value at Risk (Expected Shortfall) at the given percentile.

    Average loss in the worst ``percentile`` % of days.

    Args:
        returns: Daily arithmetic returns.
        percentile: Left-tail percentile in (0, 100).  Defaults to 5.

    Returns:
        CVaR as a non-negative loss fraction.  Returns 0.0 for an empty series.
    """
    if returns.empty:
        return 0.0
    cutoff = np.percentile(returns, percentile)
    tail = returns[returns <= cutoff]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def compute_all(
    returns: pd.Series,
    values: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute all performance metrics in one call.

    Args:
        returns: Daily arithmetic returns.
        values: Portfolio values aligned with *returns*.
        risk_free_rate: Daily risk-free rate.  Defaults to 0.

    Returns:
        Dictionary mapping metric name → float value.  Keys are always
        present regardless of whether the series is empty.
    """
    return {
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate=risk_free_rate),
        "max_drawdown": max_drawdown(values),
        "calmar_ratio": calmar_ratio(returns, values),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "annual_return": annual_return(returns),
        "annual_volatility": annual_volatility(returns),
        "var_historical": var_historical(returns),
        "cvar_historical": cvar_historical(returns),
    }
