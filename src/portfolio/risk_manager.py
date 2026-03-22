"""Risk manager with veto power over portfolio agent actions.

Enforces position-size, drawdown, daily-loss, correlation, and leverage
limits.  Any hard-limit breach causes ``should_flatten`` to return True,
which the execution layer must honour regardless of agent intent.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.base import PortfolioAction

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: dict[str, float] = {
    "max_position_pct": 0.30,
    "max_drawdown_pct": 0.15,
    "daily_loss_limit_pct": 0.03,
    "correlation_limit": 0.70,
    "max_leverage": 1.0,
}


@dataclass
class RiskConfig:
    """Validated risk threshold configuration."""

    max_position_pct: float = 0.30
    max_drawdown_pct: float = 0.15
    daily_loss_limit_pct: float = 0.03
    correlation_limit: float = 0.70
    max_leverage: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict[str, float]) -> RiskConfig:
        merged = {**_DEFAULT_CONFIG, **cfg}
        return cls(
            max_position_pct=merged["max_position_pct"],
            max_drawdown_pct=merged["max_drawdown_pct"],
            daily_loss_limit_pct=merged["daily_loss_limit_pct"],
            correlation_limit=merged["correlation_limit"],
            max_leverage=merged["max_leverage"],
        )


class RiskManager:
    """Enforces risk limits and holds veto power over the portfolio agent.

    Args:
        risk_config: Dict of threshold overrides.  Missing keys fall back
            to defaults (max_position_pct=0.30, max_drawdown_pct=0.15,
            daily_loss_limit_pct=0.03, correlation_limit=0.70,
            max_leverage=1.0).
    """

    def __init__(self, risk_config: dict[str, float] | None = None) -> None:
        self.config = RiskConfig.from_dict(risk_config or {})
        self._peak_value: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_action(
        self,
        action: PortfolioAction,
        current_state: dict[str, Any],
    ) -> tuple[PortfolioAction, list[str]]:
        """Clip the proposed action so it respects position and leverage limits.

        Args:
            action: Proposed allocation from the portfolio agent.
            current_state: Snapshot of current portfolio state (unused
                internally, available for subclass extension).

        Returns:
            (adjusted_action, warnings) where warnings is a non-empty list
            if any limit was applied.
        """
        warnings: list[str] = []
        weights = dict(action.weights)

        # 1. Clip individual positions.
        for symbol, w in weights.items():
            if abs(w) > self.config.max_position_pct:
                clipped = float(np.sign(w)) * self.config.max_position_pct
                msg = (
                    f"Position {symbol} weight {w:.4f} exceeds "
                    f"max_position_pct {self.config.max_position_pct:.4f}; "
                    f"clipped to {clipped:.4f}"
                )
                logger.warning(msg)
                warnings.append(msg)
                weights[symbol] = clipped

        # 2. Scale down to respect max_leverage.
        total_abs = sum(abs(w) for w in weights.values())
        if total_abs > self.config.max_leverage and total_abs > 0:
            scale = self.config.max_leverage / total_abs
            weights = {s: w * scale for s, w in weights.items()}
            msg = (
                f"Total leverage {total_abs:.4f} exceeds "
                f"max_leverage {self.config.max_leverage:.4f}; "
                f"scaled by {scale:.4f}"
            )
            logger.warning(msg)
            warnings.append(msg)

        adjusted = PortfolioAction(
            weights=weights,
            confidence=action.confidence,
            regime_context=action.regime_context,
            risk_metrics=deepcopy(action.risk_metrics),
        )
        return adjusted, warnings

    def reset_peak(self, current_value: float) -> None:
        """Reset the running peak to *current_value* after a flatten event.

        Call this immediately after the engine flattens all positions so that
        the drawdown check starts fresh from the post-flatten NAV rather than
        remaining permanently triggered by the pre-flatten peak.

        Args:
            current_value: Current portfolio NAV to use as the new peak.
        """
        self._peak_value = current_value
        logger.info("Peak value reset to %.2f after flatten event", current_value)

    def check_drawdown(self, portfolio_values: list[float]) -> tuple[bool, float]:
        """Compute current drawdown from the running peak.

        Args:
            portfolio_values: Time-ordered portfolio values, most recent last.

        Returns:
            (is_breached, current_drawdown_pct) where drawdown is expressed
            as a positive fraction (0.10 = 10 % drop).
        """
        if not portfolio_values:
            return False, 0.0

        current = portfolio_values[-1]

        # Update running peak.
        candidate_peak = max(portfolio_values)
        if self._peak_value is None or candidate_peak > self._peak_value:
            self._peak_value = candidate_peak

        if self._peak_value <= 0:
            return False, 0.0

        drawdown = (self._peak_value - current) / self._peak_value
        breached = drawdown > self.config.max_drawdown_pct
        if breached:
            logger.warning(
                "Drawdown %.2f%% exceeds max_drawdown_pct %.2f%%",
                drawdown * 100,
                self.config.max_drawdown_pct * 100,
            )
        return breached, float(drawdown)

    def check_daily_loss(self, today_value: float, yesterday_value: float) -> tuple[bool, float]:
        """Check whether today's loss exceeds the daily loss limit.

        Args:
            today_value: Portfolio NAV at end of current day.
            yesterday_value: Portfolio NAV at end of previous day.

        Returns:
            (is_breached, daily_loss_pct) where loss is a positive fraction
            for a down day.
        """
        if yesterday_value <= 0:
            return False, 0.0

        daily_loss = (yesterday_value - today_value) / yesterday_value
        breached = daily_loss > self.config.daily_loss_limit_pct
        if breached:
            logger.warning(
                "Daily loss %.2f%% exceeds daily_loss_limit_pct %.2f%%",
                daily_loss * 100,
                self.config.daily_loss_limit_pct * 100,
            )
        return breached, float(daily_loss)

    def check_correlation(
        self,
        weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> tuple[bool, dict[str, float]]:
        """Detect highly correlated holdings that inflate concentration risk.

        Args:
            weights: Current or proposed symbol-to-weight mapping.
            returns_df: DataFrame of return series, one column per symbol.

        Returns:
            (is_breached, violations) where violations maps
            "SYMBOL_A/SYMBOL_B" to their correlation coefficient for every
            pair exceeding the limit.
        """
        held = [s for s, w in weights.items() if w != 0 and s in returns_df.columns]
        if len(held) < 2:
            return False, {}

        corr_matrix = returns_df[held].corr()
        violations: dict[str, float] = {}

        for i, sym_a in enumerate(held):
            for sym_b in held[i + 1 :]:
                corr_val = corr_matrix.loc[sym_a, sym_b]
                if np.isnan(corr_val):
                    continue
                if abs(corr_val) > self.config.correlation_limit:
                    key = f"{sym_a}/{sym_b}"
                    violations[key] = float(corr_val)
                    logger.warning(
                        "Correlation between %s and %s is %.4f, exceeds limit %.4f",
                        sym_a,
                        sym_b,
                        corr_val,
                        self.config.correlation_limit,
                    )

        return bool(violations), violations

    def should_flatten(
        self,
        portfolio_values: list[float],
        today_value: float,
        yesterday_value: float,
    ) -> tuple[bool, list[str]]:
        """Master hard-limit check.  Returns True if ANY limit is breached.

        Args:
            portfolio_values: Time-ordered portfolio values, most recent last.
            today_value: Portfolio NAV at end of current day.
            yesterday_value: Portfolio NAV at end of previous day.

        Returns:
            (should_flatten, reasons) — reasons is non-empty when True.
        """
        reasons: list[str] = []

        dd_breached, dd_pct = self.check_drawdown(portfolio_values)
        if dd_breached:
            reasons.append(
                f"Drawdown {dd_pct * 100:.2f}% exceeds "
                f"limit {self.config.max_drawdown_pct * 100:.2f}%"
            )

        dl_breached, dl_pct = self.check_daily_loss(today_value, yesterday_value)
        if dl_breached:
            reasons.append(
                f"Daily loss {dl_pct * 100:.2f}% exceeds "
                f"limit {self.config.daily_loss_limit_pct * 100:.2f}%"
            )

        if reasons:
            logger.critical(
                "Risk veto triggered — flattening all positions. Reasons: %s",
                "; ".join(reasons),
            )

        return bool(reasons), reasons

    def generate_risk_report(
        self,
        portfolio_values: list[float],
        weights: dict[str, float],
        returns_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Produce a snapshot risk report for logging or dashboards.

        Args:
            portfolio_values: Time-ordered portfolio values.
            weights: Current symbol-to-weight mapping.
            returns_df: Return series DataFrame.

        Returns:
            Dict with keys: current_value, peak_value, drawdown_pct,
            drawdown_breached, daily_loss_pct, daily_loss_breached,
            correlation_breached, correlation_violations,
            total_leverage, leverage_breached, num_positions,
            position_limit_violations.
        """
        current_value = portfolio_values[-1] if portfolio_values else 0.0
        yesterday_value = portfolio_values[-2] if len(portfolio_values) >= 2 else current_value

        dd_breached, dd_pct = self.check_drawdown(portfolio_values)
        dl_breached, dl_pct = self.check_daily_loss(current_value, yesterday_value)
        corr_breached, corr_violations = self.check_correlation(weights, returns_df)

        total_leverage = sum(abs(w) for w in weights.values())
        leverage_breached = total_leverage > self.config.max_leverage

        position_limit_violations = [
            s for s, w in weights.items() if abs(w) > self.config.max_position_pct
        ]

        return {
            "current_value": current_value,
            "peak_value": self._peak_value,
            "drawdown_pct": dd_pct,
            "drawdown_breached": dd_breached,
            "daily_loss_pct": dl_pct,
            "daily_loss_breached": dl_breached,
            "correlation_breached": corr_breached,
            "correlation_violations": corr_violations,
            "total_leverage": total_leverage,
            "leverage_breached": leverage_breached,
            "num_positions": sum(1 for w in weights.values() if w != 0),
            "position_limit_violations": position_limit_violations,
        }
