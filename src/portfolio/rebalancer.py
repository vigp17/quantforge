"""Portfolio rebalancer — translates weight targets into trade orders.

Given a current and target weight allocation, the Rebalancer computes the
minimal set of Orders needed to move the portfolio to the target, subject to:

- A minimum trade size filter (``min_trade_pct``) that suppresses noise trades.
- An optional turnover budget (``apply_turnover_constraint``) that limits the
  total one-way weight change per rebalance.
- A drift threshold (``should_rebalance``) that gates whether a rebalance is
  needed at all.
"""

from __future__ import annotations

import logging

from src.execution.base import Order

logger = logging.getLogger(__name__)


class Rebalancer:
    """Translates portfolio weight targets into executable trade orders.

    All weight arguments are fractions in [0, 1]; portfolio_value is in the
    base currency.  Quantities are computed as::

        quantity = |Δweight| × portfolio_value / price_per_unit

    Because the Rebalancer works in weight-space rather than price-space it
    does not need per-asset prices; ``quantity`` is expressed in *notional
    currency units* (i.e. dollar value of the trade), which is the natural
    unit for fractional-share brokers such as Alpaca.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        min_trade_pct: float = 0.01,
    ) -> list[Order]:
        """Compute the orders needed to move from current to target weights.

        Symbols present only in ``target_weights`` are treated as new
        positions (current weight = 0).  Symbols present only in
        ``current_weights`` with no entry in ``target_weights`` are treated
        as full exits (target weight = 0).

        Args:
            current_weights: Current allocation, symbol → weight fraction.
            target_weights: Desired allocation, symbol → weight fraction.
            portfolio_value: Total portfolio value in base currency.
            min_trade_pct: Minimum absolute weight change to generate an order.
                Differences smaller than this are skipped.  Defaults to 0.01
                (1 % of portfolio value).

        Returns:
            List of market orders sorted sells-first to free capital before
            buys.  Returns an empty list when no trade exceeds the minimum
            threshold or when ``portfolio_value`` is zero.

        Raises:
            ValueError: If ``portfolio_value`` is negative.
        """
        if portfolio_value < 0:
            raise ValueError(f"portfolio_value must be non-negative, got {portfolio_value}")
        if portfolio_value == 0:
            return []

        all_symbols = set(current_weights) | set(target_weights)
        orders: list[Order] = []

        for sym in all_symbols:
            current_w = current_weights.get(sym, 0.0)
            target_w = target_weights.get(sym, 0.0)
            delta = target_w - current_w

            if abs(delta) < min_trade_pct:
                continue

            notional = abs(delta) * portfolio_value
            side: str = "buy" if delta > 0 else "sell"
            orders.append(
                Order(
                    symbol=sym,
                    side=side,  # type: ignore[arg-type]
                    quantity=round(notional, 6),
                    order_type="market",
                )
            )
            logger.debug(
                "Rebalancer: %s %s notional=%.2f (Δw=%.4f)",
                side,
                sym,
                notional,
                delta,
            )

        # Execute sells before buys to free up capital first.
        orders.sort(key=lambda o: 0 if o.side == "sell" else 1)
        return orders

    def should_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        threshold: float = 0.05,
    ) -> bool:
        """Decide whether a rebalance is warranted based on weight drift.

        Returns ``True`` when the maximum absolute weight deviation between
        current and target allocations exceeds *threshold*.

        Args:
            current_weights: Current allocation.
            target_weights: Desired allocation.
            threshold: Maximum tolerated absolute weight deviation before a
                rebalance is triggered.  Defaults to 0.05 (5 pp).

        Returns:
            ``True`` if any symbol's weight deviation exceeds *threshold*.
        """
        all_symbols = set(current_weights) | set(target_weights)
        for sym in all_symbols:
            current_w = current_weights.get(sym, 0.0)
            target_w = target_weights.get(sym, 0.0)
            if abs(target_w - current_w) > threshold:
                return True
        return False

    def apply_turnover_constraint(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        max_turnover: float = 0.30,
    ) -> dict[str, float]:
        """Clip the target weights to respect a one-way turnover budget.

        Turnover is defined as half the sum of absolute weight changes
        (the standard one-way definition)::

            turnover = 0.5 × Σ |w_target_i − w_current_i|

        When the unconstrained target would exceed ``max_turnover``, each
        delta is scaled uniformly so that the total one-way turnover equals
        exactly ``max_turnover``::

            Δ_clipped = Δ × (max_turnover / unconstrained_turnover)
            w_clipped  = w_current + Δ_clipped

        Args:
            current_weights: Current allocation.
            target_weights: Desired allocation.
            max_turnover: Maximum one-way turnover fraction (0, 1].
                Defaults to 0.30.

        Returns:
            Adjusted target weights.  When turnover is within budget the
            original ``target_weights`` values are returned unchanged.

        Raises:
            ValueError: If ``max_turnover`` is not in (0, 1].
        """
        if not 0.0 < max_turnover <= 1.0:
            raise ValueError(f"max_turnover must be in (0, 1], got {max_turnover}")

        all_symbols = set(current_weights) | set(target_weights)
        deltas = {
            sym: target_weights.get(sym, 0.0) - current_weights.get(sym, 0.0) for sym in all_symbols
        }

        unconstrained_turnover = 0.5 * sum(abs(d) for d in deltas.values())

        if unconstrained_turnover <= max_turnover:
            # Already within budget — return original target as-is.
            return {sym: target_weights.get(sym, 0.0) for sym in all_symbols}

        scale = max_turnover / unconstrained_turnover
        adjusted: dict[str, float] = {}
        for sym in all_symbols:
            current_w = current_weights.get(sym, 0.0)
            adjusted[sym] = current_w + deltas[sym] * scale

        logger.debug(
            "Turnover constraint applied: unconstrained=%.4f → clipped to %.4f (scale=%.4f)",
            unconstrained_turnover,
            max_turnover,
            scale,
        )
        return adjusted
