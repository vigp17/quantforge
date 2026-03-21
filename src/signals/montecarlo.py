"""Monte Carlo simulation signal generator.

Simulates forward price paths using Geometric Brownian Motion (GBM)
calibrated from trailing historical returns.  Produces per-asset risk
and return forecasts that can feed into downstream portfolio optimisation.

GBM model:
    S_{t+dt} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    Z ~ N(0, 1)

Drift (mu) and volatility (sigma) are estimated from the trailing
``calibration_window`` of log returns.
"""

import logging

import numpy as np

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator

logger = logging.getLogger(__name__)

_MIN_OBS = 60


class MonteCarloSignal(SignalGenerator):
    """Monte Carlo GBM simulation implementing SignalGenerator.

    Args:
        n_simulations: Number of forward paths per asset.
        horizon_days: Forecast horizon in trading days.
        calibration_window: Number of trailing days to estimate mu and sigma.
            If ``None``, uses all available data.
        var_percentile: Percentile for Value-at-Risk (default 5%).
        random_state: Seed for reproducibility.  ``None`` for non-deterministic.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        horizon_days: int = 21,
        calibration_window: int | None = None,
        var_percentile: float = 5.0,
        random_state: int | None = 42,
    ) -> None:
        self._n_simulations = n_simulations
        self._horizon_days = horizon_days
        self._calibration_window = calibration_window
        self._var_percentile = var_percentile
        self._random_state = random_state

    # ------------------------------------------------------------------
    # SignalGenerator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "montecarlo"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        """Simulate forward paths and return risk/return signal.

        Assets with fewer than ``_MIN_OBS`` observations are skipped
        with a warning.

        Args:
            data: Mapping of symbol to AssetData.

        Returns:
            Signal where:
              - ``values``: expected simulated return per asset.
              - ``confidence``: ``1 - prob_loss``, clipped to [0, 1].
              - ``metadata``: var_5pct, cvar_5pct, prob_loss per asset.

        Raises:
            ValueError: If no assets have sufficient data.
        """
        rng = np.random.default_rng(self._random_state)

        symbols: list[str] = []
        expected_returns: list[float] = []
        prob_losses: list[float] = []
        var_list: list[float] = []
        cvar_list: list[float] = []

        for symbol, asset in data.items():
            close = asset.ohlcv["close"].values.astype(np.float64)
            if len(close) < _MIN_OBS:
                logger.warning(
                    "Skipping %s: only %d observations (need %d)",
                    symbol,
                    len(close),
                    _MIN_OBS,
                )
                continue

            mu, sigma = self._calibrate(close)
            terminal_returns = self._simulate(mu, sigma, rng)

            expected_ret = float(np.mean(terminal_returns))
            prob_loss = float(np.mean(terminal_returns < 0))
            var_val = float(np.percentile(terminal_returns, self._var_percentile))
            # CVaR = expected return conditional on being below VaR.
            tail = terminal_returns[terminal_returns <= var_val]
            cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val

            symbols.append(symbol)
            expected_returns.append(expected_ret)
            prob_losses.append(prob_loss)
            var_list.append(var_val)
            cvar_list.append(cvar_val)

            logger.info(
                "%s: mu=%.5f sigma=%.4f E[r]=%.4f P(loss)=%.3f VaR5=%.4f",
                symbol,
                mu,
                sigma,
                expected_ret,
                prob_loss,
                var_val,
            )

        if not symbols:
            raise ValueError(
                f"No assets have sufficient data (minimum {_MIN_OBS} observations required)"
            )

        values = np.array(expected_returns, dtype=np.float64)
        confidence = np.clip(1.0 - np.array(prob_losses), 0.0, 1.0)

        return Signal(
            name=self.name,
            values=values,
            confidence=confidence,
            regime=None,
            metadata={
                "symbols": symbols,
                "var_5pct": dict(zip(symbols, var_list)),
                "cvar_5pct": dict(zip(symbols, cvar_list)),
                "prob_loss": dict(zip(symbols, prob_losses)),
                "expected_return": dict(zip(symbols, expected_returns)),
                "n_simulations": self._n_simulations,
                "horizon_days": self._horizon_days,
            },
        )

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        """Re-run the simulation with new data.

        Monte Carlo simulations are stateless, so ``update`` simply
        delegates to ``generate``.

        Args:
            new_data: Mapping of symbol to AssetData.

        Returns:
            Updated Signal.
        """
        return self.generate(new_data)

    # ------------------------------------------------------------------
    # Internal: calibration
    # ------------------------------------------------------------------

    def _calibrate(self, close: np.ndarray) -> tuple[float, float]:
        """Estimate annualised drift and volatility from close prices.

        Args:
            close: 1-D array of close prices.

        Returns:
            Tuple of (mu, sigma) — annualised.
        """
        log_returns = np.diff(np.log(close))

        if self._calibration_window is not None:
            log_returns = log_returns[-self._calibration_window :]

        mu_daily = float(np.mean(log_returns))
        sigma_daily = float(np.std(log_returns, ddof=1))

        mu_annual = mu_daily * 252
        sigma_annual = sigma_daily * np.sqrt(252)

        return mu_annual, sigma_annual

    # ------------------------------------------------------------------
    # Internal: GBM simulation (vectorised)
    # ------------------------------------------------------------------

    def _simulate(
        self,
        mu: float,
        sigma: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate terminal returns via GBM.

        All paths are simulated in a single vectorised operation.

        Args:
            mu: Annualised drift.
            sigma: Annualised volatility.
            rng: NumPy random Generator instance.

        Returns:
            1-D array of shape ``(n_simulations,)`` containing the
            simulated total return over the horizon (e.g. 0.05 = +5%).
        """
        dt = 1.0 / 252  # daily time step
        n_steps = self._horizon_days

        # Z ~ N(0,1) with shape (n_simulations, n_steps)
        z = rng.standard_normal((self._n_simulations, n_steps))

        # log-return increments: (mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z
        drift_per_step = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z

        # Cumulative log-return over the horizon.
        cum_log_return = np.sum(drift_per_step + diffusion, axis=1)

        # Convert to simple return: S_T/S_0 - 1.
        terminal_return = np.exp(cum_log_return) - 1.0

        return terminal_return
