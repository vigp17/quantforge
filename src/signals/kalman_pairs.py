"""Kalman-filter pairs trading signal generator.

Ported from the kalman_pairs_project's KalmanFilterReg.  Uses an online
Kalman filter to estimate the time-varying hedge ratio between two assets
and produces z-score signals for mean-reversion trading.

The state-space model:
    State:       beta_t  (hedge ratio, random walk)
    Transition:  beta_t = beta_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation: y_t = beta_t * x_t + v_t,    v_t ~ N(0, R)

We intentionally drop the intercept (alpha) from the original UARC
implementation — for price-ratio pairs the intercept is near-zero and
removing it makes the z-score interpretation cleaner.
"""

import logging

import numpy as np
import pandas as pd

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator

logger = logging.getLogger(__name__)

_MIN_OBS = 30
_ZSCORE_WINDOW = 20


class KalmanPairsSignal(SignalGenerator):
    """Kalman-filter pairs trading signal implementing SignalGenerator.

    Estimates a dynamic hedge ratio between two assets using an online
    Kalman filter, then computes a z-score of the resulting spread as
    the trading signal.

    Convention: the first symbol in the data dict is Y (target), the
    second is X (reference).  Spread = Y - beta * X.

    Args:
        delta: Process noise scaling.  Controls how quickly the hedge
            ratio can change.  Small = stiff, large = flexible.
        obs_noise: Observation noise variance (R).
        zscore_window: Rolling window for z-score normalisation.
        entry_threshold: Z-score threshold for entry signals.
        exit_threshold: Z-score threshold for exit signals.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        obs_noise: float = 1e-3,
        zscore_window: int = _ZSCORE_WINDOW,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
    ) -> None:
        self._delta = delta
        self._obs_noise = obs_noise
        self._zscore_window = zscore_window
        self._entry_threshold = entry_threshold
        self._exit_threshold = exit_threshold

        # Kalman state (scalar beta, scalar variance).
        self._beta: float = 0.0
        self._p: float = 1.0  # state covariance
        self._q: float = delta / (1 - delta)  # process noise variance

        # Rolling spread history for z-score computation.
        self._spread_history: list[float] = []
        self._symbols: list[str] | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # SignalGenerator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "kalman_pairs"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        """Run the Kalman filter over the full history and return signals.

        Args:
            data: Mapping of exactly 2 symbols to AssetData.

        Returns:
            Signal where ``values`` contains z-scores per asset
            (negated for X so both legs agree on direction), and
            ``confidence`` is derived from the normalised prediction
            error.

        Raises:
            ValueError: If data does not contain exactly 2 assets or
                has fewer than ``_MIN_OBS`` overlapping observations.
        """
        symbols, y_prices, x_prices = self._validate_and_extract(data)

        n_obs = len(y_prices)
        if n_obs < _MIN_OBS:
            raise ValueError(
                f"Insufficient data: {n_obs} overlapping observations (minimum {_MIN_OBS} required)"
            )

        # Reset state for a full batch run.
        self._beta = 0.0
        self._p = 1.0
        self._spread_history = []
        self._symbols = symbols

        betas = np.empty(n_obs)
        spreads = np.empty(n_obs)
        pred_errors = np.empty(n_obs)

        for t in range(n_obs):
            beta, spread, error = self._step(y_prices[t], x_prices[t])
            betas[t] = beta
            spreads[t] = spread
            pred_errors[t] = error

        # Z-score of the spread.
        spread_series = pd.Series(spreads)
        roll_mean = spread_series.rolling(self._zscore_window, min_periods=1).mean()
        roll_std = spread_series.rolling(self._zscore_window, min_periods=1).std().replace(0, 1)
        zscores = ((spread_series - roll_mean) / roll_std).values

        # Confidence: 1 - normalised absolute prediction error.
        abs_errors = np.abs(pred_errors)
        max_err = abs_errors.max()
        if max_err > 0:
            confidence = 1.0 - abs_errors / max_err
        else:
            confidence = np.ones(n_obs)
        confidence = np.clip(confidence, 0.0, 1.0)

        self._fitted = True

        logger.info(
            "Kalman pairs fitted: %s/%s, %d obs, final beta=%.4f, final z=%.4f",
            symbols[0],
            symbols[1],
            n_obs,
            betas[-1],
            zscores[-1],
        )

        return self._build_signal(zscores[-1], confidence[-1], betas[-1], symbols)

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        """Single-step Kalman update for live trading.

        Falls back to ``generate()`` if the filter has not been fitted.

        Args:
            new_data: Mapping of exactly 2 symbols to AssetData.

        Returns:
            Updated Signal.
        """
        if not self._fitted:
            return self.generate(new_data)

        symbols, y_prices, x_prices = self._validate_and_extract(new_data)
        if len(y_prices) == 0:
            raise ValueError("No valid observations in new_data")

        # Use the last observation for the online step.
        beta, spread, error = self._step(y_prices[-1], x_prices[-1])

        # Z-score from the rolling spread history.
        if len(self._spread_history) >= self._zscore_window:
            window = self._spread_history[-self._zscore_window :]
        else:
            window = self._spread_history
        mean = np.mean(window)
        std = np.std(window, ddof=1) if len(window) > 1 else 1.0
        if std == 0:
            std = 1.0
        zscore = (spread - mean) / std

        confidence = max(0.0, 1.0 - abs(error) / (abs(spread) + 1e-10))
        confidence = min(confidence, 1.0)

        return self._build_signal(zscore, confidence, beta, symbols)

    # ------------------------------------------------------------------
    # Internal: Kalman filter step
    # ------------------------------------------------------------------

    def _step(self, y_t: float, x_t: float) -> tuple[float, float, float]:
        """Single Kalman filter predict-update cycle.

        Args:
            y_t: Target asset price at time t.
            x_t: Reference asset price at time t.

        Returns:
            Tuple of (updated_beta, spread, prediction_error).
        """
        # --- Predict ---
        beta_pred = self._beta
        p_pred = self._p + self._q

        # --- Update ---
        # Predicted observation: y_hat = beta * x
        y_hat = beta_pred * x_t
        error = y_t - y_hat

        # Innovation variance: S = H * P * H' + R  (H = x_t scalar)
        s = x_t * p_pred * x_t + self._obs_noise

        # Kalman gain: K = P * H' / S
        k = p_pred * x_t / s

        # State update.
        self._beta = beta_pred + k * error

        # Covariance update: P = (1 - K*H) * P_pred
        self._p = (1.0 - k * x_t) * p_pred

        spread = y_t - self._beta * x_t
        self._spread_history.append(spread)

        return self._beta, spread, float(error)

    # ------------------------------------------------------------------
    # Internal: data extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_extract(
        data: dict[str, AssetData],
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Extract aligned price arrays from exactly 2 assets.

        Args:
            data: Mapping of symbol to AssetData.

        Returns:
            Tuple of (symbols, y_prices, x_prices) with aligned
            DatetimeIndex.

        Raises:
            ValueError: If data does not contain exactly 2 assets.
        """
        if len(data) != 2:
            raise ValueError(f"KalmanPairsSignal requires exactly 2 assets, got {len(data)}")

        symbols = list(data.keys())
        y_close = data[symbols[0]].ohlcv["close"]
        x_close = data[symbols[1]].ohlcv["close"]

        # Align on common dates.
        aligned = pd.concat([y_close, x_close], axis=1, join="inner").dropna()
        y_prices = aligned.iloc[:, 0].values.astype(np.float64)
        x_prices = aligned.iloc[:, 1].values.astype(np.float64)

        return symbols, y_prices, x_prices

    # ------------------------------------------------------------------
    # Internal: signal construction
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        zscore: float,
        confidence: float,
        beta: float,
        symbols: list[str],
    ) -> Signal:
        """Construct a Signal from the latest z-score.

        Args:
            zscore: Spread z-score at the current time-step.
            confidence: Confidence value in [0, 1].
            beta: Current hedge ratio estimate.
            symbols: [Y_symbol, X_symbol].

        Returns:
            Signal with per-asset values and metadata.
        """
        # Signal values: +z for Y leg, -z for X leg (opposite sides).
        values = np.array([zscore, -zscore], dtype=np.float64)
        confidences = np.array([confidence, confidence], dtype=np.float64)
        confidences = np.clip(confidences, 0.0, 1.0)

        # Determine trade signal from z-score thresholds.
        if zscore > self._entry_threshold:
            trade_signal = "short_spread"
        elif zscore < -self._entry_threshold:
            trade_signal = "long_spread"
        elif abs(zscore) < self._exit_threshold + 0.1:
            trade_signal = "exit"
        else:
            trade_signal = "hold"

        return Signal(
            name=self.name,
            values=values,
            confidence=confidences,
            metadata={
                "beta": beta,
                "zscore": zscore,
                "trade_signal": trade_signal,
                "symbols": symbols,
                "spread_history_len": len(self._spread_history),
                "entry_threshold": self._entry_threshold,
                "exit_threshold": self._exit_threshold,
            },
        )
