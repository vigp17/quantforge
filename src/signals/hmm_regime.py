"""HMM-based regime detection signal generator.

Ported from the UARC project's BayesianMarketHMM.  Uses a Gaussian HMM
(via hmmlearn) to classify the market into regimes (bull / bear / sideways)
and produces a Signal containing the most-likely regime index per asset
plus the posterior confidence.

Key design decisions carried over from UARC:
- Uses the **filtering posterior** p(z_t | x_{1:t}) via the forward
  algorithm — NOT Viterbi — to avoid lookahead bias.
- Regime labels are assigned automatically by sorting HMM states by
  their mean return (highest = bull, lowest = bear).
- Log-space forward algorithm for numerical stability on long sequences.
- StandardScaler fitted on training data to normalise heterogeneous
  features (returns vs. volatilities).
"""

import logging

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator

logger = logging.getLogger(__name__)

# Minimum number of observations required to fit a meaningful HMM.
_MIN_OBS = 50


def _log_sum_exp(
    x: np.ndarray,
    axis: int = -1,
    keepdims: bool = False,
) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    m = np.max(x, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=keepdims))
    if keepdims:
        return out + m
    return out + np.squeeze(m, axis=axis)


def _extract_features(data: dict[str, AssetData]) -> tuple[np.ndarray, list[str]]:
    """Build the HMM observation matrix from AssetData.

    For each symbol, computes:
      - 1-day log return of the close price
      - 20-day realised volatility (annualised)

    Args:
        data: Mapping of symbol to AssetData.

    Returns:
        Tuple of (feature_matrix, symbol_list) where feature_matrix has
        shape ``(T, 2 * n_assets)`` and symbol_list preserves insertion
        order.
    """
    import pandas as pd

    series: list[pd.Series] = []
    symbols: list[str] = []

    for symbol, asset in data.items():
        close = asset.ohlcv["close"]
        log_ret = np.log(close / close.shift(1))
        rvol = log_ret.rolling(20).std() * np.sqrt(252)
        series.extend([log_ret, rvol])
        symbols.append(symbol)

    features = pd.concat(series, axis=1).dropna()
    return features.values.astype(np.float32), symbols


class HMMRegimeDetector(SignalGenerator):
    """Gaussian HMM regime detector implementing SignalGenerator.

    Fits a Gaussian HMM on log-return and volatility features extracted
    from the provided asset data.  The most-likely regime at the final
    time-step is reported as the signal, with posterior probabilities
    providing the confidence measure.

    Args:
        n_states: Number of hidden regimes (default 3: bull, bear, sideways).
        n_iter: Maximum Baum-Welch EM iterations.
        covariance_type: HMM covariance structure.
        random_state: Seed for reproducibility.
    """

    _REGIME_LABELS = {0: "bear", 1: "sideways", 2: "bull"}

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        covariance_type: str = "diag",
        random_state: int = 42,
    ) -> None:
        self._n_states = n_states
        self._n_iter = n_iter
        self._covariance_type = covariance_type
        self._random_state = random_state

        self._model: GaussianHMM | None = None
        self._scaler = StandardScaler()
        self._regime_order: np.ndarray | None = None  # state indices sorted by mean return
        self._prev_regime: str | None = None
        self._prev_log_alpha: np.ndarray | None = None

    # ------------------------------------------------------------------
    # SignalGenerator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this signal generator."""
        return "hmm_regime"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        """Fit HMM on historical data and return the current regime signal.

        Extracts log-return and volatility features, fits the HMM, runs
        the forward algorithm to get causal posteriors, and returns the
        regime at the last time-step.

        Args:
            data: Mapping of symbol to AssetData.

        Returns:
            Signal where:
              - ``values``: most-likely regime index (0=bear … K-1=bull)
                per asset (broadcast, same regime for all).
              - ``confidence``: posterior probability of that regime per asset.
              - ``regime``: human-readable label of the dominant regime.
              - ``metadata``: full posterior vector and regime labels.

        Raises:
            ValueError: If there are fewer than ``_MIN_OBS`` observations
                after feature computation.
        """
        features, symbols = _extract_features(data)
        n_obs = len(features)

        if n_obs < _MIN_OBS:
            raise ValueError(
                f"Insufficient data: {n_obs} observations after feature "
                f"computation (minimum {_MIN_OBS} required)"
            )

        self._fit(features)

        posteriors = self._forward_posteriors(features)
        latest_posterior = posteriors[-1]  # shape (K,)

        return self._build_signal(latest_posterior, symbols)

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        """Incrementally update the regime signal with new data.

        Uses the online forward algorithm to advance one step without
        reprocessing the full history.  Falls back to ``generate()`` if
        the model has not been fitted yet.

        Args:
            new_data: Mapping of symbol to the latest AssetData slice.

        Returns:
            Updated Signal.
        """
        if self._model is None:
            return self.generate(new_data)

        features, symbols = _extract_features(new_data)
        if len(features) == 0:
            raise ValueError("No valid observations in new_data after feature extraction")

        # Use the last row for the online update.
        x_t = features[-1]
        posterior, self._prev_log_alpha = self._online_step(x_t, self._prev_log_alpha)

        return self._build_signal(posterior, symbols)

    # ------------------------------------------------------------------
    # Internal: fitting
    # ------------------------------------------------------------------

    def _fit(self, features: np.ndarray) -> None:
        """Fit the HMM and label regimes.

        Args:
            features: Shape (T, D) observation matrix.
        """
        logger.info(
            "Fitting HMM: %d states, %d features, %d timesteps",
            self._n_states,
            features.shape[1],
            len(features),
        )

        x_sc = self._scaler.fit_transform(features)

        self._model = GaussianHMM(
            n_components=self._n_states,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            random_state=self._random_state,
            verbose=False,
            tol=1e-4,
        )
        try:
            self._model.fit(x_sc)
        except Exception as exc:
            raise RuntimeError(f"HMM fitting failed: {exc}") from exc

        self._label_regimes()
        logger.info("HMM fitted. Converged: %s", self._model.monitor_.converged)

    def _label_regimes(self) -> None:
        """Sort states by mean return and assign bull/bear/sideways labels.

        Column 0 of the (unscaled) emission means is the first asset's
        log return.  States are ranked by this value.
        """
        means_orig = self._scaler.inverse_transform(self._model.means_)
        # Column 0 = first asset's log return
        mean_returns = means_orig[:, 0]
        self._regime_order = np.argsort(mean_returns)  # [bear_idx, ..., bull_idx]

        labels = {}
        labels[int(self._regime_order[0])] = "bear"
        labels[int(self._regime_order[-1])] = "bull"
        for idx in self._regime_order[1:-1]:
            labels[int(idx)] = "sideways"
        self._regime_labels_map = labels

        logger.info("Regime labels: %s", labels)
        logger.info("Mean returns per state: %s", mean_returns)

    # ------------------------------------------------------------------
    # Internal: forward algorithm (log-space)
    # ------------------------------------------------------------------

    def _log_emission_probs(self, x_sc: np.ndarray) -> np.ndarray:
        """Compute log P(x_t | z_t = k) for all t, k.

        Args:
            x_sc: Shape (T, D) standardised observations.

        Returns:
            Array of shape (T, K).
        """
        n_obs = len(x_sc)
        n_k = self._n_states
        log_probs = np.full((n_obs, n_k), -1e10)

        for k in range(n_k):
            mean = self._model.means_[k]
            if self._covariance_type == "diag":
                cov = np.diag(self._model.covars_[k])
            elif self._covariance_type == "full":
                cov = self._model.covars_[k]
            elif self._covariance_type == "tied":
                cov = self._model.covars_
            else:  # spherical
                cov = np.eye(len(mean)) * self._model.covars_[k]

            try:
                rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
                log_probs[:, k] = rv.logpdf(x_sc)
            except Exception:
                logger.warning("Emission probability error for state %d", k, exc_info=True)

        return log_probs

    def _forward_posteriors(self, features: np.ndarray) -> np.ndarray:
        """Run the full forward algorithm and return causal posteriors.

        Args:
            features: Shape (T, D) raw features (will be scaled internally).

        Returns:
            Array of shape (T, K) — each row sums to 1.
        """
        x_sc = self._scaler.transform(features)
        n_obs = len(x_sc)
        n_k = self._n_states

        log_emission = self._log_emission_probs(x_sc)
        log_trans = np.log(self._model.transmat_ + 1e-300)
        log_pi = np.log(self._model.startprob_ + 1e-300)

        log_alpha = np.zeros((n_obs, n_k))
        log_alpha[0] = log_pi + log_emission[0]

        for t in range(1, n_obs):
            for j in range(n_k):
                log_alpha[t, j] = (
                    _log_sum_exp(log_alpha[t - 1] + log_trans[:, j]) + log_emission[t, j]
                )

        # Normalise each row to get posteriors.
        log_norm = _log_sum_exp(log_alpha, axis=1, keepdims=True)
        posteriors = np.exp(log_alpha - log_norm)
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)

        # Store the last log_alpha for subsequent online updates.
        self._prev_log_alpha = log_alpha[-1].copy()

        return posteriors.astype(np.float32)

    def _online_step(
        self,
        x_t: np.ndarray,
        prev_log_alpha: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-step online forward update.

        Args:
            x_t: Shape (D,) feature vector for current time-step.
            prev_log_alpha: Shape (K,) log forward variable from previous
                step, or None for the first step.

        Returns:
            Tuple of (posterior, log_alpha) — posterior is shape (K,).
        """
        x_scaled = self._scaler.transform(x_t.reshape(1, -1))[0]
        log_emit = self._log_emission_probs(x_scaled.reshape(1, -1))[0]
        log_trans = np.log(self._model.transmat_ + 1e-300)
        log_pi = np.log(self._model.startprob_ + 1e-300)

        if prev_log_alpha is None:
            log_alpha = log_pi + log_emit
        else:
            n_k = self._n_states
            log_alpha = np.array(
                [_log_sum_exp(prev_log_alpha + log_trans[:, j]) + log_emit[j] for j in range(n_k)]
            )

        log_posterior = log_alpha - _log_sum_exp(log_alpha)
        posterior = np.exp(log_posterior).astype(np.float32)
        return posterior, log_alpha

    # ------------------------------------------------------------------
    # Internal: signal construction
    # ------------------------------------------------------------------

    def _build_signal(self, posterior: np.ndarray, symbols: list[str]) -> Signal:
        """Construct a Signal from the latest posterior.

        Args:
            posterior: Shape (K,) probability vector.
            symbols: List of asset symbols in the universe.

        Returns:
            Signal with per-asset values and confidence.
        """
        dominant_state = int(np.argmax(posterior))
        regime_label = self._regime_labels_map.get(dominant_state, "unknown")
        confidence_val = float(posterior[dominant_state])

        # Log regime transitions.
        if self._prev_regime is not None and regime_label != self._prev_regime:
            logger.info(
                "Regime transition: %s -> %s (confidence %.2f)",
                self._prev_regime,
                regime_label,
                confidence_val,
            )
        self._prev_regime = regime_label

        n_assets = len(symbols)
        # Broadcast: same regime for all assets in the universe.
        values = np.full(n_assets, dominant_state, dtype=np.float64)
        confidences = np.full(n_assets, confidence_val, dtype=np.float64)

        return Signal(
            name=self.name,
            values=values,
            confidence=confidences,
            regime=regime_label,
            metadata={
                "posterior": posterior.tolist(),
                "regime_labels": self._regime_labels_map,
                "symbols": symbols,
                "n_states": self._n_states,
            },
        )
