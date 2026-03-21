"""Tests for src/signals/hmm_regime.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator
from src.signals.hmm_regime import HMMRegimeDetector, _extract_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_asset(
    symbol: str,
    periods: int = 300,
    drift: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
) -> AssetData:
    """Build an AssetData with a controllable drift and volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=periods)
    log_returns = drift + vol * rng.standard_normal(periods)
    price = 100.0 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame(
        {
            "open": price * (1 + rng.standard_normal(periods) * 0.001),
            "high": price * (1 + np.abs(rng.standard_normal(periods)) * 0.005),
            "low": price * (1 - np.abs(rng.standard_normal(periods)) * 0.005),
            "close": price,
            "volume": rng.integers(1_000_000, 5_000_000, size=periods).astype(float),
        },
        index=dates,
    )
    return AssetData(symbol=symbol, ohlcv=df)


def _make_regime_universe(periods: int = 600, seed: int = 0) -> dict[str, AssetData]:
    """Build a two-asset universe with enough data for HMM fitting.

    First half is bull-like (positive drift), second half is bear-like
    (negative drift), so the HMM should discover two distinct regimes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=periods)
    half = periods // 2

    # Concatenate bull then bear returns.
    ret_a = np.concatenate([
        0.001 + 0.008 * rng.standard_normal(half),
        -0.001 + 0.015 * rng.standard_normal(periods - half),
    ])
    ret_b = np.concatenate([
        0.0008 + 0.007 * rng.standard_normal(half),
        -0.0008 + 0.012 * rng.standard_normal(periods - half),
    ])

    def _to_asset(sym: str, rets: np.ndarray) -> AssetData:
        price = 100.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame(
            {
                "open": price,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": np.full(periods, 3_000_000.0),
            },
            index=dates,
        )
        return AssetData(symbol=sym, ohlcv=df)

    return {"AAA": _to_asset("AAA", ret_a), "BBB": _to_asset("BBB", ret_b)}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_shape(self) -> None:
        data = {"SPY": _make_trending_asset("SPY", periods=100)}
        features, symbols = _extract_features(data)

        # 2 features per asset, minus ~20 rows for rolling warmup.
        assert features.shape[1] == 2
        assert features.shape[0] < 100
        assert symbols == ["SPY"]

    def test_multiple_assets(self) -> None:
        data = {
            "SPY": _make_trending_asset("SPY", periods=100),
            "QQQ": _make_trending_asset("QQQ", periods=100, seed=99),
        }
        features, symbols = _extract_features(data)

        assert features.shape[1] == 4  # 2 per asset
        assert symbols == ["SPY", "QQQ"]

    def test_no_nans(self) -> None:
        data = {"SPY": _make_trending_asset("SPY", periods=100)}
        features, _ = _extract_features(data)

        assert not np.any(np.isnan(features))


# ---------------------------------------------------------------------------
# HMMRegimeDetector — interface compliance
# ---------------------------------------------------------------------------

class TestInterface:
    def test_is_signal_generator(self) -> None:
        det = HMMRegimeDetector()
        assert isinstance(det, SignalGenerator)

    def test_name_property(self) -> None:
        det = HMMRegimeDetector()
        assert det.name == "hmm_regime"


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_valid_signal(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        sig = det.generate(data)

        assert isinstance(sig, Signal)
        assert sig.name == "hmm_regime"
        assert sig.regime in ("bull", "bear", "sideways")

    def test_signal_shape_matches_assets(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        sig = det.generate(data)

        assert sig.values.shape == (2,)  # 2 assets
        assert sig.confidence.shape == (2,)

    def test_confidence_in_bounds(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        sig = det.generate(data)

        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)

    def test_metadata_contains_posterior(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        sig = det.generate(data)

        assert "posterior" in sig.metadata
        posterior = np.array(sig.metadata["posterior"])
        assert len(posterior) == 3
        assert abs(posterior.sum() - 1.0) < 1e-5

    def test_metadata_contains_regime_labels(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        sig = det.generate(data)

        labels = sig.metadata["regime_labels"]
        assert set(labels.values()) == {"bull", "bear", "sideways"}

    def test_two_state_model(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=2, random_state=42)
        sig = det.generate(data)

        assert sig.regime in ("bull", "bear")
        posterior = np.array(sig.metadata["posterior"])
        assert len(posterior) == 2


# ---------------------------------------------------------------------------
# Regime labeling
# ---------------------------------------------------------------------------

class TestRegimeLabeling:
    def test_bull_has_highest_mean_return(self) -> None:
        """After fitting, the state labeled 'bull' should have the highest
        mean return in the first feature column."""
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.generate(data)

        means_orig = det._scaler.inverse_transform(det._model.means_)
        mean_rets = means_orig[:, 0]

        bull_idx = [k for k, v in det._regime_labels_map.items() if v == "bull"][0]
        bear_idx = [k for k, v in det._regime_labels_map.items() if v == "bear"][0]

        assert mean_rets[bull_idx] > mean_rets[bear_idx]

    def test_bear_has_lowest_mean_return(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.generate(data)

        means_orig = det._scaler.inverse_transform(det._model.means_)
        mean_rets = means_orig[:, 0]

        bear_idx = [k for k, v in det._regime_labels_map.items() if v == "bear"][0]
        assert mean_rets[bear_idx] == min(mean_rets)

    def test_all_labels_assigned(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.generate(data)

        assert len(det._regime_labels_map) == 3
        assert "bull" in det._regime_labels_map.values()
        assert "bear" in det._regime_labels_map.values()
        assert "sideways" in det._regime_labels_map.values()


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_too_few_observations_raises(self) -> None:
        data = {"SPY": _make_trending_asset("SPY", periods=30)}
        det = HMMRegimeDetector()

        with pytest.raises(ValueError, match="Insufficient data"):
            det.generate(data)

    def test_empty_data_raises(self) -> None:
        det = HMMRegimeDetector()

        with pytest.raises((ValueError, KeyError)):
            det.generate({})


# ---------------------------------------------------------------------------
# update() — online inference
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_returns_signal_after_generate(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.generate(data)

        # Build a small "new data" slice.
        new_data = {
            sym: AssetData(symbol=sym, ohlcv=asset.ohlcv.iloc[-30:])
            for sym, asset in data.items()
        }
        sig = det.update(new_data)

        assert isinstance(sig, Signal)
        assert sig.regime in ("bull", "bear", "sideways")
        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)

    def test_update_without_prior_fit_falls_back_to_generate(self) -> None:
        data = _make_regime_universe(periods=600)
        det = HMMRegimeDetector(n_states=3, random_state=42)

        # update() on an unfitted detector should fall back to generate().
        sig = det.update(data)
        assert isinstance(sig, Signal)
