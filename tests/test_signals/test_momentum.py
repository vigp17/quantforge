"""Tests for src/signals/momentum.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator
from src.signals.momentum import MomentumSignal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_asset(
    symbol: str,
    periods: int = 300,
    drift: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
) -> AssetData:
    """Build an AssetData with a controllable drift and volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=periods)
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


def _make_universe(
    n_assets: int = 4,
    periods: int = 300,
    drifts: list[float] | None = None,
) -> dict[str, AssetData]:
    """Build a multi-asset universe with distinct drifts."""
    if drifts is None:
        drifts = [0.001 * (i - n_assets // 2) for i in range(n_assets)]
    names = [f"ASSET_{i}" for i in range(n_assets)]
    return {
        name: _make_asset(name, periods=periods, drift=d, seed=i * 17)
        for i, (name, d) in enumerate(zip(names, drifts))
    }


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------


class TestInterface:
    def test_is_signal_generator(self) -> None:
        gen = MomentumSignal()
        assert isinstance(gen, SignalGenerator)

    def test_name_property(self) -> None:
        gen = MomentumSignal()
        assert gen.name == "momentum"


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_valid_signal(self) -> None:
        data = _make_universe(n_assets=3, periods=300)
        gen = MomentumSignal()
        sig = gen.generate(data)

        assert isinstance(sig, Signal)
        assert sig.name == "momentum"
        assert sig.regime is None

    def test_signal_shape_matches_eligible_assets(self) -> None:
        data = _make_universe(n_assets=4, periods=300)
        gen = MomentumSignal()
        sig = gen.generate(data)

        assert sig.values.shape == (4,)
        assert sig.confidence.shape == (4,)

    def test_confidence_in_bounds(self) -> None:
        data = _make_universe(n_assets=3, periods=300)
        gen = MomentumSignal()
        sig = gen.generate(data)

        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)

    def test_metadata_contains_factors(self) -> None:
        data = _make_universe(n_assets=3, periods=300)
        gen = MomentumSignal()
        sig = gen.generate(data)

        assert "symbols" in sig.metadata
        assert "factor_names" in sig.metadata
        assert "raw_factors" in sig.metadata
        assert "zscored_factors" in sig.metadata
        assert set(sig.metadata["factor_names"]) == {"mom_12_1", "high_52w", "mom_1m"}

    def test_metadata_zscored_factors_per_asset(self) -> None:
        data = _make_universe(n_assets=3, periods=300)
        gen = MomentumSignal()
        sig = gen.generate(data)

        for sym in sig.metadata["symbols"]:
            zf = sig.metadata["zscored_factors"][sym]
            assert len(zf) == 3  # three factors


# ---------------------------------------------------------------------------
# Directional correctness
# ---------------------------------------------------------------------------


class TestDirectional:
    def test_uptrend_has_positive_momentum(self) -> None:
        """A strong uptrending asset should have positive composite score."""
        data = {
            "UP": _make_asset("UP", periods=300, drift=0.002, seed=1),
            "FLAT": _make_asset("FLAT", periods=300, drift=0.0, seed=2),
        }
        gen = MomentumSignal()
        sig = gen.generate(data)

        symbols = sig.metadata["symbols"]
        up_idx = symbols.index("UP")
        flat_idx = symbols.index("FLAT")

        # UP should have higher momentum than FLAT.
        assert sig.values[up_idx] > sig.values[flat_idx]

    def test_downtrend_has_negative_momentum(self) -> None:
        """A strong downtrending asset should have negative composite score."""
        data = {
            "DOWN": _make_asset("DOWN", periods=300, drift=-0.002, seed=1),
            "FLAT": _make_asset("FLAT", periods=300, drift=0.0, seed=2),
        }
        gen = MomentumSignal()
        sig = gen.generate(data)

        symbols = sig.metadata["symbols"]
        down_idx = symbols.index("DOWN")
        flat_idx = symbols.index("FLAT")

        assert sig.values[down_idx] < sig.values[flat_idx]

    def test_relative_ranking_preserved(self) -> None:
        """Assets with higher drift should rank higher in composite score."""
        data = {
            "STRONG": _make_asset("STRONG", periods=300, drift=0.005, vol=0.005, seed=10),
            "MILD": _make_asset("MILD", periods=300, drift=0.001, vol=0.005, seed=20),
            "WEAK": _make_asset("WEAK", periods=300, drift=-0.003, vol=0.005, seed=30),
        }
        gen = MomentumSignal()
        sig = gen.generate(data)

        symbols = sig.metadata["symbols"]
        strong = sig.values[symbols.index("STRONG")]
        mild = sig.values[symbols.index("MILD")]
        weak = sig.values[symbols.index("WEAK")]

        assert strong > mild > weak


# ---------------------------------------------------------------------------
# Composite score normalisation
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_cross_sectional_zscore_zero_mean(self) -> None:
        """With multiple assets the composite scores should have ~zero mean."""
        data = _make_universe(n_assets=5, periods=300)
        gen = MomentumSignal()
        sig = gen.generate(data)

        assert abs(sig.values.mean()) < 1.0  # loosely centred

    def test_composite_is_finite(self) -> None:
        data = _make_universe(n_assets=3, periods=300)
        gen = MomentumSignal()
        sig = gen.generate(data)

        assert np.all(np.isfinite(sig.values))

    def test_single_asset_produces_bounded_score(self) -> None:
        """A single-asset universe should still produce a valid signal."""
        data = {"SPY": _make_asset("SPY", periods=300, drift=0.001)}
        gen = MomentumSignal()
        sig = gen.generate(data)

        assert sig.values.shape == (1,)
        assert np.isfinite(sig.values[0])
        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)


# ---------------------------------------------------------------------------
# Confidence and factor agreement
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_strong_trend_high_confidence(self) -> None:
        """When all factors agree (strong uptrend), confidence should be high."""
        data = {
            "UP": _make_asset("UP", periods=300, drift=0.003, seed=1),
            "DOWN": _make_asset("DOWN", periods=300, drift=-0.003, seed=2),
        }
        gen = MomentumSignal()
        sig = gen.generate(data)

        # Both assets should have high confidence — factors all agree.
        assert np.all(sig.confidence >= 0.5)

    def test_agreement_equals_one_when_all_factors_same_sign(self) -> None:
        """If all 3 z-scored factors are positive, agreement should be 1.0."""
        # Directly test the agreement function.
        zscored = np.array([0.5, 1.2, 0.3])  # all positive
        agreement = MomentumSignal._factor_agreement(zscored)
        assert agreement == 1.0

    def test_agreement_low_when_mixed_signs(self) -> None:
        """If factors disagree on sign, agreement drops."""
        zscored = np.array([1.0, -0.5, 0.3])  # 2 positive, 1 negative
        agreement = MomentumSignal._factor_agreement(zscored)
        assert agreement < 1.0

    def test_agreement_zero_when_perfectly_split(self) -> None:
        """With an even split (impossible for 3, but test 0-sum case)."""
        zscored = np.array([1.0, -1.0, 0.0])  # sum = 0
        agreement = MomentumSignal._factor_agreement(zscored)
        assert agreement == 0.0


# ---------------------------------------------------------------------------
# Insufficient data handling
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_short_asset_skipped_with_warning(self, caplog) -> None:
        """Assets with < 252 rows should be skipped."""
        data = {
            "GOOD": _make_asset("GOOD", periods=300, drift=0.001),
            "SHORT": _make_asset("SHORT", periods=100, drift=0.001),
        }
        gen = MomentumSignal()
        sig = gen.generate(data)

        assert sig.values.shape == (1,)
        assert sig.metadata["symbols"] == ["GOOD"]
        assert "Skipping SHORT" in caplog.text

    def test_all_assets_insufficient_raises(self) -> None:
        data = {"A": _make_asset("A", periods=100), "B": _make_asset("B", periods=50)}
        gen = MomentumSignal()

        with pytest.raises(ValueError, match="No assets have sufficient data"):
            gen.generate(data)

    def test_empty_data_raises(self) -> None:
        gen = MomentumSignal()

        with pytest.raises(ValueError, match="No assets have sufficient data"):
            gen.generate({})


# ---------------------------------------------------------------------------
# update() delegates to generate()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_returns_signal(self) -> None:
        data = _make_universe(n_assets=3, periods=300)
        gen = MomentumSignal()
        sig = gen.update(data)

        assert isinstance(sig, Signal)
        assert sig.name == "momentum"

    def test_update_matches_generate(self) -> None:
        """update() should produce the same result as generate()."""
        data = _make_universe(n_assets=3, periods=300)
        gen = MomentumSignal()

        sig_gen = gen.generate(data)
        sig_upd = gen.update(data)

        np.testing.assert_array_almost_equal(sig_gen.values, sig_upd.values)
        np.testing.assert_array_almost_equal(sig_gen.confidence, sig_upd.confidence)


# ---------------------------------------------------------------------------
# Individual factor computation
# ---------------------------------------------------------------------------


class TestFactors:
    def test_high_52w_at_high_equals_one(self) -> None:
        """If price is at the 52-week high, high_52w should be 1.0."""
        # Monotonically increasing prices — last price is the 52w high.
        dates = pd.bdate_range("2021-01-04", periods=300)
        price = np.linspace(100, 200, 300)
        df = pd.DataFrame(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": np.ones(300),
            },
            index=dates,
        )
        asset = AssetData(symbol="UP", ohlcv=df)
        gen = MomentumSignal()
        factors = gen._compute_factors(asset.ohlcv["close"])

        assert factors["high_52w"] == pytest.approx(1.0)

    def test_high_52w_below_one_after_drawdown(self) -> None:
        """If price has fallen from its 52-week high, ratio < 1."""
        dates = pd.bdate_range("2021-01-04", periods=300)
        price = np.concatenate([np.linspace(100, 200, 200), np.linspace(200, 150, 100)])
        df = pd.DataFrame(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": np.ones(300),
            },
            index=dates,
        )
        asset = AssetData(symbol="DD", ohlcv=df)
        gen = MomentumSignal()
        factors = gen._compute_factors(asset.ohlcv["close"])

        assert factors["high_52w"] < 1.0
        assert factors["high_52w"] == pytest.approx(150.0 / 200.0)

    def test_mom_12_1_positive_for_sustained_trend(self) -> None:
        """A steady uptrend should have positive mom_12_1."""
        dates = pd.bdate_range("2021-01-04", periods=300)
        price = np.linspace(100, 200, 300)
        df = pd.DataFrame(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": np.ones(300),
            },
            index=dates,
        )
        asset = AssetData(symbol="UP", ohlcv=df)
        gen = MomentumSignal()
        factors = gen._compute_factors(asset.ohlcv["close"])

        assert factors["mom_12_1"] > 0
