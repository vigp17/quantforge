"""Tests for src/portfolio/position_sizer.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.position_sizer import PositionSizer
from src.signals.base import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_signal(n: int, value: float = 1.0, conf: float = 1.0) -> Signal:
    """All-positive, full-confidence signal for *n* assets."""
    return Signal(
        name="test",
        values=np.full(n, value),
        confidence=np.full(n, conf),
    )


def _returns(
    symbols: list[str],
    n: int = 120,
    seed: int = 0,
    vols: list[float] | None = None,
) -> pd.DataFrame:
    """Synthetic daily returns. *vols* sets the daily std per asset."""
    rng = np.random.default_rng(seed)
    vols = vols or [0.01] * len(symbols)
    data = {sym: rng.normal(0.0, v, n) for sym, v in zip(symbols, vols)}
    return pd.DataFrame(data)


def _biased_returns(symbols: list[str], win_rates: list[float], n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Returns where each asset has a controlled win rate."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for sym, wr in zip(symbols, win_rates):
        outcomes = np.where(rng.random(n) < wr, 0.01, -0.01)
        data[sym] = outcomes
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ps() -> PositionSizer:
    return PositionSizer(max_position=0.30)


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------

class TestKellyFraction:
    def test_known_values(self, ps: PositionSizer) -> None:
        # p=0.6, avg_win=0.01, avg_loss=0.01 → b=1.0
        # f* = (0.6*1 - 0.4) / 1 = 0.2
        returns = _biased_returns(["A"], [0.60], n=2000)["A"]
        frac = ps.kelly_fraction(returns, half_kelly=1.0)
        assert pytest.approx(frac, abs=0.03) == 0.20

    def test_half_kelly_is_half_full_kelly(self, ps: PositionSizer) -> None:
        returns = _biased_returns(["A"], [0.60], n=2000)["A"]
        full = ps.kelly_fraction(returns, half_kelly=1.0)
        half = ps.kelly_fraction(returns, half_kelly=0.5)
        assert half == pytest.approx(full * 0.5, rel=1e-6)

    def test_negative_kelly_returns_zero(self, ps: PositionSizer) -> None:
        # More losses than wins → f* < 0 → floored to 0
        returns = _biased_returns(["A"], [0.30], n=2000)["A"]
        frac = ps.kelly_fraction(returns, half_kelly=1.0)
        assert frac == 0.0

    def test_clips_to_max_position(self) -> None:
        ps = PositionSizer(max_position=0.10)
        # Very favourable bet → raw Kelly likely > 0.10
        returns = _biased_returns(["A"], [0.90], n=2000)["A"]
        frac = ps.kelly_fraction(returns, half_kelly=1.0)
        assert frac <= 0.10

    def test_all_zero_returns_gives_zero(self, ps: PositionSizer) -> None:
        returns = pd.Series(np.zeros(100))
        frac = ps.kelly_fraction(returns)
        assert frac == 0.0

    def test_insufficient_data_gives_zero(self, ps: PositionSizer) -> None:
        frac = ps.kelly_fraction(pd.Series([0.01]))
        assert frac == 0.0

    def test_all_winning_returns(self, ps: PositionSizer) -> None:
        # p=1.0, q=0.0 → f* = 1.0 (full bet), clipped to max_position
        returns = pd.Series([0.01] * 100)
        frac = ps.kelly_fraction(returns, half_kelly=1.0)
        assert frac == pytest.approx(ps.max_position)


# ---------------------------------------------------------------------------
# target_vol_weights
# ---------------------------------------------------------------------------

class TestTargetVolWeights:
    def test_low_vol_asset_gets_higher_weight(self, ps: PositionSizer) -> None:
        # A: daily vol 0.005, B: daily vol 0.050 → 10× spread ensures A stays
        # above B even after max_position clipping (B is too volatile to hit the cap).
        df = _returns(["A", "B"], vols=[0.005, 0.050])
        weights = ps.target_vol_weights(df, target_vol=0.15)
        assert weights["A"] > weights["B"]

    def test_equal_vol_equal_weight(self, ps: PositionSizer) -> None:
        df = _returns(["X", "Y"], vols=[0.01, 0.01])
        weights = ps.target_vol_weights(df, target_vol=0.15)
        assert weights["X"] == pytest.approx(weights["Y"], rel=0.05)

    def test_weights_sum_leq_one(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B", "C", "D"], vols=[0.01, 0.02, 0.015, 0.008])
        weights = ps.target_vol_weights(df, target_vol=0.15)
        assert sum(abs(w) for w in weights.values()) <= 1.0 + 1e-9

    def test_no_weight_exceeds_max_position(self, ps: PositionSizer) -> None:
        # Very low vol single asset would get huge weight without clipping
        df = _returns(["A"], vols=[0.001])
        weights = ps.target_vol_weights(df, target_vol=0.15)
        for w in weights.values():
            assert abs(w) <= ps.max_position + 1e-9

    def test_high_vol_asset_gets_lower_weight(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B"], vols=[0.01, 0.10])
        weights = ps.target_vol_weights(df, target_vol=0.15)
        assert weights["A"] > weights["B"]

    def test_empty_dataframe_all_zeros(self, ps: PositionSizer) -> None:
        df = pd.DataFrame({"A": [], "B": []})
        weights = ps.target_vol_weights(df)
        assert all(w == 0.0 for w in weights.values())

    def test_all_zero_returns_all_zero_weights(self, ps: PositionSizer) -> None:
        df = pd.DataFrame({"A": np.zeros(100), "B": np.zeros(100)})
        weights = ps.target_vol_weights(df)
        assert all(w == 0.0 for w in weights.values())

    def test_very_high_volatility(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B"], vols=[0.50, 0.50])
        weights = ps.target_vol_weights(df, target_vol=0.15)
        for w in weights.values():
            assert abs(w) <= ps.max_position + 1e-9


# ---------------------------------------------------------------------------
# size – equal_weight
# ---------------------------------------------------------------------------

class TestSizeEqualWeight:
    def test_four_assets_equal_weight(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B", "C", "D"])
        sig = _uniform_signal(4)
        weights = ps.size(sig, df, method="equal_weight")
        for sym in ["A", "B", "C", "D"]:
            assert weights[sym] == pytest.approx(0.25, abs=1e-9)

    def test_single_asset_gets_full_weight_capped(self, ps: PositionSizer) -> None:
        df = _returns(["SPY"])
        sig = _uniform_signal(1)
        weights = ps.size(sig, df, method="equal_weight")
        assert weights["SPY"] == pytest.approx(min(1.0, ps.max_position), abs=1e-9)

    def test_weights_sum_leq_one(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B", "C"])
        sig = _uniform_signal(3)
        weights = ps.size(sig, df, method="equal_weight")
        assert sum(abs(w) for w in weights.values()) <= 1.0 + 1e-9

    def test_zero_confidence_gives_zero_weight(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B"])
        sig = Signal(
            name="test",
            values=np.array([1.0, 1.0]),
            confidence=np.array([0.0, 1.0]),
        )
        weights = ps.size(sig, df, method="equal_weight")
        assert weights["A"] == 0.0
        assert weights["B"] > 0.0

    def test_negative_signal_gives_negative_weight(self, ps: PositionSizer) -> None:
        df = _returns(["A"])
        sig = Signal(name="test", values=np.array([-1.0]), confidence=np.array([1.0]))
        weights = ps.size(sig, df, method="equal_weight")
        assert weights["A"] < 0.0

    def test_max_position_clipping_equal_weight(self) -> None:
        ps = PositionSizer(max_position=0.10)
        # 3 assets × 1/3 each = 0.333 > max 0.10 → each clipped then sum normalised
        df = _returns(["A", "B", "C"])
        sig = _uniform_signal(3)
        weights = ps.size(sig, df, method="equal_weight")
        for w in weights.values():
            assert abs(w) <= 0.10 + 1e-9


# ---------------------------------------------------------------------------
# size – volatility_scaled
# ---------------------------------------------------------------------------

class TestSizeVolatilityScaled:
    def test_low_vol_asset_gets_higher_weight(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B"], vols=[0.005, 0.025])
        sig = _uniform_signal(2)
        weights = ps.size(sig, df, method="volatility_scaled")
        assert weights["A"] > weights["B"]

    def test_weights_sum_leq_one(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B", "C"], vols=[0.01, 0.02, 0.015])
        sig = _uniform_signal(3)
        weights = ps.size(sig, df, method="volatility_scaled")
        assert sum(abs(w) for w in weights.values()) <= 1.0 + 1e-9

    def test_no_weight_exceeds_max_position(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B", "C"])
        sig = _uniform_signal(3)
        weights = ps.size(sig, df, method="volatility_scaled")
        for w in weights.values():
            assert abs(w) <= ps.max_position + 1e-9

    def test_confidence_scales_weight(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B"], vols=[0.01, 0.01])
        sig_full = _uniform_signal(2, conf=1.0)
        sig_half = _uniform_signal(2, conf=0.5)
        w_full = ps.size(sig_full, df, method="volatility_scaled")
        w_half = ps.size(sig_half, df, method="volatility_scaled")
        # Half confidence → half weight before normalisation
        # After normalisation ratios are preserved; sum is smaller
        total_full = sum(abs(w) for w in w_full.values())
        total_half = sum(abs(w) for w in w_half.values())
        assert total_half < total_full + 1e-9


# ---------------------------------------------------------------------------
# size – kelly
# ---------------------------------------------------------------------------

class TestSizeKelly:
    def test_positive_signal_positive_weight(self, ps: PositionSizer) -> None:
        df = _biased_returns(["A", "B"], [0.60, 0.65])
        sig = _uniform_signal(2)
        weights = ps.size(sig, df, method="kelly")
        assert weights["A"] >= 0.0
        assert weights["B"] >= 0.0

    def test_unfavourable_assets_zero_weight(self, ps: PositionSizer) -> None:
        # A is favourable, B is not → B gets Kelly=0 → zero weight
        df = _biased_returns(["A", "B"], [0.65, 0.30], n=2000)
        sig = _uniform_signal(2)
        weights = ps.size(sig, df, method="kelly")
        assert weights["B"] == pytest.approx(0.0, abs=1e-9)

    def test_weights_sum_leq_one(self, ps: PositionSizer) -> None:
        df = _biased_returns(["A", "B", "C"], [0.60, 0.55, 0.65])
        sig = _uniform_signal(3)
        weights = ps.size(sig, df, method="kelly")
        assert sum(abs(w) for w in weights.values()) <= 1.0 + 1e-9

    def test_no_weight_exceeds_max_position(self, ps: PositionSizer) -> None:
        df = _biased_returns(["A"], [0.90], n=2000)
        sig = _uniform_signal(1)
        weights = ps.size(sig, df, method="kelly")
        for w in weights.values():
            assert abs(w) <= ps.max_position + 1e-9

    def test_all_zero_returns_zero_weights(self, ps: PositionSizer) -> None:
        df = pd.DataFrame({"A": np.zeros(100), "B": np.zeros(100)})
        sig = _uniform_signal(2)
        weights = ps.size(sig, df, method="kelly")
        assert all(w == 0.0 for w in weights.values())


# ---------------------------------------------------------------------------
# size – unknown method
# ---------------------------------------------------------------------------

class TestSizeUnknownMethod:
    def test_raises_on_unknown_method(self, ps: PositionSizer) -> None:
        df = _returns(["A"])
        sig = _uniform_signal(1)
        with pytest.raises(ValueError, match="Unknown method"):
            ps.size(sig, df, method="magic")


# ---------------------------------------------------------------------------
# size – signal/returns mismatch
# ---------------------------------------------------------------------------

class TestSizeMismatch:
    def test_raises_on_length_mismatch(self, ps: PositionSizer) -> None:
        df = _returns(["A", "B", "C"])
        sig = _uniform_signal(2)  # wrong length
        with pytest.raises(ValueError, match="does not match"):
            ps.size(sig, df, method="equal_weight")


# ---------------------------------------------------------------------------
# PositionSizer constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_invalid_max_position_zero(self) -> None:
        with pytest.raises(ValueError):
            PositionSizer(max_position=0.0)

    def test_invalid_max_position_above_one(self) -> None:
        with pytest.raises(ValueError):
            PositionSizer(max_position=1.1)

    def test_max_position_one_is_valid(self) -> None:
        ps = PositionSizer(max_position=1.0)
        assert ps.max_position == 1.0
