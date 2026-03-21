"""Integration test: Yahoo data → HMM / Momentum / MonteCarlo → SignalEnsemble.

Exercises the full signal pipeline against live Yahoo Finance data.
Run with:  pytest tests/integration/ -v -m network
"""

import numpy as np
import pytest

from src.data.cache import SQLiteCacheProvider
from src.data.yahoo import YahooFinanceProvider
from src.signals.ensemble import SignalEnsemble
from src.signals.hmm_regime import HMMRegimeDetector
from src.signals.momentum import MomentumSignal
from src.signals.montecarlo import MonteCarloSignal

# ~18 months to satisfy 252-row momentum warmup.
_START = "2024-01-02"
_END = "2025-06-30"
_SYMBOLS = ["SPY", "QQQ"]


@pytest.mark.network
class TestSignalPipeline:
    """End-to-end: fetch → individual signals → ensemble."""

    def setup_method(self) -> None:
        yahoo = YahooFinanceProvider(rate_limit=0.1)
        self.cache = SQLiteCacheProvider(yahoo, db_path=":memory:")
        self.data = self.cache.fetch_universe(_SYMBOLS, _START, _END)

    # ------------------------------------------------------------------
    # Stage 1: Individual signal generators
    # ------------------------------------------------------------------

    def test_hmm_regime_signal(self) -> None:
        """HMM produces a signal with correct shape and a regime label."""
        gen = HMMRegimeDetector(n_regimes=3, random_state=42)
        sig = gen.generate(self.data)

        assert sig.values.shape == (len(_SYMBOLS),)
        assert sig.confidence.shape == (len(_SYMBOLS),)
        assert sig.regime in {"bull", "neutral", "bear"}
        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)

    def test_momentum_signal(self) -> None:
        """Momentum produces a signal with correct shape and finite values."""
        gen = MomentumSignal()
        sig = gen.generate(self.data)

        assert sig.values.shape == (len(_SYMBOLS),)
        assert sig.confidence.shape == (len(_SYMBOLS),)
        assert np.all(np.isfinite(sig.values))
        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)

    def test_montecarlo_signal(self) -> None:
        """Monte Carlo produces a signal with risk metadata."""
        gen = MonteCarloSignal(n_simulations=100, random_state=42)
        sig = gen.generate(self.data)

        assert sig.values.shape == (len(_SYMBOLS),)
        assert sig.confidence.shape == (len(_SYMBOLS),)
        assert np.all(np.isfinite(sig.values))
        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)
        assert "var_5pct" in sig.metadata
        assert "cvar_5pct" in sig.metadata

    # ------------------------------------------------------------------
    # Stage 2: Ensemble combination
    # ------------------------------------------------------------------

    def test_ensemble_combines_all_children(self) -> None:
        """Ensemble combines HMM, Momentum, and MonteCarlo signals."""
        hmm = HMMRegimeDetector(n_regimes=3, random_state=42)
        mom = MomentumSignal()
        mc = MonteCarloSignal(n_simulations=100, random_state=42)

        ens = SignalEnsemble([hmm, mom, mc], method="confidence_weighted")
        sig = ens.generate(self.data)

        assert sig.values.shape == (len(_SYMBOLS),)
        assert sig.confidence.shape == (len(_SYMBOLS),)
        assert np.all(np.isfinite(sig.values))
        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)

    def test_ensemble_metadata_attribution(self) -> None:
        """Ensemble metadata lists all contributing child signals."""
        hmm = HMMRegimeDetector(n_regimes=3, random_state=42)
        mom = MomentumSignal()
        mc = MonteCarloSignal(n_simulations=100, random_state=42)

        ens = SignalEnsemble([hmm, mom, mc], method="confidence_weighted")
        sig = ens.generate(self.data)

        assert sig.metadata["method"] == "confidence_weighted"
        assert sig.metadata["n_signals"] == 3
        assert set(sig.metadata["contributors"]) == {"hmm_regime", "momentum", "montecarlo"}
        assert len(sig.metadata["contributions"]) == 3

    def test_ensemble_regime_passthrough(self) -> None:
        """Ensemble adopts the regime from the HMM child."""
        hmm = HMMRegimeDetector(n_regimes=3, random_state=42)
        mom = MomentumSignal()
        mc = MonteCarloSignal(n_simulations=100, random_state=42)

        # HMM is first → its regime should propagate.
        ens = SignalEnsemble([hmm, mom, mc], method="confidence_weighted")
        sig = ens.generate(self.data)

        assert sig.regime in {"bull", "neutral", "bear"}

    def test_ensemble_confidence_in_bounds(self) -> None:
        """All per-asset confidences are in [0, 1]."""
        hmm = HMMRegimeDetector(n_regimes=3, random_state=42)
        mom = MomentumSignal()
        mc = MonteCarloSignal(n_simulations=100, random_state=42)

        ens = SignalEnsemble([hmm, mom, mc], method="confidence_weighted")
        sig = ens.generate(self.data)

        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)
