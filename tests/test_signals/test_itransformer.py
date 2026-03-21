"""Tests for src/signals/itransformer.py."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator
from src.signals.itransformer import (
    ITransformerConfig,
    ITransformerEncoder,
    ITransformerSignal,
    _extract_features,
    _FeedForward,
    _InvertedMultiHeadAttention,
    _ITransformerBlock,
    _VariableEmbedding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_asset(
    symbol: str,
    periods: int = 200,
    drift: float = 0.0,
    vol: float = 0.01,
    seed: int = 42,
) -> AssetData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=periods)
    log_returns = drift + vol * rng.standard_normal(periods)
    price = 100.0 * np.exp(np.cumsum(log_returns))
    volume = rng.integers(1_000_000, 5_000_000, size=periods).astype(float)
    df = pd.DataFrame(
        {
            "open": price * (1 + rng.standard_normal(periods) * 0.001),
            "high": price * (1 + np.abs(rng.standard_normal(periods)) * 0.005),
            "low": price * (1 - np.abs(rng.standard_normal(periods)) * 0.005),
            "close": price,
            "volume": volume,
        },
        index=dates,
    )
    return AssetData(symbol=symbol, ohlcv=df)


def _make_universe(
    n_assets: int = 3,
    periods: int = 200,
) -> dict[str, AssetData]:
    drifts = [0.001 * (i - n_assets // 2) for i in range(n_assets)]
    return {
        f"ASSET_{i}": _make_asset(f"ASSET_{i}", periods=periods, drift=d, seed=i * 13)
        for i, d in enumerate(drifts)
    }


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------


class TestInterface:
    def test_is_signal_generator(self) -> None:
        gen = ITransformerSignal()
        assert isinstance(gen, SignalGenerator)

    def test_name_property(self) -> None:
        gen = ITransformerSignal()
        assert gen.name == "itransformer"


# ---------------------------------------------------------------------------
# PyTorch module shapes
# ---------------------------------------------------------------------------


class TestModuleShapes:
    """Validate tensor shapes through each layer."""

    def test_variable_embedding_shape(self) -> None:
        emb = _VariableEmbedding(seq_length=60, d_model=128, dropout=0.0)
        x = torch.randn(4, 5, 60)  # (batch, n_vars, seq_length)
        out = emb(x)
        assert out.shape == (4, 5, 128)

    def test_inverted_mha_shape(self) -> None:
        mha = _InvertedMultiHeadAttention(d_model=128, n_heads=4, dropout=0.0)
        x = torch.randn(4, 5, 128)
        out, attn_w = mha(x)
        assert out.shape == (4, 5, 128)
        assert attn_w.shape == (4, 4, 5, 5)

    def test_feedforward_shape(self) -> None:
        ff = _FeedForward(d_model=128, d_ff=256, dropout=0.0)
        x = torch.randn(4, 5, 128)
        assert ff(x).shape == (4, 5, 128)

    def test_block_shape(self) -> None:
        block = _ITransformerBlock(d_model=128, n_heads=4, d_ff=256, dropout=0.0)
        x = torch.randn(4, 5, 128)
        out, attn_w = block(x)
        assert out.shape == (4, 5, 128)
        assert attn_w.shape == (4, 4, 5, 5)

    def test_encoder_output_shape(self) -> None:
        cfg = ITransformerConfig(seq_length=60, d_model=128, n_heads=4, n_layers=2, d_ff=256)
        enc = ITransformerEncoder(cfg, n_vars=5)
        x = torch.randn(4, 5, 60)
        embeddings, all_attn = enc(x)
        assert embeddings.shape == (4, 5, 128)
        assert len(all_attn) == 2
        assert all_attn[0].shape == (4, 4, 5, 5)

    def test_single_sample(self) -> None:
        cfg = ITransformerConfig(seq_length=60, d_model=64, n_heads=4, n_layers=1)
        enc = ITransformerEncoder(cfg, n_vars=3)
        x = torch.randn(1, 3, 60)
        embeddings, _ = enc(x)
        assert embeddings.shape == (1, 3, 64)


# ---------------------------------------------------------------------------
# Attention properties
# ---------------------------------------------------------------------------


class TestAttention:
    def test_attention_across_variables_not_time(self) -> None:
        """Attention should be (batch, heads, n_vars, n_vars)."""
        cfg = ITransformerConfig(seq_length=60, d_model=64, n_heads=4, n_layers=1)
        enc = ITransformerEncoder(cfg, n_vars=5)
        x = torch.randn(2, 5, 60)
        _, all_attn = enc(x)
        assert all_attn[0].shape == (2, 4, 5, 5)

    def test_attention_weights_sum_to_one(self) -> None:
        cfg = ITransformerConfig(seq_length=60, d_model=64, n_heads=4, n_layers=1, dropout=0.0)
        enc = ITransformerEncoder(cfg, n_vars=5)
        enc.eval()
        x = torch.randn(2, 5, 60)
        _, all_attn = enc(x)
        row_sums = all_attn[0].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_attention_confidence_high_for_peaked(self) -> None:
        """A peaked (one-hot-like) attention should give high confidence."""
        n_vars = 5
        attn = torch.zeros(1, 1, n_vars, n_vars)
        # Make each row attend to only one variable (identity matrix).
        for i in range(n_vars):
            attn[0, 0, i, i] = 1.0
        conf = ITransformerSignal._attention_confidence(attn)
        assert conf > 0.9

    def test_attention_confidence_low_for_uniform(self) -> None:
        """Uniform attention should give low confidence."""
        n_vars = 5
        attn = torch.ones(1, 1, n_vars, n_vars) / n_vars
        conf = ITransformerSignal._attention_confidence(attn)
        assert conf < 0.15


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradients:
    def test_gradients_flow_through_encoder(self) -> None:
        cfg = ITransformerConfig(seq_length=60, d_model=64, n_heads=4, n_layers=2)
        enc = ITransformerEncoder(cfg, n_vars=5)
        x = torch.randn(2, 5, 60, requires_grad=True)
        embeddings, _ = enc(x)
        loss = embeddings.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_in_output(self) -> None:
        cfg = ITransformerConfig(seq_length=60, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
        enc = ITransformerEncoder(cfg, n_vars=5)
        enc.eval()
        x = torch.randn(4, 5, 60)
        embeddings, _ = enc(x)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    def test_shape(self) -> None:
        data = _make_universe(n_assets=3, periods=200)
        features, symbols = _extract_features(data, seq_length=60)
        assert features.shape == (3, 5, 60)
        assert len(symbols) == 3

    def test_no_nans(self) -> None:
        data = _make_universe(n_assets=2, periods=200)
        features, _ = _extract_features(data, seq_length=60)
        assert not np.any(np.isnan(features))

    def test_insufficient_data_skipped(self) -> None:
        data = {
            "GOOD": _make_asset("GOOD", periods=200),
            "SHORT": _make_asset("SHORT", periods=30),
        }
        features, symbols = _extract_features(data, seq_length=60)
        assert symbols == ["GOOD"]
        assert features.shape[0] == 1

    def test_all_insufficient_raises(self) -> None:
        data = {"A": _make_asset("A", periods=30)}
        with pytest.raises(ValueError, match="No assets have sufficient data"):
            _extract_features(data, seq_length=60)


# ---------------------------------------------------------------------------
# generate() — happy path
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_valid_signal(self) -> None:
        data = _make_universe(n_assets=3, periods=200)
        gen = ITransformerSignal()
        sig = gen.generate(data)

        assert isinstance(sig, Signal)
        assert sig.name == "itransformer"
        assert sig.regime is None

    def test_signal_shape_matches_assets(self) -> None:
        data = _make_universe(n_assets=4, periods=200)
        gen = ITransformerSignal()
        sig = gen.generate(data)

        assert sig.values.shape == (4,)
        assert sig.confidence.shape == (4,)

    def test_confidence_in_bounds(self) -> None:
        data = _make_universe(n_assets=3, periods=200)
        gen = ITransformerSignal()
        sig = gen.generate(data)

        assert np.all(sig.confidence >= 0.0)
        assert np.all(sig.confidence <= 1.0)

    def test_values_are_finite(self) -> None:
        data = _make_universe(n_assets=3, periods=200)
        gen = ITransformerSignal()
        sig = gen.generate(data)

        assert np.all(np.isfinite(sig.values))

    def test_metadata_keys(self) -> None:
        data = _make_universe(n_assets=3, periods=200)
        gen = ITransformerSignal()
        sig = gen.generate(data)

        assert "symbols" in sig.metadata
        assert "feature_names" in sig.metadata
        assert "n_assets" in sig.metadata
        assert "d_model" in sig.metadata
        assert sig.metadata["n_assets"] == 3

    def test_single_asset(self) -> None:
        data = {"SPY": _make_asset("SPY", periods=200)}
        gen = ITransformerSignal()
        sig = gen.generate(data)

        assert sig.values.shape == (1,)
        assert sig.confidence.shape == (1,)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        data = _make_universe(n_assets=3, periods=200)
        cfg = ITransformerConfig(random_state=42)

        gen1 = ITransformerSignal(config=cfg)
        gen2 = ITransformerSignal(config=cfg)

        sig1 = gen1.generate(data)
        sig2 = gen2.generate(data)

        np.testing.assert_array_equal(sig1.values, sig2.values)

    def test_different_seed_different_result(self) -> None:
        data = _make_universe(n_assets=3, periods=200)

        gen1 = ITransformerSignal(config=ITransformerConfig(random_state=42))
        gen2 = ITransformerSignal(config=ITransformerConfig(random_state=99))

        sig1 = gen1.generate(data)
        sig2 = gen2.generate(data)

        assert not np.array_equal(sig1.values, sig2.values)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_returns_signal(self) -> None:
        data = _make_universe(n_assets=3, periods=200)
        gen = ITransformerSignal()
        sig = gen.update(data)

        assert isinstance(sig, Signal)
        assert sig.name == "itransformer"

    def test_update_matches_generate(self) -> None:
        data = _make_universe(n_assets=3, periods=200)
        cfg = ITransformerConfig(random_state=42)

        gen1 = ITransformerSignal(config=cfg)
        gen2 = ITransformerSignal(config=cfg)

        sig_gen = gen1.generate(data)
        sig_upd = gen2.update(data)

        np.testing.assert_array_almost_equal(sig_gen.values, sig_upd.values)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_custom_config(self) -> None:
        cfg = ITransformerConfig(d_model=64, n_heads=2, n_layers=1, d_ff=128)
        data = _make_universe(n_assets=2, periods=200)
        gen = ITransformerSignal(config=cfg)
        sig = gen.generate(data)

        assert sig.values.shape == (2,)
        assert sig.metadata["d_model"] == 64

    def test_d_model_must_divide_n_heads(self) -> None:
        cfg = ITransformerConfig(d_model=65, n_heads=4)
        data = _make_universe(n_assets=2, periods=200)
        gen = ITransformerSignal(config=cfg)

        with pytest.raises(ValueError, match="divisible"):
            gen.generate(data)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_empty_data_raises(self) -> None:
        gen = ITransformerSignal()
        with pytest.raises(ValueError, match="No assets have sufficient data"):
            gen.generate({})

    def test_all_insufficient_raises(self) -> None:
        data = {"A": _make_asset("A", periods=30)}
        gen = ITransformerSignal()
        with pytest.raises(ValueError, match="No assets have sufficient data"):
            gen.generate(data)

    def test_get_encoder_before_generate_raises(self) -> None:
        gen = ITransformerSignal()
        with pytest.raises(RuntimeError, match="Call generate"):
            gen.get_encoder()

    def test_load_weights_before_generate_raises(self) -> None:
        gen = ITransformerSignal()
        with pytest.raises(RuntimeError, match="Call generate"):
            gen.load_weights("/nonexistent/path.pt")
