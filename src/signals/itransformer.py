"""iTransformer signal generator.

Ported from the UARC project's iTransformerEncoder.  The "inverted"
transformer treats each **variable** (feature) as a token rather than
each time step, so self-attention captures cross-variable dependencies
instead of temporal patterns.  This is the right inductive bias for
multi-asset portfolios because cross-asset correlations matter more
than individual temporal dynamics.

Architecture (per the UARC design):
    AssetEmbedding   —  Linear(seq_len → d_model) + LayerNorm + Dropout
    iTransformerBlock ×n_layers  —  Pre-norm MHA + FFN with residual
    Mean pooling over tokens     →  (batch, d_model)
    Signal head                  →  (batch, 1) score per asset

Reference: Liu et al. (2024) "iTransformer: Inverted Transformers
           Are Effective for Time Series Forecasting" — ICLR 2024
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.data.base import AssetData
from src.signals.base import Signal, SignalGenerator

logger = logging.getLogger(__name__)

_MIN_OBS = 80  # need seq_length + warmup for features


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class ITransformerConfig:
    """Hyperparameters for the iTransformer encoder.

    Args:
        seq_length: Number of time steps in the lookback window.
        n_features: Number of input features per time step.
        d_model: Internal embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of iTransformer blocks.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
        random_state: Seed for weight initialisation reproducibility.
    """

    seq_length: int = 60
    n_features: int = 5
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    random_state: int = 42


# ======================================================================
# PyTorch modules
# ======================================================================


class _VariableEmbedding(nn.Module):
    """Projects each variable's time-series into d_model space.

    Each variable is a token whose features are its ``seq_length``
    historical values.  This linear map is analogous to a patch
    embedding in Vision Transformers.
    """

    def __init__(self, seq_length: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(seq_length, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, n_vars, seq_length) → (batch, n_vars, d_model)."""
        return self.dropout(self.norm(self.projection(x)))


class _InvertedMultiHeadAttention(nn.Module):
    """Multi-head self-attention across the **variable** dimension.

    In a standard transformer the sequence dimension is time;
    here it is variables (features / assets).  Each variable
    attends to all others, capturing cross-variable dependencies.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute self-attention and return (output, attn_weights).

        Args:
            x: (batch, n_vars, d_model).

        Returns:
            Tuple of (output, attn_weights) where output has the same
            shape as *x* and attn_weights is (batch, n_heads, n_vars, n_vars).
        """
        b, n, _ = x.shape

        q = self.w_q(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        out = torch.matmul(attn_weights_dropped, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.d_model)
        return self.w_o(out), attn_weights


class _FeedForward(nn.Module):
    """Position-wise FFN applied independently per variable."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ITransformerBlock(nn.Module):
    """Single iTransformer layer (pre-norm formulation)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = _InvertedMultiHeadAttention(d_model, n_heads, dropout)
        self.ff = _FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, attn_weights)."""
        attn_out, attn_w = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_w


class ITransformerEncoder(nn.Module):
    """Full iTransformer encoder.

    Input:  (batch, n_vars, seq_length)
    Output: (batch, n_vars, d_model)  — per-variable embeddings
    """

    def __init__(self, cfg: ITransformerConfig, n_vars: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_vars = n_vars

        self.var_embedding = _VariableEmbedding(cfg.seq_length, cfg.d_model, cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                _ITransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the encoder.

        Args:
            x: (batch, n_vars, seq_length).

        Returns:
            Tuple of (embeddings, all_attn_weights) where embeddings
            is (batch, n_vars, d_model) and all_attn_weights is a list
            of (batch, n_heads, n_vars, n_vars) per layer.
        """
        tokens = self.var_embedding(x)
        all_attn: list[torch.Tensor] = []
        for block in self.blocks:
            tokens, attn_w = block(tokens)
            all_attn.append(attn_w)
        tokens = self.norm(tokens)
        return tokens, all_attn


# ======================================================================
# Feature extraction
# ======================================================================

_FEATURE_NAMES = ["log_return", "volume_chg", "volatility_20d", "rsi_14", "macd"]


def _extract_features(data: dict[str, AssetData], seq_length: int) -> tuple[np.ndarray, list[str]]:
    """Build the feature tensor from AssetData.

    For each asset computes 5 features and returns the last
    ``seq_length`` rows (after dropping warmup NaNs).

    Returns:
        Tuple of (features, symbols) where features has shape
        ``(n_assets, n_features, seq_length)``.
    """
    symbols: list[str] = []
    asset_features: list[np.ndarray] = []

    for symbol, asset in data.items():
        close = asset.ohlcv["close"].astype(np.float64)
        volume = asset.ohlcv["volume"].astype(np.float64)

        log_ret = np.log(close / close.shift(1))
        vol_chg = np.log(volume / volume.shift(1)).fillna(0.0)
        rvol = log_ret.rolling(20).std() * np.sqrt(252)
        rsi = _rsi(close, 14) / 50.0 - 1.0  # normalise to [-1, 1]
        macd = _macd_signal(close)

        feat_df = pd.concat([log_ret, vol_chg, rvol, rsi, macd], axis=1).dropna()

        if len(feat_df) < seq_length:
            logger.warning(
                "Skipping %s: only %d valid rows (need %d)",
                symbol,
                len(feat_df),
                seq_length,
            )
            continue

        # Take last seq_length rows.  Shape (seq_length, n_features).
        arr = feat_df.iloc[-seq_length:].values.astype(np.float32)
        # Transpose to (n_features, seq_length) — each feature is a token.
        asset_features.append(arr.T)
        symbols.append(symbol)

    if not asset_features:
        raise ValueError("No assets have sufficient data for iTransformer")

    # Stack: (n_assets, n_features, seq_length)
    return np.stack(asset_features, axis=0), symbols


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd_signal(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


# ======================================================================
# SignalGenerator wrapper
# ======================================================================


class ITransformerSignal(SignalGenerator):
    """iTransformer encoder wrapped as a SignalGenerator.

    Runs the iTransformer in **eval mode** (no training).  The encoder
    produces per-asset embeddings which are projected to scalar signal
    scores via a learned linear head.  Confidence is derived from the
    entropy of the last-layer attention weights (low entropy = high
    confidence that the model knows which variables matter).

    Args:
        config: Architecture hyperparameters.
    """

    def __init__(self, config: ITransformerConfig | None = None) -> None:
        self._cfg = config or ITransformerConfig()
        self._encoder: ITransformerEncoder | None = None
        self._signal_head: nn.Linear | None = None
        self._symbols: list[str] | None = None

    # ------------------------------------------------------------------
    # SignalGenerator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "itransformer"

    def generate(self, data: dict[str, AssetData]) -> Signal:
        """Extract features, run the iTransformer, and return a Signal.

        Args:
            data: Mapping of symbol to AssetData.

        Returns:
            Signal with one score per eligible asset.

        Raises:
            ValueError: If no assets have enough data.
        """
        features, symbols = _extract_features(data, self._cfg.seq_length)
        self._symbols = symbols
        n_assets = len(symbols)
        n_features = features.shape[1]

        # Lazily build the model to match the actual number of variables.
        # n_vars = n_features (each feature is a token, per-asset).
        self._ensure_model(n_features)

        # features: (n_assets, n_features, seq_length)
        x = torch.from_numpy(features).float()
        # Run each asset through the encoder independently.
        # The encoder's "variable" dimension is the feature dimension.
        scores, confidences = self._run_inference(x)

        return Signal(
            name=self.name,
            values=scores,
            confidence=confidences,
            regime=None,
            metadata={
                "symbols": symbols,
                "feature_names": _FEATURE_NAMES,
                "n_assets": n_assets,
                "d_model": self._cfg.d_model,
                "seq_length": self._cfg.seq_length,
            },
        )

    def update(self, new_data: dict[str, AssetData]) -> Signal:
        """Recompute the signal from new data.

        The iTransformer is stateless in eval mode, so ``update``
        simply delegates to ``generate``.
        """
        return self.generate(new_data)

    # ------------------------------------------------------------------
    # Internal: model management
    # ------------------------------------------------------------------

    def _ensure_model(self, n_vars: int) -> None:
        """Build or rebuild the encoder and signal head if needed."""
        if self._encoder is not None and self._encoder.n_vars == n_vars:
            return

        torch.manual_seed(self._cfg.random_state)
        self._encoder = ITransformerEncoder(self._cfg, n_vars=n_vars)
        self._encoder.eval()

        # Signal head: pool across variables, then project to scalar.
        self._signal_head = nn.Linear(self._cfg.d_model, 1)
        nn.init.xavier_uniform_(self._signal_head.weight)
        nn.init.zeros_(self._signal_head.bias)
        self._signal_head.eval()

    @torch.no_grad()
    def _run_inference(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Run the encoder and signal head.

        Args:
            x: (n_assets, n_features, seq_length).

        Returns:
            Tuple of (scores, confidences) — both 1-D arrays of length
            ``n_assets``.
        """
        n_assets = x.shape[0]
        scores_list: list[float] = []
        confidences_list: list[float] = []

        for i in range(n_assets):
            # Single asset: (1, n_features, seq_length)
            xi = x[i].unsqueeze(0)
            embeddings, all_attn = self._encoder(xi)
            # embeddings: (1, n_features, d_model)

            # Pool across features → (1, d_model)
            pooled = embeddings.mean(dim=1)

            # Project to scalar score.
            score = self._signal_head(pooled).squeeze()
            scores_list.append(float(score.item()))

            # Confidence from attention entropy (last layer).
            # Lower entropy → more peaked attention → higher confidence.
            attn = all_attn[-1]  # (1, n_heads, n_vars, n_vars)
            confidence = self._attention_confidence(attn)
            confidences_list.append(confidence)

        return np.array(scores_list, dtype=np.float64), np.array(confidences_list, dtype=np.float64)

    @staticmethod
    def _attention_confidence(attn_weights: torch.Tensor) -> float:
        """Derive a [0, 1] confidence from attention entropy.

        Low entropy means the model has peaked attention (knows which
        variables matter); we map that to high confidence.

        Args:
            attn_weights: (1, n_heads, n_vars, n_vars).

        Returns:
            Confidence value in [0, 1].
        """
        # Average over batch and heads.
        avg_attn = attn_weights.mean(dim=(0, 1))  # (n_vars, n_vars)
        n_vars = avg_attn.shape[0]

        # Shannon entropy of each row, normalised by max entropy.
        eps = 1e-10
        entropy_per_row = -(avg_attn * torch.log(avg_attn + eps)).sum(dim=-1)
        max_entropy = math.log(n_vars) if n_vars > 1 else 1.0
        normalised_entropy = float(entropy_per_row.mean().item()) / max_entropy

        # Invert: low entropy → high confidence.
        return float(np.clip(1.0 - normalised_entropy, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public: model access for downstream training
    # ------------------------------------------------------------------

    def get_encoder(self) -> ITransformerEncoder:
        """Return the underlying PyTorch encoder for training."""
        if self._encoder is None:
            raise RuntimeError("Call generate() first to initialise the encoder")
        return self._encoder

    def load_weights(self, path: str) -> None:
        """Load pretrained encoder weights from a checkpoint.

        Args:
            path: Path to a ``state_dict`` saved via ``torch.save``.
        """
        if self._encoder is None:
            raise RuntimeError("Call generate() first to initialise the encoder")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._encoder.load_state_dict(state)
        self._encoder.eval()
        logger.info("Loaded encoder weights from %s", path)
