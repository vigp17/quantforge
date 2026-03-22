# QuantForge

Modular quant trading system for research, backtesting, and paper trading.

## What is this

QuantForge consolidates four research projects — UARC (regime-adaptive RL portfolio), statistical arbitrage via Kalman filter pairs trading, HMM-based regime detection, and Monte Carlo simulation — into a single, testable system. The signal layer feeds into a portfolio agent, which is subject to a hard risk veto before any orders are sent to the broker. Each layer is defined by an abstract interface, making it straightforward to swap signals or allocation strategies via config.

## Backtest Results

Monthly rebalance, 5 bps transaction costs, 2020–2025.

| Universe | Total Return | Annual Return | Sharpe | Max DD |
|---|---|---|---|---|
| ETFs (SPY / QQQ / TLT / GLD / SHY) | +81.7% | +10.5% | 1.28 | -17.6% |
| Tech (SPY / QQQ / AAPL / NVDA / TLT) | +165.2% | +17.7% | 1.05 | -24.6% |

## Architecture

```
Yahoo Finance ──→  HMM Regime     ──→  IQN RL Agent    ──→  Paper Broker
FRED macro    ──→  Kalman Pairs   ──→  Risk Manager    ──→  Alpaca (live)
Alpaca API    ──→  iTransformer   ──→  Position Sizer  ──→  Order Manager
SQLite cache  ──→  Momentum       ──→  MV Optimizer    ──→  Fill Simulator
              ──→  Monte Carlo    ──→  Rebalancer
              ──→  Ensemble
```

Each layer has a `base.py` defining the abstract interface. Concrete implementations are swappable via YAML strategy configs.

## Signals

- **HMM Regime Detector** — Gaussian HMM over price/volume features, classifies market into N hidden states
- **Kalman Pairs** — Adaptive spread estimation for mean-reversion pairs trading
- **iTransformer** — Inverted-attention transformer encoder for multi-asset return forecasting
- **Momentum** — Cross-sectional momentum with configurable lookback and top-K selection
- **Monte Carlo** — Forward simulation of price paths for tail-risk estimation
- **Ensemble** — Weighted combination of the above with regime-conditioned blending

## Quick Start

```bash
pip install -r requirements.txt

# Run a backtest (fetches data from Yahoo Finance, caches locally)
python scripts/run_backtest.py --universe SPY QQQ TLT GLD SHY --start 2020-01-01 --end 2025-12-31

# Suppress walk-forward for a faster single-pass run
python scripts/run_backtest.py --no-walk-forward
```

HTML report is written to `data/results/backtest_report.html`.

## Tests

758 tests covering signals, portfolio engine, backtest framework, and integration pipelines.

```bash
pytest tests/ -v -m "not network"
```

## Built With

Python 3.13 · PyTorch · hmmlearn · yfinance · scipy · plotly · pytest
