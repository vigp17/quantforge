# Unified Quantitative Trading System — Architecture Blueprint

## Project: QuantForge
**Author:** Vignesh Pai  
**GitHub:** vigp17  
**Date:** March 2026  

---

## 1. Vision

Consolidate existing quant projects (UARC, Statistical Arbitrage, HMM Regime Detection, Monte Carlo Forecasting) into a single modular platform for research, backtesting, paper trading, and eventual live deployment. Designed to serve double duty: a functional trading research platform and a portfolio centerpiece for quant firm applications.

---

## 2. Existing Assets to Integrate

| Project | Core Components | What It Contributes |
|---------|----------------|---------------------|
| **UARC** (vigp17/UARC-Portfolio-RL) | Bayesian HMM, iTransformer encoder, FiLM conditioning, IQN distributional RL | Portfolio allocation agent, regime detection, uncertainty-aware decision making |
| **Statistical Arbitrage** | Kalman filter pairs trading | Mean-reversion signal generation, spread modeling |
| **HMM Regime Detection** | Hidden Markov Model (Sharpe 0.86 vs 0.44 buy-and-hold) | Market state classification (bull/bear/sideways) |
| **Monte Carlo Forecasting** | Monte Carlo simulation for SPY | Forward price distribution, risk scenario modeling |
| **MAAT** | Multi-agent system via LangGraph | Agent orchestration patterns (reuse architecture, not LangGraph dependency) |
| **CAIF** (vigp17/Caif) | Constitutional AI pipeline, GRPO, LoRA | Potential use for NLP-based sentiment signals or LLM-driven research agents |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUANTFORGE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  DATA LAYER  │  │   SIGNAL     │  │   PORTFOLIO ENGINE   │   │
│  │              │  │   ENGINE     │  │                      │   │
│  │ • Yahoo Fin  │→ │              │→ │ • IQN RL Agent       │   │
│  │ • FRED       │  │ • HMM Regime │  │ • Risk Manager       │   │
│  │ • Alpaca API │  │ • Kalman     │  │ • Position Sizer     │   │
│  │ • Alt Data   │  │ • iTransfmr  │  │ • Rebalancer         │   │
│  │              │  │ • Monte Carlo│  │                      │   │
│  └──────────────┘  │ • Momentum   │  └──────────┬───────────┘   │
│                     │ • Sentiment  │             │               │
│                     └──────────────┘             ▼               │
│                                        ┌──────────────────┐     │
│  ┌──────────────────────────────────┐  │  EXECUTION LAYER │     │
│  │        MONITORING & DASHBOARD    │  │                  │     │
│  │                                  │  │ • Paper Trading  │     │
│  │ • Live vs Backtest comparison    │← │ • Alpaca Broker  │     │
│  │ • Regime state visualization     │  │ • Order Manager  │     │
│  │ • Risk metrics (VaR, drawdown)   │  │ • Fill Simulator │     │
│  │ • Signal attribution             │  │                  │     │
│  │ • Alerting (Slack/email)         │  └──────────────────┘     │
│  └──────────────────────────────────┘                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    RESEARCH SANDBOX                       │   │
│  │  • Strategy prototyping    • Walk-forward optimization    │   │
│  │  • Hypothesis testing      • Regime-conditioned backtest  │   │
│  │  • Feature importance      • Monte Carlo stress testing   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Directory Structure

```
quantforge/
├── CLAUDE.md                     # Claude Code project config
├── pyproject.toml                # Project metadata, dependencies
├── README.md
├── .env.example                  # API keys template
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                     # DATA LAYER
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract DataProvider interface
│   │   ├── yahoo.py              # Yahoo Finance provider
│   │   ├── fred.py               # FRED macro data provider
│   │   ├── alpaca.py             # Alpaca real-time + historical
│   │   ├── cache.py              # Local SQLite caching layer
│   │   └── features.py           # Feature engineering pipeline
│   │
│   ├── signals/                  # SIGNAL ENGINE
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract Signal interface
│   │   ├── hmm_regime.py         # Bayesian HMM regime detection
│   │   ├── kalman_pairs.py       # Kalman filter stat arb signals
│   │   ├── itransformer.py       # iTransformer encoder features
│   │   ├── momentum.py           # Momentum / trend signals
│   │   ├── montecarlo.py         # Monte Carlo forward simulation
│   │   └── ensemble.py           # Signal combination / voting
│   │
│   ├── portfolio/                # PORTFOLIO ENGINE
│   │   ├── __init__.py
│   │   ├── iqn_agent.py          # IQN distributional RL agent
│   │   ├── film_conditioning.py  # FiLM regime conditioning layer
│   │   ├── risk_manager.py       # VaR, drawdown limits, correlation
│   │   ├── position_sizer.py     # Kelly criterion, volatility-scaled
│   │   ├── rebalancer.py         # Rebalancing logic + constraints
│   │   └── optimizer.py          # Mean-variance / risk parity fallback
│   │
│   ├── execution/                # EXECUTION LAYER
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract Broker interface
│   │   ├── paper.py              # Paper trading with simulated fills
│   │   ├── alpaca_broker.py      # Live Alpaca execution
│   │   ├── order_manager.py      # Order lifecycle, retry, cancel
│   │   └── fill_simulator.py     # Slippage + market impact model
│   │
│   ├── backtest/                 # BACKTESTING FRAMEWORK
│   │   ├── __init__.py
│   │   ├── engine.py             # Core backtest loop
│   │   ├── metrics.py            # Sharpe, Sortino, max DD, Calmar
│   │   ├── walk_forward.py       # Walk-forward validation
│   │   └── report.py             # Generate backtest reports
│   │
│   ├── monitoring/               # MONITORING & DASHBOARD
│   │   ├── __init__.py
│   │   ├── dashboard.py          # Streamlit/Dash dashboard
│   │   ├── alerts.py             # Slack/email alerting
│   │   ├── drift_detector.py     # Live vs backtest drift detection
│   │   └── logger.py             # Structured logging
│   │
│   └── research/                 # RESEARCH SANDBOX
│       ├── __init__.py
│       ├── notebooks/            # Jupyter notebooks for exploration
│       ├── hypothesis.py         # Hypothesis testing framework
│       └── feature_importance.py # SHAP / permutation importance
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Shared fixtures (sample data, etc.)
│   ├── test_data/
│   ├── test_signals/
│   ├── test_portfolio/
│   ├── test_execution/
│   ├── test_backtest/
│   └── integration/              # End-to-end pipeline tests
│
├── configs/
│   ├── default.yaml              # Default strategy parameters
│   ├── uarc_strategy.yaml        # UARC-style RL allocation
│   ├── pairs_strategy.yaml       # Kalman pairs trading
│   └── risk_limits.yaml          # Risk management thresholds
│
├── scripts/
│   ├── run_backtest.py           # CLI backtest runner
│   ├── run_paper_trading.py      # Start paper trading session
│   ├── train_agent.py            # Train RL agent
│   └── generate_report.py        # Generate performance report
│
└── data/
    ├── cache/                    # SQLite cached market data
    ├── models/                   # Trained model checkpoints
    └── results/                  # Backtest results, logs
```

---

## 5. Core Interfaces (Design Contracts)

### 5.1 Data Provider

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class AssetData:
    symbol: str
    ohlcv: pd.DataFrame        # columns: open, high, low, close, volume
    metadata: dict              # exchange, sector, etc.

class DataProvider(ABC):
    @abstractmethod
    def fetch_historical(self, symbol: str, start: str, end: str) -> AssetData:
        ...

    @abstractmethod
    def fetch_realtime(self, symbol: str) -> AssetData:
        ...

    @abstractmethod
    def fetch_universe(self, symbols: list[str], start: str, end: str) -> dict[str, AssetData]:
        ...
```

### 5.2 Signal Generator

```python
import torch
import numpy as np

@dataclass
class Signal:
    name: str
    values: np.ndarray          # signal strength per asset
    confidence: np.ndarray      # [0, 1] confidence per signal
    regime: str | None          # current detected regime
    metadata: dict

class SignalGenerator(ABC):
    @abstractmethod
    def generate(self, data: dict[str, AssetData]) -> Signal:
        ...

    @abstractmethod
    def update(self, new_data: dict[str, AssetData]) -> Signal:
        """Incremental update for live trading."""
        ...
```

### 5.3 Portfolio Agent

```python
@dataclass
class PortfolioAction:
    weights: dict[str, float]   # symbol -> target weight
    confidence: float           # agent confidence in allocation
    regime_context: str         # regime used for decision
    risk_metrics: dict          # VaR, expected drawdown

class PortfolioAgent(ABC):
    @abstractmethod
    def decide(self, signals: list[Signal], current_portfolio: dict) -> PortfolioAction:
        ...

    @abstractmethod
    def train(self, historical_signals: list[Signal], returns: pd.DataFrame) -> dict:
        """Returns training metrics."""
        ...
```

### 5.4 Broker Interface

```python
@dataclass
class Order:
    symbol: str
    side: str                   # "buy" | "sell"
    quantity: float
    order_type: str             # "market" | "limit"
    limit_price: float | None

@dataclass
class Fill:
    order: Order
    fill_price: float
    fill_quantity: float
    slippage: float
    timestamp: str

class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Returns order ID."""
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        ...

    @abstractmethod
    def get_portfolio_value(self) -> float:
        ...
```

---

## 6. Key Integration Points

### 6.1 UARC → QuantForge

| UARC Component | QuantForge Location | Migration Notes |
|----------------|--------------------|--------------------|
| Bayesian HMM | `signals/hmm_regime.py` | Extract from UARC, make stateless with `generate()` / `update()` interface |
| iTransformer encoder | `signals/itransformer.py` | Key finding: subsumes HMM regime info. Keep as standalone encoder, but note redundancy |
| FiLM conditioning | `portfolio/film_conditioning.py` | Modulates RL agent; keep coupled with IQN agent |
| IQN agent | `portfolio/iqn_agent.py` | Core allocation agent. Add action masking for position limits |

### 6.2 Stat Arb → QuantForge

| Component | QuantForge Location | Migration Notes |
|-----------|--------------------|--------------------|
| Kalman filter | `signals/kalman_pairs.py` | Pairs selection + spread z-score signals |
| Pair universe | `data/features.py` | Cointegration screening as feature engineering step |

### 6.3 Strategy Configs

Each strategy is a YAML file that wires together signals → portfolio agent → risk limits:

```yaml
# configs/uarc_strategy.yaml
name: "UARC Regime-Conditioned RL"
universe: ["SPY", "QQQ", "TLT", "GLD", "VIX"]
rebalance_frequency: "daily"

signals:
  - type: "hmm_regime"
    params:
      n_states: 3
      lookback_days: 252
  - type: "itransformer"
    params:
      seq_length: 60
      d_model: 128

portfolio:
  agent: "iqn"
  params:
    quantile_samples: 32
    risk_aversion: 0.5
    film_conditioning: true

risk:
  max_position_pct: 0.30
  max_drawdown_pct: 0.15
  daily_loss_limit_pct: 0.03
  correlation_limit: 0.7
  auto_shutdown: true           # kill switch if limits breached

execution:
  broker: "paper"               # "paper" | "alpaca"
  slippage_bps: 5
  commission_per_trade: 0.0
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
**Goal:** Project scaffolding + data layer + port HMM signal

Claude Code prompts:
```
> Set up the quantforge project with pyproject.toml, pytest, 
  src/ layout as specified in the architecture doc. 
  Use Python 3.13, PyTorch 2.x, pydantic for configs.

> Implement the DataProvider interface and Yahoo Finance provider. 
  Add SQLite caching so we don't re-download data. Write tests.

> Port my HMM regime detection code from UARC into 
  signals/hmm_regime.py following the SignalGenerator interface. 
  The original is at ~/UARC-Portfolio-RL/src/... 
  Adapt it to the new interface and write tests.
```

### Phase 2: Signal Engine (Week 2-3)
**Goal:** All signal generators working, ensemble logic

```
> Port the Kalman filter pairs trading from my stat arb project 
  into signals/kalman_pairs.py. Follow the SignalGenerator interface.

> Implement the iTransformer encoder signal. Port from UARC and 
  make it work as a standalone feature extractor.

> Build the signal ensemble in signals/ensemble.py. Support 
  equal-weight voting, confidence-weighted, and stacking.

> Add momentum signals (12-1 month momentum, 52-week high proximity).
```

### Phase 3: Portfolio Engine (Week 3-4)
**Goal:** IQN agent ported, risk management in place

```
> Port the IQN distributional RL agent from UARC. 
  Implement the PortfolioAgent interface. Include FiLM conditioning.

> Build risk_manager.py with VaR calculation (historical + parametric), 
  max drawdown tracking, correlation monitoring, and the auto-shutdown 
  kill switch.

> Implement Kelly criterion and volatility-scaled position sizing.

> Build a mean-variance optimizer as a fallback when the RL agent 
  has low confidence.
```

### Phase 4: Backtesting (Week 4-5)
**Goal:** Full backtest framework with walk-forward validation

```
> Build the backtest engine that takes a strategy config YAML, 
  runs signals → portfolio → execution in a loop over historical data. 
  Include realistic transaction costs.

> Implement walk-forward validation: train on expanding window, 
  test on next period, roll forward.

> Add Monte Carlo stress testing: resample historical returns, 
  run 1000 paths, report distribution of outcomes.

> Generate HTML backtest reports with charts using plotly.
```

### Phase 5: Paper Trading (Week 5-6)
**Goal:** Live paper trading with monitoring

```
> Implement the paper trading broker with simulated fills 
  and slippage modeling.

> Integrate Alpaca API for real-time data feeds into the 
  paper trading loop.

> Build the Streamlit monitoring dashboard showing: 
  current positions, P&L, regime state, signal attribution, 
  live vs backtest performance comparison.

> Add drift detection: alert when live Sharpe deviates from 
  backtested Sharpe by more than 1 standard deviation.
```

### Phase 6: Hardening (Week 6-8)
**Goal:** Production-ready reliability

```
> Add comprehensive error handling: network failures, 
  data gaps, model inference errors. The system should 
  never crash — it should go to a safe state (flatten positions).

> Write integration tests that simulate a full trading day 
  from data fetch through execution.

> Add structured logging with correlation IDs for debugging.

> Set up the kill switch: if daily loss exceeds 3% or drawdown 
  exceeds 15%, flatten all positions and alert immediately.
```

---

## 8. CLAUDE.md for This Project

Save this as `CLAUDE.md` in the project root:

```markdown
# QuantForge — Unified Quant Trading System

## Tech Stack
- Python 3.13, PyTorch 2.x, NumPy, pandas
- Testing: pytest with fixtures in conftest.py
- Config: pydantic + YAML (configs/ directory)
- Dashboard: Streamlit
- Broker: Alpaca API (paper and live)
- Data storage: SQLite for caching

## Architecture
- All components follow abstract interfaces in `base.py` files
- Signals implement `SignalGenerator` (see src/signals/base.py)
- Portfolio agents implement `PortfolioAgent` (see src/portfolio/base.py — not created yet, use the interface from architecture doc)
- Brokers implement `Broker` (see src/execution/base.py)
- Strategy configs are YAML files in configs/

## Conventions
- Type hints on all functions
- Google-style docstrings
- Tests mirror src/ structure in tests/
- Use pydantic for all config/data classes
- Use logging module, never print()
- Tensor operations stay on CPU unless explicitly GPU-enabled

## Commands
- Run tests: `pytest tests/ -v`
- Run specific: `pytest tests/test_signals/ -v`
- Lint: `ruff check src/`
- Format: `ruff format src/`
- Backtest: `python scripts/run_backtest.py --config configs/uarc_strategy.yaml`

## Key Design Decisions
- iTransformer encoder implicitly subsumes HMM regime information (UARC finding). Both signals are kept for comparison/ablation, but the ensemble should note this redundancy.
- Risk management has veto power over portfolio agent. If risk limits are breached, positions are flattened regardless of agent signal.
- Paper trading is the default. Live trading requires explicit --live flag and confirmation.

## Important Context
- This consolidates code from: UARC-Portfolio-RL, stat-arb, and HMM regime projects
- The IQN agent uses quantile regression for distributional RL
- FiLM conditioning modulates the RL agent based on regime embeddings
- Kalman filter pairs trading generates mean-reversion signals
```

---

## 9. Risk Management Specification

This deserves special attention — it's what separates a toy from a real system.

### Hard Limits (Non-Negotiable)
| Limit | Threshold | Action |
|-------|-----------|--------|
| Max single position | 30% of portfolio | Reject order |
| Daily loss | 3% of portfolio value | Flatten all positions, halt trading |
| Max drawdown | 15% from peak | Flatten all, halt until manual review |
| Correlation between positions | > 0.7 pairwise | Reduce smaller position |
| Max leverage | 1.0x (no leverage initially) | Reject order |

### Soft Limits (Alerts)
| Metric | Threshold | Action |
|--------|-----------|--------|
| Sharpe drift from backtest | > 1.0 std dev | Slack/email alert |
| Signal disagreement | > 60% signals conflicting | Log warning, reduce position size |
| Data staleness | > 15 min delay | Alert, use last known data |
| Model inference latency | > 5 seconds | Fall back to simple momentum |

### Kill Switch Implementation
```python
class KillSwitch:
    """
    Emergency position flattener.
    Runs as an independent watchdog — not inside the trading loop.
    If the main process crashes, the kill switch still runs.
    """
    def check(self, portfolio_state: PortfolioState) -> bool:
        """Returns True if emergency flatten is needed."""
        if portfolio_state.daily_pnl_pct < -0.03:
            return True
        if portfolio_state.drawdown_from_peak_pct > 0.15:
            return True
        if not portfolio_state.data_feed_alive:
            return True
        return False
```

---

## 10. Interview Talking Points

This system is designed to demonstrate the following to quant firms:

1. **Full pipeline ownership** — data ingestion through execution, not just a model in a notebook
2. **Research rigor** — walk-forward validation, regime-conditioned backtesting, Monte Carlo stress testing
3. **Production mindset** — risk management, kill switches, drift detection, structured logging
4. **Original research** — the iTransformer/HMM redundancy finding from UARC, distributional RL for portfolio allocation
5. **Software engineering quality** — clean interfaces, comprehensive tests, YAML configs, modular architecture
6. **Honest performance attribution** — tracking live vs backtest divergence, not just reporting cherry-picked backtests

When discussing with firms like Jane Street, SIG, or Two Sigma, frame it as:
> "I built a modular research and paper trading platform that consolidates several of my quant research projects. The most interesting finding was from my UARC work — the iTransformer encoder implicitly captures the same regime information as an explicit Bayesian HMM, which challenges the assumption that you need separate regime detection modules. The system has been paper trading for N months with a live Sharpe of X vs backtested Y, and here's what I learned from the divergence..."

---

*This document serves as both the architecture specification and the blueprint for Claude Code-assisted development. Each phase includes the exact prompts to use.*
