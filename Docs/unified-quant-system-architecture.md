# QuantForge — Architecture

**Unified quant trading system for research, backtesting, and paper trading.**

---

## Motivation

I had several quant projects sitting in separate repos — an HMM regime detector, a Kalman filter stat arb system, an RL portfolio agent (UARC), Monte Carlo forecasting — all doing related things but not talking to each other. QuantForge consolidates them into one platform with clean interfaces so I can mix and match strategies, backtest properly, and eventually paper trade.

The goal is a system I'd actually trust to run overnight, not just a notebook that looks good in a screenshot.

---

## What's Being Consolidated

| Project | What it contributes |
|---------|-------------------|
| **UARC** (vigp17/UARC-Portfolio-RL) | Bayesian HMM, iTransformer encoder, FiLM conditioning, IQN distributional RL agent |
| **Statistical Arbitrage** | Kalman filter pairs trading, spread z-score signals |
| **HMM Regime Detection** | Market state classification — Sharpe 0.86 vs 0.44 buy-and-hold |
| **Monte Carlo Forecasting** | Forward price simulation, VaR/CVaR estimation |

Not everything from these projects is worth porting. The MAAT multi-agent system used LangGraph which added complexity without much benefit — I'm keeping the orchestration ideas but dropping the framework dependency. The CAIF project (Constitutional AI) might be useful for sentiment signals later, but that's a stretch goal.

---

## Architecture

```
DATA LAYER          SIGNAL ENGINE         PORTFOLIO ENGINE        EXECUTION
                                                                    
Yahoo Finance ──→  HMM Regime     ──→   IQN RL Agent      ──→  Paper Broker
FRED macro    ──→  Kalman Pairs   ──→   Risk Manager       ──→  Alpaca (live)
Alpaca API    ──→  iTransformer   ──→   Position Sizer     ──→  Order Manager
SQLite cache  ──→  Momentum       ──→   MV Optimizer       ──→  Fill Simulator
              ──→  Monte Carlo    ──→   Rebalancer
              ──→  Ensemble
                                            │
                                            ▼
                                    MONITORING
                                    • Drift detection (live vs backtest)
                                    • Risk dashboard
                                    • Kill switch
```

The signal engine and portfolio engine communicate through two dataclasses (`Signal` and `PortfolioAction`). Everything else is implementation detail. This makes it easy to swap in a new signal or a different allocation strategy without touching the rest of the pipeline.

---

## Directory Layout

```
quantforge/
├── src/
│   ├── data/           # Providers (Yahoo, FRED, Alpaca), SQLite cache, feature engineering
│   ├── signals/        # HMM regime, Kalman pairs, iTransformer, momentum, Monte Carlo, ensemble
│   ├── portfolio/      # IQN agent, FiLM conditioning, risk manager, position sizer, optimizer
│   ├── execution/      # Paper broker, Alpaca broker, order management, fill simulation
│   ├── backtest/       # Backtest engine, metrics, walk-forward validation, reporting
│   ├── monitoring/     # Dashboard, alerts, drift detection
│   └── research/       # Notebooks, hypothesis testing, feature importance
├── tests/              # Mirrors src/ structure + integration tests
├── configs/            # Strategy YAML files, risk limit configs
├── scripts/            # CLI runners for backtest, paper trading, training
└── data/               # Cache, model checkpoints, results
```

Each layer has a `base.py` with abstract interfaces. Concrete implementations can be swapped via strategy config files.

---

## Core Interfaces

### Signal

```python
@dataclass
class Signal:
    name: str
    values: np.ndarray       # signal strength per asset
    confidence: np.ndarray   # [0, 1] confidence per signal
    regime: str | None       # detected regime if applicable
    metadata: dict
```

All signal generators implement `generate(data) -> Signal` and `update(new_data) -> Signal` (for incremental live updates). The ensemble layer combines multiple signals using confidence-weighted averaging or majority vote.

### Portfolio Action

```python
@dataclass
class PortfolioAction:
    weights: dict[str, float]  # symbol -> target weight, sum <= 1.0
    confidence: float
    regime_context: str
    risk_metrics: dict
```

Portfolio agents implement `decide(signals, current_portfolio) -> PortfolioAction`. The risk manager validates every action before it reaches the execution layer — it has veto power over the agent.

### Execution

```python
@dataclass
class Order:
    symbol: str
    side: str           # "buy" | "sell"
    quantity: float
    order_type: str     # "market" | "limit"
    limit_price: float | None
```

Brokers implement `submit_order()`, `get_positions()`, `get_portfolio_value()`. The paper broker simulates fills with configurable slippage. Same interface for live trading via Alpaca — the only difference is a `--live` flag.

---

## Risk Management

This is the part I'm most careful about. A system that doesn't manage risk isn't a trading system, it's a random number generator with extra steps.

### Hard Limits

| Limit | Threshold | What happens |
|-------|-----------|-------------|
| Single position size | 30% of portfolio | Order rejected |
| Daily loss | 3% | Flatten everything, halt trading |
| Drawdown from peak | 15% | Flatten everything, halt until manual review |
| Pairwise correlation | > 0.7 | Reduce the smaller position |
| Leverage | > 1.0x | Order rejected |

### Soft Limits

| Metric | Threshold | What happens |
|--------|-----------|-------------|
| Sharpe drift from backtest | > 1σ | Alert |
| Signal disagreement | > 60% conflicting | Reduce position sizes |
| Data staleness | > 15 min | Alert, use last known data |
| Model inference latency | > 5s | Fall back to momentum |

### Kill Switch

Runs as an independent watchdog, not inside the trading loop. If the main process crashes, the kill switch still operates. Triggers on daily loss > 3%, drawdown > 15%, or data feed going stale. Action is always the same: flatten all positions, cancel pending orders, send alert, halt until manual restart.

I'm keeping this intentionally simple. Fancy risk systems that try to be clever about when to flatten are just adding more things that can break.

---

## Key Technical Decisions

**iTransformer subsumes HMM regime info.** This was the main finding from the UARC research. The iTransformer encoder's 60-day lookback window implicitly captures the same regime structure as an explicit Bayesian HMM. All four agent variants (with/without regime, DQN/IQN) converged to the same Sharpe. I'm keeping both signal generators for ablation, but the ensemble should downweight the HMM when the iTransformer is active.

**Filtering posteriors, not Viterbi.** The HMM uses the forward algorithm to get p(z_t | x_{1:t}) — causal posteriors that don't peek into the future. Viterbi gives you the globally optimal state sequence but requires the full observation history, which is cheating in a live trading context.

**IQN over DQN.** The IQN agent models the full return distribution, not just the expected value. This matters for risk-aware allocation — you can make decisions based on tail risk (CVaR) rather than just mean return. The quantile embedding approach (cosine basis functions) is elegant and adds minimal overhead.

**FiLM conditioning as identity init.** The FiLM layer initializes gamma=1, beta=0 so it starts as a pass-through. This means the system works without regime conditioning and FiLM only adds modulation once it's learned something useful during training. Avoids the cold-start problem.

**Paper trading is the default.** There's no way to accidentally trade real money. Live mode requires an explicit flag and confirmation.

---

## Open Questions

Things I'm still figuring out:

- **Transaction cost modeling** — using a flat slippage in bps for now, but real market impact depends on order size relative to ADV. Not sure how much this matters at the scale I'd be trading.

- **Retraining frequency** — the IQN agent and iTransformer encoder will need periodic retraining as market conditions shift. Haven't decided on the trigger yet (calendar-based vs drift-based).

- **Pair universe selection** — the Kalman filter works great for a known pair, but automated pair discovery via cointegration screening has a massive multiple testing problem. Might need to constrain the search space (same sector, related ETFs).

- **Feature engineering for the iTransformer** — UARC used 6 features (multi-scale returns, vol, RSI, MACD). This was somewhat arbitrary. Should probably do a proper feature importance study before committing to a final feature set.

- **Whether to add a sentiment signal** — could use the CAIF project's LLM pipeline to parse financial news, but the signal-to-noise ratio of NLP sentiment in trading is... debatable. Parking this for now.

---

## Strategy Config Example

```yaml
name: "UARC Regime-Conditioned RL"
universe: ["SPY", "QQQ", "TLT", "GLD", "SHY"]
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
  - type: "momentum"

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
  auto_shutdown: true

execution:
  broker: "paper"
  slippage_bps: 5
```

---

*Last updated: March 2026*