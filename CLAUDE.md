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
