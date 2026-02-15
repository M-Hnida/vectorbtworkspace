# VectorFlow

A structured workspace for organizing and running VectorBT trading strategies. Provides a clean project structure, CLI interface, and enhanced visualizations.

## What It Actually Does

- **Organizes your strategies**: Keep all your VectorBT strategies in one place with a consistent structure
- **Loads CSV data**: Simple data loader for OHLCV CSV files  
- **Runs backtests**: Wrapper around VectorBT's backtesting with YAML config support
- **Adds trade markers**: Enhanced visualizations showing entry/exit points with PnL coloring
- **CLI interface**: Interactive and command-line modes to run strategies

## What It Doesn't Do (Yet)

The README originally promised features that either don't exist or are very basic wrappers:
- "Advanced" parameter optimization - just basic grid search
- "Production-grade" - this is a personal workspace tool
- Monte Carlo validation - basic wrappers exist but aren't robust
- Walk-forward analysis - implemented but not thoroughly tested

## Project Structure

```
vectorflow/
├── core/              # Data loading and portfolio creation
│   ├── config_manager.py    # YAML config loading
│   ├── data_loader.py      # CSV data ingestion
│   └── portfolio_builder.py # Strategy discovery & portfolio creation
├── strategies/        # Your trading strategies
│   ├── vol_breakout.py     # Example strategy
│   ├── ema_rsi_trend.py    # Example strategy
│   └── strategy_template.py # Template for new strategies
├── optimization/      # Parameter optimization (basic grid search)
├── validation/       # Validation tools (MC, walk-forward)
├── utils/           # Utilities
└── visualization/   # Enhanced plotting with trade markers
```

## Installation

```bash
pip install -e .
```

## Usage

### CLI (Interactive)

```bash
python -m vectorflow.cli
```

Then follow the prompts to select strategy, time range, and analysis mode.

### CLI (Direct)

```bash
python -m vectorflow.cli vol_breakout
python -m vectorflow.cli vol_breakout --full
python -m vectorflow.cli ema_rsi_trend --walkforward
```

### Python API

```python
from vectorflow import create_portfolio, load_ohlc_csv

# Load data
data = load_ohlc_csv("data/EURUSD_1h.csv")

# Run backtest
portfolio = create_portfolio("vol_breakout", data, {
    "entry_threshold": 0.5,
    "stop_atr_mult": 2.0,
})

# Check results
print(portfolio.stats())

# Plot with trade markers
from vectorflow.visualization.indicators import add_trade_signals
import plotly.graph_objects as go

fig = portfolio.plot()
fig = add_trade_signals(portfolio, fig, show_direction=True)
fig.show()
```

## Configuration

Create a YAML config for each strategy in `config/`:

```yaml
# config/vol_breakout.yaml
parameters:
  entry_threshold: 0.5
  stop_atr_mult: 2.0
  target_atr_mult: 4.0
  initial_cash: 10000
  fee: 0.0004

csv_path: "data/your_data.csv"

optimization_grid:
  entry_threshold: [0.3, 0.7, 0.1]  # [start, end, step]
  stop_atr_mult: [1.5, 3.0, 0.5]
```

## Adding a New Strategy

1. Copy `vectorflow/strategies/strategy_template.py`
2. Rename and implement your logic
3. Create a YAML config in `config/`
4. Run it: `python -m vectorflow.cli your_strategy`

## Requirements

- Python 3.8+
- numpy, pandas
- vectorbt
- plotly
- pyyaml

## License

MIT
