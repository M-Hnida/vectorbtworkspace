# Project Structure & Organization

## Root Directory Layout

```
├── main.py                 # Main entry point and analysis pipeline
├── data_manager.py         # Unified data loading (CSV/CCXT/Freqtrade)
├── strategy_registry.py    # Auto-discovery and strategy management
├── optimizer.py           # Parameter optimization with grid search
├── plotter.py             # Plotly-based visualization system
├── walk_forward.py        # Time-series cross-validation
├── vectoragent.py         # VectorBT integration utilities
├── requirements.txt       # Python dependencies
├── STYLE_GUIDE.md        # Code style conventions (French comments)
└── run_tests.py          # Test runner
```

## Core Directories

### `/strategies/`
Strategy implementations following the plugin pattern:
- Each strategy file must implement `create_portfolio(data, params)` function
- Auto-discovered by `strategy_registry.py`
- Examples: `chop.py`, `rsi.py`, `momentum.py`, `orb.py`, `tdi.py`
- `__init__.py` for package initialization

### `/config/`
Configuration files matching strategy names:
- **YAML format**: `strategy_name.yaml` (preferred)
- **JSON format**: `strategy_name_freqtrade.json` (Freqtrade compatibility)
- Contains: parameters, optimization_grid, data_requirements
- Examples: `chop.yaml`, `rsi.yaml`, `momentum.yaml`

### `/data/`
Market data files in CSV format:
- Naming convention: `SYMBOL_TIMEFRAME_PERIOD.csv`
- Examples: `BTCUSDT_15m.csv`, `EURUSD_1H_2009-2025.csv`
- Auto-detected by data manager for backtesting

## Architecture Patterns

### Functional Strategy System
1. **Auto-Discovery**: `strategy_registry.py` scans `/strategies/` folder
2. **Simple Interface**: All strategies implement `create_portfolio(data, params)` function
3. **Configuration Binding**: Matches `config/{strategy_name}.yaml`
4. **Direct Parameter Passing**: No complex parameter objects, use simple dicts

### Data Source Abstraction
1. **Unified Interface**: `load_data_for_strategy()` handles all sources
2. **Source Detection**: Auto-detects CSV/CCXT/Freqtrade based on config
3. **Time Harmonization**: Aligns multiple symbols/timeframes
4. **Flexible Filtering**: Time range and symbol filtering

### Analysis Pipeline
1. **Entry Point**: `main.py` orchestrates full analysis
2. **Data Loading**: Multi-source data via `data_manager.py`
3. **Strategy Execution**: Portfolio creation via `strategy_registry.py`
4. **Optimization**: Grid search via `optimizer.py`
5. **Validation**: Walk-forward analysis via `walk_forward.py`
6. **Visualization**: Comprehensive plots via `plotter.py`

## File Naming Conventions

### Strategy Files
- Lowercase with underscores: `strategy_name.py`
- Match config file names exactly
- Class names use PascalCase: `CHOPStrategy`, `RSIStrategy`

### Configuration Files
- YAML preferred: `strategy_name.yaml`
- Freqtrade compatibility: `strategy_name_freqtrade.json`
- Must match strategy file name (without extension)

### Data Files
- Format: `SYMBOL_TIMEFRAME_PERIOD.csv`
- Examples: `BTCUSD_1h_2011-2025.csv`, `EURUSD_15m_2021-2025.csv`
- Auto-detected by symbol extraction from filename

## Module Dependencies

### Core Flow
```
main.py
├── data_manager.py (data loading)
├── strategy_registry.py (strategy management)
├── optimizer.py (parameter optimization)
├── walk_forward.py (validation)
└── plotter.py (visualization)
```

### Strategy Dependencies
```
strategies/*.py
├── pandas (data manipulation)
├── vectorbt (backtesting)
├── pandas-ta (indicators)
└── talib (optional, Freqtrade compatibility)
```

## Extension Points

### Adding New Strategies
1. Create `strategies/new_strategy.py` with `create_portfolio(data, params)` function
2. Add `config/new_strategy.yaml` with parameters and optimization grid
3. Strategy auto-discovered on next run - no registration needed

### Adding New Data Sources
1. Add new `_load_*_data_internal()` function to `data_manager.py`
2. Update `_detect_data_source()` logic
3. Keep it simple - direct implementation, no abstraction layers

### Adding New Analysis Types
1. Create new module with simple functions
2. Import and call directly from `main.py`
3. No complex inheritance or plugin systems