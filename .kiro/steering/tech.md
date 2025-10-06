# Technology Stack & Build System

## Core Dependencies

### Required Libraries
- **pandas** (>=2.0.0): Data manipulation and analysis
- **numpy** (>=1.20.0): Numerical computing
- **vectorbt** (>=0.25.0): Vectorized backtesting engine
- **pandas-ta** (>=0.3.14b0): Technical analysis indicators
- **plotly** (>=5.0.0): Interactive visualizations
- **PyYAML** (>=6.0.0): Configuration file parsing
- **ccxt** (>=4.0.0): Cryptocurrency exchange connectivity

### Optional Dependencies
- **TA-Lib** (>=0.4.25): For exact Freqtrade indicator compatibility
- **joblib** (>=1.0.0): Parallel processing for optimization

## Architecture Patterns

### Functional Strategy Pattern
- All strategies implement a simple `create_portfolio(data, params)` function
- Auto-discovery via `strategy_registry.py` scans `strategies/` folder
- Configuration-driven parameters via YAML/JSON files
- Minimal classes - prefer functions and direct data manipulation

### Data Source Abstraction
- Unified interface in `data_manager.py` supports multiple sources
- CSV files, CCXT exchanges, and Freqtrade data formats
- Automatic time range harmonization across symbols/timeframes

### VectorBT Integration
- Primary backtesting engine using `vbt.Portfolio.from_signals()`
- Signal-based approach with entry/exit conditions
- Built-in performance metrics and statistics

## Common Commands

### Running Analysis
```bash
# Interactive mode
python main.py

# Quick test mode
python main.py --quick <strategy_name>

# Full analysis mode
python main.py --quick <strategy_name> --full
```

### Testing
```bash
# Run all tests
python run_tests.py

# Test specific strategy
python -c "from main import quick_test; quick_test('chop')"
```

### Data Management
```bash
# Switch data source in config
python -c "from data_manager import switch_data_source; switch_data_source('config/chop.yaml', 'ccxt')"
```

## Development Guidelines

### File Structure
- `strategies/`: Strategy implementations with `create_portfolio()` function
- `config/`: YAML/JSON configuration files matching strategy names
- `data/`: CSV data files for backtesting
- Main modules: `main.py`, `data_manager.py`, `strategy_registry.py`, `optimizer.py`

### Code Style
- **Minimal OOP**: Use classes only when absolutely necessary (like CHOPStrategy for parameter management)
- **Direct execution**: Let exceptions bubble up, don't catch and handle everything
- **Simple functions**: Prefer pure functions over complex class hierarchies

### Code Philosophy
- **Direct approach**: Minimal error handling, fail fast when issues occur
- **Functional style**: Prefer functions over classes, minimal OOP
- **Simple dependencies**: Required libraries must be installed, no complex fallbacks

### Performance Considerations
- VectorBT vectorized operations for speed
- Parallel optimization using joblib
- Memory-efficient data loading with time range filtering