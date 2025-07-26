# Trading Strategy Framework

A scalable and modular framework for backtesting trading strategies with comprehensive analysis tools.

## 🏗️ Architecture

### Core Components
- **`core/io.py`** - Data loading and management (MarketData class)
- **`core/backtest.py`** - Portfolio backtesting engine
- **`core/optimizer.py`** - Grid search parameter optimization
- **`core/metrics.py`** - Performance metrics calculation
- **`core/plotting.py`** - Visualization and dashboard generation
- **`core/indicators.py`** - Technical indicator calculations
- **`core/monte_carlo.py`** - Statistical validation via permutation tests
- **`core/walkforward.py`** - Walk-forward analysis for robustness testing

### Strategy Implementation
- **`core/strategies/`** - Strategy signal generation modules
  - `momentum_signals.py` - Volatility momentum strategy
  - `lti_signals.py` - Logical Trading Indicator strategy  
  - `orb_signals.py` - Opening Range Breakout strategy

### Configuration
- **`config/`** - YAML configuration files for each strategy
  - `momentum.yaml` - Momentum strategy parameters
  - `lti.yaml` - LTI strategy parameters
  - `orb.yaml` - ORB strategy parameters

## 🚀 Usage

### Running a Strategy
```bash
# Edit runner.py to set STRATEGY_NAME
python runner.py
```

### Available Strategies
1. **Momentum** - Volatility-based momentum strategy
2. **LTI** - Logical Trading Indicator with Bollinger Bands and Keltner Channels
3. **ORB** - Opening Range Breakout strategy

### Data Format
Place CSV files in `data/` directory with naming pattern:
- `SYMBOL_TIMEFRAME_DATERANGE.csv` (e.g., `EURUSD_1H_2009-2025.csv`)
- Supports both tab-separated and comma-separated formats
- Headers optional (auto-detected)

## 📊 Analysis Pipeline

Each strategy execution includes:

1. **Data Loading** - Multi-timeframe data harmonization
2. **Optimization** - Grid search on training data (70%)
3. **Backtesting** - Train/test/full dataset analysis
4. **Walk-Forward Analysis** - Robustness testing across time windows
5. **Monte Carlo Testing** - Statistical significance validation
6. **Visualization** - Comprehensive performance dashboards

## 🎯 Key Features

- **Modular Design** - Easy to add new strategies
- **Comprehensive Analysis** - Statistical validation and robustness testing
- **Risk Management** - ATR-based position sizing and stop losses
- **Performance Metrics** - Sharpe ratio, Calmar ratio, win rate, profit factor
- **Visualization** - Interactive plots and dashboards
- **Configuration-Driven** - YAML-based parameter management

## 📈 Performance Metrics

- Sharpe Ratio
- Calmar Ratio  
- Total Return
- Maximum Drawdown
- Win Rate
- Profit Factor
- VaR (95%, 99%)
- CVaR (Conditional VaR)

## 🔧 Adding New Strategies

1. Create `core/strategies/your_strategy_signals.py`
2. Implement `generate_signals()` function
3. Add `config/your_strategy.yaml` configuration
4. Update `STRATEGY_NAME` in `runner.py`

## 📋 Code Style

Follow the established patterns in `STYLE_GUIDE.md`:
- Concise yet clear naming
- Consistent abbreviations
- Modular function design
- Comprehensive error handling