# Consolidated Trading System

A robust and comprehensive trading strategy analysis framework with advanced backtesting, optimization, and visualization capabilities. Built with Python, this system provides a complete toolkit for developing, testing, and analyzing trading strategies.

## 🚀 Features

- **Advanced Trading Pipeline**: Comprehensive suite including parameter optimization, walk-forward analysis, Monte Carlo simulation, and backtesting
- **Multiple Strategy Support**: Pre-built strategies including LTI (Logical Trading Indicator), Momentum, and ORB (Opening Range Breakout)
- **Multi-Timeframe Analysis**: Support for strategies using multiple timeframes for trend filtering and range definition
- **Time Range Control**: Flexible time range selection (e.g., '2y', '6m', '1y') with automatic data harmonization across timeframes
- **Sophisticated Analytics**: In-depth performance metrics, risk analysis, and statistical validation tools
- **Interactive Visualizations**: Professional-grade charts and dashboards using Plotly
- **Multi-Asset Capability**: Analyze multiple symbols and timeframes simultaneously
- **Modular Architecture**: Clean separation of concerns with dedicated modules for strategies, optimization, and visualization

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the System**:
   ```bash
   python trading_system.py
   ```

3. **Select Strategy**: Choose from available strategies (LTI, Momentum, ORB)

4. **View Results**: The system will automatically run optimization, validation, backtesting, and create visualizations

## 📁 Project Structure

```
├── trading_system.py      # Main system orchestration
├── core_components.py     # Base classes and core functionality
├── optimizer.py          # Parameter optimization and analysis
├── plotter.py           # Visualization and plotting
├── run_tests.py         # Test runner script
├── metrics.py           # Performance and risk metrics calculation
├── base.py             # Abstract base classes and interfaces
├── data_manager.py     # Data handling and preprocessing
├── strategies/         # Strategy implementations
│   ├── __init__.py    # Strategy registry
│   ├── lti_strategy.py     # LTI strategy
│   ├── momentum_strategy.py # Momentum strategy
│   └── orb_strategy.py     # ORB strategy
├── config/             # Strategy configurations
│   ├── data_sources.yaml   # Data source settings
│   ├── lti.yaml           # LTI parameters
│   ├── momentum.yaml      # Momentum parameters
│   └── orb.yaml          # ORB parameters
├── data/               # Market data files
│   ├── BTCUSD_1h_2011-2025.csv
│   ├── BTCUSDT_15m.csv
│   ├── ETH_1h_2017-2020.csv
│   ├── EURUSD_1D_2009-2025.csv
│   └── EURUSD_4H_2009-2025.csv
└── tests/              # Test suite
    ├── test_core_components.py
    ├── test_data_manager.py
    ├── test_optimizer.py
    └── test_plotter.py
```

## Strategy Configuration

Each strategy is configured via YAML files in the `config/` directory:

```yaml
# Example: config/lti.yaml
indicators:
  atr_period: 7
  atr_multiple: 3.0
  ma_length: 20
  ma_type: 'EMA'
  bb_std_dev: 2.0

optimization:
  atr_period: [10, 15]
  atr_multiple: [2.0, 2.5]
  ma_length: [20, 30]

analysis:
  monte_carlo_runs: 20
```

## Available Strategies

### 1. LTI (Logical Trading Indicator)
- Uses ATR, Moving Averages, and Bollinger Bands
- Multi-condition entry/exit logic
- Trend following with mean reversion elements

### 2. Momentum Strategy
- Based on price momentum and volatility
- Uses WMA (Weighted Moving Average) for trend filtering
- Adaptive to market volatility

### 3. ORB (Opening Range Breakout)
- Trades breakouts from opening range
- Configurable time periods and thresholds
- Suitable for intraday trading

## 🔄 Analysis Pipeline

The system executes a comprehensive 5-step analysis:

1. **Parameter Optimization**: Grid search optimization to find optimal parameters
2. **Walk-Forward Analysis**: Out-of-sample validation across multiple time periods
3. **Monte Carlo Simulation**: Statistical robustness testing with bootstrap resampling
4. **Full Backtesting**: Complete performance analysis on historical data
5. **Visualization**: Interactive charts and performance dashboards

## 📈 Key Metrics

- **Performance**: Total Return, Sharpe Ratio, Calmar Ratio
- **Risk**: Maximum Drawdown, Volatility, VaR (95%, 99%)
- **Trading**: Win Rate, Profit Factor, Average Win/Loss
- **Validation**: Walk-Forward Efficiency, Monte Carlo Confidence

## Data Format

The system supports CSV files with OHLC data. It automatically detects:
- Different column naming conventions
- Various separators (comma, tab)
- With or without headers
- Multiple datetime formats

### Time Range Harmonization

The system automatically harmonizes time ranges across different CSV files:
- **Automatic Alignment**: When loading multiple timeframes, the system finds the common overlapping period
- **User Control**: Users can specify custom time ranges (e.g., '2y', '6m') that apply consistently across all timeframes
- **Smart Filtering**: Time ranges are calculated from the most recent data backwards, ensuring you get the latest market conditions
- **Data Validation**: The system ensures all timeframes have data for the specified period before proceeding with analysis

Example: If you have EURUSD_1H_2009-2025.csv and EURUSD_4H_2009-2025.csv, specifying '2y' will load the last 2 years of data from both files, ensuring perfect alignment for multi-timeframe strategies.

## Customization

### Adding New Strategies

1. Create a new class inheriting from `BaseStrategy`
2. Implement `generate_signals()` and `get_required_columns()` methods
3. Add the strategy to the `strategy_classes` dictionary in `TradingSystem._load_strategy()`
4. Create a corresponding YAML configuration file

### Modifying Analysis

The analysis pipeline is modular and can be customized by modifying the `TradingSystem.run_complete_analysis()` method.

## Performance

The system is optimized for performance with:
- Vectorized operations using pandas and numpy
- Parallel processing for Monte Carlo simulations
- Efficient data structures and caching
- Minimal memory footprint

## ⚠️ Important Notes

For detailed guidelines, refer to:
- [`STYLE_GUIDE.md`](STYLE_GUIDE.md) for coding conventions
- [`DATA_FORMAT_CONVENTION.md`](DATA_FORMAT_CONVENTION.md) for data structure specifications

Key conventions:
- All metrics are strictly typed as `float` or `int`
- DataFrames must have `DatetimeIndex`
- Signals must be boolean `pd.Series`
- Configurations follow specific YAML structure
- String formats: symbols (UPPERCASE), timeframes (lowercase)

## 📦 Dependencies

Core libraries:
- **pandas**: Data manipulation and time series analysis
- **numpy**: Scientific computing and numerical operations
- **vectorbt**: High-performance backtesting engine
- **pandas-ta**: Technical analysis indicators suite
- **plotly**: Interactive visualization library
- **PyYAML**: YAML configuration parsing
- **joblib**: Parallel computing support

Optional tools:
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Static type checking

## 📄 License

This project is provided as-is for educational and research purposes. See LICENSE file for details.
## 🔧 System Architecture

### Core Components (`core_components.py`)
- **BaseStrategy**: Abstract base class for strategy implementation
- **Signals**: Data structure for trading signals
- **StrategyConfig**: Configuration management
- **Backtesting Engine**: VectorBT integration

### Optimization (`optimizer.py`)
- **ParameterOptimizer**: Grid search optimization
- **WalkForwardAnalysis**: Time-series validation
- **MonteCarloAnalysis**: Bootstrap resampling analysis

### Visualization (`plotter.py`)
- **TradingVisualizer**: Interactive charts
- **Portfolio Analysis**: Performance visualization
- **Monte Carlo Plots**: Distribution analysis

### Analytics System (`metrics.py`, `data_manager.py`)
- **Performance Metrics**: Comprehensive trading statistics
- **Risk Analysis**: Advanced risk metrics calculation
- **Data Management**: Multi-timeframe data handling

## 🚀 Usage Examples

### Quick Start
```python
from trading_system import run_strategy_pipeline

# Run complete analysis for LTI strategy
results = run_strategy_pipeline('lti')
```

### Time Range Control
```python
from trading_system import run_strategy_with_time_range

# Run strategy on last 2 years of data
results = run_strategy_with_time_range('momentum', '2y')

# Run strategy on last 6 months
results = run_strategy_with_time_range('orb', '6m')

# Run strategy with custom end date
results = run_strategy_with_time_range('momentum', '1y', '2024-12-31')

# Supported time range formats:
# '2y' - 2 years
# '1y' - 1 year  
# '6m' - 6 months
# '3m' - 3 months
# '30d' - 30 days
# '4w' - 4 weeks
```

### Custom Strategy Development
```python
from base import BaseStrategy, Signals
from typing import Dict, Any, List

class MyStrategy(BaseStrategy):
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Strategy logic implementation
        main_tf = self.required_timeframes[0]
        df = data[main_tf]
        
        # Your signal generation logic here
        signals = pd.DataFrame(index=df.index)
        signals['entries'] = your_entry_condition
        signals['exits'] = your_exit_condition
        
        return signals
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']
    
    @property
    def required_timeframes(self) -> List[str]:
        return ['1h']  # Base timeframe
```

### Advanced Usage
```python
from optimizer import ParameterOptimizer, OptimizationConfig
from metrics import calc_metrics
from data_manager import DataManager

# Load and preprocess data
dm = DataManager()
data = dm.load_data('BTCUSD', '1h')

# Configure and run optimization
opt_config = OptimizationConfig(
    split_ratio=0.7,
    window_size=504,  # 2 years daily
    step_size=63,     # Quarterly
    num_windows=5
)

optimizer = ParameterOptimizer(strategy, strategy_config, opt_config)
results = optimizer.optimize(data)
metrics = calc_metrics(results['portfolio'])
print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
```

