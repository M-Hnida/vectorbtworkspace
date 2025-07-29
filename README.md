# Consolidated Trading System

A robust and comprehensive trading strategy analysis framework with advanced backtesting, optimization, and visualization capabilities. Built with Python, this system provides a complete toolkit for developing, testing, and analyzing trading strategies.

## 🚀 Features

- **Advanced Trading Pipeline**: Comprehensive suite including parameter optimization, walk-forward analysis, Monte Carlo simulation, and backtesting
- **Multiple Strategy Support**: Pre-built strategies including TDI (Traders Dynamic Index), Momentum, ORB (Opening Range Breakout), and VectorBT
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
   # Interactive mode - select strategy from menu
   python trading_system.py

   # Quick test mode (fast, no optimization)
   python trading_system.py --quick tdi

   # Full analysis mode
   python trading_system.py --quick tdi --full
   ```

3. **Select Strategy**: Choose from available strategies (TDI, Momentum, ORB, VectorBT)

4. **View Results**: The system will automatically run optimization, validation, backtesting, and create visualizations

## 📁 Project Structure

```
├── trading_system.py      # Main system orchestration
├── core_components.py     # Base classes and core functionality
├── analysis_pipeline.py   # Analysis pipeline and workflow management
├── optimizer.py          # Parameter optimization and analysis
├── plotter.py           # Visualization and plotting
├── validation.py        # Strategy validation and testing
├── run_tests.py         # Test runner script
├── metrics.py           # Performance and risk metrics calculation
├── base.py             # Abstract base classes and interfaces
├── data_manager.py     # Data handling and preprocessing
├── strategies/         # Strategy implementations
│   ├── __init__.py    # Strategy registry
│   ├── tdi_strategy.py     # TDI strategy
│   ├── momentum_strategy.py # Momentum strategy
│   ├── orb_strategy.py     # ORB strategy
│   └── vectorbt_strategy.py # VectorBT strategy
├── config/             # Strategy configurations
│   ├── tdi.yaml           # TDI parameters
│   ├── momentum.yaml      # Momentum parameters
│   ├── orb.yaml          # ORB parameters
│   └── vectorbt.yaml     # VectorBT parameters
├── data/               # Market data files
│   ├── BTCUSD_1h_2011-2025.csv
│   ├── BTCUSDT_15m.csv
│   ├── ETH_1h_2017-2020.csv
│   ├── EURUSD_15m_2021-2025.csv
│   ├── EURUSD_30m_2017_2025.csv
│   ├── EURUSD_1H_2009-2025.csv
│   ├── EURUSD_1D_2009-2025.csv
│   └── EURUSD_4H_2009-2025.csv

```

## Strategy Configuration

Each strategy is configured via YAML files in the `config/` directory:

```yaml
# Example: config/tdi.yaml
name: "tdi_strategy"

parameters:
  # TDI Core Parameters
  rsi_period: 21
  tdi_fast_period: 2
  tdi_slow_period: 7
  tdi_middle_period: 34

  # Multi-Timeframe Settings
  required_timeframes: ["15m", "30m", "1h", "4h", "1D"]

  # Risk Management
  risk_factor: 1.0

optimization_grid:
  rsi_period: [21, 34]
  tdi_fast_period: [2, 3]
  tdi_slow_period: [7, 10]

csv_path:
  - "data/EURUSD_15m_2021-2025.csv"
  - "data/EURUSD_1H_2009-2025.csv"
```

## Available Strategies

### 1. TDI (Traders Dynamic Index)
- Multi-timeframe strategy using TDI indicators
- Combines pivot points with RSI-based signals
- Uses multiple timeframes (15m, 30m, 1h, 4h, 1D) for trend filtering
- Advanced signal configuration with cross, trend, and angle analysis

### 2. Momentum Strategy
- Based on price momentum and volatility
- Uses WMA (Weighted Moving Average) for trend filtering
- Adaptive to market volatility

### 3. ORB (Opening Range Breakout)
- Trades breakouts from opening range
- Configurable time periods and thresholds
- Suitable for intraday trading

### 4. VectorBT Strategy
- High-performance strategy using VectorBT library
- Optimized for fast backtesting and analysis
- Customizable parameters for various market conditions

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

## 🔧 Adding New Strategies

#
#### 3. Create Configuration File
Create `config/my.yaml`:

```yaml
name: "my_strategy"

parameters:
  # Strategy parameters with default values
  param1: 20
  param2: 2.0
  param3: "SMA"

  # Multi-timeframe support (if needed)
  required_timeframes: ["1h"]

# Optimization parameter ranges
optimization_grid:
  param1: [10, 20, 30]
  param2: [1.5, 2.0, 2.5]

# Data file paths (adjust for your symbol/timeframe)
csv_path:
  - "data/BTCUSD_1h_2011-2025.csv"
```

#### 4. Test Your Strategy
Run your strategy:

```bash
python trading_system.py
# Select your strategy from the menu

# Or run directly:
python trading_system.py --quick my --full
```

### Signal Generation Best Practices

**Boolean Expressions**: Signals can be created using pandas boolean expressions:
```python
# Simple conditions
entries = df['close'] > df['ema_20']
exits = df['close'] < df['ema_20']

# Complex multi-condition signals
entries = (df['ema_21'] > df['ema_50']) & (df['rsi'] < 70) & (df['volume'] > df['volume'].rolling(20).mean())

# Multi-timeframe signals
entries = (ema_df["21_EMA_4H"] > ema_df["100_EMA_4H"]) & (ema_df["50_EMA_1H"] > ema_df["200_EMA_1H"])

# Time-based conditions
entries = entries & (df.index.hour >= 9) & (df.index.hour <= 16)  # Trading hours only
```

**Signal Types Supported**:
- Boolean Series (True/False)
- Boolean expressions that evaluate to Series
- Numeric Series with 0/1 values (automatically converted)

### Strategy Development Tips

1. **Start Simple**: Begin with basic indicators and gradually add complexity
2. **Use Vectorized Operations**: Leverage pandas/numpy for performance
3. **Boolean Expressions**: Use pandas conditions directly - no need to create boolean Series manually
4. **Multi-Timeframe Support**: Use `get_required_timeframes()` for complex strategies
5. **Parameter Validation**: Add validation in `__init__()` for robustness
6. **Testing**: Create unit tests in `tests/test_my_strategy.py`

### Advanced Features

- **Boolean Expression Signals**: Use pandas conditions directly (e.g., `entries = (df['ema_21'] > df['ema_50']) & (df['rsi'] < 70)`)
- **Position Sizing**: Return `sizes` in Signals for dynamic position sizing
- **Short Selling**: Use `short_entries` and `short_exits` for short positions
- **Risk Management**: Implement stop-loss and take-profit logic
- **Multi-Asset**: Handle multiple symbols in signal generation
- **Time-Based Filtering**: Add time conditions (e.g., trading hours) to signals
- **Multi-Timeframe Logic**: Combine signals from different timeframes

### Modifying Analysis Pipeline

The analysis pipeline is modular and can be customized by modifying the `TradingSystem.run_complete_analysis()` method or creating custom analysis workflows.

## Performance & Caching

The system is optimized for performance with:
- **Vectorized Operations**: Using pandas and numpy for fast computations
- **Parallel Processing**: Monte Carlo simulations run in parallel
- **Efficient Data Structures**: Minimal memory footprint with optimized data handling
- **VectorBT Integration**: High-performance backtesting engine for rapid analysis

### Cache System
- Automatically caches processed data files to speed up subsequent runs
- Cache files are stored in `cache/` directory with `.pkl` extension
- Cache is invalidated when source data files are modified
- Manual cache clearing: Delete files in `cache/` directory

## ⚠️ Important Notes

For detailed guidelines, refer to:
- [`STYLE_GUIDE.md`](STYLE_GUIDE.md) for coding conventions
- [`DATA_FORMAT_CONVENTION.md`](DATA_FORMAT_CONVENTION.md) for data structure specifications

Key conventions:
- All metrics are strictly typed as `float` or `int`
- DataFrames must have `DatetimeIndex`
- Signals can be boolean `pd.Series` or boolean expressions (e.g., `df['close'] > df['ema']`)
- Configurations follow specific YAML structure
- String formats: symbols (UPPERCASE), timeframes (lowercase)

## 📦 Dependencies

Core libraries (see `requirements.txt`):
- **pandas>=2.0.0**: Data manipulation and time series analysis
- **numpy>=1.20.0**: Scientific computing and numerical operations
- **vectorbt>=0.25.0**: High-performance backtesting engine
- **pandas-ta>=0.3.14b0**: Technical analysis indicators suite
- **plotly>=5.0.0**: Interactive visualization library
- **PyYAML>=6.0.0**: YAML configuration parsing
- **joblib>=1.0.0**: Parallel computing support

Development tools:
- **black**: Code formatting (optional)
- **mypy**: Static type checking (optional)

## 🔧 Troubleshooting

### Common Issues

**Strategy not found error:**
- Ensure your strategy file is in the `strategies/` directory
- Check that the strategy is imported in `strategies/__init__.py`
- Verify the configuration file exists in `config/` directory

**Data loading errors:**
- Check that CSV files exist in the `data/` directory
- Verify file paths in the strategy configuration YAML
- Ensure CSV files have proper OHLC columns

**Memory issues with large datasets:**
- Use time range filtering (e.g., '2y', '1y') to reduce data size
- Clear cache directory if files become corrupted
- Consider using smaller optimization grids

**Performance issues:**
- Enable caching by ensuring `cache/` directory exists
- Use `--quick` mode for faster testing
- Reduce Monte Carlo runs in configuration

**Import errors:**
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)
- Verify all required packages are installed

### Getting Help

1. Check the error message and stack trace
2. Verify your configuration files follow the YAML format
3. Test with provided sample strategies first
4. Review the test files in `tests/` for examples

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

# Run complete analysis for TDI strategy
results = run_strategy_pipeline('tdi')
```

### Time Range Control
```python
from trading_system import run_strategy_with_time_range

# Run strategy on last 2 years of data
results = run_strategy_with_time_range('momentum', '2y')

# Run strategy on last 6 months
results = run_strategy_with_time_range('orb', '6m')

# Run strategy with custom end date
results = run_strategy_with_time_range('momentum', '1y', end_date='2024-12-31')

# Skip optimization for faster testing
results = run_strategy_with_time_range('tdi', '1y', skip_optimization=True)

# Supported time range formats:
# '2y' - 2 years
# '1y' - 1 year
# '6m' - 6 months
# '3m' - 3 months
# '30d' - 30 days
# '4w' - 4 weeks
```

### Command Line Usage
```bash
# Interactive mode
python trading_system.py

# Quick test (no optimization)
python trading_system.py --quick tdi

# Full analysis
python trading_system.py --quick momentum --full

# Available strategies
python -c "from trading_system import get_available_strategies; print(get_available_strategies())"
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
from base import Signals

# Load and preprocess data
dm = DataManager()
data = dm.load_data('BTCUSD', '1h')

# Advanced signal generation with boolean expressions
def create_complex_signals(df):
    import pandas_ta as ta

    # Calculate indicators
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.bbands(df['close'], length=20)

    # Complex boolean expressions for signals
    bullish_trend = df['ema_21'] > df['ema_50']
    oversold = df['rsi'] < 30
    near_support = df['close'] <= df['bb_lower'] * 1.02

    # Combine conditions
    entries = bullish_trend & oversold & near_support
    exits = (df['rsi'] > 70) | (df['close'] >= df['bb_upper'])

    return Signals(entries=entries, exits=exits)

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

