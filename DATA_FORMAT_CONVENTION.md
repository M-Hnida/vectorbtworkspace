# Data Format Convention & Style Guide

## 📋 **Overview**
This document defines the standard data formats and conventions used throughout the trading system to ensure consistency and prevent type errors.

## 🏗️ **Core Data Structures**

### **1. Portfolio Metrics Dictionary**
All portfolio metrics must be returned as a standardized dictionary with specific data types:

```python
metrics: Dict[str, Union[float, int]] = {
    # Performance Metrics (float)
    'return': float,           # Total return percentage
    'max_dd': float,          # Maximum drawdown percentage  
    'sharpe': float,          # Sharpe ratio
    'calmar': float,          # Calmar ratio
    'volatility': float,      # Volatility percentage
    
    # Trade Metrics (int for counts, float for ratios)
    'trades': int,            # Total number of trades
    'win_rate': float,        # Win rate percentage
    'profit_factor': float,   # Profit factor ratio
    'avg_win': float,         # Average winning trade percentage
    'avg_loss': float,        # Average losing trade percentage
    'win_loss_ratio': float,  # Win/loss ratio (avg_win / avg_loss)
    
    # Risk Metrics (float)
    'var_95': float,          # Value at Risk 95%
    'var_99': float,          # Value at Risk 99%
    'downside_vol': float,    # Downside volatility percentage
}
```

### **2. Signal Data Structure**
Trading signals must follow the Signals dataclass format:

```python
@dataclass
class Signals:
    entries: pd.Series        # Boolean series for long entries
    exits: pd.Series          # Boolean series for long exits
    short_entries: pd.Series  # Boolean series for short entries
    short_exits: pd.Series    # Boolean series for short exits
```

**Requirements:**
- All series must have the same length
- All series must have DatetimeIndex
- Values must be boolean (True/False)

### **3. Multi-Timeframe Data Structure**
Data must be organized as nested dictionaries:

```python
data: Dict[str, Dict[str, pd.DataFrame]] = {
    'SYMBOL': {
        'timeframe': pd.DataFrame  # OHLC data with DatetimeIndex
    }
}

# Example:
data = {
    'BTCUSD': {
        '15m': pd.DataFrame,  # 15-minute OHLC data
        '1h': pd.DataFrame    # 1-hour OHLC data
    },
    'EURUSD': {
        '15m': pd.DataFrame,
        '1h': pd.DataFrame
    }
}
```

### **4. OHLC DataFrame Format**
All price data must follow this standard format:

```python
df: pd.DataFrame = {
    'open': float,    # Opening price
    'high': float,    # High price
    'low': float,     # Low price
    'close': float,   # Closing price
    'volume': float   # Volume (optional)
}
```

**Requirements:**
- Index must be DatetimeIndex
- All price columns must be float type
- No missing values (NaN) allowed
- Data must be sorted by datetime ascending

## 🔧 **Data Type Conventions**

### **Numeric Types**
- **Percentages**: Always as float (e.g., 15.5 for 15.5%)
- **Ratios**: Always as float (e.g., 1.25 for 1.25:1 ratio)
- **Counts**: Always as int (e.g., 150 trades)
- **Prices**: Always as float (e.g., 45123.50)

### **Special Cases**
- **Infinite ratios**: Use `float('inf')` for division by zero cases
- **Missing data**: Use 0.0 for missing metrics, never None or NaN
- **Boolean flags**: Always True/False, never 1/0

### **String Formatting**
- **Symbols**: Uppercase (e.g., 'BTCUSD', 'EURUSD')
- **Timeframes**: Lowercase with unit (e.g., '15m', '1h', '4h', '1d')
- **Strategy names**: Lowercase (e.g., 'orb', 'momentum', 'lti')

## 📊 **Configuration Conventions**

### **Strategy Configuration**
```yaml
# config/strategy_name.yaml
name: "strategy_name"

parameters:
  param1: float/int/str
  required_timeframes: List[str]  # ['15m', '1h']
  split_ratio: float              # 0.7

optimization_grid:
  param1: List[Union[int, float]]

data_requirements:
  symbol_group: str               # 'volatile_pairs'
  timeframes: List[str]           # ['15m', '1h']
  max_symbols: int                # 3

analysis_settings:
  monte_carlo_runs: int           # 50
```

### **Data Sources Configuration**
```yaml
# config/data_sources.yaml
data_sources:
  SYMBOL:
    file_path: str
    base_timeframe: str
    asset_class: str
    description: str

symbol_groups:
  group_name: List[str]

defaults:
  symbol_group: str
  timeframes: List[str]
  max_symbols: int
```

## ⚠️ **Common Pitfalls to Avoid**

### **1. Type Mismatches**
```python
# ❌ Wrong - returning tuple instead of float
metrics['win_loss_ratio'] = (avg_win, avg_loss)

# ✅ Correct - calculate ratio as float
metrics['win_loss_ratio'] = abs(avg_win) / abs(avg_loss) if avg_loss != 0 else 0
```

### **2. Index Misalignment**
```python
# ❌ Wrong - different index lengths
signals.entries = pd.Series([True, False], index=[0, 1])
data.index = pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'])

# ✅ Correct - aligned indices
signals.entries = signals.entries.reindex(data.index, fill_value=False)
```

### **3. Missing Data Handling**
```python
# ❌ Wrong - leaving NaN values
metrics['sharpe'] = np.nan

# ✅ Correct - use default values
metrics['sharpe'] = 0.0 if np.isnan(calculated_sharpe) else calculated_sharpe
```

## 🧪 **Testing Data Formats**

### **Validation Functions**
```python
def validate_metrics(metrics: Dict[str, Any]) -> bool:
    """Validate metrics dictionary format."""
    required_keys = ['return', 'max_dd', 'sharpe', 'calmar', 'trades', 'win_rate']
    
    # Check all required keys exist
    if not all(key in metrics for key in required_keys):
        return False
    
    # Check data types
    for key, value in metrics.items():
        if key == 'trades':
            if not isinstance(value, int):
                return False
        else:
            if not isinstance(value, (int, float)):
                return False
    
    return True

def validate_signals(signals: Signals) -> bool:
    """Validate signals format."""
    # Check all series have same length
    lengths = [len(signals.entries), len(signals.exits), 
               len(signals.short_entries), len(signals.short_exits)]
    
    if len(set(lengths)) != 1:
        return False
    
    # Check all are boolean
    series_list = [signals.entries, signals.exits, 
                   signals.short_entries, signals.short_exits]
    
    for series in series_list:
        if not series.dtype == bool:
            return False
    
    return True
```

## 📝 **Implementation Checklist**

When implementing new features, ensure:

- [ ] All metrics return proper data types (float/int, never tuples)
- [ ] All DataFrames have DatetimeIndex
- [ ] All signals are boolean Series with matching indices
- [ ] Configuration files follow YAML conventions
- [ ] Error handling returns default values, not None/NaN
- [ ] String formats follow naming conventions
- [ ] Multi-timeframe data is properly nested
- [ ] All numeric calculations handle edge cases (division by zero)

## 🔄 **Version Control**

When modifying data formats:
1. Update this convention document
2. Add validation tests
3. Update all affected modules
4. Test with sample data
5. Document breaking changes

---

**Last Updated**: 2024-01-XX  
**Version**: 1.0  
**Maintainer**: Trading System Team