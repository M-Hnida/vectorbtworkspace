# Configuration Guide

## Quick Start

Every strategy needs a YAML config file in `config/` matching the strategy name.

### Minimal Config Example

```yaml
# config/my_strategy.yaml

# Strategy parameters (required)
parameters:
  rsi_period: 14
  oversold: 30
  overbought: 70
  primary_timeframe: "1H"

# Data source (pick ONE)
data_source: csv  # Options: csv, ccxt, freqtrade

# CSV data files (if using csv)
csv_path:
  - "data/BTCUSD_1h_2011-2025.csv"

# Optimization grid (optional) - supports two formats:
# Format 1: List of values
optimization_grid:
  rsi_period: [10, 14, 21]
  
# Format 2: Range with start/end/step (recommended)
optimization_grid:
  rsi_period:
    start: 10
    end: 21
    step: 3
  oversold:
    start: 20
    end: 35
    step: 5
  overbought:
    start: 65
    end: 80
    step: 5
```

## Data Source Options

### Option 1: CSV Files (Simplest)
```yaml
data_source: csv
csv_path:
  - "data/BTCUSD_1h_2011-2025.csv"
  - "data/EURUSD_1H_2009-2025.csv"
```

### Option 2: CCXT Exchange (Live Data)
```yaml
data_source: ccxt
parameters:
  # ... your strategy params ...

# CCXT configuration
ccxt:
  exchange: binance
  symbols:
    - BTC/USDT
    - ETH/USDT
  sandbox: false  # true for testnet
```

### Option 3: Freqtrade Data
```yaml
data_source: freqtrade
freqtrade_data_dir: "/path/to/freqtrade/user_data/data/binance"
symbols:
  - BTC_USDT
  - ETH_USDT
```

## Common Issues

### Issue: "No data loaded"
**Fix**: Make sure `data_source` is set and matches your data setup:
- If using CSV: set `data_source: csv` and provide `csv_path`
- If using CCXT: set `data_source: ccxt` and configure `ccxt` section
- If using Freqtrade: set `data_source: freqtrade` and set `freqtrade_data_dir`

### Issue: "Symbol not found"
**Fix**: Check your symbol format:
- CSV: Extracted from filename (e.g., `BTCUSD_1h.csv` → `BTCUSD`)
- CCXT: Use exchange format (e.g., `BTC/USDT`)
- Freqtrade: Use underscore format (e.g., `BTC_USDT`)

### Issue: "Timeframe mismatch"
**Fix**: Set `primary_timeframe` in parameters:
```yaml
parameters:
  primary_timeframe: "1H"  # Must match your data
```

## Full Example

```yaml
# config/rsi.yaml

# Strategy parameters
parameters:
  rsi_period: 14
  oversold_level: 30
  overbought_level: 70
  primary_timeframe: "1H"
  initial_cash: 10000
  fee: 0.001

# Data source
data_source: csv
csv_path:
  - "data/BTCUSD_1h_2011-2025.csv"

# Optimization (optional) - using start/end/step format
optimization_grid:
  rsi_period:
    start: 10
    end: 21
    step: 3
  oversold_level:
    start: 20
    end: 35
    step: 5
  overbought_level:
    start: 65
    end: 80
    step: 5

# Data requirements (optional - auto-detected)
data_requirements:
  required_timeframes: ["1H"]
  required_columns: ["open", "high", "low", "close"]
```

## Tips

1. **Start simple**: Use CSV data source first, it's the most reliable
2. **One parameter section**: Put ALL parameters under `parameters:`, not at root level
3. **Explicit data_source**: Always set `data_source` explicitly, don't rely on auto-detection
4. **Match timeframes**: Ensure `primary_timeframe` matches your data files
5. **Test with --quick**: Run `python main.py --quick strategy_name` to test quickly
