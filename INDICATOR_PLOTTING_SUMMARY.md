# Indicator Plotting Enhancement - Summary

## ✅ What Was Added

Created simple, chainable utilities to add custom indicators to VectorBT portfolio plots.

### New Files

1. **`vectorflow/visualization/indicators.py`** - Core utilities
2. **`docs/INDICATOR_PLOTTING.md`** - Complete documentation
3. **`scripts/nasdaqma_with_indicators.py`** - Working example
4. **`scripts/advanced_indicator_example.py`** - Advanced patterns

### New Functions

```python
from vectorflow.visualization import (
    add_indicator,      # Add any indicator (line, series, etc.)
    add_bbands,         # Add Bollinger Bands
    add_signals,        # Add entry/exit markers
    remove_date_gaps    # Remove weekend/holiday gaps
)
```

---

## 🚀 Quick Usage

### Simple Example

```python
import vectorbt as vbt
from vectorflow.visualization import add_indicator, remove_date_gaps

# Create your portfolio
portfolio = vbt.Portfolio.from_signals(...)

# Traditional plot
fig = portfolio.plot()

# ADD INDICATORS - Just chain them!
fig = add_indicator(fig, sma_200, row=1, name='SMA 200')
fig = add_indicator(fig, ema_50, row=1, name='EMA 50')
fig = remove_date_gaps(fig, price_data)

fig.show()
```

### With Your NasdaqMA Strategy

Add these lines to your `nasdaqma.py` after creating the portfolio:

```python
from vectorflow.visualization import add_indicator, remove_date_gaps

# Enhanced plot
fig = portfolio.plot(subplots=['orders', 'trade_pnl'])

# Add Kalman filter overlay
fig = add_indicator(
    fig, 
    data['close_kalman'], 
    row=1, 
    name='Kalman Filter',
    trace_kwargs=dict(line=dict(color='cyan', width=1.5))
)

# Add SMA overlay
fig = add_indicator(
    fig, 
    data['SMA_225'], 
    row=1, 
    name='SMA 225',
    trace_kwargs=dict(line=dict(color='orange', width=2))
)

# Remove gaps and show
fig = remove_date_gaps(fig, data)
fig.update_layout(title_text="NasdaqMA Strategy - Enhanced").show()
```

---

## 📋 API Overview

### `add_indicator(fig, data, row=1, name=None, trace_kwargs=None)`

Add any indicator to a specific subplot row.

- Works with `pd.Series`, `pd.DataFrame`, or any VectorBT indicator
- Use `row` to specify which subplot (1 = first subplot)
- Style with `trace_kwargs` (line color, width, dash, etc.)

### `add_bbands(fig, bbands_data, row=1, line_style=None, limits=None)`

Add Bollinger Bands from a VectorBT BBands indicator.

- Automatically plots upper, middle, lower bands
- Customize with `line_style` dict
- Optional `limits` for reference lines (e.g., `(30, 70)` for RSI)

### `add_signals(fig, entries, exits, base_data=None, row=1)`

Add entry/exit signal markers.

- Plots markers on `base_data` (e.g., plot entries on RSI)
- Customizable colors: `entry_color`, `exit_color`

### `remove_date_gaps(fig, data)`

Remove gaps from weekends/holidays in the chart.

- Detects missing dates automatically
- Makes charts cleaner and easier to read

---

## 🎯 Design Philosophy

**Simple & Explicit**
- No auto-detection overhead
- No specialized helpers for every indicator type
- You specify exactly what you want

**Chainable**
- All functions return the modified figure
- Easy to chain multiple additions

**VectorBT Native**
- Uses vectorBT's `.plot()` methods under the hood
- Compatible with all vectorBT indicators

**Minimal Dependencies**
- Just pandas, plotly, and vectorbt
- No extra packages needed

---

## 📚 Examples

### Example 1: Add Moving Averages

```python
fig = portfolio.plot()
fig = add_indicator(fig, sma_50, row=1, name='SMA 50', 
                   trace_kwargs=dict(line=dict(color='cyan')))
fig = add_indicator(fig, sma_200, row=1, name='SMA 200',
                   trace_kwargs=dict(line=dict(color='orange')))
fig.show()
```

### Example 2: Add BBands + RSI

```python
# Create indicators
bb_price = vbt.indicators.factory.talib('BBANDS').run(close, timeperiod=20)
rsi = vbt.indicators.factory.talib('RSI').run(close, timeperiod=14).rsi

# Plot with subplots
fig = portfolio.plot(subplots=['orders', 'trade_pnl'])

# Add BBands to price chart (row 1)
fig = add_bbands(fig, bb_price, row=1, 
                line_style=dict(color='white', dash='dot'))

# To add RSI, you'd need a 3rd subplot
# For now, use portfolio.plot() without predefined subplots
```

### Example 3: Full Custom Layout

See `scripts/advanced_indicator_example.py` for a complete example with:
- OHLCV candlesticks
- Price with BBands
- RSI with BBands in separate subplot
- Entry/exit signals
- Trade P/L visualization

---

## ⚠️ Current Limitations

1. **Dynamic Subplots**: Creating new subplots on-the-fly (`subplot='new'`) is not yet fully supported. **Workaround**: Define all subplots in `portfolio.plot()` first, then use explicit `row` numbers.

2. **Row Numbers**: You need to know which row number to target. Count your subplots:
   - If `portfolio.plot(subplots=['orders', 'trade_pnl'])` → orders=row 1, trade_pnl=row 2

---

## 🔄 Migration from Old Approach

**Before:**
```python
# Limited to portfolio's built-in plots
fig = portfolio.plot()
fig.show()
```

**After:**
```python
# Add any custom indicator
fig = portfolio.plot()
fig = add_indicator(fig, your_custom_indicator, row=1, name='Custom')
fig = remove_date_gaps(fig, data)
fig.show()
```

---

## 📖 See Also

- **Full Documentation**: `docs/INDICATOR_PLOTTING.md`
- **Simple Example**: `scripts/nasdaqma_with_indicators.py`
- **Advanced Example**: `scripts/advanced_indicator_example.py`
- **VectorBT Docs**: https://vectorbt.dev/

---

## 🎉 Ready to Use!

All utilities are already imported in the `vectorflow.visualization` module.  
Just import and start adding indicators to your plots!

```python
from vectorflow.visualization import add_indicator, add_bbands, remove_date_gaps
```
