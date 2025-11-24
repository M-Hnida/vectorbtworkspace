# Indicator Plotting Utilities

Simple, lightweight utilities to add custom indicators to VectorBT portfolio plots.

## Features

✅ **Simple API** - Chainable functions that work with existing `portfolio.plot()`  
✅ **No Overhead** - Explicit control, no auto-detection or magic  
✅ **VectorBT Native** - Uses vectorBT's plotting methods under the hood  
✅ **Gap Handling** - Remove weekend/holiday gaps from charts  

---

## Installation

Already included in vectorflow! Just import:

```python
from vectorflow.visualization import (
    add_indicator,
    add_bbands,
    add_signals,
    remove_date_gaps
)
```

---

## Quick Start

### Basic Example

```python
import vectorbt as vbt
from vectorflow.visualization import add_indicator, remove_date_gaps

# Create portfolio
portfolio = vbt.Portfolio.from_signals(...)

# Basic plot
fig = portfolio.plot()

# Add indicators
fig = add_indicator(fig, sma_200, row=1, name='SMA 200')
fig = add_indicator(fig, rsi, row=1, name='RSI')
fig = remove_date_gaps(fig, price_data)

fig.show()
```

---

## API Reference

### `add_indicator(fig, data, subplot='overlay', row=None, name=None, trace_kwargs=None, **plot_kwargs)`

Add any indicator to an existing portfolio plot.

**Parameters:**
- `fig` (go.Figure): Figure returned from `portfolio.plot()`
- `data` (pd.Series | pd.DataFrame | vbt.indicator): Indicator data
- `subplot` (str): `'overlay'` (add to first subplot) or `'new'` (create new subplot)
- `row` (int): Specific row number to add to (overrides `subplot`)
- `name` (str): Legend name
- `trace_kwargs` (dict): Styling for the trace (line color, width, etc.)
- `**plot_kwargs`: Additional kwargs passed to `vbt.plot()`

**Returns:** Modified figure

**Example:**

```python
# Add SMA overlay on main chart
fig = add_indicator(
    fig, 
    data['SMA_200'], 
    row=1, 
    name='SMA 200',
    trace_kwargs=dict(line=dict(color='orange', width=2))
)

# Add RSI in new subplot
fig = add_indicator(
    fig, 
    rsi_data, 
    subplot='new', 
    name='RSI',
    trace_kwargs=dict(line=dict(color='yellow'))
)
```

---

### `add_bbands(fig, bbands_data, subplot='overlay', row=None, upper_name='BB Upper', middle_name='BB Middle', lower_name='BB Lower', line_style=None, limits=None, **plot_kwargs)`

Add Bollinger Bands to an existing figure.

**Parameters:**
- `fig` (go.Figure): Figure from `portfolio.plot()`
- `bbands_data` (vbt.indicator): VectorBT BBands indicator object
- `subplot` (str): `'overlay'` or `'new'`
- `row` (int): Specific row number
- `upper_name`, `middle_name`, `lower_name` (str): Legend names for each band
- `line_style` (dict): Line styling (color, width, dash)
- `limits` (tuple): Optional (lower, upper) horizontal reference lines
- `**plot_kwargs`: Additional kwargs

**Returns:** Modified figure

**Example:**

```python
# Create BBands indicator
bb_price = vbt.indicators.factory.talib('BBANDS').run(
    data['close'], 
    timeperiod=20
)

# Add to plot
fig = portfolio.plot()
fig = add_bbands(
    fig, 
    bb_price, 
    row=1,
    line_style=dict(color='white', width=1, dash='dot')
)
```

---

### `add_signals(fig, entries, exits, base_data=None, row=1, entry_name='Entry', exit_name='Exit', entry_color='limegreen', exit_color='red', **plot_kwargs)`

Add entry/exit signal markers to a figure.

**Parameters:**
- `fig` (go.Figure): Figure from `portfolio.plot()`
- `entries` (pd.Series): Boolean series of entry signals
- `exits` (pd.Series): Boolean series of exit signals
- `base_data` (pd.Series): Optional data to plot signals on (e.g., RSI)
- `row` (int): Row number to add signals to
- `entry_name`, `exit_name` (str): Legend names
- `entry_color`, `exit_color` (str): Marker colors
- `**plot_kwargs`: Additional kwargs

**Returns:** Modified figure

**Example:**

```python
# Add signals on RSI indicator
fig = add_indicator(fig, rsi, row=2, name='RSI')
fig = add_signals(
    fig, 
    long_entries, 
    long_exits,
    base_data=rsi,
    row=2,
    entry_color='limegreen',
    exit_color='red'
)
```

---

### `remove_date_gaps(fig, data)`

Remove gaps from missing dates (weekends, holidays) in the plot.

**Parameters:**
- `fig` (go.Figure): Figure to modify
- `data` (pd.DataFrame | pd.Series | pd.DatetimeIndex): Data to detect gaps from

**Returns:** Modified figure with rangebreaks applied

**Example:**

```python
fig = portfolio.plot()
fig = add_indicator(fig, sma, name='SMA')
fig = remove_date_gaps(fig, price_data)
fig.show()
```

---

## Complete Example

```python
import pandas as pd
import vectorbt as vbt
from vectorflow.visualization import (
    add_indicator, 
    add_bbands, 
    add_signals, 
    remove_date_gaps
)

# Load data and create indicators
data = pd.read_csv('price_data.csv', parse_dates=['DateTime']).set_index('DateTime')
data['SMA_50'] = data['close'].rolling(50).mean()
data['SMA_200'] = data['close'].rolling(200).mean()

# Create RSI
rsi = vbt.indicators.factory.talib('RSI').run(data['close'], timeperiod=14).rsi

# Create Bollinger Bands
bb_price = vbt.indicators.factory.talib('BBANDS').run(data['close'], timeperiod=20)

# Run backtest
portfolio = vbt.Portfolio.from_signals(
    close=data['close'],
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.001
)

# Create enhanced plot
fig = portfolio.plot(
    subplots=['orders', 'trade_pnl']
)

# Add moving averages to main chart (row 1)
fig = add_indicator(
    fig, 
    data['SMA_50'], 
    row=1, 
    name='SMA 50',
    trace_kwargs=dict(line=dict(color='cyan', width=1))
)

fig = add_indicator(
    fig, 
    data['SMA_200'], 
    row=1, 
    name='SMA 200',
    trace_kwargs=dict(line=dict(color='orange', width=2))
)

# Add Bollinger Bands to main chart (row 1)
fig = add_bbands(
    fig, 
    bb_price, 
    row=1,
    line_style=dict(color='white', width=1, dash='dot')
)

# Note: For subplot='new', you need to specify row numbers manually
# since dynamic subplot creation is not yet supported
# Instead, define subplots in portfolio.plot() first

# Remove gaps
fig = remove_date_gaps(fig, data)

# Style and show
fig.update_layout(
    title_text="Trading Strategy with Indicators",
    height=900,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.show()
```

---

## Tips & Tricks

### 1. **Specify Row Numbers Explicitly**

For best results, define your subplots upfront and use explicit row numbers:

```python
fig = portfolio.plot(
    subplots=['orders', 'trade_pnl', 'drawdowns']
)

# orders = row 1
# trade_pnl = row 2
# drawdowns = row 3

fig = add_indicator(fig, sma, row=1)  # Overlay on orders
fig = add_indicator(fig, rsi, row=4)  # Would be a new row
```

### 2. **Chain Multiple Indicators**

```python
fig = (portfolio.plot()
       .pipe(add_indicator, sma_50, row=1, name='SMA 50')
       .pipe(add_indicator, sma_200, row=1, name='SMA 200')
       .pipe(add_bbands, bb_price, row=1)
       .pipe(remove_date_gaps, data))
```

Or traditional chaining:

```python
fig = portfolio.plot()
fig = add_indicator(fig, sma_50, row=1, name='SMA 50')
fig = add_indicator(fig, sma_200, row=1, name='SMA 200')
fig = add_bbands(fig, bb_price, row=1)
fig = remove_date_gaps(fig, data)
fig.show()
```

### 3. **Styling Indicators**

Use `trace_kwargs` for complete control:

```python
fig = add_indicator(
    fig,
    data['ema'],
    row=1,
    name='EMA 20',
    trace_kwargs=dict(
        line=dict(
            color='rgba(255, 165, 0, 0.8)',
            width=2,
            dash='dash'
        ),
        connectgaps=True
    )
)
```

### 4. **Working with VectorBT Indicators**

Any indicator with a `.plot()` method works:

```python
# RSI with BBands on it
rsi = vbt.indicators.factory.talib('RSI').run(close, timeperiod=14).rsi
bb_rsi = vbt.indicators.factory.talib('BBANDS').run(rsi, timeperiod=14)

fig = portfolio.plot(subplots=['orders'])
fig = add_indicator(fig, rsi, row=2, name='RSI')
fig = add_bbands(fig, bb_rsi, row=2, limits=(30, 70))
```

---

## Limitations

1. **Dynamic Subplot Creation**: `subplot='new'` is not yet fully supported due to Plotly limitations. Use explicit row numbers instead.

2. **Subplot Definition**: For best results, define all subplots in `portfolio.plot()` first, then add indicators to specific rows.

---

## See Also

- [VectorBT Documentation](https://vectorbt.dev/)
- [Plotly Figure Reference](https://plotly.com/python/reference/)
- Example: `scripts/nasdaqma_with_indicators.py`
