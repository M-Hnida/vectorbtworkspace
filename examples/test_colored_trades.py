#!/usr/bin/env python3
"""Test script to verify profit/loss coloring works correctly"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import vectorbt as vbt
import pandas as pd
import numpy as np
from vectorflow.visualization.indicators import add_trade_signals

print("Testing profit/loss colored trade lines...")

# Create data with obvious winning and losing trades
np.random.seed(42)
dates = pd.date_range("2023-01-01", periods=200, freq="D")

# Create a price series with ups and downs
price = 100 + np.cumsum(np.random.randn(200) * 2)
close = pd.Series(price, index=dates)

# Manual signals to create both winning and losing trades
entries = pd.Series(False, index=dates)
exits = pd.Series(False, index=dates)

# Trade 1: Buy low, sell high (PROFIT - should be GREEN)
entries.iloc[20] = True
exits.iloc[40] = True

# Trade 2: Buy high, sell low (LOSS - should be RED)
entries.iloc[60] = True
exits.iloc[80] = True

# Trade 3: Another profit
entries.iloc[100] = True
exits.iloc[120] = True

portfolio = vbt.Portfolio.from_signals(
    close=close, entries=entries, exits=exits, init_cash=10000, fees=0.001
)

# Show trade PnLs
trades_df = portfolio.trades.records_readable

# Create plot with colored lines
fig = portfolio.plot(template="plotly_dark")
fig = add_trade_signals(portfolio, fig)
fig.update_layout(title="Trade Lines: Green=Profit, Red=Loss")

print("\n✅ Plot created! Opening in browser...")
print("   Green lines = Profitable trades")
print("   Red lines = Losing trades")

fig.show()
