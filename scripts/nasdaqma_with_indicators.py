# %%
"""
Example: Using the new indicator plotting utilities
Shows how to add indicators to portfolio plots with minimal code
"""

import pandas as pd
import vectorbt as vbt
import numpy as np
import sys
import os
from numba import njit

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorflow.visualization import add_indicator, remove_date_gaps

# Silence warnings
pd.set_option('future.no_silent_downcasting', True)

# %%
# ===== CONFIGURATION =====
MA_PERIOD = 250
TF = "4h"
INITIAL_CASH = 1000
FEE = 0.0004
TARGET_VOLATILITY = 0.25
VOLATILITY_WINDOW = 10

# %%
# ===== DATA PREPARATION =====
data = pd.read_csv('data/4h_NASDAQ.csv', sep='\t', parse_dates=['DateTime']).set_index('DateTime').iloc[::-1]
data.columns = [c.lower() for c in data.columns]

# %%
# ===== KALMAN FILTER =====
@njit
def get_kalman_filter(data, Q=1e-5, R=0.001):
    n = len(data)
    xhat = np.zeros(n)      
    P = np.zeros(n)         
    xhatminus = np.zeros(n) 
    Pminus = np.zeros(n)    
    K = np.zeros(n)         

    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat

data['close_kalman'] = get_kalman_filter(data['close'].values)

# %%
# ===== TECHNICAL INDICATORS =====
data['SMA_225'] = data['close_kalman'].rolling(window=MA_PERIOD).mean()

# Volatility sizing
returns = data['close'].pct_change()
bars_per_year = 365 * 6 
rolling_volatility = returns.rolling(window=VOLATILITY_WINDOW).std() * np.sqrt(bars_per_year)
vol_weights = (TARGET_VOLATILITY / rolling_volatility).vbt.fshift(1)
vol_weights = vol_weights.fillna(0).clip(upper=1.0)

# %%
# ===== ENTRY/EXIT SIGNALS =====
price_above_ma = data["close_kalman"] > data["SMA_225"]
price_below_ma = data["close_kalman"] < data["SMA_225"]

long_entries = price_above_ma & ~price_above_ma.shift(1).fillna(False).astype(bool)
long_exits = price_below_ma & ~price_below_ma.shift(1).fillna(False).astype(bool)

short_entries = pd.Series(False, index=data.index)
short_exits = pd.Series(False, index=data.index)

sizing = vol_weights.mask(~price_above_ma, 0) 

# %%
# ===== PORTFOLIO =====
portfolio = vbt.Portfolio.from_signals(
    close=data["close"],
    entries=long_entries,
    exits=long_exits,
    short_entries=short_entries,
    short_exits=short_exits,
    size=sizing,
    size_type="percent",
    init_cash=INITIAL_CASH,
    fees=FEE,
    freq=TF
)

vbt.settings.set_theme("dark")
# %%
# ===== ENHANCED PLOT (New Way - Add Indicators) =====
print("\n===== Enhanced Portfolio Plot with Indicators =====")

# Start with portfolio plot
fig = portfolio.plot(
    subplots=['orders', 'trade_pnl',"cum_returns"],
    subplot_settings={
        "orders": {"close_trace_kwargs": {"visible": True}}
    }
)

# Add Kalman filtered price (overlay on orders subplot - row 1)
fig = add_indicator(
    fig, 
    data['close_kalman'], 
    row=1, 
    name='Kalman Filter',
    trace_kwargs=dict(line=dict(color='cyan', width=1.5))
)

# Add SMA (overlay on orders subplot - row 1)
fig = add_indicator(
    fig, 
    data['SMA_225'], 
    row=1, 
    name='SMA 225',
    trace_kwargs=dict(line=dict(color='orange', width=2))
)

# Remove weekend gaps
fig = remove_date_gaps(fig, data)

# Final styling
fig.update_layout(
    title_text="NasdaqMA Strategy - Enhanced with Indicators",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.show()

# %%
print("\n===== Portfolio Stats =====")
stats = portfolio.stats()
print(stats)
