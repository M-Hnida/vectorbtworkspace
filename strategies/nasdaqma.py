# %%
import pandas as pd
import vectorbt as vbt
import numpy as np
import sys
import os
from numba import njit

# CORRECTION 1: Gestion du FutureWarning pandas
pd.set_option('future.no_silent_downcasting', True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
# ===== CONFIGURATION =====
MA_PERIOD = 225
TF = "4h"
INITIAL_CASH = 1000
FEE = 0.0004

# --- VOLATILITY TARGET CONFIGURATION ---
# Instead of fixed risk $, we target a specific Portfolio Volatility.
# 0.30 (30%) is aggressive but standard for Tech/Crypto. 
# It allows 100% exposure during normal times, but cuts risk during crashes.
TARGET_VOLATILITY = 0.35 
VOLATILITY_WINDOW = 5  # Lookback period to calculate current volatility

# %%
# ===== DATA PREPARATION =====

data = pd.read_csv('data/4h_NASDAQ.csv', sep='\t', parse_dates=['DateTime']).set_index('DateTime').iloc[::-1]

data.columns = [c.lower() for c in data.columns]

# %%
# ===== KALMAN FILTER =====
print("\n===== Applying Kalman Filter =====")
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
ma_col = "SMA_225"

# --- CALCULATING TARGET VOLATILITY SIZING ---
# 1. Calculate Returns
returns = data['close'].pct_change()

# 2. Calculate Annualized Rolling Volatility
# TF is 4h. Assuming Crypto/Forex 24/7 ~ 6 bars per day. 
# If Stocks (6.5h open) ~ 2 bars per day. 
# Adjust 'bars_per_year' based on your specific data source nature.
# Assuming 24/7 market for generic 4h calculation: 365 * 6 = 2190 bars
bars_per_year = 365 * 6 
rolling_volatility = returns.rolling(window=VOLATILITY_WINDOW).std() * np.sqrt(bars_per_year)

# 3. Calculate Target Weights
# If Vol is low (10%), Weight = 30%/10% = 3.0 (Leverage). 
# If Vol is high (60%), Weight = 30%/60% = 0.5 (Half position).
# We shift(1) because we can only size based on Yesterday's volatility.
vol_weights = (TARGET_VOLATILITY / rolling_volatility).vbt.fshift(1)

# 4. Cap at 1.0 (100% Cash) - Remove this line if you want Margin/Leverage
vol_weights = vol_weights.fillna(0).clip(upper=1.0)

print("\n===== Volatility Sizing Stats =====")
print(f"Target Volatility: {TARGET_VOLATILITY*100}%")
print(f"Avg Position Size: {vol_weights.mean():.2%}")
print(f"Min Position Size (Peak Fear): {vol_weights.min():.2%}")

# %%
# ===== ENTRY/EXIT SIGNALS =====
price_above_ma = data["close_kalman"] > data[ma_col]
price_below_ma = data["close_kalman"] < data[ma_col]

long_entries = price_above_ma & ~price_above_ma.shift(1).fillna(False).astype(bool)
long_exits = price_below_ma & ~price_below_ma.shift(1).fillna(False).astype(bool)

short_entries = pd.Series(False, index=data.index)
short_exits = pd.Series(False, index=data.index)

# ===== APPLY SIZING TO SIGNALS =====
# We use the vol_weights as the "TargetPercent" of the portfolio
# When NOT in a trade, weight is 0.
sizing = vol_weights.mask(~price_above_ma, 0) 
# Note: We use 'price_above_ma' (the state) rather than just 'long_entries' 
# because TargetPercent resizing runs on every bar to adjust to volatility changes.

# %%
# ===== PORTFOLIO =====
portfolio = vbt.Portfolio.from_signals(
    close=data["close"],
    entries=long_entries,
    exits=long_exits,
    short_entries=short_entries,
    short_exits=short_exits,
    size=sizing,
    size_type="percent",  # KEY CHANGE: We target a % of equity, not a fixed amount
    init_cash=INITIAL_CASH,
    fees=FEE,
    freq=TF
)

# %%
# ===== STATS & PLOTS =====
print("\n===== Portfolio Stats =====")

# Beta Calc
beta = portfolio.beta()
beta_stat = pd.Series(beta, index=["Beta vs Benchmark"])
stats = portfolio.stats()
stats = pd.concat([stats, beta_stat])
print(stats)

# Buy & Hold
buy_hold = vbt.Portfolio.from_holding(close=data["close"], init_cash=INITIAL_CASH, fees=FEE, freq=TF)
print(f"\nBuy & Hold Sharpe: {buy_hold.sharpe_ratio():.2f}")

# Visualization
vbt.settings.set_theme("dark")
fig = portfolio.plot(subplot_settings={"orders": {"close_trace_kwargs": {"visible": False}}})
fig = data.vbt.ohlcv.plot(plot_type="candlestick", fig=fig, show_volume=False, xaxis_rangeslider_visible=False)
fig = data[['close_kalman', ma_col]].vbt.plot(fig=fig)
fig.update_layout(title_text="225-Day MA - Target Volatility Sizing").show()