import pandas as pd
import vectorbt as vbt
import numpy as np
from numba import njit

# ===== CONFIGURATION =====
TF = "4h"
INITIAL_CASH = 1000
FEE = 0.0004
# 4h = 6 bars/day * 365 days = 2190 bars/year
ANNUALIZATION_FACTOR = np.sqrt(365 * 6)

# --- PARAMETER RANGES ---
# Strategy Parameters
MA_PERIODS = range(100, 300, 25)
Q_VALUES = [1e-5, 1e-4, 1e-3]
R_VALUES = [1e-3, 1e-2, 1e-1]

# Sizing Parameters
TARGET_VOLS = np.arange(0.10, 0.65, 0.05)
VOL_WINDOWS = np.arange(5, 100, 5)

pd.set_option('future.no_silent_downcasting', True)

# ===== LOAD DATA =====
print("Loading Data...")
try:
    data = pd.read_csv('data/4h_NASDAQ.csv', sep='\t', parse_dates=['DateTime']).set_index('DateTime').iloc[::-1]
except FileNotFoundError:
    data = vbt.YFData.download("QQQ", period="2y", interval="1h").get()
    
data.columns = data.columns.str.lower()
close = data['close']
returns = close.pct_change()

# ===== KALMAN FILTER =====
@njit
def get_kalman_filter(data, Q=1e-5, R=0.001):
    n = len(data); xhat = np.zeros(n); P = np.zeros(n); xhatminus = np.zeros(n); Pminus = np.zeros(n); K = np.zeros(n)
    xhat[0] = data[0]; P[0] = 1.0
    for k in range(1, n):
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q; K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k]); P[k] = (1 - K[k]) * Pminus[k]
    return xhat

# ===== STEP 1: GENERATE SIGNALS (MA, Q, R) =====
print(f"⚡ Generating Signals for {len(MA_PERIODS)*len(Q_VALUES)*len(R_VALUES)} Strategy Combinations...")

signal_cols = []
trend_signals = []

for q in Q_VALUES:
    for r in R_VALUES:
        # Calculate Kalman once per Q/R pair
        kalman_line = get_kalman_filter(close.values, Q=q, R=r)
        kalman_series = pd.Series(kalman_line, index=close.index)
        
        for ma in MA_PERIODS:
            ma_line = kalman_series.rolling(window=ma).mean()
            
            # Trend State: Price > MA
            is_bullish = (kalman_series > ma_line)
            
            trend_signals.append(is_bullish)
            signal_cols.append((ma, q, r))

# DataFrame of Trend Signals (True/False)
trend_df = pd.concat(trend_signals, axis=1)
trend_df.columns = pd.MultiIndex.from_tuples(signal_cols, names=['MA', 'Q', 'R'])

# ===== STEP 2: GENERATE VOLATILITY WEIGHTS (TargetVol, Window) =====
print(f"⚡ Generating Weights for {len(TARGET_VOLS)*len(VOL_WINDOWS)} Sizing Combinations...")

weight_cols = []
raw_weights = []

for w_bars in VOL_WINDOWS:
    # Calculate Rolling Volatility
    vol_rolling = returns.rolling(window=w_bars).std() * ANNUALIZATION_FACTOR
    
    for target_vol in TARGET_VOLS:
        # Weight Formula
        w = (target_vol / vol_rolling).vbt.fshift(1)
        w = w.fillna(0.0).replace([np.inf, -np.inf], 0.0).clip(upper=1.0)
        
        raw_weights.append(w)
        weight_cols.append((target_vol, w_bars))

# DataFrame of Unmasked Weights
weight_df = pd.concat(raw_weights, axis=1)
weight_df.columns = pd.MultiIndex.from_tuples(weight_cols, names=['Target_Vol', 'Vol_Window'])

# ===== STEP 3: BROADCAST & COMBINE =====
print("⚡ Broadcasting Signals x Weights...")

n_strat = trend_df.shape[1]
m_sizing = weight_df.shape[1]

# Tile/Repeat indices to create the cross-product
strat_indices = np.repeat(np.arange(n_strat), m_sizing)
sizing_indices = np.tile(np.arange(m_sizing), n_strat)

# Combine values (Boolean Mask * Float Weight)
final_weights_values = trend_df.iloc[:, strat_indices].values * weight_df.iloc[:, sizing_indices].values

# Combine Column Indices
strat_cols = trend_df.columns[strat_indices]
sizing_cols = weight_df.columns[sizing_indices]
new_col_tuples = [s + w for s, w in zip(strat_cols, sizing_cols)]

# Create Final DataFrame
final_weights = pd.DataFrame(
    final_weights_values, 
    index=close.index, 
    columns=pd.MultiIndex.from_tuples(new_col_tuples, names=['MA', 'Q', 'R', 'Target_Vol', 'Vol_Window'])
)

print(f"  -> Total Combinations: {final_weights.shape[1]}")

# ===== STEP 4: RUN SIMULATION =====
print("🚀 Running Portfolio Simulation (from_signals)...")

# Flatten columns to strings for VBT
flat_cols = final_weights.columns.to_flat_index().map(str)
final_weights_flat = final_weights.copy()
final_weights_flat.columns = flat_cols

# Broadcast entries and exits to match final_weights shape
# Re-construct broadcasted signals using the trend_df
trend_bool = trend_df.astype(bool)
entries_bool = trend_bool & ~trend_bool.shift(1).fillna(False)
exits_bool = ~trend_bool & trend_bool.shift(1).fillna(False)

# Broadcast to (N*M) columns using the same strat_indices
entries_broadcast = entries_bool.iloc[:, strat_indices]
exits_broadcast = exits_bool.iloc[:, strat_indices]

# Assign the flat columns to match weights
entries_broadcast.columns = flat_cols
exits_broadcast.columns = flat_cols

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries_broadcast,
    exits=exits_broadcast,
    size=final_weights_flat,
    size_type='Percent',
    init_cash=INITIAL_CASH,
    fees=FEE,
    freq=TF,
    cash_sharing=True,
    group_by=flat_cols
)

# ===== STEP 5: ANALYZE RESULTS =====
print("📊 Analyzing Results...")

stats = pd.DataFrame({
    'Calmar': pf.calmar_ratio(),
    'Sharpe': pf.sharpe_ratio(),
    'Total_Return': pf.total_return(),
    'Max_DD': pf.max_drawdown()
})

# Restore MultiIndex for grouping/analysis
stats.index = final_weights.columns

# Find Global Best
best_idx = stats['Calmar'].idxmax()
best_global = stats.loc[best_idx]

print("\n" + "="*60)
print("🏆 GLOBAL BEST PARAMETERS 🏆")
print("="*60)
print(f"MA Period:       {best_idx[0]}")
print(f"Kalman Q:        {best_idx[1]}")
print(f"Kalman R:        {best_idx[2]}")
print(f"Target Vol:      {best_idx[3]:.2f}")
print(f"Vol Window:      {best_idx[4]}")
print("-" * 40)
print(f"Calmar:          {best_global['Calmar']:.4f}")
print(f"Sharpe:          {best_global['Sharpe']:.4f}")
print(f"Return:          {best_global['Total_Return']:.2%}")
print("="*60)

# Find Best Sizing per Strategy Config
print("\n🔍 Best Target Vol & Window for each Strategy Config (Top 10 by Calmar):")
best_per_config_idx = stats.groupby(level=['MA', 'Q', 'R'])['Calmar'].idxmax()
best_per_config = stats.loc[best_per_config_idx].sort_values('Calmar', ascending=False)
print(best_per_config.head(10).to_string())

stats.to_csv('optimization_results.csv')
print("\n✅ Results saved to 'optimization_results.csv'")