import pandas as pd
import vectorbt as vbt
import numpy as np
from numba import njit

# ===== CONFIGURATION =====
MA_PERIOD = 225
TF = "4h"
INITIAL_CASH = 1000
FEE = 0.0004

# INFO : On fixe l'annualisation pour que le calcul de volatilité soit stable
# 4h = 6 barres par jour * 365 jours = 2190 barres par an
ANNUALIZATION_FACTOR = np.sqrt(365 * 6)

# PLAGES D'OPTIMISATION (EN BARRES BRUTES)
TARGET_VOL_RANGE = np.arange(0.10, 0.65, 0.05) 

# On teste des fenêtres de 5 barres à 100 barres (plus besoin de multiplier par 6)
WINDOW_BARS_RANGE = np.arange(5, 100, 5) 

pd.set_option('future.no_silent_downcasting', True)

# ===== LOAD DATA =====
print("Loading Data...")
try:
    data = pd.read_csv('data/4h_NASDAQ.csv', sep='\t', parse_dates=['DateTime']).set_index('DateTime').iloc[::-1]
except FileNotFoundError:
    data = vbt.YFData.download("QQQ", period="2y", interval="1h").get()
data.columns = data.columns.str.lower()
close = data['close']

# ===== INDICATORS =====
@njit
def get_kalman_filter(data, Q=1e-5, R=0.001):
    n = len(data); xhat = np.zeros(n); P = np.zeros(n); xhatminus = np.zeros(n); Pminus = np.zeros(n); K = np.zeros(n)
    xhat[0] = data[0]; P[0] = 1.0
    for k in range(1, n):
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q; K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k]); P[k] = (1 - K[k]) * Pminus[k]
    return xhat

data['close_kalman'] = get_kalman_filter(close.values)
ma_line = data['close_kalman'].rolling(window=MA_PERIOD).mean()
price_above_ma = data['close_kalman'] > ma_line

# Signals bases
entries_base = (price_above_ma & ~price_above_ma.shift(1).fillna(False)).astype(bool)
exits_base = (~price_above_ma & price_above_ma.shift(1).fillna(False)).astype(bool)

# ===== BUILD MATRIX =====
print("Building Optimization Matrix...")
returns = close.pct_change()
sizers = []
combo_labels = []

for w_bars in WINDOW_BARS_RANGE:
    # Ici on utilise w_bars DIRECTEMENT. Pas de multiplication.
    vol_rolling = returns.rolling(window=w_bars).std() * ANNUALIZATION_FACTOR
    
    for target_vol in TARGET_VOL_RANGE:
        weight = (target_vol / vol_rolling).vbt.fshift(1)
        weight = weight.fillna(0.0).replace([np.inf, -np.inf], 0.0).clip(upper=1.0)
        weight = weight.mask(~price_above_ma, 0.0)
        
        sizers.append(weight)
        combo_labels.append((target_vol, w_bars))

sizing_df = pd.concat(sizers, axis=1)
sizing_df.columns = pd.MultiIndex.from_tuples(combo_labels, names=['Target_Vol', 'Window_Bars'])

entries_broadcast = entries_base.vbt.tile(len(combo_labels), keys=sizing_df.columns)
exits_broadcast = exits_base.vbt.tile(len(combo_labels), keys=sizing_df.columns)

# ===== RUN =====
print("Running Simulation...")
pf = vbt.Portfolio.from_signals(
    close=close, entries=entries_broadcast, exits=exits_broadcast,
    size=sizing_df, size_type='Percent',
    init_cash=INITIAL_CASH, fees=FEE, freq=TF
)

# ===== RESULTS =====
res = pd.DataFrame({
    'Calmar': pf.calmar_ratio(),
    'Sharpe': pf.sharpe_ratio(),
    'Total_Return': pf.total_return(),
    'Max_DD': pf.max_drawdown()
}).reset_index()

best_calmar = res.loc[res['Calmar'].idxmax()]

print("\n" + "="*50)
print("🏆 VRAI RÉSULTAT (EN BARRES) 🏆")
print("="*50)
print(f"Target Vol:      {best_calmar['Target_Vol']:.2f}")
print(f"Window Bars:     {int(best_calmar['Window_Bars'])}")
print("-" * 30)
print(f"Calmar:          {best_calmar['Calmar']:.4f}")
print(f"Sharpe:          {best_calmar['Sharpe']:.4f}")
print(f"Return:          {best_calmar['Total_Return']:.2%}")
print("="*50)