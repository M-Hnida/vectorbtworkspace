import pandas as pd
import numpy as np
import vectorbt as vbt

vbt.settings.set_theme('dark')
vbt.settings['array_wrapper']["freq"] = "4h"

symbols = ['BTCUSDT', 'ETHUSDT']

# ---------------- Data ----------------
data = {
    sym: vbt.CCXTData.download(sym, timeframe='4h', limit=1000).get()
    for sym in symbols
}

close = pd.concat([data[s]['Close'].rename(s) for s in symbols], axis=1).sort_index()
high = pd.concat([data[s]['High'].rename(s) for s in symbols], axis=1).reindex_like(close)
low = pd.concat([data[s]['Low'].rename(s) for s in symbols], axis=1).reindex_like(close)

for _df in (close, high, low):
    _df.columns.name = None
    _df.index.name = None

# ---------------- ATR (compute per symbol to avoid alignment issues) ----------------
atr_cols = {}
for sym in symbols:
    atr_sym = vbt.IndicatorFactory.from_talib('ATR').run(
        high[sym], low[sym], close[sym], timeperiod=14
    ).real
    atr_cols[sym] = atr_sym
atr = pd.DataFrame(atr_cols, index=close.index).reindex_like(close)

# Normalize to ATR% and clean
atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)

# ---------------- Volatility factor ----------------
atr_mean200 = atr_pct.rolling(200, min_periods=50).mean()
vol_factor = (atr_pct / atr_mean200).clip(lower=0.5, upper=2.0)
vol_factor = vol_factor.fillna(1.0)

# Adaptive target windows (float -> int later where needed)
tema_fast_window = (10 * vol_factor).round().clip(lower=6, upper=30)
tema_slow_window = (80 * vol_factor).round().clip(lower=30, upper=150)
tema_long_fast_window = (20 * vol_factor).round().clip(lower=10, upper=50)
tema_long_slow_window = (70 * vol_factor).round().clip(lower=30, upper=140)

# ---------------- Adaptive EMA implementation (per-bar alpha) ----------------
def adaptive_ema_recursive(price_df: pd.DataFrame, window_df: pd.DataFrame, min_periods: int = 5) -> pd.DataFrame:
    """Compute per-bar adaptive EMA with time-varying window -> alpha = 2/(w+1)."""
    out = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)
    for sym in price_df.columns:
        p = price_df[sym].astype(float)
        w = window_df[sym].round().clip(lower=1).fillna(method='ffill').astype(int)
        # Initialize EMA when we have enough data; otherwise, start from first non-nan
        ema = np.empty(len(p))
        ema[:] = np.nan
        # find first valid index
        first_idx = p.first_valid_index()
        if first_idx is None:
            out[sym] = ema
            continue
        start_i = max(p.index.get_loc(first_idx), 0)
        # Warm-up: simple mean over min_periods if available
        i0 = start_i
        warm_end = min(len(p), i0 + min_periods)
        if warm_end - i0 >= 1:
            ema[warm_end - 1] = np.nanmean(p.iloc[i0:warm_end])
            prev = ema[warm_end - 1]
            k = warm_end
        else:
            ema[i0] = p.iloc[i0]
            prev = ema[i0]
            k = i0 + 1

        for i in range(k, len(p)):
            px = p.iloc[i]
            if np.isnan(px):
                ema[i] = prev
                continue
            ww = int(w.iloc[i])
            alpha = 2.0 / (ww + 1.0)
            val = alpha * px + (1.0 - alpha) * prev
            ema[i] = val
            prev = val
        out[sym] = ema
    return out

tema_fast = adaptive_ema_recursive(close, tema_fast_window, min_periods=5)
tema_slow = adaptive_ema_recursive(close, tema_slow_window, min_periods=10)
tema_long_fast = adaptive_ema_recursive(close, tema_long_fast_window, min_periods=8)
tema_long_slow = adaptive_ema_recursive(close, tema_long_slow_window, min_periods=12)

# ---------------- ATR absolute, cleaned ----------------
atr_abs = (atr_pct * close).replace([np.inf, -np.inf], np.nan)
atr_abs = atr_abs.ffill().bfill()
atr_abs = atr_abs.clip(lower=1e-8)

# ---------------- Momentum filters ----------------
slope = tema_long_fast.pct_change(5).replace([np.inf, -np.inf], np.nan)

r = close.diff()
up = r.clip(lower=0).rolling(14, min_periods=7).sum()
dn = -r.clip(upper=0).rolling(14, min_periods=7).sum()
cmo_den = (up + dn).replace(0, np.nan)
cmo = 100 * (up - dn) / cmo_den

cmo_mean = cmo.rolling(100, min_periods=25).mean()
cmo_std = cmo.rolling(100, min_periods=25).std()
cmo_z = ((cmo - cmo_mean) / cmo_std).replace([np.inf, -np.inf], np.nan)

trend_short_ok = (tema_fast > tema_slow)
trend_long_ok = (tema_long_fast > tema_long_slow)
momentum_ok = (slope > 0) & (cmo_z > 0.5)

enter_long = (trend_short_ok & trend_long_ok & momentum_ok).fillna(False)

exit_long_trend_break = (tema_fast < tema_slow).fillna(False)

# ---------------- Market regime filter (BTC must be trending) ----------------
btc_filter = (tema_long_fast['BTCUSDT'] > tema_long_slow['BTCUSDT']).fillna(False)

# Explicit construction to ensure clear alignment
market_filter = pd.DataFrame({sym: btc_filter for sym in symbols}, index=close.index)

# Remove deduplication: allow entries whenever signal+filter are true; portfolio exposure/Stops control re-entries
entries = (enter_long & market_filter).fillna(False).astype(bool)

# Trend-break exits remain for pf1 (TP+SL) and pf2 (SL-only keeps exits too per request)
exits = exit_long_trend_break.fillna(False).astype(bool)

# ---------------- Risk sizing and stops ----------------
atr_mult_sl = 3.0
tp_mult = 4.5
risk_per_trade = 0.01
equity = 10000.0  # start capital in vectorbt defaults is 100. Match sizing scale.

size_units = (equity * risk_per_trade / (atr_mult_sl * atr_abs))
size_units = size_units.replace([np.inf, -np.inf], np.nan).fillna(0.0)

entry_price = close
sl_price = (entry_price - atr_mult_sl * atr_abs)
tp_price = (entry_price + tp_mult * atr_abs)

# Ensure stops have no NaNs; use forward-fill only to avoid contaminating early bars
sl_price = sl_price.ffill()
tp_price = tp_price.ffill()

# Guard entries until indicators and ATR are ready
ready_mask = (
    tema_long_slow.notna() &
    tema_long_fast.notna() &
    atr.notna()
).all(axis=1)

# Apply readiness across all symbols
entries = (entries & ready_mask.values[:, None]).astype(bool)

# ---------------- Build portfolios ----------------
pf1 = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    sl_stop=sl_price,
    tp_stop=tp_price,
    size=size_units * 0.5,  # half allocated here
    fees=0.0004,
    slippage=0.0005,
    init_cash=equity
)

# If you intended trailing stop, vectorbt uses ts_stop (distance) or sl_trail depending on version.
# Here we keep SL-only as requested, but remove unused variables.
pf2 = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    sl_stop=sl_price,
    size=size_units * 0.5,  # other half
    fees=0.0004,
    slippage=0.0005,
    init_cash=equity
)

print("Portfolio 1 (TP+SL):")
for sym in symbols:
    print(f"--- {sym} ---")
    pf1.plot(column=sym).show()
    print(pf1[sym].stats())

print("\nPortfolio 2 (SL only):")
for sym in symbols:
    print(f"--- {sym} ---")
    print(pf2[sym].stats())

