import numpy as np
import pandas as pd
import vectorbt as vbt

# 1. PARSE USER DATA (Cost Analysis)
# ==============================================================================
data_str = """
Symbol,Price,Spread,Comm
EURUSD,1.1584,0.00002,5.00
GBPUSD,1.3226,0.00004,5.00
USDJPY,156.26,0.009,5.00
US30,47401,1.58,0.00
US100,25251,1.40,0.00
US500,6812,0.46,0.00
BTCUSD,91359,1.00,113.86
ETHUSD,3026,0.60,38.48
XAUUSD,4160,0.29,11.60
USOIL,58.97,0.010,0.00
"""
# Note: Simplified parsing for demo. 'Comm' assumed per standard lot or contract equivalent.


# 2. THE GENERIC "EV+" SIGNAL ENGINE
# ==============================================================================
def generate_adaptive_signals(close, high, low, window=50):
    """
    Returns a signal between -1 (Strong Short) and +1 (Strong Long).
    Logic: Volatility-Adjusted Momentum (Sharpe-like signal).
    """
    # 1. Trend Component (Drift)
    # distance from MA normalized by volatility
    ma = vbt.talib("SMA").run(close, timeperiod=window).real
    std = close.rolling(window).std()

    # Z-Score of price relative to MA (Generic Trend)
    trend_score = (close - ma) / std

    # 2. Regime Component (Volatility Filter)
    # If recent volatility is 2x higher than average, reduce exposure (Defensive)
    vol_ma = std.rolling(window * 2).mean()
    vol_regime = np.where(std > vol_ma, 0.5, 1.0)  # Penalty for high vol

    # Combined Signal (Capped at -1 to 1 for sizing)
    raw_signal = trend_score * vol_regime
    return np.clip(raw_signal, -1.0, 1.0)  # Standardized Alpha


# 3. PORTFOLIO CONSTRUCTION (MPT / Risk Parity)
# ==============================================================================
def allocate_portfolio(close, signals, vol_lookback=20):
    """
    Allocates weights based on Signal Strength / Volatility.
    This creates a 'Risk Parity' portfolio with a directional bias.
    """
    # Calculate realized volatility (risk)
    returns = close.pct_change()
    asset_vol = returns.rolling(vol_lookback).std()

    # Inverse Volatility Weighting (The Safety Layer)
    # Safe assets get more weight, Risky assets get less
    inv_vol = 1 / asset_vol

    # Combine Signal (Alpha) with Safety (Risk)
    # We use abs(signals) because direction is handled by the sign later
    raw_weights = inv_vol * signals.abs()

    # Normalize weights so they sum to 1.0 (or 100% exposure)
    total_weight = raw_weights.sum(axis=1)
    normalized_weights = raw_weights.div(total_weight, axis=0)

    # Apply Direction (Long/Short)
    final_weights = normalized_weights * np.sign(signals)

    return final_weights


# ==============================================================================
# SIMULATION HARNESS
# ==============================================================================

# Mock Data Generation for the backtest (Since we don't have CSVs loaded)
# We will simulate price paths with properties matching your asset classes
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=300, freq="4h")
assets = ["EURUSD", "US30", "BTCUSD", "XAUUSD"]

# Generate random walk with drift (US30/BTC uptrend, EURUSD choppy)
prices = pd.DataFrame(index=dates, columns=assets)
prices["EURUSD"] = 1.0 + np.random.randn(300).cumsum() * 0.001
prices["US30"] = 30000 + np.random.randn(300).cumsum() * 100 + np.linspace(0, 5000, 300)
prices["BTCUSD"] = (
    40000 + np.random.randn(300).cumsum() * 500 + np.linspace(0, 40000, 300)
)  # High Vol
prices["XAUUSD"] = 2000 + np.random.randn(300).cumsum() * 10

# Run Strategy
signals = generate_adaptive_signals(prices, None, None, window=50)
weights = allocate_portfolio(prices, signals)

# Run VectorBT Backtest
pf = vbt.Portfolio.from_orders(
    prices,
    size=weights,
    size_type="targetpercent",
    freq="4h",
    init_cash=100_000,
    fees=0.0005,  # Approx 5bps avg friction across universe
)

print(f"Combined Portfolio Sortino: {pf.sortino_ratio():.2f}")
print(f"Combined Portfolio Sharpe:  {pf.sharpe_ratio():.2f}")
print("\n--- Allocation Snapshot (Last Row) ---")
print(weights.iloc[-1].round(3))
