"""Debug path MC - check if values are truly zero or just display issue"""

from monte_carlo_path import run_path_randomization_mc
from strategy_registry import create_portfolio
import pandas as pd
import numpy as np

print("="*80)
print("DEBUG PATH MC")
print("="*80)

# Load data
try:
    data = pd.read_csv('data/1h_NASDAQ.csv', sep='\t', parse_dates=['DateTime'])
    data.set_index('DateTime', inplace=True)
    data = data.iloc[::-1].iloc[-2000:]
    data.columns = data.columns.str.lower()
    print(f"✅ Loaded {len(data)} bars")
except:
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(2000) * 0.5)
    data = pd.DataFrame({
        'open': close, 'high': close + 1, 'low': close - 1,
        'close': close, 'volume': 1e6
    }, index=pd.date_range('2022', periods=2000, freq='1H'))

# Create portfolio
portfolio = create_portfolio('supertrend_grid', data, {
    'st_period': 10, 'st_multiplier': 3.0,
    'grid_levels': 5, 'grid_range': 0.05,
    'initial_cash': 10000, 'fee': 0.001, 'freq': '1H'
})

print(f"\n📊 Portfolio stats:")
print(f"   Return: {portfolio.total_return()}")
print(f"   Sharpe: {portfolio.sharpe_ratio()}")
print(f"   Trades: {portfolio.trades.count()}")

returns = portfolio.returns()
print(f"\n📊 Returns analysis:")
print(f"   Shape: {returns.shape}")
print(f"   Mean: {returns.mean():.10f}")  # More precision
print(f"   Std: {returns.std():.10f}")    # More precision
print(f"   Min: {returns.min():.10f}")
print(f"   Max: {returns.max():.10f}")
print(f"   Non-zero: {(returns != 0).sum()}")
print(f"   Unique values: {len(returns.unique())}")

# Run small MC
print("\n🎲 Running MC with 50 simulations...")
results = run_path_randomization_mc(portfolio, n_simulations=50, method='shuffle_returns', seed=42)

stats = results['statistics']
mc_returns = results['simulated_returns']

print(f"\n📊 MC Results (RAW VALUES):")
print(f"   Original return: {stats['original_return']}")
print(f"   MC returns array shape: {mc_returns.shape}")
print(f"   MC returns mean: {np.mean(mc_returns):.10f}")
print(f"   MC returns std: {np.std(mc_returns):.10f}")  # Raw numpy std
print(f"   MC returns min: {mc_returns.min():.10f}")
print(f"   MC returns max: {mc_returns.max():.10f}")
print(f"   MC returns unique: {len(np.unique(mc_returns))}")

print(f"\n📊 First 10 MC returns:")
for i, ret in enumerate(mc_returns[:10]):
    print(f"   Sim {i+1}: {ret:.10f}%")

print(f"\n📊 Stats dict:")
print(f"   mean_mc_return: {stats['mean_mc_return']}")
print(f"   std_mc_return: {stats['std_mc_return']}")

# Check if all returns are identical
if len(np.unique(mc_returns)) == 1:
    print(f"\n❌ PROBLEM: All MC returns are IDENTICAL!")
    print(f"   This means shuffling is not working")
else:
    print(f"\n✅ MC returns are varied ({len(np.unique(mc_returns))} unique values)")
