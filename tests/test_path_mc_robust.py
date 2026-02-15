from vectorflow.validation.path_randomization import run_path_randomization_mc
from vectorflow.visualization.plotters import plot_path_mc_results
from vectorflow.core.portfolio_builder import create_portfolio
import numpy as np
import pandas as pd

# Load real data
print("\n📊 Loading data...")
try:
    data = pd.read_csv('data/1h_NASDAQ.csv', sep='\t', parse_dates=['DateTime'])
    data.set_index('DateTime', inplace=True)
    data = data.iloc[::-1].iloc[-2000:]  # Chronological, last 2000
    data.columns = data.columns.str.lower()
    print(f"   ✅ {len(data)} bars NASDAQ 1H")
except Exception as e:
    print(f"   Fallback to synthetic data")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(2000) * 0.5)
    data = pd.DataFrame({
        'open': close + np.random.randn(2000) * 0.3,
        'high': close + abs(np.random.randn(2000)),
        'low': close - abs(np.random.randn(2000)),
        'close': close,
        'volume': abs(np.random.randn(2000)) * 1e6 + 1e6
    }, index=pd.date_range('2022', periods=2000, freq='1H'))

# Create portfolio with Supertrend Grid
print("\n🔧 Creating Supertrend Grid portfolio...")
portfolio = create_portfolio('supertrend_grid', data, {
    'st_period': 10, 'st_multiplier': 3.0,
    'grid_levels': 5, 'grid_range': 0.05,
    'initial_cash': 10000, 'fee': 0.001, 'freq': '1H'
})

if not portfolio:
    print("❌ Strategy failed")
    exit(1)

# Stats
print(f"\n📊 Portfolio:")
print(f"   Return: {portfolio.total_return():.2f}%")
print(f"   Sharpe: {portfolio.sharpe_ratio():.3f}")
print(f"   Trades: {portfolio.trades.count()}")

if portfolio.returns().std() == 0:
    print("❌ No variance")
    exit(1)

# Run Path MC
print("\n" + "="*80)
print("PATH RANDOMIZATION MONTE CARLO")
print("="*80)

results = run_path_randomization_mc(portfolio, n_simulations=500, method='shuffle_returns', seed=42)

stats = results['statistics']
print(f"\n✅ Completed!")
print(f"   Original: {stats['original_return']:.2f}%")
print(f"   MC Mean: {stats['mean_mc_return']:.2f}% (±{stats['std_mc_return']:.2f}%)")
print(f"   Percentile: {stats['percentile_rank_return']:.1f}%")
print(f"   P-value: {stats['p_value_return']:.4f} {'✅' if stats['is_significant_return'] else '⚠️'}")

# Plot
try:
    plot_path_mc_results(results)
    print("\n✅ Plot displayed")
except Exception as e:
    print(f"⚠️ Plot failed: {e}")

print("\n" + "="*80)
print("✅ TEST COMPLETE - SUPERTREND GRID")
print("="*80)