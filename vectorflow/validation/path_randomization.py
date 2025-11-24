#!/usr/bin/env python3
"""
Path Randomization Monte Carlo Analysis

Tests portfolio robustness by randomizing the sequence of returns/trades.
This is different from parameter Monte Carlo - it tests if the results are
dependent on the specific order of market outcomes.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Optional
from scipy import stats


def run_path_randomization_mc(
    portfolio: vbt.Portfolio,
    n_simulations: int = 1000,
    method: str = "shuffle_returns",
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run path randomization Monte Carlo on an existing portfolio.
    
    This tests if the strategy's performance is robust to the sequence
    of market events, or if it's just lucky timing.
    
    Args:
        portfolio: Existing VectorBT portfolio object
        n_simulations: Number of randomized paths to generate
        method: 'shuffle_returns' or 'bootstrap_trades'
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with simulation results and statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"🎲 Path Randomization Monte Carlo ({n_simulations} simulations)")
    print(f"   Method: {method}")
    
    # Extract original portfolio data
    original_returns = portfolio.returns()
    original_total_return = portfolio.total_return()
    original_sharpe = portfolio.sharpe_ratio()
    original_max_dd = portfolio.max_drawdown()
    
    if original_returns is None or len(original_returns) == 0:
        raise ValueError("Portfolio has no returns data")
    
    print(f"   Original Total Return: {original_total_return:.2f}%")
    print(f"   Original Sharpe: {original_sharpe:.3f}")
    print(f"   Original Max DD: {original_max_dd:.2f}%")
    
    # Run simulations based on method
    if method == "shuffle_returns":
        results = _shuffle_returns_method(original_returns, n_simulations)
    elif method == "bootstrap_trades":
        results = _bootstrap_trades_method(portfolio, n_simulations)
    elif method == "block_bootstrap":
        results = _block_bootstrap_method(original_returns, n_simulations)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate statistics
    mc_returns = results['simulated_returns']
    mc_sharpes = results['simulated_sharpes']
    mc_max_dds = results['simulated_max_dds']
    
    statistics = {
        'original_return': float(original_total_return) if original_total_return is not None else 0.0,
        'original_sharpe': float(original_sharpe) if original_sharpe is not None else 0.0,
        'original_max_dd': float(original_max_dd) if original_max_dd is not None else 0.0,
        'mean_mc_return': float(np.mean(mc_returns)),
        'std_mc_return': float(np.std(mc_returns)),
        'mean_mc_sharpe': float(np.mean(mc_sharpes)),
        'std_mc_sharpe': float(np.std(mc_sharpes)),
        'mean_mc_max_dd': float(np.mean(mc_max_dds)),
        'percentile_return_5': float(np.percentile(mc_returns, 5)),
        'percentile_return_95': float(np.percentile(mc_returns, 95)),
        'percentile_sharpe_5': float(np.percentile(mc_sharpes, 5)),
        'percentile_sharpe_95': float(np.percentile(mc_sharpes, 95)),
    }
    
    # Calculate p-values (how extreme is the original result?)
    percentile_rank_return = stats.percentileofscore(mc_returns, original_total_return)
    percentile_rank_sharpe = stats.percentileofscore(mc_sharpes, original_sharpe)
    
    statistics['percentile_rank_return'] = float(percentile_rank_return)
    statistics['percentile_rank_sharpe'] = float(percentile_rank_sharpe)
    
    # Two-tailed p-value
    p_value_return = 2 * min(percentile_rank_return, 100 - percentile_rank_return) / 100
    p_value_sharpe = 2 * min(percentile_rank_sharpe, 100 - percentile_rank_sharpe) / 100
    
    statistics['p_value_return'] = float(p_value_return)
    statistics['p_value_sharpe'] = float(p_value_sharpe)
    statistics['is_significant_return'] = p_value_return < 0.05
    statistics['is_significant_sharpe'] = p_value_sharpe < 0.05
    
    # Print summary
    print(f"\n📊 Monte Carlo Results:")
    print(f"   Mean MC Return: {statistics['mean_mc_return']:.2f}% (±{statistics['std_mc_return']:.2f}%)")
    print(f"   Mean MC Sharpe: {statistics['mean_mc_sharpe']:.3f} (±{statistics['std_mc_sharpe']:.3f})")
    print(f"   Original Return Percentile: {percentile_rank_return:.1f}%")
    print(f"   Original Sharpe Percentile: {percentile_rank_sharpe:.1f}%")
    print(f"   P-value (Return): {p_value_return:.4f} {'✅ Significant' if p_value_return < 0.05 else '⚠️ Not significant'}")
    print(f"   P-value (Sharpe): {p_value_sharpe:.4f} {'✅ Significant' if p_value_sharpe < 0.05 else '⚠️ Not significant'}")
    
    return {
        'statistics': statistics,
        'simulated_returns': mc_returns,
        'simulated_sharpes': mc_sharpes,
        'simulated_max_dds': mc_max_dds,
        'equity_paths': results.get('equity_paths'),
        'method': method,
        'n_simulations': n_simulations
    }


def _shuffle_returns_method(returns: pd.Series, n_simulations: int) -> Dict[str, Any]:
    """
    Method 1: Shuffle Returns
    
    Randomly permute the order of returns to see if sequence matters.
    This preserves the distribution of returns but randomizes timing.
    """
    print("   Using shuffle returns method...")
    
    returns_array = returns.values
    simulated_returns = []
    simulated_sharpes = []
    simulated_max_dds = []
    equity_paths = []
    
    for i in range(n_simulations):
        # Shuffle returns
        shuffled = np.random.permutation(returns_array)
        
        # Calculate equity curve
        equity = (1 + shuffled).cumprod()
        equity_paths.append(equity)
        
        # Calculate metrics
        total_return = (equity[-1] - 1) * 100  # Percentage
        
        # Sharpe ratio
        if len(shuffled) > 1 and np.std(shuffled) > 0:
            sharpe = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_dd = np.min(drawdown)
        
        simulated_returns.append(total_return)
        simulated_sharpes.append(sharpe)
        simulated_max_dds.append(max_dd)
        
        if (i + 1) % 200 == 0:
            print(f"   Progress: {i + 1}/{n_simulations}")
    
    return {
        'simulated_returns': np.array(simulated_returns),
        'simulated_sharpes': np.array(simulated_sharpes),
        'simulated_max_dds': np.array(simulated_max_dds),
        'equity_paths': np.array(equity_paths)
    }


def _bootstrap_trades_method(portfolio: vbt.Portfolio, n_simulations: int) -> Dict[str, Any]:
    """
    Method 2: Bootstrap Trades
    
    Randomly sample trades with replacement to create alternate histories.
    This tests if the strategy works with different combinations of trades.
    """
    print("   Using bootstrap trades method...")
    
    # Extract trades
    trades_df = portfolio.trades.records_readable
    
    if len(trades_df) == 0:
        raise ValueError("Portfolio has no trades to bootstrap")
    
    # Get P&L per trade
    trade_returns = trades_df['PnL'].values / portfolio.init_cash
    
    simulated_returns = []
    simulated_sharpes = []
    simulated_max_dds = []
    equity_paths = []
    
    for i in range(n_simulations):
        # Bootstrap: sample with replacement
        sampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
        
        # Calculate equity curve
        equity = (1 + sampled_returns).cumprod()
        equity_paths.append(equity)
        
        # Calculate metrics
        total_return = (equity[-1] - 1) * 100
        
        # Sharpe
        if len(sampled_returns) > 1 and np.std(sampled_returns) > 0:
            sharpe = np.mean(sampled_returns) / np.std(sampled_returns) * np.sqrt(len(trade_returns))
        else:
            sharpe = 0.0
        
        # Max DD
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_dd = np.min(drawdown)
        
        simulated_returns.append(total_return)
        simulated_sharpes.append(sharpe)
        simulated_max_dds.append(max_dd)
        
        if (i + 1) % 200 == 0:
            print(f"   Progress: {i + 1}/{n_simulations}")
    
    return {
        'simulated_returns': np.array(simulated_returns),
        'simulated_sharpes': np.array(simulated_sharpes),
        'simulated_max_dds': np.array(simulated_max_dds),
        'equity_paths': np.array(equity_paths)
    }


def _block_bootstrap_method(returns: pd.Series, n_simulations: int, block_size: int = 20) -> Dict[str, Any]:
    """
    Method 3: Block Bootstrap
    
    Sample blocks of consecutive returns to preserve short-term correlation.
    More realistic than pure shuffling for time series data.
    """
    print(f"   Using block bootstrap method (block_size={block_size})...")
    
    returns_array = returns.values
    n_returns = len(returns_array)
    
    simulated_returns = []
    simulated_sharpes = []
    simulated_max_dds = []
    equity_paths = []
    
    for i in range(n_simulations):
        # Create blocks
        resampled = []
        while len(resampled) < n_returns:
            # Random starting point
            start = np.random.randint(0, max(1, n_returns - block_size + 1))
            block = returns_array[start:start + block_size]
            resampled.extend(block)
        
        # Trim to original length
        resampled = np.array(resampled[:n_returns])
        
        # Calculate equity curve
        equity = (1 + resampled).cumprod()
        equity_paths.append(equity)
        
        # Calculate metrics
        total_return = (equity[-1] - 1) * 100
        
        # Sharpe
        if len(resampled) > 1 and np.std(resampled) > 0:
            sharpe = np.mean(resampled) / np.std(resampled) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max DD
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_dd = np.min(drawdown)
        
        simulated_returns.append(total_return)
        simulated_sharpes.append(sharpe)
        simulated_max_dds.append(max_dd)
        
        if (i + 1) % 200 == 0:
            print(f"   Progress: {i + 1}/{n_simulations}")
    
    return {
        'simulated_returns': np.array(simulated_returns),
        'simulated_sharpes': np.array(simulated_sharpes),
        'simulated_max_dds': np.array(simulated_max_dds),
        'equity_paths': np.array(equity_paths)
    }





# Example usage
if __name__ == "__main__":
    import vectorbt as vbt
    import pandas as pd
    import numpy as np
    
    # Example: Create a simple portfolio
    np.random.seed(42)
    close = pd.Series(np.random.randn(252).cumsum() + 100, index=pd.date_range('2020', periods=252))
    
    # Simple MA crossover
    fast_ma = close.rolling(10).mean()
    slow_ma = close.rolling(50).mean()
    
    entries = fast_ma > slow_ma
    exits = fast_ma < slow_ma
    
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001
    )
    
    # Run path randomization Monte Carlo
    mc_results = run_path_randomization_mc(
        portfolio,
        n_simulations=1000,
        method="shuffle_returns",
        seed=42
    )
    
    # Plot results
    from vectorflow.visualization.plotters import plot_path_mc_results
    plot_path_mc_results(mc_results)
