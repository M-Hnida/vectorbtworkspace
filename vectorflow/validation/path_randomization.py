#!/usr/bin/env python3
"""
Path Randomization Monte Carlo Analysis

Tests portfolio robustness by randomizing the sequence of trade outcomes.
Different from parameter Monte Carlo - tests path dependency, not parameter sensitivity.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Optional
from scipy import stats


def run_path_randomization_mc(
    portfolio: vbt.Portfolio,
    n_simulations: int = 1000,
    method: str = "bootstrap_trades",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run path randomization Monte Carlo on an existing portfolio.

    Tests if strategy performance is robust to trade sequence or just lucky timing.

    Args:
        portfolio: VectorBT portfolio object to analyze
        n_simulations: Number of randomized paths to generate (default: 1000)
        method: 'bootstrap_trades' (recommended) or 'shuffle_returns'
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results and statistics
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract original metrics (vectorbt returns fractions, so multiply by 100)
    original_total_return = portfolio.total_return() * 100  # Convert to percentage
    original_sharpe = portfolio.sharpe_ratio()
    original_max_dd = portfolio.max_drawdown() * 100  # Convert to percentage

    print(f"🎲 Path Randomization Monte Carlo ({n_simulations} simulations)")
    print(f"   Method: {method}")
    print(f"   Original Total Return: {original_total_return:.2f}%")
    print(f"   Original Sharpe: {original_sharpe:.3f}")
    print(f"   Original Max DD: {original_max_dd:.2f}%")

    # Run simulations
    if method == "bootstrap_trades":
        results = _bootstrap_trades(portfolio, n_simulations)
    elif method == "shuffle_returns":
        print("   ⚠️  Warning: shuffle_returns includes zero-return periods")
        results = _shuffle_returns(portfolio, n_simulations)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'bootstrap_trades' or 'shuffle_returns'"
        )

    # Calculate statistics
    mc_returns = results["simulated_returns"]
    mc_sharpes = results["simulated_sharpes"]
    mc_max_dds = results["simulated_max_dds"]

    statistics = {
        "original_return": float(original_total_return),
        "original_sharpe": float(original_sharpe),
        "original_max_dd": float(original_max_dd),
        "mean_mc_return": float(np.mean(mc_returns)),
        "std_mc_return": float(np.std(mc_returns)),
        "mean_mc_sharpe": float(np.mean(mc_sharpes)),
        "std_mc_sharpe": float(np.std(mc_sharpes)),
        "mean_mc_max_dd": float(np.mean(mc_max_dds)),
        "percentile_return_5": float(np.percentile(mc_returns, 5)),
        "percentile_return_95": float(np.percentile(mc_returns, 95)),
    }

    # Calculate p-values
    pct_rank_return = stats.percentileofscore(mc_returns, original_total_return)
    pct_rank_sharpe = stats.percentileofscore(mc_sharpes, original_sharpe)

    statistics["percentile_rank_return"] = float(pct_rank_return)
    statistics["percentile_rank_sharpe"] = float(pct_rank_sharpe)

    # Two-tailed p-values
    p_val_return = 2 * min(pct_rank_return, 100 - pct_rank_return) / 100
    p_val_sharpe = 2 * min(pct_rank_sharpe, 100 - pct_rank_sharpe) / 100

    statistics["p_value_return"] = float(p_val_return)
    statistics["p_value_sharpe"] = float(p_val_sharpe)

    # Print summary
    print(f"\n📊 Monte Carlo Results:")
    print(
        f"   Mean MC Return: {statistics['mean_mc_return']:.2f}% (±{statistics['std_mc_return']:.2f}%)"
    )
    print(
        f"   Mean MC Sharpe: {statistics['mean_mc_sharpe']:.3f} (±{statistics['std_mc_sharpe']:.3f})"
    )
    print(f"   Original Return Percentile: {pct_rank_return:.1f}%")
    print(f"   Original Sharpe Percentile: {pct_rank_sharpe:.1f}%")
    print(
        f"   P-value (Return): {p_val_return:.4f} {'✅ Significant' if p_val_return < 0.05 else '⚠️ Not significant'}"
    )
    print(
        f"   P-value (Sharpe): {p_val_sharpe:.4f} {'✅ Significant' if p_val_sharpe < 0.05 else '⚠️ Not significant'}"
    )

    return {
        "statistics": statistics,
        "simulated_returns": mc_returns,
        "simulated_sharpes": mc_sharpes,
        "simulated_max_dds": mc_max_dds,
        "equity_paths": results.get("equity_paths"),
        "method": method,
        "n_simulations": n_simulations,
    }


def _bootstrap_trades(portfolio: vbt.Portfolio, n_simulations: int) -> Dict[str, Any]:
    """
    Bootstrap method: Randomly resample trades with replacement.

    This is the recommended method for trading strategies as it:
    - Uses actual trade outcomes (ignores zero-return periods)
    - Creates realistic divergent paths
    - Properly tests robustness to trade sequence
    """
    print("   Using bootstrap trades method...")

    # Extract trades
    trades_df = portfolio.trades.records_readable

    if len(trades_df) == 0:
        raise ValueError("Portfolio has no trades to bootstrap")

    # Get per-trade returns as fractions
    trade_returns = trades_df["PnL"].values / portfolio.init_cash
    n_trades = len(trade_returns)

    simulated_returns = []
    simulated_sharpes = []
    simulated_max_dds = []
    equity_paths = []

    for i in range(n_simulations):
        # Bootstrap: sample with replacement
        sampled = np.random.choice(trade_returns, size=n_trades, replace=True)

        # Calculate equity curve
        equity = (1 + sampled).cumprod()
        equity_paths.append(equity)

        # Total return (percentage)
        total_return = (equity[-1] - 1) * 100

        # Sharpe ratio (annualized by sqrt of number of trades)
        if np.std(sampled) > 0:
            sharpe = (np.mean(sampled) / np.std(sampled)) * np.sqrt(n_trades)
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_dd = np.abs(np.min(drawdown))

        simulated_returns.append(total_return)
        simulated_sharpes.append(sharpe)
        simulated_max_dds.append(max_dd)

        if (i + 1) % 200 == 0:
            print(f"   Progress: {i + 1}/{n_simulations}")

    return {
        "simulated_returns": np.array(simulated_returns),
        "simulated_sharpes": np.array(simulated_sharpes),
        "simulated_max_dds": np.array(simulated_max_dds),
        "equity_paths": np.array(equity_paths),
    }


def _shuffle_returns(portfolio: vbt.Portfolio, n_simulations: int) -> Dict[str, Any]:
    """
    Shuffle method: Randomly permute portfolio returns.

    WARNING: This includes zero-return periods (when not in position),
    which can give misleading results. Use bootstrap_trades instead.
    """
    print("   Using shuffle returns method...")

    returns = portfolio.returns()
    if returns is None or len(returns) == 0:
        raise ValueError("Portfolio has no returns data")

    returns_array = returns.values
    n_periods = len(returns_array)

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

        # Total return (percentage)
        total_return = (equity[-1] - 1) * 100

        # Sharpe ratio (annualized)
        if np.std(shuffled) > 0:
            sharpe = (np.mean(shuffled) / np.std(shuffled)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_dd = np.abs(np.min(drawdown))

        simulated_returns.append(total_return)
        simulated_sharpes.append(sharpe)
        simulated_max_dds.append(max_dd)

        if (i + 1) % 200 == 0:
            print(f"   Progress: {i + 1}/{n_simulations}")

    return {
        "simulated_returns": np.array(simulated_returns),
        "simulated_sharpes": np.array(simulated_sharpes),
        "simulated_max_dds": np.array(simulated_max_dds),
        "equity_paths": np.array(equity_paths),
    }


# Example usage for testing
if __name__ == "__main__":
    import vectorbt as vbt
    import pandas as pd
    import numpy as np

    # Create simple test portfolio
    np.random.seed(42)
    close = pd.Series(
        np.random.randn(252).cumsum() + 100, index=pd.date_range("2020", periods=252)
    )

    # MA crossover signals
    fast_ma = close.rolling(10).mean()
    slow_ma = close.rolling(50).mean()
    entries = fast_ma > slow_ma
    exits = fast_ma < slow_ma

    portfolio = vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits, init_cash=10000, fees=0.001
    )

    # Run path MC
    results = run_path_randomization_mc(portfolio, n_simulations=1000, seed=42)
    print(f"\n✅ Test completed. Simulated {results['n_simulations']} paths.")
