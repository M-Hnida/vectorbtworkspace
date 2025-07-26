"""Portfolio metrics calculation and analysis."""
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any
from .indicators import get_scalar

# Constants for metric calculations
TRADING_DAYS_PER_YEAR = 252
PERCENTAGE_MULTIPLIER = 100
VAR_95_PERCENTILE = 5
VAR_99_PERCENTILE = 1


def calc_metrics(portfolio: vbt.Portfolio, name: str = "Portfolio") -> Dict[str, Any]:
    """Calculate comprehensive portfolio metrics."""
    metrics = {}

    # Get returns and check volatility
    returns = portfolio.returns()

    # For multi-column portfolios, returns() aggregates, so we check the result
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    vol = returns.std() if len(returns) > 1 else 0
    if vol < 1e-6:
        print(f"⚠️ Very low volatility ({vol:.2e}) for {name} - metrics adjusted to 0")
        metrics['sharpe'] = 0
        metrics['calmar'] = 0
    else:
        # stats() called on a multi-column object will aggregate.
        # This is the desired behavior for 'test' and 'full' sets.
        sharpe = get_scalar(portfolio.sharpe_ratio())
        metrics['sharpe'] = 0 if not np.isfinite(sharpe) else sharpe
        calmar = get_scalar(portfolio.calmar_ratio())
        metrics['calmar'] = 0 if not np.isfinite(calmar) else calmar

    # Basic performance metrics
    metrics['return'] = get_scalar(portfolio.total_return()) * PERCENTAGE_MULTIPLIER
    metrics['max_dd'] = get_scalar(portfolio.max_drawdown()) * PERCENTAGE_MULTIPLIER

    # Trade analysis
    trades = portfolio.trades.records_readable
    metrics['trades'] = len(trades)

    if metrics['trades'] > 0:
        wins = trades[trades['PnL'] > 0]
        losses = trades[trades['PnL'] < 0]

        metrics['win_rate'] = len(wins) / metrics['trades'] * PERCENTAGE_MULTIPLIER
        metrics['profit_factor'] = abs(wins['PnL'].sum() / losses['PnL'].sum()) if len(losses) > 0 and wins['PnL'].sum() > 0 else np.inf
        metrics['profit_factor'] = 0 if not np.isfinite(metrics['profit_factor']) else metrics['profit_factor']

        metrics['avg_win'] = wins['PnL'].mean() if len(wins) > 0 else 0
        metrics['avg_loss'] = losses['PnL'].mean() if len(losses) > 0 else 0
        metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else np.inf
        metrics['win_loss_ratio'] = 0 if not np.isfinite(metrics['win_loss_ratio']) else metrics['win_loss_ratio']
    else:
        metrics.update({
            'win_rate': 0, 'profit_factor': 0, 'avg_win': 0,
            'avg_loss': 0, 'win_loss_ratio': 0
        })

    # Risk metrics
    if len(returns) > 0:
        metrics['volatility'] = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * PERCENTAGE_MULTIPLIER
        metrics['volatility'] = 0 if not np.isfinite(metrics['volatility']) else metrics['volatility']

        metrics['var_95'] = np.percentile(returns, VAR_95_PERCENTILE) * PERCENTAGE_MULTIPLIER
        metrics['var_95'] = 0 if not np.isfinite(metrics['var_95']) else metrics['var_95']

        metrics['var_99'] = np.percentile(returns, VAR_99_PERCENTILE) * PERCENTAGE_MULTIPLIER
        metrics['var_99'] = 0 if not np.isfinite(metrics['var_99']) else metrics['var_99']

        metrics['cvar_95'] = returns[returns <= np.percentile(returns, VAR_95_PERCENTILE)].mean() * PERCENTAGE_MULTIPLIER
        metrics['cvar_95'] = 0 if not np.isfinite(metrics['cvar_95']) else metrics['cvar_95']

        downside_vol = returns[returns < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR) * PERCENTAGE_MULTIPLIER if len(returns[returns < 0]) > 0 else 0
        metrics['downside_vol'] = 0 if not np.isfinite(downside_vol) else downside_vol
    else:
        metrics.update({
            'volatility': 0, 'var_95': 0, 'var_99': 0,
            'cvar_95': 0, 'downside_vol': 0
        })

    return metrics


def print_metrics(metrics: Dict[str, Any], name: str = "Portfolio"):
    """Print formatted metrics."""
    print(f"\n📊 {name} Analysis:")
    print(f"   Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"   Calmar Ratio: {metrics['calmar']:.3f}")
    print(f"   Total Return: {metrics['return']:.2f}%")
    print(f"   Max Drawdown: {metrics['max_dd']:.2f}%")
    print(f"   Total Trades: {metrics['trades']}")

    if metrics['trades'] > 0:
        print(f"   Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Average Win: ${metrics['avg_win']:.2f}")
        print(f"   Average Loss: ${metrics['avg_loss']:.2f}")
        print(f"   Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")

    print(f"   Volatility: {metrics['volatility']:.2f}%")
    print(f"   Downside Volatility: {metrics['downside_vol']:.2f}%")
    print(f"   VaR (95%): {metrics['var_95']:.2f}%")
    print(f"   VaR (99%): {metrics['var_99']:.2f}%")
    print(f"   CVaR (95%): {metrics['cvar_95']:.2f}%")


def create_performance_summary(train_metrics: Dict, test_metrics: Dict, full_metrics: Dict) -> pd.DataFrame:
    """Create performance comparison DataFrame."""
    performance_data = {
        'Train': train_metrics,
        'Test': test_metrics,
        'Full': full_metrics
    }

    return pd.DataFrame(performance_data).loc[[
        'return', 'sharpe', 'max_dd', 'trades', 'win_rate'
    ]].rename(index={
        'return': 'Total Return [%]',
        'sharpe': 'Sharpe Ratio',
        'max_dd': 'Max Drawdown [%]',
        'trades': 'Total Trades',
        'win_rate': 'Win Rate [%]'
    })


if __name__ == '__main__':
    print("This script is a module and is not meant to be run directly.")
    print("Please run the main pipeline via 'runner.py' from the project root.")
