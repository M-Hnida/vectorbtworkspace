"""Portfolio metrics calculation and analysis."""
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Union
from .indicators import get_scalar

# Constants for metric calculations
TRADING_DAYS_PER_YEAR = 252
PERCENTAGE_MULTIPLIER = 100
VAR_95_PERCENTILE = 5
VAR_99_PERCENTILE = 1


def safe_value(value: Union[float, np.ndarray], default: float = 0) -> float:
    """Return the scalar value or default if not finite."""
    scalar = get_scalar(value) if not np.isscalar(value) else value
    return scalar if np.isfinite(scalar) else default


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safely divide two numbers, returning default if denominator is zero or result is not finite."""
    if denominator == 0 or not np.isfinite(numerator) or not np.isfinite(denominator):
        return default
    return numerator / denominator


def calc_metrics(portfolio: vbt.Portfolio, name: str = "Portfolio") -> Dict[str, Any]:
    """Calculate comprehensive portfolio metrics."""
    metrics = {}

    # Get portfolio stats
    stats = portfolio.stats()
    trades_stats = portfolio.trades.stats()

    # Basic performance and risk metrics (directly from portfolio)
    metrics['return'] = safe_value(stats['Total Return [%]'])
    metrics['max_dd'] = safe_value(stats['Max Drawdown [%]'])
    metrics['sharpe'] = safe_value(portfolio.sharpe_ratio())
    metrics['calmar'] = safe_value(portfolio.calmar_ratio())
    
    # Trade metrics (from trades stats)
    metrics['trades'] = stats['Total Trades']
    if metrics['trades'] > 0:
        metrics['win_rate'] = safe_value(trades_stats['Win Rate [%]'])
        metrics['profit_factor'] = safe_value(trades_stats['Profit Factor'])
        metrics['avg_win'] = safe_value(trades_stats['Avg Winning Trade [%]'])
        metrics['avg_loss'] = safe_value(trades_stats['Avg Losing Trade [%]'])
        metrics['win_loss_ratio'] = safe_divide(
            abs(metrics['avg_win']), 
            abs(metrics['avg_loss'])
        )
    else:
        metrics.update({
            'win_rate': 0, 'profit_factor': 0, 'avg_win': 0,
            'avg_loss': 0, 'win_loss_ratio': 0
        })

    # Risk metrics
    returns_stats = portfolio.returns_stats()
    metrics['volatility'] = safe_value(returns_stats.get('Volatility [%]', 0))
    metrics['var_95'] = safe_value(portfolio.value_at_risk(cutoff=VAR_95_PERCENTILE) * PERCENTAGE_MULTIPLIER)
    metrics['var_99'] = safe_value(portfolio.value_at_risk(cutoff=VAR_99_PERCENTILE) * PERCENTAGE_MULTIPLIER)
    metrics['cvar_95'] = safe_value(portfolio.value_at_risk(cutoff=VAR_95_PERCENTILE, var_type='cvar') * PERCENTAGE_MULTIPLIER)
    metrics['downside_vol'] = safe_value(returns_stats.get('Downside Volatility [%]', 0))

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
