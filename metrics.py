"""
Portfolio Metrics Module
Provides comprehensive portfolio performance analysis and metrics calculation.
"""

from typing import Dict, Union
import pandas as pd
import vectorbt as vbt

# Constants
DEFAULT_VAR_CUTOFF_95 = 0.05
DEFAULT_VAR_CUTOFF_99 = 0.01
ZERO_THRESHOLD = 1e-10


def calc_metrics(portfolio: vbt.Portfolio) -> Dict[str, Union[float, int]]:
    """Calculate comprehensive portfolio metrics with robust error handling.
    
    Args:
        portfolio: VectorBT Portfolio object
        
    Returns:
        Dictionary containing calculated metrics
        
    Raises:
        ValueError: If portfolio is invalid or empty
    """

    try:
        stats = portfolio.stats()
        if stats is None:
            raise ValueError("Portfolio statistics are empty")
            
        metrics = _calculate_basic_metrics(portfolio, stats)
        metrics.update(_calculate_trade_metrics(portfolio, stats))
        metrics.update(_calculate_risk_metrics(portfolio))
        
        return metrics
        
    except Exception as e:
        print(f"⚠️ Error calculating metrics: {e}")
        return _get_default_metrics()


def _calculate_basic_metrics(portfolio: vbt.Portfolio, stats: pd.Series) -> Dict[str, Union[float, int]]:
    """Calculate basic performance metrics."""
    metrics = {}
    
    # Basic performance metrics
    metrics['return'] = stats.get('Total Return [%]', 0)
    metrics['max_dd'] = stats.get('Max Drawdown [%]', 0)
    
    # Sharpe ratio with zero division protection
    returns_std = portfolio.returns().std()
    if abs(returns_std) > ZERO_THRESHOLD:
        metrics['sharpe'] = portfolio.sharpe_ratio()
    else:
        metrics['sharpe'] = 0.0
        
    metrics['calmar'] = portfolio.calmar_ratio()
    
    return metrics


def _calculate_trade_metrics(portfolio: vbt.Portfolio, stats: pd.Series) -> Dict[str, Union[float, int]]:
    """Calculate trade-related metrics."""
    metrics = {}
    
    total_trades = stats.get('Total Trades', 0)
    metrics['trades'] = total_trades
    
    if total_trades > 0:
        try:
            trades_stats = portfolio.trades.stats()
            metrics['win_rate'] = trades_stats.get('Win Rate [%]', 0)
            metrics['profit_factor'] = trades_stats.get('Profit Factor', 0)
            metrics['avg_win'] = trades_stats.get('Avg Winning Trade [%]', 0)
            metrics['avg_loss'] = trades_stats.get('Avg Losing Trade [%]', 0)
            
            # Calculate win/loss ratio with zero division protection
            avg_loss = abs(metrics['avg_loss'])
            if avg_loss > ZERO_THRESHOLD:
                metrics['win_loss_ratio'] = abs(metrics['avg_win']) / avg_loss
            else:
                metrics['win_loss_ratio'] = float('inf') if metrics['avg_win'] > 0 else 0
                
        except Exception as e:
            print(f"Warning: Could not calculate trade metrics: {e}")
            metrics.update(_get_default_trade_metrics())
    else:
        metrics.update(_get_default_trade_metrics())
    
    return metrics


def _calculate_risk_metrics(portfolio: vbt.Portfolio) -> Dict[str, Union[float, int]]:
    """Calculate risk-related metrics."""
    metrics = {}
    
    try:
        returns_stats = portfolio.returns_stats()
        metrics['volatility'] = returns_stats.get('Volatility [%]', 0)
        metrics['downside_vol'] = returns_stats.get('Downside Volatility [%]', 0)
        
        # VaR calculations with error handling
        try:
            metrics['var_95'] = portfolio.value_at_risk(cutoff=DEFAULT_VAR_CUTOFF_95)
            metrics['var_99'] = portfolio.value_at_risk(cutoff=DEFAULT_VAR_CUTOFF_99)
        except Exception as e:
            print(f"Warning: Could not calculate VaR: {e}")
            metrics['var_95'] = 0
            metrics['var_99'] = 0
            
    except Exception as e:
        print(f"Warning: Could not calculate risk metrics: {e}")
        metrics.update({
            'volatility': 0,
            'downside_vol': 0,
            'var_95': 0,
            'var_99': 0
        })
    
    return metrics


def _get_default_metrics() -> Dict[str, Union[float, int]]:
    """Return default metrics when calculation fails."""
    return {
        'return': 0.0,
        'max_dd': 0.0,
        'sharpe': 0.0,
        'calmar': 0.0,
        'trades': 0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'win_loss_ratio': 0.0,
        'volatility': 0.0,
        'var_95': 0.0,
        'var_99': 0.0,
        'downside_vol': 0.0
    }


def _get_default_trade_metrics() -> Dict[str, Union[float, int]]:
    """Return default trade metrics."""
    return {
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'win_loss_ratio': 0.0
    }


def print_metrics(metrics: Dict[str, Union[float, int]], name: str = "Portfolio") -> None:
    """Print formatted metrics with consistent naming.
    
    Args:
        metrics: Dictionary of calculated metrics
        name: Name to display in the header
    """
    print(f"\n📊 {name} Analysis:")
    print(f"   Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"   Calmar Ratio: {metrics['calmar']:.3f}")
    print(f"   Total Return: {metrics['return']:.2f}%")
    print(f"   Max Drawdown: {metrics['max_dd']:.2f}%")
    print(f"   Total Trades: {metrics['trades']}")

    if metrics['trades'] > 0:
        print(f"   Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")

    print(f"   Volatility: {metrics['volatility']:.2f}%")
    print(f"   VaR (95%): {metrics['var_95']:.2f}%")
