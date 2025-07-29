#!/usr/bin/env python3
"""
Functional Parameter Optimizer - Fully Decoupled
Works directly with signal functions, not strategy classes.
Maximum flexibility for any indicator or signal generator.
"""

import copy
import logging
import warnings
from typing import Dict, Any, List, Optional, Tuple, NamedTuple, Callable
import numpy as np
import pandas as pd
import vectorbt as vbt
from itertools import product

# Import your functional signal generators
from strategies import (
    SIGNAL_FUNCTIONS, 
    get_signal_function,
    get_required_columns,
    list_available_strategies
)

# Suppress pandas FutureWarnings about frequency strings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class FunctionalOptimizationResult(NamedTuple):
    """Functional optimization result."""
    best_portfolio: vbt.Portfolio
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_results: pd.DataFrame
    signal_function_name: str


class VectorizedResult(NamedTuple):
    """Vectorized optimization result."""
    portfolios: vbt.Portfolio
    param_combinations: List[Dict[str, Any]]
    metrics_df: pd.DataFrame
    best_idx: Any
    execution_time: float


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _align_signals_with_data(signals, data_index):
    """Align signals with data index to prevent shape mismatches."""
    aligned_signals = type(signals)(
        entries=signals.entries.reindex(data_index, fill_value=False),
        exits=signals.exits.reindex(data_index, fill_value=False),
        short_entries=getattr(signals, 'short_entries', pd.Series(False, index=data_index)).reindex(data_index, fill_value=False),
        short_exits=getattr(signals, 'short_exits', pd.Series(False, index=data_index)).reindex(data_index, fill_value=False)
    )
    return aligned_signals


def _prepare_data_for_optimization(data):
    """Prepare data for optimization by ensuring proper frequency and handling missing values."""
    # Ensure consistent frequency
    if pd.infer_freq(data.index) is None:
        # Try to infer and set frequency
        try:
            data = data.asfreq('1h')
        except:
            pass
    
    # Forward fill missing values
    data = data.ffill()
    
    # Drop any remaining NaN rows
    data = data.dropna()
    
    return data


# =============================================================================
# CORE FUNCTIONAL OPTIMIZER
# =============================================================================

def optimize_signal_function(data: pd.DataFrame,
                            signal_function: Callable,
                            param_grid: Dict[str, List[Any]],
                            optimization_config: Optional[Dict[str, Any]] = None) -> FunctionalOptimizationResult:
    """
    Optimize any signal function with parameter grid.
    
    Args:
        data: OHLC DataFrame
        signal_function: Function that takes (df, **params) and returns Signals
        param_grid: Dictionary of parameter lists to optimize
        optimization_config: Optimization settings
        
    Returns:
        FunctionalOptimizationResult with best parameters and metrics
    """
    config = {
        'split_ratio': 0.8,
        'verbose': True,
        'init_cash': 50000,
        'fees': 0.0004,
    }
    if optimization_config:
        config.update(optimization_config)
    
    if config['verbose']:
        logger.info(f"Optimizing signal function with {len(param_grid)} parameter types")
    
    # Prepare data
    data = _prepare_data_for_optimization(data)
    
    # Split data
    split_idx = int(len(data) * config['split_ratio'])
    train_data = data.iloc[:split_idx].copy()
    
    # Try vectorized optimization first
    try:
        return _optimize_vectorized(train_data, signal_function, param_grid, config)
    except Exception as e:
        if config['verbose']:
            logger.warning(f"Vectorized optimization failed: {e}")
            logger.info("Falling back to sequential optimization")
        return _optimize_sequential(train_data, signal_function, param_grid, config)


def _optimize_vectorized(data: pd.DataFrame,
                        signal_function: Callable,
                        param_grid: Dict[str, List[Any]],
                        config: Dict[str, Any]) -> FunctionalOptimizationResult:
    """Vectorized optimization using IndicatorFactory approach."""
    
    # Create vectorized strategy factory
    def strategy_wrapper(close, **params):
        """Strategy wrapper for vectorization."""
        # Create full OHLC data for signal function
        full_data = data.copy()
        
        # Generate signals
        signals = signal_function(full_data, **params)
        
        # Align signals with close data to prevent shape mismatches
        entries = signals.entries.reindex(close.index, fill_value=False)
        exits = signals.exits.reindex(close.index, fill_value=False)
        short_entries = signals.short_entries.reindex(close.index, fill_value=False) if hasattr(signals, 'short_entries') else pd.Series(False, index=close.index)
        short_exits = signals.short_exits.reindex(close.index, fill_value=False) if hasattr(signals, 'short_exits') else pd.Series(False, index=close.index)
        
        return entries, exits, short_entries, short_exits
    
    # Execute vectorized optimization using manual approach (more reliable)
    import time
    start_time = time.time()
    
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = list(product(*param_values))
    
    if config['verbose']:
        logger.info(f"Running vectorized optimization with {len(param_combinations)} combinations")
    
    all_entries = []
    all_exits = []
    all_short_entries = []
    all_short_exits = []
    
    for combo in param_combinations:
        params = dict(zip(param_names, combo))
        entries, exits, short_entries, short_exits = strategy_wrapper(data['close'], **params)
        all_entries.append(entries)
        all_exits.append(exits)
        all_short_entries.append(short_entries)
        all_short_exits.append(short_exits)
    
    # Stack results
    entries_df = pd.concat(all_entries, axis=1, keys=param_combinations)
    exits_df = pd.concat(all_exits, axis=1, keys=param_combinations)
    short_entries_df = pd.concat(all_short_entries, axis=1, keys=param_combinations)
    short_exits_df = pd.concat(all_short_exits, axis=1, keys=param_combinations)
    
    # Create portfolios
    portfolios = vbt.Portfolio.from_signals(
        close=data['close'],
        entries=entries_df,
        exits=exits_df,
        short_entries=short_entries_df,
        short_exits=short_exits_df,
        init_cash=config['init_cash'],
        fees=config['fees'],
        freq=pd.infer_freq(data.index) or '1h'
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate metrics
    sharpe_ratios = portfolios.sharpe_ratio()
    total_returns = portfolios.total_return() * 100
    max_drawdowns = portfolios.max_drawdown() * 100
    
    # Handle single vs multiple results
    if isinstance(sharpe_ratios, (int, float)):
        sharpe_ratios = pd.Series([sharpe_ratios])
        total_returns = pd.Series([total_returns])
        max_drawdowns = pd.Series([max_drawdowns])
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sharpe_ratio': sharpe_ratios,
        'total_return': total_returns,
        'max_drawdown': max_drawdowns
    })
    
    # Add composite score
    results_df['composite_score'] = (
        0.7 * results_df['sharpe_ratio'] + 
        0.3 * (results_df['total_return'] / np.maximum(np.abs(results_df['max_drawdown']), 1))
    )
    
    # Find best combination
    best_idx = results_df['composite_score'].idxmax()
    best_metrics = results_df.loc[best_idx].to_dict()
    
    # Extract best parameters
    param_names = list(param_grid.keys())
    if isinstance(best_idx, tuple):
        best_params = dict(zip(param_names, best_idx))
    else:
        # Handle single parameter case
        if len(param_names) == 1:
            best_params = {param_names[0]: best_idx}
        else:
            best_params = {}
    
    # Get best portfolio
    if hasattr(portfolios, 'iloc') and len(results_df) > 1:
        best_portfolio = portfolios.iloc[best_idx] if isinstance(best_idx, int) else portfolios[best_idx]
    else:
        best_portfolio = portfolios
    
    if config['verbose']:
        total_combinations = np.prod([len(param_grid[name]) for name in param_names])
        logger.info(f"Vectorized optimization completed in {execution_time:.2f}s")
        logger.info(f"Tested {total_combinations} combinations ({total_combinations/execution_time:.1f} comb/s)")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best Sharpe: {best_metrics['sharpe_ratio']:.3f}")
        logger.info(f"Best Return: {best_metrics['total_return']:.2f}%")
    
    return FunctionalOptimizationResult(
        best_portfolio=best_portfolio,
        best_params=best_params,
        best_metrics=best_metrics,
        all_results=results_df,
        signal_function_name=signal_function.__name__
    )


def _optimize_sequential(data: pd.DataFrame,
                        signal_function: Callable,
                        param_grid: Dict[str, List[Any]],
                        config: Dict[str, Any]) -> FunctionalOptimizationResult:
    """Sequential optimization fallback."""
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    param_combinations = [dict(zip(param_names, combo)) for combo in product(*param_values)]
    
    if config['verbose']:
        logger.info(f"Testing {len(param_combinations)} parameter combinations sequentially")
    
    results = []
    
    for i, params in enumerate(param_combinations):
        try:
            # Generate signals
            signals = signal_function(data, **params)
            
            # Align signals with data to prevent shape mismatches
            aligned_signals = _align_signals_with_data(signals, data.index)
            
            # Create portfolio
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=aligned_signals.entries,
                exits=aligned_signals.exits,
                short_entries=aligned_signals.short_entries,
                short_exits=aligned_signals.short_exits,
                init_cash=config['init_cash'],
                fees=config['fees'],
                freq=pd.infer_freq(data.index) or '1h'  # Fix: use lowercase 'h'
            )
            
            # Calculate metrics
            stats = portfolio.stats()
            metrics = {
                'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
                'total_return': float(stats.get('Total Return [%]', 0)),
                'max_drawdown': float(stats.get('Max Drawdown [%]', 0))
            }
            
            results.append({
                'params': params,
                'metrics': metrics,
                'portfolio': portfolio,
                'combination_id': i
            })
            
            if config['verbose'] and (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(param_combinations)} combinations")
                
        except Exception as e:
            logger.warning(f"Error with combination {i}: {e}")
            continue
    
    if not results:
        raise ValueError("No successful parameter combinations found")
    
    # Find best result
    best_result = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {**r['params'], **r['metrics'], 'combination_id': r['combination_id']}
        for r in results
    ])
    
    if config['verbose']:
        logger.info(f"Sequential optimization completed")
        logger.info(f"Best parameters: {best_result['params']}")
        logger.info(f"Best Sharpe: {best_result['metrics']['sharpe_ratio']:.3f}")
    
    return FunctionalOptimizationResult(
        best_portfolio=best_result['portfolio'],
        best_params=best_result['params'],
        best_metrics=best_result['metrics'],
        all_results=results_df,
        signal_function_name=signal_function.__name__
    )


# =============================================================================
# STRATEGY-SPECIFIC OPTIMIZERS
# =============================================================================

def optimize_strategy_by_name(data: pd.DataFrame,
                             strategy_name: str,
                             param_grid: Dict[str, List[Any]],
                             optimization_config: Optional[Dict[str, Any]] = None) -> FunctionalOptimizationResult:
    """
    Optimize a strategy by name using its registered signal function.
    
    Args:
        data: OHLC DataFrame
        strategy_name: Name of strategy (e.g., 'momentum', 'orb', 'tdi', 'vectorbt')
        param_grid: Parameter grid to optimize
        optimization_config: Optimization settings
        
    Returns:
        FunctionalOptimizationResult
    """
    try:
        signal_function = get_signal_function(strategy_name)
        return optimize_signal_function(data, signal_function, param_grid, optimization_config)
    except ValueError as e:
        available = list_available_strategies()
        raise ValueError(f"{e}. Available strategies: {available}")


def optimize_momentum_strategy(data: pd.DataFrame,
                              param_grid: Optional[Dict[str, List[Any]]] = None,
                              optimization_config: Optional[Dict[str, Any]] = None) -> FunctionalOptimizationResult:
    """Optimize momentum strategy with default parameter grid."""
    default_grid = {
        'rsi_period': [10, 14, 20, 25],
        'rsi_overbought': [70, 75, 80],
        'rsi_oversold': [20, 25, 30],
        'ma_period': [20, 50, 100, 200]
    }
    
    param_grid = param_grid or default_grid
    return optimize_strategy_by_name(data, 'momentum', param_grid, optimization_config)


def optimize_orb_strategy(data: pd.DataFrame,
                         param_grid: Optional[Dict[str, List[Any]]] = None,
                         optimization_config: Optional[Dict[str, Any]] = None) -> FunctionalOptimizationResult:
    """Optimize ORB strategy with default parameter grid."""
    default_grid = {
        'orb_period_minutes': [30, 60, 90, 120],
        'breakout_threshold': [0.001, 0.002, 0.003, 0.005],
        'stop_loss_pct': [0.01, 0.015, 0.02, 0.025]
    }
    
    param_grid = param_grid or default_grid
    return optimize_strategy_by_name(data, 'orb', param_grid, optimization_config)


def optimize_tdi_strategy(data: pd.DataFrame,
                         param_grid: Optional[Dict[str, List[Any]]] = None,
                         optimization_config: Optional[Dict[str, Any]] = None) -> FunctionalOptimizationResult:
    """Optimize TDI strategy with default parameter grid."""
    default_grid = {
        'rsi_period': [8, 13, 21],
        'price_period': [2, 5, 8],
        'signal_period': [5, 8, 13],
        'volatility_band': [34, 55, 89]
    }
    
    param_grid = param_grid or default_grid
    return optimize_strategy_by_name(data, 'tdi', param_grid, optimization_config)


def optimize_bollinger_strategy(data: pd.DataFrame,
                               param_grid: Optional[Dict[str, List[Any]]] = None,
                               optimization_config: Optional[Dict[str, Any]] = None) -> FunctionalOptimizationResult:
    """Optimize Bollinger Bands strategy with default parameter grid."""
    default_grid = {
        'bbands_period': [10, 15, 20, 25, 30],
        'bbands_std': [1.5, 2.0, 2.5, 3.0],
        'adx_period': [10, 14, 20],
        'adx_threshold': [15, 20, 25, 30]
    }
    
    param_grid = param_grid or default_grid
    return optimize_strategy_by_name(data, 'vectorbt', param_grid, optimization_config)


# =============================================================================
# CUSTOM INDICATOR OPTIMIZER
# =============================================================================

def optimize_custom_indicator(data: pd.DataFrame,
                             indicator_function: Callable,
                             signal_logic: Callable,
                             param_grid: Dict[str, List[Any]],
                             optimization_config: Optional[Dict[str, Any]] = None) -> FunctionalOptimizationResult:
    """
    Optimize any custom indicator with custom signal logic.
    
    Args:
        data: OHLC DataFrame
        indicator_function: Function that calculates indicator values
        signal_logic: Function that converts indicator values to signals
        param_grid: Parameter grid to optimize
        optimization_config: Optimization settings
        
    Example:
        def my_indicator(data, period=20):
            return data['close'].rolling(period).mean()
            
        def my_signals(data, indicator_values, threshold=0.02):
            entries = data['close'] > indicator_values * (1 + threshold)
            exits = data['close'] < indicator_values * (1 - threshold)
            return Signals(entries=entries, exits=exits)
            
        result = optimize_custom_indicator(data, my_indicator, my_signals, param_grid)
    """
    
    def combined_signal_function(df, **params):
        """Combine indicator calculation and signal logic."""
        # Split parameters for indicator and signal logic
        indicator_params = {k: v for k, v in params.items() if k.startswith('ind_')}
        signal_params = {k: v for k, v in params.items() if k.startswith('sig_')}
        
        # Remove prefixes
        indicator_params = {k[4:]: v for k, v in indicator_params.items()}
        signal_params = {k[4:]: v for k, v in signal_params.items()}
        
        # Calculate indicator
        indicator_values = indicator_function(df, **indicator_params)
        
        # Generate signals
        signals = signal_logic(df, indicator_values, **signal_params)
        
        return signals
    
    return optimize_signal_function(data, combined_signal_function, param_grid, optimization_config)


# =============================================================================
# MULTI-STRATEGY COMPARISON
# =============================================================================

def compare_strategies(data: pd.DataFrame,
                      strategies_config: Dict[str, Dict[str, Any]],
                      optimization_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Compare multiple strategies with their respective parameter grids.
    
    Args:
        data: OHLC DataFrame
        strategies_config: Dict of {strategy_name: {'param_grid': {...}, 'config': {...}}}
        optimization_config: Global optimization settings
        
    Returns:
        DataFrame comparing all strategies
    """
    results = []
    
    for strategy_name, strategy_config in strategies_config.items():
        try:
            param_grid = strategy_config.get('param_grid', {})
            local_config = {**optimization_config, **strategy_config.get('config', {})}
            
            logger.info(f"Optimizing {strategy_name}...")
            result = optimize_strategy_by_name(data, strategy_name, param_grid, local_config)
            
            results.append({
                'strategy': strategy_name,
                'best_sharpe': result.best_metrics['sharpe_ratio'],
                'best_return': result.best_metrics['total_return'],
                'best_drawdown': result.best_metrics['max_drawdown'],
                'best_params': str(result.best_params),
                'num_combinations': len(result.all_results)
            })
            
        except Exception as e:
            logger.error(f"Error optimizing {strategy_name}: {e}")
            results.append({
                'strategy': strategy_name,
                'best_sharpe': np.nan,
                'best_return': np.nan,
                'best_drawdown': np.nan,
                'best_params': 'ERROR',
                'num_combinations': 0
            })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('best_sharpe', ascending=False)
    
    logger.info("Strategy comparison completed:")
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    return comparison_df


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_usage():
    """Examples of how to use the functional optimizer."""
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='1h')  # Fix: use lowercase 'h'
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    
    data = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(len(dates)) * 0.001),
        'high': close_prices * (1 + np.abs(np.random.randn(len(dates))) * 0.002),
        'low': close_prices * (1 - np.abs(np.random.randn(len(dates))) * 0.002),
        'close': close_prices
    }, index=dates)
    
    print("=== Functional Optimizer Examples ===\n")
    
    # Example 1: Optimize specific strategy
    print("1. Optimizing Bollinger Bands strategy:")
    result = optimize_bollinger_strategy(data, optimization_config={'verbose': True})
    print(f"Best params: {result.best_params}")
    print(f"Best Sharpe: {result.best_metrics['sharpe_ratio']:.3f}\n")
    
    # Example 2: Compare multiple strategies
    print("2. Comparing multiple strategies:")
    strategies_config = {
        'vectorbt': {
            'param_grid': {
                'bbands_period': [15, 20, 25],
                'bbands_std': [2.0, 2.5]
            }
        },
        'momentum': {
            'param_grid': {
                'rsi_period': [14, 20],
                'ma_period': [50, 100]
            }
        }
    }
    
    comparison = compare_strategies(data, strategies_config, {'verbose': False})
    print(comparison)


if __name__ == "__main__":
    example_usage()