#!/usr/bin/env python3
"""
Simplified Functional Parameter Optimizer
Works directly with signal functions with maximum flexibility.
"""

import time
import warnings
from typing import Dict, Any, List, Optional, NamedTuple, Callable
import numpy as np
import pandas as pd
import vectorbt as vbt
from itertools import product

# Import strategy functions
from strategies import get_strategy_signal_function, list_available_strategies

warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {}

DEFAULT_PARAM_GRIDS = {}

# =============================================================================
# CORE CLASSES
# =============================================================================

class OptimizationResult(NamedTuple):
    best_portfolio: vbt.Portfolio
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_results: pd.DataFrame
    execution_time: float

class Optimizer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
    
    def optimize(self, data: pd.DataFrame, signal_function: Callable, 
                param_grid: Dict[str, List]) -> OptimizationResult:
        """Main optimization entry point."""
        data = self._prepare_data(data)
        split_idx = int(len(data) * self.config['split_ratio'])
        train_data = data.iloc[:split_idx]
        
        try:
            return self._vectorized_optimize(train_data, signal_function, param_grid)
        except Exception as e:
            if self.config['verbose']:
                print(f"Vectorized failed: {e}, using sequential")
            return self._sequential_optimize(train_data, signal_function, param_grid)
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for optimization."""
        return data.asfreq('1H').ffill().dropna()
    
    def _vectorized_optimize(self, data: pd.DataFrame, signal_function: Callable,
                           param_grid: Dict[str, List]) -> OptimizationResult:
        """Fast vectorized optimization."""
        start_time = time.time()
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        # Generate all signals
        all_entries, all_exits = [], []
        for combo in param_combinations:
            params = dict(zip(param_names, combo))
            signals = signal_function(data, **params)
            all_entries.append(signals.entries.reindex(data.index, fill_value=False))
            all_exits.append(signals.exits.reindex(data.index, fill_value=False))
        
        # Create vectorized portfolios
        entries_df = pd.concat(all_entries, axis=1, keys=param_combinations)
        exits_df = pd.concat(all_exits, axis=1, keys=param_combinations)
        
        portfolios = vbt.Portfolio.from_signals(
            close=data['close'], entries=entries_df, exits=exits_df,
            init_cash=self.config['init_cash'], fees=self.config['fees']
        )
        
        # Calculate metrics
        sharpe_ratios = portfolios.sharpe_ratio()
        total_returns = portfolios.total_return() * 100
        max_drawdowns = portfolios.drawdown.max_drawdown() * 100
        
        # Handle single result case
        if not hasattr(sharpe_ratios, '__len__'):
            sharpe_ratios = pd.Series([sharpe_ratios])
            total_returns = pd.Series([total_returns])
            max_drawdowns = pd.Series([max_drawdowns])
        
        # Create results
        results_df = pd.DataFrame({
            'sharpe_ratio': sharpe_ratios,
            'total_return': total_returns,
            'max_drawdown': max_drawdowns
        })
        results_df['composite_score'] = (
            0.7 * results_df['sharpe_ratio'] + 
            0.3 * results_df['total_return'] / np.maximum(np.abs(results_df['max_drawdown']), 1)
        )
        
        # Find best
        best_idx = results_df['composite_score'].idxmax()
        best_params = dict(zip(param_names, best_idx if isinstance(best_idx, tuple) else param_combinations[0]))
        best_portfolio = portfolios.iloc[best_idx] if hasattr(portfolios, 'iloc') else portfolios
        
        return OptimizationResult(
            best_portfolio=best_portfolio,
            best_params=best_params,
            best_metrics=results_df.loc[best_idx].to_dict(),
            all_results=results_df,
            execution_time=time.time() - start_time
        )
    
    def _sequential_optimize(self, data: pd.DataFrame, signal_function: Callable,
                           param_grid: Dict[str, List]) -> OptimizationResult:
        """Sequential fallback optimization."""
        start_time = time.time()
        param_combinations = [dict(zip(param_grid.keys(), combo)) 
                            for combo in product(*param_grid.values())]
        results = []
        
        for params in param_combinations:
            try:
                signals = signal_function(data, **params)
                portfolio = vbt.Portfolio.from_signals(
                    close=data['close'],
                    entries=signals.entries.reindex(data.index, fill_value=False),
                    exits=signals.exits.reindex(data.index, fill_value=False),
                    init_cash=self.config['init_cash'], fees=self.config['fees']
                )
                
                stats = portfolio.stats()
                results.append({
                    **params,
                    'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
                    'total_return': float(stats.get('Total Return [%]', 0)),
                    'max_drawdown': float(stats.get('Max Drawdown [%]', 0)),
                    'portfolio': portfolio
                })
            except Exception:
                continue
        
        if not results:
            raise ValueError("No successful parameter combinations")
        
        # Find best
        best_result = max(results, key=lambda x: x['sharpe_ratio'])
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'portfolio'} for r in results])
        
        return OptimizationResult(
            best_portfolio=best_result['portfolio'],
            best_params={k: v for k, v in best_result.items() if k not in ['sharpe_ratio', 'total_return', 'max_drawdown', 'portfolio']},
            best_metrics={k: best_result[k] for k in ['sharpe_ratio', 'total_return', 'max_drawdown']},
            all_results=results_df,
            execution_time=time.time() - start_time
        )

# =============================================================================
# MONTE CARLO ANALYSIS
# =============================================================================

def monte_carlo_analysis(data: pd.DataFrame, signal_function: Callable, 
                        best_params: Dict, n_simulations: int = 1000) -> Dict:
    """Monte Carlo analysis of strategy robustness."""
    np.random.seed(42)
    returns = []
    # NEW: Add parameter tracking
    param_values = []
    
    for _ in range(n_simulations):
        # Bootstrap sampling
        sample_data = data.sample(n=len(data), replace=True).sort_index()
        
        try:
            signals = signal_function(sample_data, **best_params)
            portfolio = vbt.Portfolio.from_signals(
                close=sample_data['close'],
                entries=signals.entries.reindex(sample_data.index, fill_value=False),
                exits=signals.exits.reindex(sample_data.index, fill_value=False),
                **DEFAULT_CONFIG
            )
            returns.append(portfolio.total_return())
            # NEW: Track parameters
            param_values.append({k: v for k, v in best_params.items() if not isinstance(v, (dict, list))})
        except:
            continue

    # NEW: Validate parameter values
    if not param_values:
        param_values = [{k: 0 for k in best_params} for _ in range(n_simulations)]

    returns = np.array(returns)
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'var_95': np.percentile(returns, 5),
        'var_99': np.percentile(returns, 1),
        'success_rate': len(returns) / n_simulations,
        'simulations': [{**p, 'total_return': r} for p, r in zip(param_values, returns)]
    }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def optimize_strategy(data: pd.DataFrame, strategy_name: str, 
                     param_grid: Optional[Dict] = None, config: Optional[Dict] = None) -> OptimizationResult:
    """Optimize strategy by name."""
    param_grid = param_grid or DEFAULT_PARAM_GRIDS.get(strategy_name, {})
    signal_function = get_strategy_signal_function(strategy_name)
    
    # Wrapper for multi-timeframe strategies
    def wrapper(df, **params):
        tf_data = {params.get('primary_timeframe', '1h'): df}
        return signal_function(tf_data, params)
    
    optimizer = FunctionalOptimizer(config)
    return optimizer.optimize(data, wrapper, param_grid)

def compare_strategies(data: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    """Compare multiple strategies."""
    results = []
    for strategy in strategies:
        try:
            result = optimize_strategy(data, strategy)
            results.append({
                'strategy': strategy,
                'sharpe': result.best_metrics['sharpe_ratio'],
                'return': result.best_metrics['total_return'],
                'drawdown': result.best_metrics['max_drawdown'],
                'time': result.execution_time
            })
        except Exception as e:
            results.append({'strategy': strategy, 'error': str(e)})
    
    return pd.DataFrame(results).sort_values('sharpe', ascending=False)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='1H')
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    
    data = pd.DataFrame({
        'open': close * (1 + np.random.randn(len(dates)) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(len(dates))) * 0.002),
        'low': close * (1 - np.abs(np.random.randn(len(dates))) * 0.002),
        'close': close
    }, index=dates)
    
    # Optimize single strategy
    result = optimize_strategy(data, 'vectorbt')
    print(f"Best Sharpe: {result.best_metrics['sharpe_ratio']:.3f}")
    
    # Monte Carlo analysis
    mc_results = monte_carlo_analysis(data, get_strategy_signal_function('vectorbt'), result.best_params)
    print(f"MC Mean Return: {mc_results['mean_return']:.3f}")
    
    # Compare strategies
    comparison = compare_strategies(data, ['vectorbt', 'momentum'])
    print(comparison)