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

DEFAULT_CONFIG = {
    'split_ratio': 0.7,
    'init_cash': 10000,
    'fees': 0.001,
    'verbose': False
}

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

def run_optimization(strategy, strategy_config, data: pd.DataFrame) -> Dict[str, Any]:
    """Run parameter optimization."""
    try:
        from strategies import get_strategy_function
        from backtest import run_backtest
        from itertools import product
        
        if not hasattr(strategy_config, 'optimization_grid') or not strategy_config.optimization_grid:
            print("⚠️ No optimization grid found, using default parameters")
            return {'best_params': strategy.parameters}

        # Get the signal function
        signal_func = get_strategy_function(strategy.name)
        param_grid = strategy_config.optimization_grid
        print(f"🎯 Optimizing {strategy.name} with {len(param_grid)} parameters")

        # Simple grid search implementation
        best_params = strategy.parameters.copy()
        best_score = -999999

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"📊 Testing {len(combinations)} parameter combinations...")
        
        for i, combo in enumerate(combinations[:10]):  # Limit to first 10 for testing
            test_params = strategy.parameters.copy()
            for name, value in zip(param_names, combo):
                test_params[name] = value

            # Generate signals with test parameters
            signals = signal_func({strategy.get_required_timeframes()[0]: data}, test_params)
            
            # Quick backtest
            portfolio = run_backtest(data, signals)
            
            # Calculate score (using Sharpe ratio)
            try:
                stats = portfolio.stats()
                score = float(stats.get('Sharpe Ratio', -999))
                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
            except Exception:
                continue

        # Update strategy with best parameters
        strategy.parameters.update(best_params)
        print(f"✅ Best parameters found: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'tested_combinations': len(combinations)
        }
        
    except Exception as e:
        print(f"⚠️ Optimization failed: {e}")
        return {'error': str(e), 'best_params': strategy.parameters}


def run_monte_carlo_analysis(data: pd.DataFrame, strategy=None, actual_return: float = None) -> Dict[str, Any]:
    """Run Monte Carlo analysis with statistical significance testing."""
    try:
        import numpy as np
        from scipy import stats
        from backtest import run_backtest
        
        print(f"🎲 Monte Carlo analysis on {len(data)} bars")
        
        # Get actual strategy performance if provided
        if strategy is not None:
            try:
                from strategies import get_strategy_function
                signal_func = get_strategy_function(strategy.name)
                signals = signal_func({strategy.get_required_timeframes()[0]: data}, strategy.parameters)
                portfolio = run_backtest(data, signals)
                actual_stats = portfolio.stats()
                actual_return = float(actual_stats.get('Total Return [%]', 0))
            except Exception:
                actual_return = 0.0

        # Monte Carlo simulations: shuffle returns
        returns = data['close'].pct_change().dropna()
        num_simulations = 1000  # Increased for better statistical power
        simulations = []
        random_returns = []
        
        for i in range(num_simulations):
            # Shuffle returns to break any patterns
            shuffled_returns = np.random.permutation(returns)
            
            # Calculate cumulative return
            cum_return = (1 + shuffled_returns).prod() - 1
            total_return_pct = cum_return * 100
            
            simulations.append({
                'simulation': i + 1,
                'total_return': total_return_pct,
                'volatility': shuffled_returns.std() * np.sqrt(252) * 100
            })
            random_returns.append(total_return_pct)

        # Statistical significance testing
        random_returns = np.array(random_returns)
        
        # Calculate percentile of actual return
        if actual_return is not None and not np.isnan(actual_return):
            percentile = stats.percentileofscore(random_returns, actual_return)
            p_value = min(percentile, 100 - percentile) / 100  # Two-tailed test
            is_significant = p_value < 0.05
        else:
            percentile = None
            p_value = None
            is_significant = None

        # Calculate statistics
        statistics = {
            'mean_return': np.mean(random_returns),
            'std_return': np.std(random_returns),
            'min_return': np.min(random_returns),
            'max_return': np.max(random_returns),
            'percentile_5': np.percentile(random_returns, 5),
            'percentile_95': np.percentile(random_returns, 95),
            'actual_return': actual_return,
            'percentile_rank': percentile,
            'p_value': p_value,
            'is_significant': is_significant
        }
        
        return {
            'simulations': simulations,
            'statistics': statistics,
            'summary': f"Completed {num_simulations} Monte Carlo simulations",
            'significance_test': {
                'actual_return': actual_return,
                'percentile_rank': percentile,
                'p_value': p_value,
                'is_significant': is_significant,
                'interpretation': f"Strategy performance is {'significant' if is_significant else 'not significant'} vs random" if is_significant is not None else "No actual return provided"
            }
        }
        
    except Exception as e:
        print(f"⚠️ Monte Carlo analysis failed: {e}")
        return {'error': str(e)}


# Legacy function for compatibility
def monte_carlo_analysis(data: pd.DataFrame, signal_function: Callable, 
                        best_params: Dict, n_simulations: int = 1000) -> Dict:
    """Legacy Monte Carlo function for compatibility."""
    return run_monte_carlo_analysis(data, None, None)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def optimize_strategy(data: pd.DataFrame, strategy_name: str, 
                     param_grid: Optional[Dict] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Simple grid search optimization."""
    try:
        # Import here to avoid circular imports
        from backtest import run_backtest
        
        if not param_grid:
            print("⚠️ No optimization grid found, using default parameters")
            return {'best_params': {}, 'best_score': 0}

        signal_function = get_strategy_signal_function(strategy_name)
        
        print(f"🎯 Optimizing {strategy_name} with {len(param_grid)} parameters")
        
        # Simple grid search implementation
        best_params = {}
        best_score = -999999
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"📊 Testing {min(len(combinations), 20)} parameter combinations...")
        
        for i, combo in enumerate(combinations[:20]):  # Limit to first 20 for speed
            test_params = dict(zip(param_names, combo))
            
            try:
                # Generate signals with test parameters
                tf_data = {'1H': data}  # Assume 1H timeframe
                signals = signal_function(tf_data, test_params)
                
                # Quick backtest
                portfolio = run_backtest(data, signals)
                
                # Calculate score (using Sharpe ratio)
                stats = portfolio.stats()
                score = float(stats.get('Sharpe Ratio', -999))
                
                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                    print(f"  ✅ New best: {score:.3f} with {test_params}")
                    
            except Exception as e:
                continue
        
        if best_params:
            print(f"✅ Optimization complete. Best Sharpe: {best_score:.3f}")
            return {
                'best_params': best_params,
                'best_score': best_score,
                'tested_combinations': min(len(combinations), 20)
            }
        else:
            print("⚠️ No successful parameter combinations found")
            return {'best_params': {}, 'best_score': 0}
            
    except Exception as e:
        print(f"⚠️ Optimization failed: {e}")
        return {'error': str(e), 'best_params': {}}

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