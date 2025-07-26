"""Parameter optimization for trading strategies."""
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from itertools import product
import vectorbt as vbt
from .base import BaseStrategy
from .portfolio import PortfolioManager
from .config import StrategyConfig


class ParameterOptimizer:
    """Handles parameter optimization using grid search."""
    
    def __init__(self, strategy: BaseStrategy, config: StrategyConfig):
        self.strategy = strategy
        self.config = config
        self.optimization_grid = config.optimization_grid
        
    def optimize(self, data: pd.DataFrame, split_ratio: float = 0.7) -> Dict[str, Any]:
        """Run grid search optimization on training data."""
        print("🔍 Running Grid Search Optimization on training data")
        
        # Split data for training
        split_idx = int(len(data) * split_ratio)
        train_data = data.iloc[:split_idx].copy()
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        
        if not param_combinations:
            print("⚠️ No optimization parameters defined, using default parameters")
            return self._run_single_optimization(train_data, self.strategy.parameters)
        
        print(f"🧪 Testing {len(param_combinations)} parameter combinations...")
        
        # Test all combinations
        results = []
        for i, params in enumerate(param_combinations):
            try:
                result = self._run_single_optimization(train_data, params)
                result['param_combination'] = params
                result['combination_id'] = i
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   Completed {i + 1}/{len(param_combinations)} combinations")
                    
            except Exception as e:
                print(f"⚠️ Error with parameter combination {i}: {e}")
                continue
        
        if not results:
            raise ValueError("No successful parameter combinations found")
        
        # Find best parameters
        best_result = self._select_best_parameters(results)
        
        print("🧬 Optimal parameters found:")
        for param, value in best_result['param_combination'].items():
            print(f"   - {param.replace('_', ' ').title()}: {value}")
        
        return best_result
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from optimization grid."""
        if not self.optimization_grid:
            return []
        
        # Get parameter names and values
        param_names = list(self.optimization_grid.keys())
        param_values = []
        
        for name in param_names:
            values = self.optimization_grid[name]
            # Ensure values is a list
            if not isinstance(values, list):
                values = [values]
            param_values.append(values)
        
        # Generate all combinations
        combinations = []
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _run_single_optimization(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run optimization for a single parameter combination."""
        # Update strategy parameters
        original_params = self.strategy.parameters.copy()
        self.strategy.parameters.update(params)
        
        try:
            # Generate signals
            entries, exits = self.strategy.generate_signals(data)
            
            # Create portfolio
            portfolio_manager = PortfolioManager([self.strategy.name], self.config.portfolio)
            portfolio = portfolio_manager.create_portfolio(data, entries, exits)
            
            # Calculate metrics
            metrics = self._calculate_optimization_metrics(portfolio)
            
            return {
                'portfolio': portfolio,
                'metrics': metrics,
                'sharpe_ratio': metrics.get('sharpe', -999),
                'total_return': metrics.get('return', -999),
                'max_drawdown': metrics.get('max_dd', 999),
                'profit_factor': metrics.get('profit_factor', 0)
            }
            
        finally:
            # Restore original parameters
            self.strategy.parameters = original_params
    
    def _calculate_optimization_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """Calculate key metrics for optimization."""
        try:
            returns = portfolio.returns()
            
            # Handle empty returns
            if returns.empty or returns.isna().all():
                return {
                    'sharpe_ratio': -999,
                    'total_return': -999,
                    'max_drawdown': 999,
                    'profit_factor': 0
                }
            
            # Calculate metrics
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else -999
            total_return = portfolio.total_return() * 100
            max_dd = abs(portfolio.max_drawdown()) * 100
            
            # Calculate profit factor
            wins = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            profit_factor = wins / losses if losses > 0 else 0
            
            return {
                'sharpe': sharpe,
                'return': total_return,
                'max_dd': max_dd,
                'profit_factor': profit_factor,
                'trades': len(portfolio.trades.records),
                'win_rate': (returns > 0).mean() * 100
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating metrics: {e}")
            return {
                'sharpe': -999,
                'return': -999,
                'max_dd': 999,
                'profit_factor': 0
            }
    
    def _select_best_parameters(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best parameters based on Sharpe ratio."""
        # Sort by Sharpe ratio (descending)
        sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
        
        # Return best result
        best = sorted_results[0]
        
        print(f"📊 Best Sharpe Ratio: {best['sharpe_ratio']:.3f}")
        print(f"📊 Best Total Return: {best['total_return']:.2f}%")
        print(f"📊 Best Max Drawdown: {best['max_drawdown']:.2f}%")
        
        return best