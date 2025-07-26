"""Monte Carlo permutation testing for statistical validation."""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from .base import BaseStrategy
from .portfolio import PortfolioManager
from .config import StrategyConfig


class MonteCarloAnalyzer:
    """Performs Monte Carlo permutation testing for statistical validation."""
    
    def __init__(self, strategy: BaseStrategy, config: StrategyConfig):
        self.strategy = strategy
        self.config = config
        
    def run_analysis(self, data: pd.DataFrame, best_params: Dict[str, Any], 
                    original_portfolio, runs: int = 20, 
                    train_split: float = 0.7, seed: Optional[int] = 42) -> Dict[str, Any]:
        """Run Monte Carlo permutation test."""
        print("🎲 Running Advanced Monte Carlo Permutation Test...")
        
        # Calculate original metrics
        original_metrics = self._calculate_portfolio_metrics(original_portfolio)
        original_sharpe = original_metrics.get('sharpe_ratio', 0)
        original_profit_factor = original_metrics.get('profit_factor', 0)
        
        print(f"📊 Best Sharpe: {original_sharpe:.4f}, Best Profit Factor: {original_profit_factor:.4f}")
        
        # Run permutation tests in parallel with timeout
        print(f"🔀 Running Monte Carlo Permutation Test ({runs} iterations) in parallel...")
        
        try:
            permutation_results = Parallel(n_jobs=4, verbose=1, timeout=300)(  # 5 minute timeout, 4 jobs max
                delayed(self._run_single_permutation)(
                    data, best_params, i, train_split, seed
                ) for i in range(runs)
            )
        except Exception as e:
            print(f"⚠️ Monte Carlo analysis timed out or failed: {e}")
            return {"success": False, "error": "Monte Carlo analysis timed out"}
        
        # Filter successful results
        successful_results = [r for r in permutation_results if r is not None]
        
        if not successful_results:
            return {"success": False, "error": "No successful permutation runs"}
        
        # Calculate statistics
        permuted_sharpes = [r['sharpe_ratio'] for r in successful_results]
        permuted_profit_factors = [r['profit_factor'] for r in successful_results]
        
        # Calculate p-values
        sharpe_p_value = self._calculate_p_value(original_sharpe, permuted_sharpes)
        pf_p_value = self._calculate_p_value(original_profit_factor, permuted_profit_factors)
        
        # Determine significance
        significance_threshold = 0.05
        is_significant = sharpe_p_value < significance_threshold or pf_p_value < significance_threshold
        
        print("✅ Monte Carlo Permutation test completed.")
        print(f"   Sharpe Ratio - Best: {original_sharpe:.4f}, Mean Permuted: {np.mean(permuted_sharpes):.4f}, p-value: {sharpe_p_value:.4f}")
        print(f"   Profit Factor - Best: {original_profit_factor:.4f}, Mean Permuted: {np.mean(permuted_profit_factors):.4f}, p-value: {pf_p_value:.4f}")
        
        interpretation = "✅ SIGNIFICANT" if is_significant else "❌ NOT SIGNIFICANT"
        print(f"   Interpretation: {interpretation}")
        
        return {
            "success": True,
            "original_sharpe": original_sharpe,
            "original_profit_factor": original_profit_factor,
            "permuted_sharpes": permuted_sharpes,
            "permuted_profit_factors": permuted_profit_factors,
            "sharpe_p_value": sharpe_p_value,
            "profit_factor_p_value": pf_p_value,
            "is_significant": is_significant,
            "interpretation": interpretation,
            "successful_runs": len(successful_results),
            "total_runs": runs
        }
    
    def _run_single_permutation(self, data: pd.DataFrame, params: Dict[str, Any], 
                               iteration: int, train_split: float, 
                               seed: Optional[int]) -> Optional[Dict[str, Any]]:
        """Run a single permutation test."""
        try:
            # Set random seed for reproducibility
            if seed is not None:
                np.random.seed(seed + iteration)
            
            # Create permuted data
            permuted_data = self._permute_data(data)
            
            # Split data
            split_idx = int(len(permuted_data) * train_split)
            test_data = permuted_data.iloc[split_idx:].copy()
            
            # Update strategy parameters
            original_params = self.strategy.parameters.copy()
            self.strategy.parameters.update(params)
            
            try:
                # Generate signals
                entries, exits = self.strategy.generate_signals(test_data)
                
                # Create portfolio
                portfolio_manager = PortfolioManager([self.strategy.name], self.config.portfolio)
                portfolio = portfolio_manager.create_portfolio(test_data, entries, exits)
                
                # Calculate metrics
                metrics = self._calculate_portfolio_metrics(portfolio)
                
                return metrics
                
            finally:
                # Restore original parameters
                self.strategy.parameters = original_params
                
        except Exception as e:
            # Silently handle errors in permutation runs
            return None
    
    def _permute_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create permuted version of the data - simplified for speed."""
        permuted_data = data.copy()
        
        # Simple approach: just shuffle the close prices
        permuted_close = np.random.permutation(data['close'].values)
        permuted_data['close'] = permuted_close
        
        # Quick OHLC adjustment based on close
        permuted_data['open'] = permuted_close * np.random.uniform(0.995, 1.005, len(permuted_close))
        permuted_data['high'] = permuted_close * np.random.uniform(1.0, 1.01, len(permuted_close))
        permuted_data['low'] = permuted_close * np.random.uniform(0.99, 1.0, len(permuted_close))
        
        # Ensure high >= low
        permuted_data['high'] = np.maximum(permuted_data['high'], permuted_data['low'])
        
        return permuted_data
    
    def _calculate_portfolio_metrics(self, portfolio) -> Dict[str, Any]:
        """Calculate key metrics for a portfolio."""
        try:
            returns = portfolio.returns()
            
            if returns.empty or returns.isna().all():
                return {
                    'sharpe_ratio': -999,
                    'profit_factor': 0,
                    'total_return': -999
                }
            
            # Calculate Sharpe ratio
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else -999
            
            # Calculate profit factor
            wins = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            profit_factor = wins / losses if losses > 0 else 0
            
            # Total return
            total_return = portfolio.total_return() * 100
            
            return {
                'sharpe_ratio': sharpe,
                'profit_factor': profit_factor,
                'total_return': total_return
            }
            
        except Exception:
            return {
                'sharpe_ratio': -999,
                'profit_factor': 0,
                'total_return': -999
            }
    
    def _calculate_p_value(self, original_value: float, permuted_values: List[float]) -> float:
        """Calculate p-value for permutation test."""
        if not permuted_values:
            return 1.0
        
        # Count how many permuted values are better than original
        better_count = sum(1 for pv in permuted_values if pv >= original_value)
        
        # P-value is the proportion of permuted values that are better
        p_value = better_count / len(permuted_values)
        
        return p_value