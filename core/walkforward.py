"""Walk-forward analysis for strategy robustness testing."""
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from .base import BaseStrategy
from .config import StrategyConfig


class WalkForwardAnalyzer:
    """Performs walk-forward analysis to test strategy robustness."""
    
    def __init__(self, strategy: BaseStrategy, config: StrategyConfig):
        self.strategy = strategy
        self.config = config
        
    def run_analysis(self, data: pd.DataFrame, best_params: Dict[str, Any], 
                    n_splits: int = 5) -> Dict[str, Any]:
        """Run walk-forward analysis."""
        print(f"📈 Running Walk-Forward Analysis ({n_splits} windows)...")
        
        # Calculate window size
        window_size = len(data) // n_splits
        results = []
        
        for i in range(n_splits - 1):  # n_splits - 1 because we need data for testing
            start_idx = i * window_size
            end_idx = start_idx + window_size
            test_start_idx = end_idx
            test_end_idx = min(test_start_idx + window_size, len(data))
            
            # Skip if not enough test data
            if test_end_idx - test_start_idx < window_size // 2:
                continue
            
            train_data = data.iloc[start_idx:end_idx].copy()
            test_data = data.iloc[test_start_idx:test_end_idx].copy()
            
            try:
                # Run analysis on this window
                window_result = self._analyze_window(train_data, test_data, best_params, i + 1)
                results.append(window_result)
                
            except Exception as e:
                print(f"⚠️ Error in window {i + 1}: {e}")
                continue
        
        if not results:
            return {"success": False, "error": "No successful walk-forward windows"}
        
        # Calculate stability metrics
        stability_ratio = self._calculate_stability_ratio(results)
        
        # Create results summary
        results_df = pd.DataFrame(results)
        
        print("📊 Walk-Forward Results:")
        print(results_df[['window', 'sharpe', 'return', 'max_dd']].to_string(index=False))
        
        print(f"📈 Walk-Forward Stability Ratio: {stability_ratio:.3f}")
        print("   (Lower is better, < 0.5 is considered stable)")
        
        stability_assessment = "STABLE" if stability_ratio < 0.5 else "UNSTABLE"
        print(f"🎯 Stability Assessment: {stability_assessment}")
        
        return {
            "success": True,
            "results": results_df,
            "stability_ratio": stability_ratio,
            "stability_assessment": stability_assessment,
            "mean_sharpe": results_df['sharpe'].mean(),
            "std_sharpe": results_df['sharpe'].std(),
            "mean_return": results_df['return'].mean(),
            "std_return": results_df['return'].std()
        }
    
    def _analyze_window(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                       params: Dict[str, Any], window_num: int) -> Dict[str, Any]:
        """Analyze a single walk-forward window."""
        
        # Update strategy parameters
        original_params = self.strategy.parameters.copy()
        self.strategy.parameters.update(params)
        
        try:
            # Generate signals on test data
            entries, exits = self.strategy.generate_signals(test_data)
            
            # Create portfolio
            
            # Calculate metrics
            returns = portfolio.returns()
            
            if returns.empty or returns.isna().all():
                sharpe = -999
                total_return = -999
                max_dd = 999
            else:
                sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else -999
                total_return = portfolio.total_return() * 100
                max_dd = abs(portfolio.max_drawdown()) * 100
            
            return {
                'window': window_num,
                'sharpe': sharpe,
                'return': total_return,
                'max_dd': max_dd,
                'trades': len(portfolio.trades.records),
                'win_rate': (returns > 0).mean() * 100 if not returns.empty else 0
            }
            
        finally:
            # Restore original parameters
            self.strategy.parameters = original_params
    
    def _calculate_stability_ratio(self, results: List[Dict[str, Any]]) -> float:
        """Calculate stability ratio from walk-forward results."""
        if len(results) < 2:
            return 1.0
        
        sharpe_values = [r['sharpe'] for r in results if r['sharpe'] != -999]
        
        if len(sharpe_values) < 2:
            return 1.0
        
        mean_sharpe = np.mean(sharpe_values)
        std_sharpe = np.std(sharpe_values)
        
        # Stability ratio: lower is better
        stability_ratio = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else 1.0
        
        return stability_ratio