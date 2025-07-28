#!/usr/bin/env python3
"""
Analysis Pipeline Module
Handles the orchestration of different analysis steps (optimization, walk-forward, monte carlo).
"""

from typing import Dict, Any
import multiprocessing as mp
import pandas as pd

from optimizer import ParameterOptimizer, WalkForwardAnalysis, MonteCarloAnalysis, OptimizationConfig
from base import BaseStrategy, StrategyConfig


class AnalysisPipeline:
    """Orchestrates different analysis steps for trading strategies."""
    
    def __init__(self, strategy: BaseStrategy, strategy_config: StrategyConfig):
        """Initialize analysis pipeline.
        
        Args:
            strategy: Strategy instance
            strategy_config: Strategy configuration
        """
        self.strategy = strategy
        self.strategy_config = strategy_config
    
    def run_optimization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run parameter optimization.
        
        Args:
            data: Primary timeframe data for optimization
            
        Returns:
            Optimization results dictionary
        """
        print(f"🎯 Optimizing on {len(data)} bars")
        
        # Get optimization config from strategy config
        opt_settings = self.strategy_config.analysis_settings.get('optimization', {})
        opt_config = OptimizationConfig(
            enable_parallel=opt_settings.get('enable_parallel', True),
            max_workers=opt_settings.get('max_workers', min(4, mp.cpu_count())),
            early_stopping=opt_settings.get('early_stopping', True),
            early_stopping_patience=opt_settings.get('early_stopping_patience', 10)
        )
        
        optimizer = ParameterOptimizer(self.strategy, self.strategy_config, opt_config)
        result = optimizer.optimize(data)
        
        # Update strategy with optimal parameters
        if hasattr(result, 'param_combination') and result.param_combination:
            self.strategy.parameters.update(result.param_combination)
            print("✅ Strategy updated with optimal parameters")
        
        return result.__dict__ if hasattr(result, '__dict__') else result
    
    def run_walkforward_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run walk-forward analysis.
        
        Args:
            data: Primary timeframe data for analysis
            
        Returns:
            Walk-forward analysis results
        """
        print(f"📊 Walk-forward analysis on {len(data)} bars")
        
        wf_analyzer = WalkForwardAnalysis(self.strategy, self.strategy_config)
        result = wf_analyzer.run_analysis(data)
        return result.__dict__ if hasattr(result, '__dict__') else result
    
    def run_monte_carlo_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run Monte Carlo analysis.
        
        Args:
            data: Primary timeframe data for analysis
            
        Returns:
            Monte Carlo analysis results
        """
        print(f"🎲 Monte Carlo analysis on {len(data)} bars")
        
        mc_analyzer = MonteCarloAnalysis(self.strategy)
        result = mc_analyzer.run_analysis(data)
        return result.__dict__ if hasattr(result, '__dict__') else result
    
    def run_complete_analysis(self, data: pd.DataFrame, skip_optimization: bool = False) -> Dict[str, Any]:
        """Run complete analysis pipeline.
        
        Args:
            data: Primary timeframe data for analysis
            skip_optimization: If True, skip optimization steps
            
        Returns:
            Complete analysis results
        """
        results = {}
        
        if skip_optimization:
            print("\n⚡ FAST MODE: Skipping optimization steps")
            return results
        
        try:
            # Step 1: Parameter Optimization
            print("\n🔧 STEP 1: Parameter Optimization")
            optimization_results = self.run_optimization(data)
            results['optimization'] = optimization_results
            
            # Step 2: Walk-Forward Analysis
            print("\n📈 STEP 2: Walk-Forward Analysis")
            walkforward_results = self.run_walkforward_analysis(data)
            results['walkforward'] = walkforward_results
            
            # Step 3: Monte Carlo Analysis
            print("\n🎲 STEP 3: Monte Carlo Analysis")
            monte_carlo_results = self.run_monte_carlo_analysis(data)
            results['monte_carlo'] = monte_carlo_results
            
            return results
            
        except Exception as e:
            print(f"⚠️ Analysis pipeline failed: {e}")
            raise


# ============================================================================
# DATA PROCESSING FUNCTIONS (Simplified from DataProcessor class)
# ============================================================================

def get_primary_data(data: Dict[str, Dict[str, pd.DataFrame]],
                    strategy_config: StrategyConfig) -> tuple:
    """Get primary symbol and timeframe data for analysis.

    Args:
        data: Multi-timeframe data dictionary
        strategy_config: Strategy configuration

    Returns:
        Tuple of (primary_symbol, primary_timeframe, primary_data)
    """
    primary_symbol = strategy_config.parameters.get('primary_symbol', list(data.keys())[0])
    primary_timeframe = strategy_config.parameters.get('primary_timeframe',
                                                      list(data[primary_symbol].keys())[0])
    primary_data = data[primary_symbol][primary_timeframe]
    return primary_symbol, primary_timeframe, primary_data


def validate_data_structure(data: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Validate multi-timeframe data structure.

    Args:
        data: Multi-timeframe data dictionary

    Raises:
        ValueError: If data structure is invalid
    """
    if not isinstance(data, dict) or not data:
        raise ValueError("Data must be a non-empty dictionary")

    for symbol, timeframes in data.items():
        if not isinstance(timeframes, dict) or not timeframes:
            raise ValueError(f"Timeframes for {symbol} must be a non-empty dictionary")

        for tf, df in timeframes.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError(f"Data for {symbol} {tf} must be a non-empty DataFrame")


# ============================================================================
# STRATEGY MANAGEMENT FUNCTIONS (Simplified from StrategyManager class)
# ============================================================================

def create_strategy(strategy_name: str, strategy_config: StrategyConfig) -> BaseStrategy:
    """Create functional strategy instance from name and configuration.

    Args:
        strategy_name: Name of the strategy
        strategy_config: Strategy configuration

    Returns:
        Functional strategy instance

    Raises:
        ValueError: If strategy cannot be created
    """
    from strategies import get_strategy_class

    try:
        strategy_factory = get_strategy_class(strategy_name)
        return strategy_factory(strategy_config)
    except ValueError as e:
        raise ValueError(f"Unknown strategy: {strategy_name}. {e}") from e


# Note: load_strategy_config functionality is already available in core_components.py
# Use: from core_components import load_strategy_config
