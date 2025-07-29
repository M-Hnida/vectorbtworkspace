#!/usr/bin/env python3
"""
Analysis Pipeline Module - Functional Implementation
Handles the orchestration of different analysis steps (optimization, walk-forward, monte carlo).
"""

from typing import Dict, Any
import multiprocessing as mp
import pandas as pd

from optimizer import (optimize_strategy_parameters, run_walkforward_analysis_functional, 
                      run_monte_carlo_analysis_functional, OptimizationConfig)
from base import StrategyConfig


def run_optimization(strategy, strategy_config: StrategyConfig, data: pd.DataFrame) -> Dict[str, Any]:
    """Run parameter optimization.
    
    Args:
        strategy: Strategy instance
        strategy_config: Strategy configuration
        data: Primary timeframe data for optimization
        
    Returns:
        Optimization results dictionary
    """
    print(f"🎯 Optimizing on {len(data)} bars")
    
    # Get optimization config from strategy config
    opt_settings = strategy_config.analysis_settings.get('optimization', {})
    opt_config = OptimizationConfig(
        enable_parallel=opt_settings.get('enable_parallel', True),
        max_workers=opt_settings.get('max_workers', min(4, mp.cpu_count())),
        early_stopping=opt_settings.get('early_stopping', True),
        early_stopping_patience=opt_settings.get('early_stopping_patience', 10)
    )
    
    result = optimize_strategy_parameters(strategy, strategy_config, data, opt_config)
    
    # Update strategy with optimal parameters
    if hasattr(result, 'param_combination') and result.param_combination:
        strategy.parameters.update(result.param_combination)
        print("✅ Strategy updated with optimal parameters")
    
    # Convert result to dict for consistent return type
    if hasattr(result, '__dict__'):
        return vars(result)
    return result if isinstance(result, dict) else {}


def run_walkforward_analysis(strategy, strategy_config: StrategyConfig, data: pd.DataFrame) -> Dict[str, Any]:
    """Run walk-forward analysis.
    
    Args:
        strategy: Strategy instance
        strategy_config: Strategy configuration
        data: Primary timeframe data for analysis
        
    Returns:
        Walk-forward analysis results
    """
    print(f"📊 Walk-forward analysis on {len(data)} bars")
    
    result = run_walkforward_analysis_functional(strategy, strategy_config, data)
    # Convert result to dict for consistent return type
    if hasattr(result, '__dict__'):
        return vars(result)
    return result if isinstance(result, dict) else {}


def run_monte_carlo_analysis(strategy, data: pd.DataFrame) -> Dict[str, Any]:
    """Run Monte Carlo analysis.
    
    Args:
        strategy: Strategy instance
        data: Primary timeframe data for analysis
        
    Returns:
        Monte Carlo analysis results
    """
    print(f"🎲 Monte Carlo analysis on {len(data)} bars")
    
    result = run_monte_carlo_analysis_functional(strategy, data)
    # Convert result to dict for consistent return type
    if hasattr(result, '__dict__'):
        return vars(result)
    return result if isinstance(result, dict) else {}


def run_complete_analysis(strategy, strategy_config: StrategyConfig, data: pd.DataFrame, 
                         skip_optimization: bool = False) -> Dict[str, Any]:
    """Run complete analysis pipeline.
    
    Args:
        strategy: Strategy instance
        strategy_config: Strategy configuration
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
        optimization_results = run_optimization(strategy, strategy_config, data)
        results['optimization'] = optimization_results
        
        # Step 2: Walk-Forward Analysis
        print("\n📈 STEP 2: Walk-Forward Analysis")
        walkforward_results = run_walkforward_analysis(strategy, strategy_config, data)
        results['walkforward'] = walkforward_results
        
        # Step 3: Monte Carlo Analysis
        print("\n🎲 STEP 3: Monte Carlo Analysis")
        monte_carlo_results = run_monte_carlo_analysis(strategy, data)
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

def create_strategy(strategy_name: str, strategy_config: StrategyConfig):
    """Create functional strategy wrapper from name and configuration.

    Args:
        strategy_name: Name of the strategy
        strategy_config: Strategy configuration

    Returns:
        Strategy wrapper object with functional interface

    Raises:
        ValueError: If strategy cannot be created
    """
    try:
        from strategies import get_strategy_signal_function, get_required_timeframes, get_required_columns
        
        # Create a simple strategy wrapper object
        class StrategyWrapper:
            """Simple wrapper to provide strategy interface."""
            
            def __init__(self, name, config):
                self.name = name
                self.config = config
                self.parameters = config.parameters.copy()
                self._signal_func = get_strategy_signal_function(name)
                self._required_tfs = get_required_timeframes(name, self.parameters)
                self._required_cols = get_required_columns(name)
            
            def generate_signals(self, tf_data):
                """Generate trading signals."""
                return self._signal_func(tf_data, self.parameters)
            
            def get_required_timeframes(self):
                """Get required timeframes."""
                return self._required_tfs
            
            def get_required_columns(self):
                """Get required columns."""
                return self._required_cols
            
            def get_parameter(self, key, default=None):
                """Get parameter value."""
                return self.parameters.get(key, default)
        
        return StrategyWrapper(strategy_name, strategy_config)
    except ValueError as e:
        raise ValueError(f"Unknown strategy: {strategy_name}. {e}") from e


# Note: load_strategy_config functionality is already available in core_components.py
# Use: from core_components import load_strategy_config
