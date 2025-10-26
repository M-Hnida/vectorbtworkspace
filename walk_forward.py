#!/usr/bin/env python3
"""
Walk-forward analysis: simple, clean, native VectorBT implementation.
Optimizes on train window, tests on hold-out window.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Tuple
from itertools import product

from constants import (
    TRAIN_WINDOW_DAYS,
    TEST_WINDOW_DAYS,
    MAX_WINDOWS,
    MAX_PARAM_COMBINATIONS
)

# =============================================================================
# DOMAIN CONSTANTS - Walk-Forward Module
# These constants are specific to walk-forward analysis and not used elsewhere
# =============================================================================

# Simple walk-forward split ratio
SIMPLE_WALKFORWARD_TRAIN_RATIO = 0.8  # 80/20 train/test split

# Parameter stability thresholds
STABILITY_THRESHOLD_STABLE = 0.7      # >70% similarity = stable
STABILITY_THRESHOLD_MODERATE = 0.4    # 40-70% similarity = moderate


def run_walkforward_analysis(strategy, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Walk-forward analysis: optimize on train, test on hold-out.
    
    This is a standard walk-forward optimization approach:
    1. Split data into overlapping train/test windows
    2. Optimize parameters on each train window
    3. Test optimized parameters on the following test window
    4. Measure out-of-sample performance
    
    This helps detect overfitting - if train performance is much better
    than test performance, the strategy is likely overfit.
    """
    from strategy_registry import create_portfolio, get_optimization_grid
    from optimizer import expand_parameter_grid
    
    print(f"📊 Walk-forward analysis on {len(data)} bars")
    
    # Get optimization grid
    param_grid = get_optimization_grid(strategy.name)
    if not param_grid:
        print("⚠️ No optimization grid, using fixed parameters")
        return simple_walkforward(strategy, data)
    
    expanded_grid = expand_parameter_grid(param_grid)
    
    # Get price data
    if isinstance(data, pd.DataFrame) and 'close' in data.columns:
        price = data['close']
    else:
        price = data
    
    # Calculate number of windows
    total_bars = len(price)
    window_size = TRAIN_WINDOW_DAYS + TEST_WINDOW_DAYS
    num_windows = min(MAX_WINDOWS, (total_bars - TRAIN_WINDOW_DAYS) // TEST_WINDOW_DAYS)
    
    if num_windows < 1:
        raise ValueError(f"Insufficient data: need at least {window_size} bars, have {total_bars}")
    
    print(f"   Running {num_windows} walk-forward windows")
    print(f"   Train: {TRAIN_WINDOW_DAYS} days, Test: {TEST_WINDOW_DAYS} days")
    
    # Run walk-forward windows
    windows = []
    for i in range(num_windows):
        start_idx = i * TEST_WINDOW_DAYS
        train_end = start_idx + TRAIN_WINDOW_DAYS
        test_end = train_end + TEST_WINDOW_DAYS
        
        if test_end > len(price):
            break
        
        # Split data
        train_data = price.iloc[start_idx:train_end]
        test_data = price.iloc[train_end:test_end]
        
        # Convert to DataFrame if needed
        if isinstance(train_data, pd.Series):
            train_df = pd.DataFrame({'close': train_data})
            test_df = pd.DataFrame({'close': test_data})
        else:
            train_df = train_data
            test_df = test_data
        
        # Optimize on train set
        best_params, train_sharpe = optimize_window(
            strategy.name, train_df, expanded_grid
        )
        
        # Test on out-of-sample
        test_portfolio = create_portfolio(strategy.name, test_df, best_params)
        if test_portfolio is None:
            print(f"   ⚠️ Window {i+1}: Failed to create test portfolio")
            continue
            
        test_stats = test_portfolio.stats()
        if test_stats is not None:
            sharpe_value = test_stats.get('Sharpe Ratio', 0.0)
            test_sharpe = float(sharpe_value) if sharpe_value is not None else 0.0
        else:
            test_sharpe = 0.0
        
        # Benchmark: hold
        try:
            hold_portfolio = vbt.Portfolio.from_holding(test_df['close'], freq='1H')
            hold_stats = hold_portfolio.stats()
            if hold_stats is not None:
                sharpe_value = hold_stats.get('Sharpe Ratio', 0.0)
                hold_sharpe = float(sharpe_value) if sharpe_value is not None else 0.0
            else:
                hold_sharpe = 0.0
        except Exception:
            hold_sharpe = 0.0
        
        window_result = {
            'window': i + 1,
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'best_params': best_params,
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'hold_sharpe': hold_sharpe
        }
        
        windows.append(window_result)
        print(f"✅ Window {i+1}: Train={train_sharpe:.3f}, Test={test_sharpe:.3f}, Hold={hold_sharpe:.3f}")
    
    if not windows:
        raise ValueError("No valid walk-forward windows completed")
    
    # Calculate summary statistics
    test_sharpes = [w['test_sharpe'] for w in windows]
    hold_sharpes = [w['hold_sharpe'] for w in windows]
    
    return {
        'windows': windows,
        'avg_test_sharpe': float(np.mean(test_sharpes)),
        'avg_hold_sharpe': float(np.mean(hold_sharpes)),
        'parameter_stability': calculate_stability(windows),
        'summary': f"Completed {len(windows)} windows, avg test Sharpe: {np.mean(test_sharpes):.3f}"
    }


def optimize_window(strategy_name: str, data: pd.DataFrame, param_grid: Dict) -> Tuple[Dict, float]:
    """
    Optimize parameters on a single window using grid search.
    
    Returns:
        Tuple of (best_params, best_sharpe)
    """
    from strategy_registry import create_portfolio, get_default_parameters
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    best_params = None
    best_sharpe = -float('inf')
    success_count = 0
    
    # Test all combinations (limit for speed)
    test_limit = min(len(combinations), MAX_PARAM_COMBINATIONS)
    
    for combo in combinations[:test_limit]:
        params = dict(zip(param_names, combo))
        
        try:
            portfolio = create_portfolio(strategy_name, data, params)
            if portfolio is None:
                continue
                
            stats = portfolio.stats()
            if stats is None:
                continue
                
            sharpe = float(stats.get('Sharpe Ratio', -float('inf')))
            
            if np.isfinite(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
                success_count += 1
        except Exception:
            continue
    
    # If no valid params found, use defaults
    if best_params is None:
        print("   ⚠️ No valid parameter combinations found, using defaults")
        best_params = get_default_parameters(strategy_name)
        if not best_params:
            best_params = {}
        best_sharpe = 0.0
    
    return best_params, best_sharpe


def simple_walkforward(strategy, data: pd.DataFrame) -> Dict[str, Any]:
    """Simple 80/20 train/test split without optimization."""
    from strategy_registry import create_portfolio
    
    if isinstance(data, pd.DataFrame) and 'close' in data.columns:
        price = data['close']
    else:
        price = data
    
    split_point = int(len(price) * SIMPLE_WALKFORWARD_TRAIN_RATIO)
    train_data = pd.DataFrame({'close': price.iloc[:split_point]})
    test_data = pd.DataFrame({'close': price.iloc[split_point:]})
    
    train_pf = create_portfolio(strategy.name, train_data, strategy.parameters)
    test_pf = create_portfolio(strategy.name, test_data, strategy.parameters)
    
    if train_pf is None or test_pf is None:
        raise ValueError("Failed to create portfolios for simple walkforward")
    
    train_stats = train_pf.stats()
    test_stats = test_pf.stats()
    
    if train_stats is not None:
        sharpe_value = train_stats.get('Sharpe Ratio', 0.0)
        train_sharpe = float(sharpe_value) if sharpe_value is not None else 0.0
    else:
        train_sharpe = 0.0
        
    if test_stats is not None:
        sharpe_value = test_stats.get('Sharpe Ratio', 0.0)
        test_sharpe = float(sharpe_value) if sharpe_value is not None else 0.0
    else:
        test_sharpe = 0.0
    
    return {
        'windows': [{
            'window': 1,
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
        }],
        'avg_test_sharpe': test_sharpe,
        'parameter_stability': 'no_optimization',
        'summary': "Simple train/test split (no optimization)"
    }


def calculate_stability(windows: list) -> str:
    """
    Calculate parameter stability across windows.
    
    Returns:
        'stable': Parameters are consistent across windows (>70% similarity)
        'moderate': Some variation in parameters (40-70% similarity)
        'unstable': High variation in parameters (<40% similarity)
        'insufficient_data': Not enough windows to assess
        'no_optimization': No parameter optimization was performed
    """
    if len(windows) < 2:
        return "insufficient_data"
    
    param_sets = [w.get('best_params', {}) for w in windows if 'best_params' in w]
    if not param_sets:
        return "no_optimization"
    
    # Count unique parameter combinations
    unique_combinations = len(set(str(sorted(p.items())) for p in param_sets))
    stability_ratio = 1.0 - (unique_combinations / len(param_sets))
    
    if stability_ratio > STABILITY_THRESHOLD_STABLE:
        return "stable"
    elif stability_ratio > STABILITY_THRESHOLD_MODERATE:
        return "moderate"
    else:
        return "unstable"
