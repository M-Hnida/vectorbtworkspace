#!/usr/bin/env python3
"""
Core Trading System - Streamlined Implementation
Core functionality without optimization and plotting modules.
"""

import warnings
from typing import Dict, Any
import pandas as pd

# Core imports
from backtest import run_backtest, get_available_strategies as get_strategies_from_config, load_strategy_config
from data_manager import load_data_for_strategy
from optimizer import monte_carlo_analysis, optimize_strategy
from walk_forward import walk_forward_optimize
from base import StrategyConfig

# Plotting imports
from plotter import (
    plot_comprehensive_analysis,
    create_comparison_plot
)


# Constants
DEFAULT_SPLIT_RATIO = 0.7
DEFAULT_CONFIG_DIR = 'config'
EXCLUDED_CONFIG_FILES = {'data_sources.yaml', 'global_config.yaml', 'settings.yaml'}
PROGRESS_UPDATE_INTERVAL = 20

# Analysis step names
STEP_OPTIMIZATION = "Parameter Optimization"
STEP_WALKFORWARD = "Walk-Forward Analysis"
STEP_MONTE_CARLO = "Monte Carlo Analysis"
STEP_FULL_BACKTEST = "Full Backtest"
STEP_VISUALIZATION = "Generating Visualizations"

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")




# ============================================================================
# MAIN TRADING SYSTEM
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


def run_strategy_analysis(data, strat, fast_mode=False, time_range=None, end_date=None):
    """Run strategy analysis with optimal execution order."""
    # Load strategy and data
    strategy_config = load_strategy_config(strat)
    strategy = create_strategy(strat, strategy_config)
    
    print(f"📊 Loading data with time range: {time_range or 'full dataset'}")
    # Store original data before loading strategy-specific data
    data = load_data_for_strategy(strategy, time_range, end_date)
    
    results = {}
    
    # Get primary data before entering any mode-specific block
    try:
        _, _, primary_data = get_primary_data(data, strategy_config)
    except Exception as e:
        print(f"⚠️ Failed to get primary data: {e}")
        return {"success": False, "error": str(e)}
    
    results = {}
    
    if fast_mode:
        # Fast mode: Just backtest with default parameters
        print("\n🚀 Running Backtest (Default Parameters)")
        full_backtest_results = run_full_backtest(data, strategy)
        results['full_backtest'] = full_backtest_results
    else:
        # Full analysis mode: OPTIMAL ORDER
        _, _, primary_data = get_primary_data(data, strategy_config)
        
        # Step 1: Backtest with DEFAULT parameters (for comparison)
        print("\n🔧 STEP 1: Backtest with Default Parameters")
        default_strategy = create_strategy(strat, strategy_config)  # Fresh copy with defaults
        default_backtest_results = run_full_backtest(data, default_strategy)
        results['default_backtest'] = default_backtest_results
        
        # Step 2: Optimization (find best parameters)
        print("\n🔧 STEP 2: Parameter Optimization")
        optimization_results = optimize_strategy(strategy, strategy_config, primary_data)
        results['optimization'] = optimization_results
        
        # Check if optimization failed and handle fallback
        if 'error' in optimization_results:
            print("⚠️ Optimization failed, continuing with default parameters")
            strategy = create_strategy(strat, strategy_config)  # Reset to defaults
        else:
            # Update strategy with optimized parameters if optimization succeeded
            if 'best_params' in optimization_results:
                strategy.parameters.update(optimization_results['best_params'])
        
        # Step 3: Backtest with OPTIMIZED parameters (or defaults if optimization failed)
        print("\n🚀 STEP 3: Backtest with Final Parameters")
        optimized_backtest_results = run_full_backtest(data, strategy)
        results['full_backtest'] = optimized_backtest_results
        
        # Step 4: Walk-Forward Analysis (test robustness)
        print("\n📈 STEP 4: Walk-Forward Analysis")
        walkforward_results = walk_forward_optimize(strategy, primary_data)
        results['walkforward'] = walkforward_results
        
        # Step 5: Monte Carlo Analysis (test against randomness)
        print("\n🎲 STEP 5: Monte Carlo Analysis")
        # Get actual return from optimized backtest results for significance testing
        actual_return = None
        if 'full_backtest' in results:
            for _, timeframes in results['full_backtest'].items():
                for _, result in timeframes.items():
                    if 'portfolio' in result:
                        stats = result['portfolio'].stats()
                        actual_return = float(stats.get('Total Return [%]', 0))
                        break
                if actual_return is not None:
                    break
        
        monte_carlo_results = monte_carlo_analysis(primary_data, strategy, actual_return)
        results['monte_carlo'] = monte_carlo_results
    
    # Final Step: Generate plots
    print("\n📊 FINAL STEP: Generating Visualizations")
    visualization_results = plot_comprehensive_analysis(results, strat)
    results['visualizations'] = visualization_results
    
    return results


# TradingSystem class removed - use run_trading_system_analysis() directly
    
def run_full_backtest(data: Dict[str, Dict[str, pd.DataFrame]], strategy) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run full backtest on all csv and timeframes."""
    print("Running full backtest on all symbols and timeframes...")
    results = {}
    
    for symbol, timeframes in data.items():
        results[symbol] = {}
        
        # For multi-timeframe strategies, run backtest on the primary (entry) timeframe
        # but provide all timeframes to the strategy
        required_tfs = strategy.get_required_timeframes()
        
        if len(required_tfs) > 1:
            # Multi-timeframe strategy - use primary timeframe for backtest
            primary_tf = required_tfs[0]  # Entry timeframe
            
            if primary_tf in timeframes:
                try:
                    # Provide all available timeframes to the strategy
                    tf_data = {}
                    for req_tf in required_tfs:
                        if req_tf in timeframes:
                            tf_data[req_tf] = timeframes[req_tf]
                    
                    signals = strategy.generate_signals(tf_data)
                    print(f"✅ Multi-timeframe signals generated for {symbol}")
                    
                    # Run backtest on primary timeframe
                    primary_data = timeframes[primary_tf]
                    portfolio = run_backtest(primary_data, signals)
                    
                    results[symbol][primary_tf] = {
                        'portfolio': portfolio
                    }
                    
                    # Print portfolio stats
                    print(f"\n📊 {symbol} {primary_tf} (Multi-TF) Stats:")
                    print(portfolio.stats())
                except Exception as e:
                    print(f"⚠️ Multi-timeframe backtest failed for {symbol}: {e}")
                    raise e
            else:
                print(f"⚠️ Primary timeframe {primary_tf} not available for {symbol}")
        else:
            # Single timeframe strategy
            for timeframe, df in timeframes.items():
                try:
                    tf_data = {timeframe: df}
                    signals = strategy.generate_signals(tf_data)
                    print(f"✅ Signals generated for {symbol} {timeframe}")
                    
                    # Run backtest
                    portfolio = run_backtest(df, signals)
                    
                    results[symbol][timeframe] = {
                        'portfolio': portfolio
                    }
                    
                    # Print portfolio stats
                    print(f"\n📊 {symbol} {timeframe} Stats:")
                    print(portfolio.stats())
                except Exception as e:
                    print(f"⚠️ Metrics calculation failed for {symbol} {timeframe}: {e}")
                    raise e
    return results

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Use the function from core_components instead of duplicating
get_available_strategies = get_strategies_from_config

def run_strategy_pipeline(strategy_name: str, time_range: str = None, end_date: str = None, 
                         skip_optimization: bool = False) -> Dict[str, Any]:
    """Run strategy pipeline - simplified."""
    try:
        print(f"\n🚀 Starting {strategy_name} strategy...")
        results = run_strategy_analysis(None, strategy_name, None, skip_optimization, time_range, end_date)
        return {"success": True, "results": results}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Main entry point."""
    print("🚀 Trading Strategy Analysis Pipeline")
    print("="*50)

    available_strategies = get_available_strategies()
    
    if not available_strategies:
        print("❌ No strategies found in config directory")
        return

    print("\n📊 Available Strategies:")
    for i, strategy in enumerate(available_strategies, 1):
        print(f"{i}. {strategy}")

    try:
        choice = int(input("\nSelect strategy number: ")) - 1
        if choice < 0 or choice >= len(available_strategies):
            raise IndexError("Invalid choice")
        strategy_name = available_strategies[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return

    # Ask for time range preference
    print("\n📅 Time Range Options:")
    print("1. Full dataset (default)")
    print("2. Last 2 years")
    print("3. Last 1 year")
    print("4. Last 6 months")
    print("5. Last 3 months")
    print("6. Custom time range")
    
    time_range = None
    end_date = None
    
    try:
        time_choice = input("\nSelect time range (press Enter for full dataset): ").strip()
        if time_choice == '2':
            time_range = '2y'
        elif time_choice == '3':
            time_range = '1y'
        elif time_choice == '4':
            time_range = '6m'
        elif time_choice == '5':
            time_range = '3m'
        elif time_choice == '6':
            custom_range = input("Enter time range (e.g., '18m', '2y', '90d'): ").strip()
            if custom_range:
                time_range = custom_range
            custom_end = input("Enter end date (YYYY-MM-DD, press Enter for most recent): ").strip()
            if custom_end:
                end_date = custom_end
    except Exception as e:
        print(f"⚠️ Invalid time range input: {e}, using full dataset")

    # Ask for analysis mode
    print("\n⚙️ Analysis Mode:")
    print("1. Full analysis with optimization (default)")
    print("2. Fast mode - skip optimization")
    
    skip_optimization = False
    try:
        mode_choice = input("\nSelect analysis mode (press Enter for full analysis): ").strip()
        if mode_choice == '2':
            skip_optimization = True
    except Exception as e:
        print(f"⚠️ Invalid mode choice: {e}, using full analysis")

    results = run_strategy_pipeline(strategy_name, time_range, end_date, skip_optimization)

    if results["success"]:
        print("\n✅ Strategy pipeline completed successfully!")
    else:
        print(f"\n❌ Strategy pipeline failed: {results['error']}")
def quick_test(strategy_name: str, time_range: str = '3m', fast_mode: bool = True):
    """Quick test function for development and debugging with no plotting.
    
    Args:
        strategy_name: Name of the strategy to test
        time_range: Time range for testing (default: 3m)
        fast_mode: If True, skip optimization (default: True)
    
    Example:
        python trading_system.py --quick vectorbt
        python trading_system.py --quick momentum --full
    """
    print(f"🧪 Quick Test: {strategy_name} strategy")
    print(f"📅 Time range: {time_range}")
    print(f"⚡ Mode: {'Fast (no optimization)' if fast_mode else 'Full analysis'}")
    print("=" * 50)
    
    results = run_strategy_pipeline(strategy_name, time_range, fast_mode=fast_mode)
    
    if results['success']:
        print(f"\n✅ {strategy_name} strategy test completed!")
        
        # Stats already printed in run_full_backtest
    else:
        print(f"❌ Test failed: {results['error']}")

if __name__ == "__main__":
    import sys
    
    # Simple CLI for quick testing
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        if len(sys.argv) < 3:
            print("Usage: python trading_system.py --quick <strategy_name> [--full]")
            print("Available strategies:", get_available_strategies())
            sys.exit(1)
        
        strategy = sys.argv[2]
        fast_mode = '--full' not in sys.argv
        quick_test(strategy, fast_mode=fast_mode)
    else:
        main()
