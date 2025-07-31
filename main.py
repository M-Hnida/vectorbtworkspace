#!/usr/bin/env python3
"""
Core Trading System - Functional Implementation
Clean, functional approach without unnecessary OOP.
"""

import warnings
from typing import Dict, Any, Optional
import pandas as pd

# Core imports
from backtest import run_backtest, get_available_strategies, load_strategy_config
from data_manager import load_data_for_strategy
from base import StrategyConfig
from strategies import get_strategy_signal_function, get_required_timeframes, get_required_columns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_primary_data(data: Dict[str, Dict[str, pd.DataFrame]], 
                    strategy_config: StrategyConfig) -> tuple:
    """Get primary symbol and timeframe data for analysis."""
    if not data:
        raise ValueError("No data provided")
    
    primary_symbol = strategy_config.parameters.get('primary_symbol', list(data.keys())[0])
    if primary_symbol not in data:
        raise ValueError(f"Primary symbol '{primary_symbol}' not found in data")
    
    primary_timeframe = strategy_config.parameters.get('primary_timeframe', 
                                                      list(data[primary_symbol].keys())[0])
    if primary_timeframe not in data[primary_symbol]:
        raise ValueError(f"Primary timeframe '{primary_timeframe}' not found for {primary_symbol}")
    
    primary_data = data[primary_symbol][primary_timeframe]
    return primary_symbol, primary_timeframe, primary_data


def generate_signals_for_strategy(strategy_name: str, parameters: dict, tf_data: Dict[str, pd.DataFrame]):
    """Generate signals using strategy function."""
    signal_func = get_strategy_signal_function(strategy_name)
    return signal_func(tf_data, parameters)


def run_single_backtest(symbol: str, timeframe: str, timeframes: Dict[str, pd.DataFrame],
                       strategy_name: str, parameters: dict, required_tfs: list, is_multi_tf: bool = False):
    """Run backtest for a single symbol/timeframe combination."""
    try:
        if is_multi_tf:
            # Prepare multi-timeframe data
            tf_data = {tf: timeframes[tf] for tf in required_tfs if tf in timeframes}
            primary_data = timeframes[timeframe]
        else:
            # Single timeframe data
            tf_data = {timeframe: timeframes[timeframe]}
            primary_data = timeframes[timeframe]
        
        # Generate signals and run backtest
        signals = generate_signals_for_strategy(strategy_name, parameters, tf_data)
        portfolio = run_backtest(primary_data, signals)
        
        # Log results
        total_return = portfolio.stats()['Total Return [%]']
        tf_label = f" (Multi-TF)" if is_multi_tf else ""
        print(f"✅ {symbol} {timeframe}{tf_label}: {total_return:.2f}%")
        
        return portfolio
        
    except Exception as e:
        print(f"❌ {symbol} {timeframe} failed: {e}")
        return None


def run_full_backtest(data: Dict[str, Dict[str, pd.DataFrame]], 
                     strategy_name: str, parameters: dict) -> Dict[str, Dict[str, Any]]:
    """Run backtest for all symbols and timeframes."""
    print(f"🔄 Running backtest for strategy: {strategy_name}")
    results = {}
    required_tfs = get_required_timeframes(strategy_name, parameters)
    
    for symbol, timeframes in data.items():
        results[symbol] = {}
        
        if len(required_tfs) > 1:
            # Multi-timeframe strategy
            primary_tf = required_tfs[0]
            if primary_tf in timeframes:
                portfolio = run_single_backtest(
                    symbol, primary_tf, timeframes, strategy_name, parameters, required_tfs, is_multi_tf=True
                )
                if portfolio:
                    results[symbol][primary_tf] = portfolio
        else:
            # Single timeframe strategy - run on all available timeframes
            for timeframe, df in timeframes.items():
                portfolio = run_single_backtest(
                    symbol, timeframe, timeframes, strategy_name, parameters, required_tfs, is_multi_tf=False
                )
                if portfolio:
                    results[symbol][timeframe] = portfolio
    
    return results


def flatten_portfolios(portfolio_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten nested portfolio structure for plotting."""
    flattened = {}
    for symbol, timeframes in portfolio_results.items():
        for timeframe, portfolio in timeframes.items():
            key = f"{symbol}_{timeframe}" if len(timeframes) > 1 else symbol
            flattened[key] = portfolio
    return flattened


def run_fast_analysis(strategy_name: str, strategy_config: StrategyConfig, data: Dict) -> Dict[str, Any]:
    """Run fast analysis - just backtest and basic plots."""
    print("\n🚀 Running Strategy (Default Parameters)")
    portfolio_results = run_full_backtest(data, strategy_name, strategy_config.parameters)
    
    print("\n📊 Generating Portfolio Plots")
    from plotter import plot_comprehensive_analysis
    flattened_portfolios = flatten_portfolios(portfolio_results)
    visualization_results = plot_comprehensive_analysis(flattened_portfolios, strategy_config.name)
    
    return {
        'portfolios': portfolio_results,
        'visualizations': visualization_results
    }


def run_full_analysis(strategy_name: str, strategy_config: StrategyConfig, 
                     data: Dict, primary_data: pd.DataFrame) -> Dict[str, Any]:
    """Run full analysis with optimization and robustness tests."""
    results = {}
    
    # Step 1: Default strategy performance
    print("\n🔧 STEP 1: Default Strategy Performance")
    default_results = run_full_backtest(data, strategy_name, strategy_config.parameters)
    results['default_portfolios'] = default_results
    
    # Step 2: Parameter optimization
    print("\n🔧 STEP 2: Parameter Optimization")
    from optimizer import run_optimization
    
    # Create a simple strategy context for optimizer
    class OptimizerStrategyContext:
        def __init__(self):
            self.name = strategy_name
            self.parameters = strategy_config.parameters.copy()
            self.signal_func = get_strategy_signal_function(strategy_name)
        
        def get_required_timeframes(self):
            return get_required_timeframes(strategy_name, self.parameters)
    
    strategy_context = OptimizerStrategyContext()
    
    optimization_results = run_optimization(strategy_context, strategy_config, primary_data)
    results['optimization'] = optimization_results
    
    # Step 3: Optimized strategy performance
    optimized_params = strategy_config.parameters.copy()
    if 'best_params' in optimization_results and optimization_results['best_params']:
        optimized_params.update(optimization_results['best_params'])
        print(f"✅ Using optimized parameters: {optimization_results['best_params']}")
    else:
        print("⚠️ Optimization failed, using default parameters")
    
    print("\n🚀 STEP 3: Optimized Strategy Performance")
    optimized_results = run_full_backtest(data, strategy_name, optimized_params)
    results['optimized_portfolios'] = optimized_results
    
    # Step 4: Robustness tests
    print("\n📈 STEP 4: Walk-Forward Analysis (Robustness)")
    from walk_forward import run_walkforward_analysis
    
    class OptimizedStrategyContext:
        def __init__(self):
            self.name = strategy_name
            self.parameters = optimized_params
            self.signal_func = get_strategy_signal_function(strategy_name)
        
        def get_required_timeframes(self):
            return get_required_timeframes(strategy_name, self.parameters)
    
    optimized_context = OptimizedStrategyContext()
    
    walkforward_results = run_walkforward_analysis(optimized_context, primary_data)
    results['walkforward'] = walkforward_results
    
    print("\n🎲 STEP 5: Monte Carlo Analysis (Overfitting Test)")
    from optimizer import run_monte_carlo_analysis
    monte_carlo_results = run_monte_carlo_analysis(primary_data, optimized_context)
    results['monte_carlo'] = monte_carlo_results
    
    # Step 5: Comprehensive visualization
    print("\n📊 Generating Comprehensive Analysis Plots")
    visualization_data = {
        'default_backtest': default_results,
        'full_backtest': optimized_results,
        'monte_carlo': monte_carlo_results,
        'walkforward': walkforward_results,
        'optimization': optimization_results
    }
    
    from plotter import create_visualizations
    visualization_results = create_visualizations(visualization_data, strategy_config.name)
    results['visualizations'] = visualization_results
    
    return results


def run_strategy_analysis(strategy_name: str, fast_mode: bool = False, 
                         time_range: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """Run complete strategy analysis pipeline."""
    try:
        # Load strategy and data
        strategy_config = load_strategy_config(strategy_name)
        print(f"📊 Loading data with time range: {time_range or 'full dataset'}")
        
        # Create simple strategy context for data loading
        class StrategyContext:
            name = strategy_name
            def get_required_timeframes(self):
                return get_required_timeframes(strategy_name, strategy_config.parameters)
            def get_required_columns(self):
                return get_required_columns(strategy_name)
            def get_parameter(self, key, default=None):
                return strategy_config.parameters.get(key, default)
        
        strategy_context = StrategyContext()
        data = load_data_for_strategy(strategy_context, time_range, end_date)
        _, _, primary_data = get_primary_data(data, strategy_config)
        
        if fast_mode:
            results = run_fast_analysis(strategy_name, strategy_config, data)
        else:
            results = run_full_analysis(strategy_name, strategy_config, data, primary_data)
        
        return {"success": True, "results": results}
        
    except Exception as e:
        error_msg = f"Strategy analysis failed: {e}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def run_strategy_pipeline(strategy_name: str, time_range: Optional[str] = None, 
                         end_date: Optional[str] = None, skip_optimization: bool = False) -> Dict[str, Any]:
    """Run complete strategy pipeline."""
    try:
        print(f"\n🚀 Starting {strategy_name} strategy...")
        results = run_strategy_analysis(strategy_name, skip_optimization, time_range, end_date)
        
        if results["success"]:
            print("✅ Strategy pipeline completed successfully!")
        else:
            print(f"❌ Strategy pipeline failed: {results['error']}")
            
        return results
        
    except Exception as e:
        error_msg = f"Pipeline error: {e}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}


def get_user_strategy_choice() -> Optional[str]:
    """Get strategy choice from user input."""
    available_strategies = get_available_strategies()
    
    if not available_strategies:
        print("❌ No strategies found in config directory")
        return None

    print("\n📊 Available Strategies:")
    for i, strategy in enumerate(available_strategies, 1):
        print(f"{i}. {strategy}")

    try:
        choice = int(input("\nSelect strategy number: ")) - 1
        if 0 <= choice < len(available_strategies):
            return available_strategies[choice]
        else:
            print("❌ Invalid selection")
            return None
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return None


def get_user_time_range() -> tuple:
    """Get time range preferences from user input."""
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
        time_map = {'2': '2y', '3': '1y', '4': '6m', '5': '3m'}
        
        if time_choice in time_map:
            time_range = time_map[time_choice]
        elif time_choice == '6':
            custom_range = input("Enter time range (e.g., '18m', '2y', '90d'): ").strip()
            if custom_range:
                time_range = custom_range
            custom_end = input("Enter end date (YYYY-MM-DD, press Enter for most recent): ").strip()
            if custom_end:
                end_date = custom_end
                
    except Exception as e:
        print(f"⚠️ Invalid time range input: {e}, using full dataset")
    
    return time_range, end_date


def get_user_analysis_mode() -> bool:
    """Get analysis mode preference from user input."""
    print("\n⚙️ Analysis Mode:")
    print("1. Full analysis with optimization (default)")
    print("2. Fast mode - skip optimization")
    
    try:
        mode_choice = input("\nSelect analysis mode (press Enter for full analysis): ").strip()
        return mode_choice == '2'
    except Exception as e:
        print(f"⚠️ Invalid mode choice: {e}, using full analysis")
        return False


def main():
    """Main interactive entry point."""
    print("🚀 Trading Strategy Analysis Pipeline")
    print("=" * 50)

    # Get user inputs
    strategy_name = get_user_strategy_choice()
    if not strategy_name:
        return
    
    time_range, end_date = get_user_time_range()
    skip_optimization = get_user_analysis_mode()

    # Run pipeline
    results = run_strategy_pipeline(strategy_name, time_range, end_date, skip_optimization)
    return results


def quick_test(strategy_name: str, time_range: str = '3m', fast_mode: bool = True):
    """Quick test function for development and debugging.
    
    Args:
        strategy_name: Name of the strategy to test
        time_range: Time range for testing (default: 3m)
        fast_mode: If True, skip optimization (default: True)
    
    Example:
        python main.py --quick vectorbt
        python main.py --quick momentum --full
    """
    print(f"🧪 Quick Test: {strategy_name} strategy")
    print(f"📅 Time range: {time_range}")
    print(f"⚡ Mode: {'Fast (no optimization)' if fast_mode else 'Full analysis'}")
    print("=" * 50)
    
    results = run_strategy_pipeline(strategy_name, time_range, skip_optimization=fast_mode)
    
    if results['success']:
        print(f"\n✅ {strategy_name} strategy test completed!")
    else:
        print(f"❌ Test failed: {results['error']}")


if __name__ == "__main__":
    import sys
    
    # Simple CLI for quick testing
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        if len(sys.argv) < 3:
            print("Usage: python main.py --quick <strategy_name> [--full]")
            print("Available strategies:", get_available_strategies())
            sys.exit(1)
        
        strategy = sys.argv[2]
        fast_mode = '--full' not in sys.argv
        quick_test(strategy, '3m', fast_mode)
    else:
        main()