#!/usr/bin/env python3
"""
Core Trading System - Functional Implementation
Clean, functional approach without unnecessary OOP.
"""

import warnings
from typing import Dict, Any, Optional
import pandas as pd

# Core imports
from backtest import run_backtest, get_available_strategies
from data_manager import load_data_for_strategy, load_strategy_config
from base import StrategyConfig
from strategies import get_required_timeframes, get_required_columns
from plotter import create_visualizations
from walk_forward import run_walkforward_analysis
from optimizer import run_optimization,run_monte_carlo_analysis

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_primary_data(data: Dict[str, Dict[str, pd.DataFrame]],
                    strategy_config: StrategyConfig) -> tuple:
    """Get primary symbol/timeframe data, case-insensitive with order fallback."""
    if not data:
        raise ValueError("No data provided")

    params = strategy_config.parameters or {}

    # Select primary symbol by parameter or first key if not found (case-insensitive)
    primary_symbol = params.get('primary_symbol')
    if primary_symbol:
        sym_map = {k.lower(): k for k in data.keys()}
        primary_symbol = sym_map.get(str(primary_symbol).lower(), primary_symbol)
    if primary_symbol not in data:
        primary_symbol = next(iter(data.keys()))

    # Select timeframe by case-insensitive match, fallback to first available
    available_tfs = data[primary_symbol]
    requested_tf = params.get('primary_timeframe')
    if requested_tf:
        tf_map = {k.lower(): k for k in available_tfs.keys()}
        chosen_tf = tf_map.get(str(requested_tf).lower(), next(iter(available_tfs.keys())))
    else:
        chosen_tf = next(iter(available_tfs.keys()))

    primary_data = available_tfs[chosen_tf]
    # Persist chosen primary timeframe for downstream consistency
    strategy_config.parameters['primary_timeframe'] = chosen_tf
    return primary_symbol, chosen_tf, primary_data


def generate_signals_for_strategy(strategy_name: str, parameters: dict, tf_data: Dict[str, pd.DataFrame]):
    """Generate signals using strategy function."""
    # Use canonical registry accessor; alias kept in strategies for compatibility
    from strategies import get_strategy_function
    return get_strategy_function(strategy_name)(tf_data, parameters)


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
        
        # Generate signals
        signals = generate_signals_for_strategy(strategy_name, parameters, tf_data)

        # Get portfolio params via unified hook
        from strategies import get_portfolio_params
        vbt_params = get_portfolio_params(strategy_name, primary_data, parameters) or {}
        if not isinstance(vbt_params, dict):
            vbt_params = {}

        # Run backtest
        portfolio = run_backtest(primary_data, signals, vbt_params=vbt_params)
        
        # Log results
        total_return = portfolio.stats()['Total Return [%]'] #type:ignore
        tf_label = " (Multi-TF)" if is_multi_tf else ""
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
            req = parameters.get('primary_timeframe', required_tfs[0] if required_tfs else None)
            if req:
                tf_map = {k.lower(): k for k in timeframes.keys()}
                chosen_tf = tf_map.get(str(req).lower(), next(iter(timeframes.keys())))
            else:
                chosen_tf = next(iter(timeframes.keys()))

            portfolio = run_single_backtest(
                symbol, chosen_tf, timeframes, strategy_name, parameters, required_tfs, is_multi_tf=True
            )
            if portfolio:
                results[symbol][chosen_tf] = portfolio
        else:
            # Single timeframe strategy - run on all available timeframes
            for timeframe in timeframes.keys():
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


def run_fast_analysis(strategy_name: str, strategy_config: StrategyConfig, data: Dict):
    """Run fast analysis - just backtest and basic plots."""
    print("\n🚀 Running Strategy (Default Parameters)")
    portfolio_results = run_full_backtest(data, strategy_name, strategy_config.parameters)
    
    print("\n📊 Generating Portfolio Plots")
    from plotter import plot_comprehensive_analysis
    flattened_portfolios = flatten_portfolios(portfolio_results)
    plot_comprehensive_analysis(flattened_portfolios, strategy_config.name)
    

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
    
    # Create a simple strategy context for optimizer
    class OptimizerStrategyContext:
        """Strategy context for optimization operations."""
        
        def __init__(self):
            """Initialize optimizer strategy context."""
            self.name = strategy_name
            self.parameters = strategy_config.parameters.copy()
            from strategies import get_strategy_function
            self.signal_func = get_strategy_function(strategy_name)
        
        def get_required_timeframes(self):
            """Get required timeframes for the strategy."""
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
    
    class OptimizedStrategyContext:
        """Strategy context for optimized parameters."""
        
        def __init__(self):
            """Initialize optimized strategy context."""
            self.name = strategy_name
            self.parameters = optimized_params
            from strategies import get_strategy_function
            self.signal_func = get_strategy_function(strategy_name)
        
        def get_required_timeframes(self):
            """Get required timeframes for the strategy."""
            return get_required_timeframes(strategy_name, self.parameters)
    
    optimized_context = OptimizedStrategyContext()
    
    walkforward_results = run_walkforward_analysis(optimized_context, primary_data)
    results['walkforward'] = walkforward_results
    
    print("\n🎲 STEP 5: Monte Carlo Analysis (Overfitting Test)")
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
    
    # Prefer showing Monte Carlo by default when available
    visualization_results = create_visualizations(visualization_data, strategy_config.name)
    results['visualizations'] = visualization_results
    
    return results


def run_strategy_analysis(strategy_name: str, fast_mode: bool = False,
                         time_range: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """Run complete strategy analysis pipeline."""
    try:
        # Load strategy and data
        raw_config = load_strategy_config(strategy_name)  # returns dict
        # Normalize to StrategyConfig-like shape for this module
        parameters = (raw_config or {}).get('parameters', {})
        strategy_config = StrategyConfig(
            name=strategy_name,
            parameters=parameters,
            optimization_grid=(raw_config or {}).get('optimization_grid', {}),
            analysis_settings=(raw_config or {}).get('analysis_settings', {}),
            data_requirements=(raw_config or {}).get('data_requirements', {})
        )
        print(f"📊 Loading data with time range: {time_range or 'full dataset'}")

        # Create simple strategy context for data loading
        class StrategyContext:
            """Simple strategy context for data loading operations."""
            name = strategy_name

            def get_required_timeframes(self):
                return get_required_timeframes(strategy_name, strategy_config.parameters)

            def get_required_columns(self):
                return get_required_columns(strategy_name)

            def get_parameter(self, key, default=None):
                return strategy_config.parameters.get(key, default)

        strategy_context = StrategyContext()
        data = load_data_for_strategy(strategy_context, time_range, end_date)
        # get_primary_data returns a tuple; bind explicitly to avoid linter confusion
        _sym, _tf, primary_data = get_primary_data(data, strategy_config)

        if fast_mode:
            results = run_fast_analysis(strategy_name, strategy_config, data)
        else:
            _ = run_full_analysis  # hint for linter about function usage
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
    """Run complete strategy pipeline.

    Note: run_strategy_analysis already handles exceptions and returns a structured result.
    Avoid redundant try/except here to prevent duplicated error handling.
    """
    print(f"\n🚀 Starting {strategy_name} strategy...")
    results = run_strategy_analysis(strategy_name, skip_optimization, time_range, end_date)

    if isinstance(results, dict) and results.get("success", False):
        print("✅ Strategy pipeline completed successfully!")
    else:
        # Defensive extract of error message if present
        err = None
        if isinstance(results, dict):
            err = results.get("error")
        print(f"❌ Strategy pipeline failed{f': {err}' if err else ''}")
    return results


def get_user_strategy_choice() -> Optional[str]:
    """Get strategy choice from user input."""
    available_strategies = get_available_strategies()
    
    if not available_strategies:
        print("❌ No strategies found in config directory")
        return None

    print("\n📊 Available Strategies:")
    for i ,strategy  in enumerate(available_strategies, 1):
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
