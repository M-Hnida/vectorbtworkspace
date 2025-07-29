#!/usr/bin/env python3
"""
Core Trading System - Streamlined Implementation
Core functionality without optimization and plotting modules.
"""

import traceback
import warnings
from typing import Dict, Any

import pandas as pd

from core_components import run_backtest, get_available_strategies as get_strategies_from_config, load_strategy_config
from data_manager import load_data_for_strategy
from analysis_pipeline import run_complete_analysis, get_primary_data, create_strategy
from plotter import plot_comprehensive_analysis

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

def run_trading_system_analysis(strategy_name: str = None, symbol: str = None,
                               time_range: str = None, end_date: str = None, 
                               skip_optimization: bool = False) -> Dict[str, Any]:
    """Run complete trading system analysis pipeline.
    
    Args:
        strategy_name: Name of the strategy to use
        symbol: Trading symbol (extracted from data files)
        time_range: Time range for analysis (e.g., '2y', '6m', '1y')
        end_date: End date for the time range (defaults to most recent data)
        skip_optimization: If True, skip optimization and use default parameters
        
    Returns:
        Dictionary containing analysis results
    """
    strategy_name = strategy_name or 'momentum'
    symbol = symbol or 'EURUSD'  # Using default symbol

    # Load strategy configuration and create strategy
    strategy_config = load_strategy_config(strategy_name)
    strategy = create_strategy(strategy_name, strategy_config)

    results = {}

    # Load data for the strategy with time range control
    print(f"📊 Loading data with time range: {time_range or 'full dataset'}")
    data = load_data_for_strategy(strategy, time_range, end_date)
    
    try:
        if skip_optimization:
            print("\n⚡ FAST MODE: Skipping optimization steps")
            
            # Skip to full backtest with default parameters
            print("\n🚀 Running Full Backtest (No Optimization)")
            full_backtest_results = run_full_backtest(, strategy)
            results['full_backtest'] = full_backtest_results
            
            # Create basic visualizations
            print("\n📊 Generating Basic Visualizations")
            visualization_results = _create_basic_visualizations_functional(results, strategy_name)
            results['visualizations'] = visualization_results
            
        else:
            # Run analysis pipeline
            _, _, primary_data = get_primary_data(data, strategy_config)
            analysis_results = run_complete_analysis(strategy, strategy_config, primary_data, skip_optimization)
            results.update(analysis_results)

            # Step 4: Full Backtest with Optimal Parameters
            print("\n🚀 STEP 4: Full Backtest")
            full_backtest_results = run_full_backtest(, strategy)
            results['full_backtest'] = full_backtest_results
            
            # Step 5: Visualization
            print("\n📊 STEP 5: Generating Visualizations")
            visualization_results = _create_visualizations_functional(results, strategy_name)
            results['visualizations'] = visualization_results

        return results
        
    except Exception as e:
        print(f"❌ Error in complete analysis: {e}")
        raise


# TradingSystem class removed - use run_trading_system_analysis() directly
    
def run_full_backtest(symbol:   Dict[str, Dict[str, pd.DataFrame]], strategy) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run full backtest on all symbols and timeframes."""
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
                    
                    # Calculate metrics
                    metrics = get_metrics(portfolio)
                    
                    results[symbol][primary_tf] = {
                        'portfolio': portfolio,
                        'metrics': metrics
                    }
                    print_metrics(metrics, f"{symbol} {primary_tf} (Multi-TF)")
                    
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
                    
                    # Calculate metrics
                    metrics = get_metrics(portfolio)
                    
                    results[symbol][timeframe] = {
                        'portfolio': portfolio,
                        'metrics': metrics
                    }
                    print_metrics(metrics, f"{symbol} {timeframe}")
                except Exception as e:
                    print(f"⚠️ Metrics calculation failed for {symbol} {timeframe}: {e}")
                    raise e
    return results
    
def _create_visualizations_functional(results: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """Create performance visualizations."""
    try:
        print("📊 Creating performance visualizations...")
        
        portfolios = {}
        if 'full_backtest' in results:
            for symbol, timeframes in results['full_backtest'].items():
                for timeframe, result in timeframes.items():
                    if 'portfolio' in result:
                        portfolios[f"{symbol}_{timeframe}"] = result['portfolio']
        
        if portfolios:
            # Use the functional interface for comprehensive analysis
            viz_result = plot_comprehensive_analysis(
                portfolios=portfolios,
                strategy_name=strategy_name,
                mc_results=results.get('monte_carlo', {}),
                wf_results=results.get('walkforward', {})
            )
            print("✅ Visualizations created successfully")
            return viz_result

        print("⚠️ No portfolios available for visualization")
        return {}
            
    except Exception as e:
        print(f"⚠️ Visualization creation failed: {e}")
        return {"error": str(e)}
    
def _create_basic_visualizations_functional(results: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """Create basic performance visualizations without optimization results."""
    try:
        print("📊 Creating basic performance visualizations...")
        
        portfolios = {}
        if 'full_backtest' in results:
            for symbol, timeframes in results['full_backtest'].items():
                for timeframe, result in timeframes.items():
                    if 'portfolio' in result:
                        portfolios[f"{symbol}_{timeframe}"] = result['portfolio']
        
        if portfolios:
            # Create basic portfolio analysis without optimization results
            viz_result = plot_comprehensive_analysis(
                portfolios=portfolios,
                strategy_name=strategy_name,
                mc_results={},  # Empty for basic mode
                wf_results={}   # Empty for basic mode
            )
            print("✅ Basic visualizations created successfully")
            return viz_result

        print("⚠️ No portfolios available for visualization")
        return {}
            
    except Exception as e:
        print(f"⚠️ Basic visualization creation failed: {e}")
        return {"error": str(e)}


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Use the function from core_components instead of duplicating
get_available_strategies = get_strategies_from_config

def run_strategy_pipeline(strategy_name: str, time_range: str = None, end_date: str = None, 
                         skip_optimization: bool = False) -> Dict[str, Any]:
    """Run complete strategy pipeline with all features."""
    try:
        mode_text = "fast mode" if skip_optimization else "full analysis"
        print(f"\n🚀 Starting {strategy_name} strategy pipeline ({mode_text})...")
        
        print("📊 Loading market data...")
        results = run_trading_system_analysis(strategy_name, time_range=time_range, end_date=end_date, 
                                            skip_optimization=skip_optimization)
        
        return {"success": True, "results": results}
        
    except Exception as e:
        print(f"❌ Error running {strategy_name} strategy: {e}")
        traceback.print_exc()
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

def run_strategy_with_time_range(strategy_name: str, time_range: str = None, 
                                end_date: str = None, symbol: str = None, 
                                skip_optimization: bool = False) -> Dict[str, Any]:
    """Convenience function to run a strategy with specific time range parameters.
    
    Args:
        strategy_name: Name of the strategy to run
        time_range: Time range specification (e.g., '2y', '6m', '1y', '3m')
        end_date: End date for the time range (YYYY-MM-DD format)
        symbol: Trading symbol (optional, extracted from data files)
        skip_optimization: If True, skip optimization and run basic backtest only
    
    Returns:
        Dictionary containing analysis results
    
    Example:
        # Run momentum strategy on last 2 years of data
        results = run_strategy_with_time_range('momentum', '2y')
        
        # Run with custom end date
        results = run_strategy_with_time_range('orb', '1y', '2024-12-31')
        
        # Run without optimization (fast mode)
        results = run_strategy_with_time_range('vectorbt', '3m', skip_optimization=True)
    """
    return run_strategy_pipeline(strategy_name, time_range, end_date, skip_optimization)


def quick_test(strategy_name: str, time_range: str = '3m', fast_mode: bool = True):
    """Quick test function for development and debugging.
    
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
    
    results = run_strategy_with_time_range(strategy_name, time_range, skip_optimization=fast_mode)
    
    if results['success']:
        print(f"\n✅ {strategy_name} strategy test completed!")
        
        # Print key metrics
        fb = results['results']['full_backtest']
        for symbol, timeframes in fb.items():
            for tf, result in timeframes.items():
              print(f"\n{symbol} {tf} metrics:{result.stats()}")

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
