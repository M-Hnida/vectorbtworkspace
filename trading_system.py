#!/usr/bin/env python3
"""
Core Trading System - Streamlined Implementation
Core functionality without optimization and plotting modules.
"""

import os
import traceback
import warnings
from typing import Dict, Any, List

import pandas as pd


from core_components import run_backtest
from data_manager import load_data,DataManager
from metrics import calc_metrics, print_metrics
from optimizer import ParameterOptimizer, WalkForwardAnalysis, MonteCarloAnalysis
from plotter import TradingVisualizer

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

class TradingSystem:
    """Main trading system that orchestrates the complete analysis pipeline."""
    
    def __init__(self, strategy_name: str = None, symbol: str = None):
        """Initialize trading system with strategy and symbol."""
        self.strategy_name = strategy_name or 'momentum'
        self.symbol = symbol or DataManager().defaults.get('symbol', 'EURUSD')  # Using default from config
        
        # Initialize strategy
        self.strategy_config = {
            'name': self.strategy_name,
            'symbol': self.symbol,
            # Add any strategy-specific parameters here from config if needed
        }
        self.strategy = self._init_strategy()
        
    def _init_strategy(self):
        """Initialize strategy based on name."""
        strategies = {
            'lti': lti_strategy.LTIStrategy,
            'momentum': momentum_strategy.MomentumStrategy,
            'orb': orb_strategy.ORBStrategy
        }
        
        if self.strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
            
        return strategies[self.strategy_name](self.strategy_config)  # Pass config to strategy
    
    def run_complete_analysis(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Run complete trading system analysis pipeline."""
        results = {}
        
        try:
            # Step 1: Parameter Optimization
            print("\n🔧 STEP 1: Parameter Optimization")
            optimization_results = self._run_optimization(data)
            results['optimization'] = optimization_results
            
            # Step 2: Walk-Forward Analysis
            print("\n📈 STEP 2: Walk-Forward Analysis")
            walkforward_results = self._run_walkforward_analysis(data)
            results['walkforward'] = walkforward_results
            
            # Step 3: Monte Carlo Analysis
            print("\n🎲 STEP 3: Monte Carlo Analysis")
            monte_carlo_results = self._run_monte_carlo_analysis(data)
            results['monte_carlo'] = monte_carlo_results
            
            # Step 4: Full Backtest with Optimal Parameters
            print("\n🚀 STEP 4: Full Backtest")
            full_backtest_results = self._run_full_backtest(data)
            results['full_backtest'] = full_backtest_results
            
            # Step 5: Visualization
            print("\n📊 STEP 5: Generating Visualizations")
            visualization_results = self._create_visualizations(results)
            results['visualizations'] = visualization_results
            
            # Summary
            self._print_final_summary(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in complete analysis: {e}")
            raise
    
    def _get_primary_data(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> tuple:
        """Get primary symbol and timeframe data for analysis."""
        primary_symbol = getattr(self.strategy_config, 'primary_symbol', list(data.keys())[0])
        primary_timeframe = getattr(self.strategy_config, 'primary_timeframe', list(data[primary_symbol].keys())[0])
        primary_data = data[primary_symbol][primary_timeframe]
        return primary_symbol, primary_timeframe, primary_data

    def _run_optimization(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Run parameter optimization."""
        primary_symbol, primary_timeframe, optimization_data = self._get_primary_data(data)

        print(f"🎯 Optimizing on {primary_symbol} {primary_timeframe} ({len(optimization_data)} bars)")
        
        # Get split ratio from config
        split_ratio = getattr(self.strategy_config, 'split_ratio', DEFAULT_SPLIT_RATIO)
        
        optimizer = ParameterOptimizer(self.strategy, self.strategy_config)
        optimization_result = optimizer.optimize(optimization_data, split_ratio=split_ratio)
        
        if 'param_combination' in optimization_result:
            self.strategy.parameters.update(optimization_result['param_combination'])
            print("✅ Strategy updated with optimal parameters")
        
        return optimization_result
    
    def _run_walkforward_analysis(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Run walk-forward analysis."""
        try:
            primary_symbol, primary_timeframe, wf_data = self._get_primary_data(data)
            print(f"📊 Walk-forward analysis on {primary_symbol} {primary_timeframe}")
            
            wf_analyzer = WalkForwardAnalysis(self.strategy, self.strategy_config)
            return wf_analyzer.run_analysis(wf_data)
            
        except Exception as e:
            print(f"⚠️ Walk-forward analysis failed: {e}")
            raise
    
    def _run_monte_carlo_analysis(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Run Monte Carlo analysis."""
        try:
            primary_symbol, primary_timeframe, mc_data = self._get_primary_data(data)
            print(f"🎲 Monte Carlo analysis on {primary_symbol} {primary_timeframe}")
            
            mc_analyzer = MonteCarloAnalysis(self.strategy)
            return mc_analyzer.run_analysis(mc_data)
            
        except Exception as e:
            print(f"⚠️ Monte Carlo analysis failed: {e}")
            raise
    
    def _run_full_backtest(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Run full backtest on all symbols and timeframes."""
        print("Running full backtest on all symbols and timeframes...")
        results = {}
        
        for symbol, timeframes in data.items():
            results[symbol] = {}
            
            # For multi-timeframe strategies, run backtest on the primary (entry) timeframe
            # but provide all timeframes to the strategy
            required_tfs = getattr(self.strategy, 'required_timeframes', list(timeframes.keys()))
            
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
                        
                        signals = self.strategy.generate_signals(tf_data)
                        print(f"✅ Multi-timeframe signals generated for {symbol}")
                        
                        # Run backtest on primary timeframe
                        primary_data = timeframes[primary_tf]
                        portfolio = run_backtest(primary_data, signals)
                        
                        # Calculate metrics
                        metrics = calc_metrics(portfolio)
                        
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
                        signals = self.strategy.generate_signals(tf_data)
                        print(f"✅ Signals generated for {symbol} {timeframe}")
                        
                        # Run backtest
                        portfolio = run_backtest(df, signals)
                        
                        # Calculate metrics
                        metrics = calc_metrics(portfolio)
                        
                        results[symbol][timeframe] = {
                            'portfolio': portfolio,
                            'metrics': metrics
                        }
                        print_metrics(metrics, f"{symbol} {timeframe}")
                    except Exception as e:
                        print(f"⚠️ Metrics calculation failed for {symbol} {timeframe}: {e}")
                        raise e
        return results
    
    def _create_visualizations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance visualizations."""
        visualizer = TradingVisualizer()
        try:
            print("📊 Creating performance visualizations...")
            
            portfolios = {}
            if 'full_backtest' in results:
                for symbol, timeframes in results['full_backtest'].items():
                    for timeframe, result in timeframes.items():
                        if 'portfolio' in result:
                            portfolios[f"{symbol}_{timeframe}"] = result['portfolio']
            
            if portfolios:
                # Use the public interface for comprehensive analysis
                viz_result = visualizer.plot_comprehensive_analysis(
                    portfolios=portfolios,
                    strategy_name=self.strategy_name,
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

    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final summary of the analysis."""
        print("\n📊 Final Summary:")
        for symbol, timeframes in results['full_backtest'].items():
            for timeframe, result in timeframes.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    sharpe = metrics['sharpe']
                    total_return = metrics['total_return']
                    max_dd = metrics['max_drawdown']
                    trades = metrics['total_trades']
                    win_rate = metrics['win_rate']
                    print(f"📊 {symbol} {timeframe}: Sharpe={sharpe:.3f}, Return={total_return:.1f}%, DD={max_dd:.1f}%, Trades={trades}, WinRate={win_rate:.1f}%")

        print("\n✅ Analysis completed!")
        print("="*50)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def get_available_strategies() -> List[str]:
    """Get list of available strategies from config directory."""
    config_dir = 'config'
    if not os.path.exists(config_dir):
        return []
    
    # Exclude non-strategy config files
    excluded_files = {'data_sources.yaml', 'global_config.yaml', 'settings.yaml'}
    
    strategies = []
    for filename in os.listdir(config_dir):
        if filename.endswith('.yaml') and filename not in excluded_files:
            strategies.append(os.path.splitext(filename)[0])
    
    return strategies

def run_strategy_pipeline(strategy_name: str) -> Dict[str, Any]:
    """Run complete strategy pipeline with all features."""
    try:
        print(f"\n🚀 Starting {strategy_name} strategy pipeline...")
        
        trading_system = TradingSystem(strategy_name)
        
        print("📊 Loading market data...")
        # Extract data requirements from strategy config
        data_requirements = getattr(trading_system.strategy_config, 'data_requirements', {})
        
        # Use existing load_data function with parameters from config
        data = load_data(
            symbols=data_requirements.get('symbols'),
            timeframes=data_requirements.get('timeframes'),
            symbol_group=data_requirements.get('symbol_group'),
            max_symbols=data_requirements.get('max_symbols')
        )
        
        if not data:
            return {"success": False, "error": "No data loaded"}
        
        results = trading_system.run_complete_analysis(data)
        
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

    results = run_strategy_pipeline(strategy_name)

    if results["success"]:
        print("\n✅ Strategy pipeline completed successfully!")
    else:
        print(f"\n❌ Strategy pipeline failed: {results['error']}")

if __name__ == "__main__":
    main()
