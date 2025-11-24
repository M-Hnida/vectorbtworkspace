#!/usr/bin/env python3
"""
VectorFlow CLI & Strategy Engine
================================

This module serves as the main entry point and orchestration engine for the
VectorFlow trading system. It handles:
1. User interaction (CLI)
2. Strategy orchestration (Backtesting, Optimization, Validation)
3. Result aggregation and reporting

The design follows a clean separation between the `StrategyEngine` (logic)
and the `CLI` (user interface).
"""

import sys
import logging
import warnings
from typing import Dict, Any, Optional, Tuple, List
from types import SimpleNamespace
import pandas as pd

# Core Imports
from vectorflow.core.data_loader import load_data_for_strategy, load_strategy_config
from vectorflow.core.portfolio_builder import (
    create_portfolio,
    get_optimization_grid,
    strategy_needs_multi_timeframe,
    get_available_strategies,
)
from vectorflow.core.constants import (
    STAT_TOTAL_RETURN,
    STAT_SHARPE_RATIO,
    STAT_MAX_DRAWDOWN,
    STAT_WIN_RATE,
    STAT_TOTAL_TRADES,
)

# Analysis Modules
from vectorflow.optimization.walk_forward import run_walkforward_analysis
from vectorflow.optimization.grid_search import run_optimization
from vectorflow.optimization.param_monte_carlo import run_monte_carlo_analysis
from vectorflow.validation.path_randomization import run_path_randomization_mc
from vectorflow.utils.config_validator import quick_validate
from vectorflow.visualization.plotters import create_visualizations, plot_comprehensive_analysis

# Configuration
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("VectorFlow")


class StrategyEngine:
    """
    The core engine responsible for running trading strategies.
    It encapsulates the complexity of data loading, backtesting, optimization,
    and validation pipelines.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.config = self._load_config()
        self.data = {}
        self.primary_data = None
        self.primary_symbol = None
        self.primary_timeframe = None

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate strategy configuration."""
        raw_config = load_strategy_config(self.strategy_name) or {}
        config = raw_config.copy()
        config["name"] = self.strategy_name
        
        # Quick validation (non-blocking)
        if not quick_validate(self.strategy_name, config, auto_fix=False):
            logger.warning(f"Configuration for '{self.strategy_name}' has potential issues.")
            
        return config

    def load_data(self, time_range: Optional[str] = None, end_date: Optional[str] = None):
        """Load data required for the strategy."""
        logger.info(f"Loading data for {self.strategy_name} ({time_range or 'Full History'})...")
        
        # Create a context object expected by data_loader
        def get_parameter(key, default=None):
            if key in self.config:
                return self.config[key]
            return self.config.get("parameters", {}).get(key, default)

        strategy_context = SimpleNamespace(
            name=self.strategy_name,
            config=self.config,
            get_required_timeframes=lambda: ["1h"], # Default assumption
            get_required_columns=lambda: ["open", "high", "low", "close", "volume"],
            get_parameter=get_parameter,
        )

        self.data = load_data_for_strategy(strategy_context, time_range, end_date)
        if not self.data:
            raise ValueError("No data loaded. Check your data directory and configuration.")
            
        self._resolve_primary_data()

    def _resolve_primary_data(self):
        """Determine the primary symbol and timeframe for optimization/analysis."""
        params = self.config.get("parameters", {})
        
        # Resolve Symbol
        requested_symbol = str(params.get("primary_symbol", "")).lower()
        sym_map = {k.lower(): k for k in self.data.keys()}
        self.primary_symbol = sym_map.get(requested_symbol)
        
        if not self.primary_symbol:
            self.primary_symbol = next(iter(self.data.keys()))
            
        # Resolve Timeframe
        available_tfs = self.data[self.primary_symbol]
        requested_tf = str(params.get("primary_timeframe", "")).lower()
        tf_map = {k.lower(): k for k in available_tfs.keys()}
        chosen_tf = tf_map.get(requested_tf)
        
        if not chosen_tf:
            chosen_tf = next(iter(available_tfs.keys()))
            
        self.primary_timeframe = chosen_tf
        self.primary_data = available_tfs[chosen_tf]
        
        # Update config to reflect reality
        self.config["parameters"]["primary_symbol"] = self.primary_symbol
        self.config["parameters"]["primary_timeframe"] = self.primary_timeframe

    def run_backtest(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run a batch backtest across all loaded symbols and timeframes."""
        results = {}
        required_tfs = ["1h"] # Could be dynamic based on strategy

        for symbol, timeframes in self.data.items():
            results[symbol] = {}
            
            # Determine which timeframe(s) to test
            if len(required_tfs) > 1:
                # Multi-timeframe strategy: pass all data, but key by primary TF
                tf_to_use = self.primary_timeframe
                portfolio = self._create_portfolio_safe(symbol, tf_to_use, timeframes, params)
                if portfolio:
                    results[symbol][tf_to_use] = portfolio
            else:
                # Single timeframe strategy: test on ALL available timeframes independently
                for tf, data in timeframes.items():
                    # For single TF, we pass just the dataframe
                    portfolio = self._create_portfolio_safe(symbol, tf, timeframes, params)
                    if portfolio:
                        results[symbol][tf] = portfolio
                        
        return results

    def _create_portfolio_safe(self, symbol, tf, timeframes, params):
        """Helper to create portfolio with error handling."""
        try:
            if strategy_needs_multi_timeframe(self.strategy_name):
                return create_portfolio(self.strategy_name, timeframes, params)
            else:
                return create_portfolio(self.strategy_name, timeframes[tf], params)
        except Exception as e:
            logger.error(f"Backtest failed for {symbol} {tf}: {e}")
            return None

    def run_optimization(self) -> Dict[str, Any]:
        """Run parameter optimization."""
        logger.info("🔧 Running Parameter Optimization...")
        grid = get_optimization_grid(self.strategy_name)
        if not grid:
            logger.warning("No optimization grid found. Skipping.")
            return {}
            
        return run_optimization(self.strategy_name, self.primary_data, grid)

    def run_monte_carlo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo parameter sensitivity analysis."""
        logger.info("🎲 Running Monte Carlo Analysis...")
        return run_monte_carlo_analysis(self.primary_data, self.strategy_name, params)

    def run_walk_forward(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Walk-Forward analysis."""
        logger.info("📈 Running Walk-Forward Analysis...")
        wf_context = SimpleNamespace(
            name=self.strategy_name,
            parameters=params,
            get_required_timeframes=lambda: ["1h"],
        )
        return run_walkforward_analysis(wf_context, self.primary_data)

    def run_pipeline(self, mode: str = "full") -> Dict[str, Any]:
        """
        Execute the full analysis pipeline based on the selected mode.
        
        Modes:
        - 'fast': Backtest only (default params)
        - 'full': Optimization -> Backtest -> MC -> WF
        - 'monte_carlo': Optimization -> MC
        - 'walkforward': Optimization -> WF
        """
        results = {
            "success": False,
            "config": self.config,
            "results": {}
        }
        
        try:
            # 1. Initial Backtest (Default Params)
            default_params = self.config.get("parameters", {})
            results["results"]["default_portfolios"] = self.run_backtest(default_params)
            
            if mode == "fast":
                self._generate_plots(results["results"], mode)
                results["success"] = True
                return results

            # 2. Optimization
            optimized_params = default_params.copy()
            opt_results = self.run_optimization()
            if opt_results.get("best_params"):
                optimized_params.update(opt_results["best_params"])
            results["results"]["optimization"] = opt_results
            
            # 3. Optimized Backtest
            results["results"]["optimized_portfolios"] = self.run_backtest(optimized_params)

            # 4. Advanced Analysis
            if mode in ["full", "monte_carlo"]:
                results["results"]["monte_carlo"] = self.run_monte_carlo(optimized_params)
                
            if mode in ["full", "walkforward"]:
                results["results"]["walkforward"] = self.run_walk_forward(optimized_params)

            # 5. Visualization
            self._generate_plots(results["results"], mode)
            
            results["success"] = True
            return results

        except Exception as e:
            logger.exception("Pipeline execution failed")
            results["error"] = str(e)
            return results

    def _generate_plots(self, results_dict: Dict, mode: str):
        """Generate all relevant plots."""
        logger.info("📊 Generating Visualizations...")
        
        # Flatten portfolios for the plotter
        # The plotter expects { "Symbol_TF": portfolio, ... }
        flattened_portfolios = {}
        source_key = "optimized_portfolios" if "optimized_portfolios" in results_dict else "default_portfolios"
        
        for symbol, tfs in results_dict.get(source_key, {}).items():
            for tf, pf in tfs.items():
                key = f"{symbol}_{tf}" if len(tfs) > 1 or len(results_dict.get(source_key, {})) > 1 else symbol
                flattened_portfolios[key] = pf

        # Use the comprehensive plotter
        plot_comprehensive_analysis(
            flattened_portfolios,
            self.strategy_name,
            mc_results=results_dict.get("monte_carlo"),
            wf_results=results_dict.get("walkforward")
        )


class CLI:
    """
    Handles user interaction and command-line arguments.
    """
    
    @staticmethod
    def get_user_choice(options: List[str], prompt: str) -> int:
        while True:
            try:
                choice = input(f"{prompt} (1-{len(options)}): ").strip()
                if not choice: return 0 # Default
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return idx
                print(f"❌ Invalid choice. Please enter 1-{len(options)}")
            except ValueError:
                print("❌ Please enter a number.")

    @staticmethod
    def interactive_mode():
        print("\n🚀 VectorFlow Strategy Runner")
        print("=============================")
        
        # 1. Select Strategy
        strategies = get_available_strategies()
        if not strategies:
            print("❌ No strategies found in 'vectorflow/strategies/'")
            return

        print("\n📊 Available Strategies:")
        for i, s in enumerate(strategies, 1):
            print(f"{i}. {s}")
            
        s_idx = CLI.get_user_choice(strategies, "Select Strategy")
        strategy_name = strategies[s_idx]

        # 2. Select Time Range
        print("\n📅 Time Range:")
        ranges = ["3m", "6m", "1y", "2y", "Full History"]
        for i, r in enumerate(ranges, 1):
            print(f"{i}. {r}")
            
        t_idx = CLI.get_user_choice(ranges, "Select Range (Default: 2y)")
        time_range = ranges[t_idx] if t_idx < 4 else None
        if time_range == "Full History": time_range = None
        if not time_range and t_idx == 0: time_range = "2y" # Default handling

        # 3. Select Mode
        print("\n⚙️  Analysis Mode:")
        modes = [
            ("Quick Analysis (Backtest Only)", "fast"),
            ("Full Analysis (Opt + MC + WF)", "full"),
            ("Monte Carlo Only", "monte_carlo"),
            ("Walk-Forward Only", "walkforward")
        ]
        for i, (desc, _) in enumerate(modes, 1):
            print(f"{i}. {desc}")
            
        m_idx = CLI.get_user_choice(modes, "Select Mode (Default: Quick)")
        mode = modes[m_idx][1]

        # Run
        CLI.run(strategy_name, time_range, mode)

    @staticmethod
    def run(strategy_name: str, time_range: str, mode: str):
        print(f"\n▶ Starting {mode.upper()} analysis for '{strategy_name}'...")
        
        engine = StrategyEngine(strategy_name)
        try:
            engine.load_data(time_range=time_range)
            results = engine.run_pipeline(mode=mode)
            
            if results["success"]:
                CLI.print_summary(results["results"])
                print("\n✅ Analysis Completed Successfully!")
            else:
                print(f"\n❌ Analysis Failed: {results.get('error')}")
                
        except KeyboardInterrupt:
            print("\n⚠️  Analysis interrupted by user.")
        except Exception as e:
            logger.exception("Critical Error")
            print(f"\n❌ Critical Error: {e}")

    @staticmethod
    def print_summary(results: Dict):
        """Print a nice summary of the results."""
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        
        # Determine which portfolios to show (Optimized > Default)
        pfs = results.get("optimized_portfolios") or results.get("default_portfolios")
        
        if not pfs:
            print("No portfolio results to display.")
            return

        for symbol, tfs in pfs.items():
            for tf, pf in tfs.items():
                stats = pf.stats()
                print(f"\n🔸 {symbol} [{tf}]")
                print(f"   Return:      {stats[STAT_TOTAL_RETURN]:.2f}%")
                print(f"   Sharpe:      {stats[STAT_SHARPE_RATIO]:.3f}")
                print(f"   Max DD:      {stats[STAT_MAX_DRAWDOWN]:.2f}%")
                print(f"   Win Rate:    {stats[STAT_WIN_RATE]:.1f}%")
                print(f"   Trades:      {stats[STAT_TOTAL_TRADES]}")

        # Benchmark Comparison (Simple Buy & Hold)
        # Note: This is a placeholder. Real B&H requires raw data access here or in engine.
        print("\n" + "="*60)


def main():
    """Entry point."""
    if len(sys.argv) > 1:
        # Simple argument parsing for quick testing
        # Usage: python cli.py <strategy_name> [--full]
        strategy = sys.argv[1]
        mode = "fast"
        if "--full" in sys.argv: mode = "full"
        elif "--monte-carlo" in sys.argv: mode = "monte_carlo"
        elif "--walkforward" in sys.argv: mode = "walkforward"
        
        CLI.run(strategy, "1y", mode)
    else:
        CLI.interactive_mode()


if __name__ == "__main__":
    main()
