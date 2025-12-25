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
import os
import logging
import warnings
from typing import Dict, Any, List
from types import SimpleNamespace


# Core Imports
from vectorflow.core.config_manager import load_strategy_config
from vectorflow.core.data_loader import load_ohlc_csv
from vectorflow.core.portfolio_builder import (
    create_portfolio,
    get_optimization_grid,
    get_available_strategies,
)

# Analysis Modules
from vectorflow.validation.walk_forward import run_walkforward_analysis
from vectorflow.optimization.grid_search import run_optimization
from vectorflow.optimization.param_monte_carlo import run_monte_carlo_analysis
from vectorflow.validation.path_randomization import run_path_randomization_mc
from vectorflow.utils.config_validator import quick_validate
from vectorflow.visualization.plotters import plot_comprehensive_analysis

# Configuration
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
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
            logger.warning(
                f"Configuration for '{self.strategy_name}' has potential issues."
            )

        return config

    def load_data(self, time_range: str = None, end_date: str = None):
        """Load data required for the strategy from CSV files."""
        logger.info(
            f"Loading data for {self.strategy_name} ({time_range or 'Full History'})..."
        )

        # Get CSV paths from config
        csv_paths = self.config.get("csv_path", [])
        if not csv_paths:
            csv_paths = self.config.get("parameters", {}).get("csv_path", [])

        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]

        if not csv_paths:
            # Auto-discover CSV files in data/ directory
            data_dir = "data"
            if os.path.exists(data_dir):
                csv_paths = [
                    os.path.join(data_dir, f)
                    for f in os.listdir(data_dir)
                    if f.endswith(".csv")
                ]

        if not csv_paths:
            raise ValueError(
                "No CSV files specified in config and none found in data/ directory"
            )

        # Just load the first CSV file (simplest approach)
        # Strategies can handle multi-asset if needed via config
        try:
            self.primary_data = load_ohlc_csv(csv_paths[0])
            logger.info(f"✅ Loaded {len(self.primary_data)} bars from {csv_paths[0]}")
        except Exception as e:
            raise ValueError(f"Failed to load data from {csv_paths[0]}: {e}")

    def run_backtest(self, params: Dict[str, Any]):
        """Run backtest with current parameters."""
        try:
            portfolio = create_portfolio(self.strategy_name, self.primary_data, params)
            return portfolio
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise

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
        logger.info("🎲 Running Parameter Monte Carlo Analysis...")
        return run_monte_carlo_analysis(self.primary_data, self.strategy_name, params)

    def run_path_monte_carlo(self, portfolio) -> Dict[str, Any]:
        """Run Path Randomization Monte Carlo analysis."""
        logger.info("🎲 Running Path Monte Carlo Analysis...")
        return run_path_randomization_mc(portfolio)

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
        - 'full': Optimization -> Backtest -> Param MC -> WF
        - 'param_monte_carlo': Optimization -> Param MC
        - 'path_monte_carlo': Backtest -> Path MC
        - 'walkforward': Optimization -> WF
        """
        results = {"success": False, "config": self.config, "results": {}}

        try:
            # 1. Initial Backtest (Default Params)
            default_params = self.config.get("parameters", {})
            default_portfolio = self.run_backtest(default_params)
            results["results"]["default_portfolio"] = default_portfolio

            if mode == "fast":
                self._generate_plots(results["results"], mode)
                results["success"] = True
                return results

            # 2. Optimization (Required for Full, Param MC, WF)
            optimized_params = default_params.copy()
            if mode in ["full", "param_monte_carlo", "walkforward"]:
                opt_results = self.run_optimization()
                if opt_results.get("best_params"):
                    optimized_params.update(opt_results["best_params"])
                results["results"]["optimization"] = opt_results

            # 3. Optimized Backtest
            optimized_portfolio = self.run_backtest(optimized_params)
            results["results"]["optimized_portfolio"] = optimized_portfolio

            # 4. Advanced Analysis
            if mode in ["full", "param_monte_carlo"]:
                results["results"]["monte_carlo"] = self.run_monte_carlo(
                    optimized_params
                )

            if mode == "path_monte_carlo":
                # Use the optimized portfolio for path randomization
                results["results"]["monte_carlo"] = self.run_path_monte_carlo(
                    optimized_portfolio
                )

            if mode in ["full", "walkforward"]:
                results["results"]["walkforward"] = self.run_walk_forward(
                    optimized_params
                )

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

        # Get portfolio to plot (prefer optimized over default)
        portfolio = results_dict.get("optimized_portfolio") or results_dict.get(
            "default_portfolio"
        )

        if portfolio:
            plot_comprehensive_analysis(
                {self.strategy_name: portfolio},
                self.strategy_name,
                mc_results=results_dict.get("monte_carlo"),
                wf_results=results_dict.get("walkforward"),
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
                if not choice:
                    return 0  # Default
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
        if time_range == "Full History":
            time_range = None
        if not time_range and t_idx == 0:
            time_range = "2y"  # Default handling

        # 3. Select Mode
        print("\n⚙️  Analysis Mode:")
        modes = [
            ("Quick Analysis (Backtest Only)", "fast"),
            ("Full Analysis (Opt + Param MC + WF)", "full"),
            ("Parameter Monte Carlo", "param_monte_carlo"),
            ("Path Randomization Monte Carlo", "path_monte_carlo"),
            ("Walk-Forward", "walkforward"),
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
                # For fast mode, just print stats and beta
                if mode == "fast":
                    pf = results["results"].get("optimized_portfolio") or results[
                        "results"
                    ].get("default_portfolio")
                    if pf:
                        print("\n" + "=" * 60)
                        print("📊 PERFORMANCE SUMMARY")
                        print("=" * 60)
                        print(pf.stats())
                        print(f"\nBeta: {pf.beta():.3f}")
                        print("=" * 60)
                else:
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
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)

        # Get portfolio (prefer optimized over default)
        pf = results.get("optimized_portfolio") or results.get("default_portfolio")

        if not pf:
            print("No portfolio results to display.")
            return

        print(f"\n   Return:      {pf.total_return() * 100:.2f}%")
        print(f"   Sharpe:      {pf.sharpe_ratio():.3f}")
        print(f"   Max DD:      {pf.max_drawdown() * 100:.2f}%")

        # Win rate and trades
        trades_count = len(pf.trades.records)
        if trades_count > 0:
            winning_trades = len(pf.trades.records[pf.trades.records["pnl"] > 0])
            win_rate = (winning_trades / trades_count) * 100
            print(f"   Win Rate:    {win_rate:.1f}%")
            print(f"   Trades:      {trades_count}")
        else:
            raise ValueError("No trades found in portfolio.")

        print("\n" + "=" * 60)


def main():
    """Entry point."""
    if len(sys.argv) > 1:
        # Simple argument parsing for quick testing
        # Usage: python cli.py <strategy_name> [--full]
        strategy = sys.argv[1]
        mode = "fast"
        if "--full" in sys.argv:
            mode = "full"
        elif "--param-monte-carlo" in sys.argv:
            mode = "param_monte_carlo"
        elif "--path-monte-carlo" in sys.argv:
            mode = "path_monte_carlo"
        elif "--walkforward" in sys.argv:
            mode = "walkforward"

        CLI.run(strategy, "1y", mode)
    else:
        CLI.interactive_mode()


if __name__ == "__main__":
    main()
