#!/usr/bin/env python3
"""Trading System - Streamlined Implementation"""

import warnings
import sys
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from data_manager import load_data_for_strategy, load_strategy_config
from strategy_registry import (
    create_portfolio,
    get_optimization_grid,
    strategy_needs_multi_timeframe,
)
from plotter import create_visualizations, plot_comprehensive_analysis
from walk_forward import run_walkforward_analysis
from optimizer import run_optimization, run_monte_carlo_analysis
from config_validator import quick_validate

warnings.filterwarnings("ignore")


def get_primary_data(
    data: Dict, strategy_config: Dict
) -> Tuple[str, str, pd.DataFrame]:
    """Get primary symbol/timeframe data with fallbacks."""
    if not data:
        raise ValueError("No data provided")

    params = strategy_config.get("parameters", {})

    # Get primary symbol (case-insensitive)
    primary_symbol_param = params.get("primary_symbol")
    if primary_symbol_param:
        sym_map = {k.lower(): k for k in data.keys()}
        primary_symbol = sym_map.get(str(primary_symbol_param).lower())
        if not primary_symbol or primary_symbol not in data:
            primary_symbol = next(iter(data.keys()))
    else:
        primary_symbol = next(iter(data.keys()))

    # Get timeframe (case-insensitive)
    available_tfs = data[primary_symbol]
    requested_tf = params.get("primary_timeframe")
    if requested_tf:
        tf_map = {k.lower(): k for k in available_tfs.keys()}
        chosen_tf = tf_map.get(str(requested_tf).lower())
        if not chosen_tf:
            chosen_tf = next(iter(available_tfs.keys()))
    else:
        chosen_tf = next(iter(available_tfs.keys()))

    strategy_config["parameters"]["primary_timeframe"] = chosen_tf
    return primary_symbol, chosen_tf, available_tfs[chosen_tf]


def run_backtest_for_strategy(
    symbol: str,
    timeframe: str,
    timeframes: Dict[str, pd.DataFrame],
    strategy_name: str,
    parameters: dict,
) -> Any:
    """Run backtest for a strategy using portfolio-direct approach."""
    try:
        # Check if strategy needs multi-timeframe data
        if strategy_needs_multi_timeframe(strategy_name):
            # Pass all timeframes as dict
            portfolio = create_portfolio(strategy_name, timeframes, parameters)
        else:
            # Use the primary timeframe data only
            primary_data = timeframes[timeframe]
            portfolio = create_portfolio(strategy_name, primary_data, parameters)

        # Check if portfolio creation was successful
        if portfolio is None:
            print(f"❌ {symbol} {timeframe}: Portfolio creation returned None")
            return None

        # Get portfolio statistics
        try:
            from constants import STAT_TOTAL_RETURN

            stats = portfolio.stats()
            if stats is not None and STAT_TOTAL_RETURN in stats.index:
                total_return = stats[STAT_TOTAL_RETURN]
                print(f"✅ {symbol} {timeframe}: Return {total_return:.2f}%")
            else:
                print(f"✅ {symbol} {timeframe}: Portfolio created")
        except Exception as stats_error:
            print(
                f"✅ {symbol} {timeframe}: Portfolio created but stats failed: {stats_error}"
            )

        return portfolio

    except Exception as e:
        print(f"❌ {symbol} {timeframe} failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_full_backtest(data: Dict, strategy_name: str, parameters: dict) -> Dict:
    """Run backtest for all symbols/timeframes."""
    print(f"🔄 Running backtest for strategy: {strategy_name}")
    results = {}
    required_tfs = ["1h"]  # Simplified - most strategies use 1h

    for symbol, timeframes in data.items():
        results[symbol] = {}

        if len(required_tfs) > 1:
            # Multi-timeframe: use primary timeframe
            req = parameters.get("primary_timeframe") or required_tfs[0]
            tf_map = {k.lower(): k for k in timeframes.keys()}
            chosen_tf = str(tf_map.get(str(req).lower(), next(iter(timeframes.keys()))))

            portfolio = run_backtest_for_strategy(
                symbol, chosen_tf, timeframes, strategy_name, parameters
            )
            if portfolio:
                results[symbol][chosen_tf] = portfolio
        else:
            # Single timeframe: run on all available
            for tf_key in timeframes.keys():
                portfolio = run_backtest_for_strategy(
                    symbol, tf_key, timeframes, strategy_name, parameters
                )
                if portfolio:
                    results[symbol][tf_key] = portfolio

    # Calculate and display aggregated portfolio statistics
    if results:
        print_aggregated_portfolio_stats(results)

    return results


def print_aggregated_portfolio_stats(results: Dict):
    """Print full pf.stats() for each portfolio and aggregated statistics."""
    from constants import (
        STAT_TOTAL_RETURN,
        STAT_SHARPE_RATIO,
        STAT_MAX_DRAWDOWN,
        STAT_WIN_RATE,
        STAT_TOTAL_TRADES,
    )

    print("\n" + "=" * 80)
    print("📊 INDIVIDUAL PORTFOLIO STATISTICS (Full pf.stats())")
    print("=" * 80)

    all_portfolios = []

    # Print full stats for each individual portfolio
    for symbol, timeframes in results.items():
        for timeframe, portfolio in timeframes.items():
            if portfolio is not None:
                try:
                    print(f"\n🔸 {symbol} {timeframe} - Full Portfolio Statistics:")
                    print("-" * 50)
                    stats = portfolio.stats()
                    print(stats)
                    all_portfolios.append((symbol, timeframe, portfolio, stats))
                except Exception as e:
                    print(f"⚠️ Could not get stats for {symbol} {timeframe}: {e}")

    if not all_portfolios:
        print("❌ No valid portfolio data found")
        return

    # Print aggregated summary
    print("\n" + "=" * 80)
    print("📊 AGGREGATED SUMMARY")
    print("=" * 80)

    # Helper function for safe stat extraction
    def safe_stat(stats, key: str, default=0.0):
        try:
            if key in stats.index:
                value = stats[key]
                return float(value) if pd.notna(value) else default
            return default
        except Exception:
            return default

    # Calculate aggregated metrics from full stats
    total_portfolios = len(all_portfolios)
    returns = []
    sharpe_ratios = []
    max_drawdowns = []
    win_rates = []
    total_trades = []

    for symbol, timeframe, portfolio, stats in all_portfolios:
        returns.append(safe_stat(stats, STAT_TOTAL_RETURN))
        sharpe_ratios.append(safe_stat(stats, STAT_SHARPE_RATIO))
        max_drawdowns.append(safe_stat(stats, STAT_MAX_DRAWDOWN))
        win_rates.append(safe_stat(stats, STAT_WIN_RATE))
        total_trades.append(int(safe_stat(stats, STAT_TOTAL_TRADES)))

    if returns:
        avg_return = sum(returns) / len(returns)
        avg_sharpe = sum(s for s in sharpe_ratios if s != 0) / max(
            1, len([s for s in sharpe_ratios if s != 0])
        )
        avg_max_drawdown = sum(max_drawdowns) / len(max_drawdowns)
        avg_win_rate = sum(win_rates) / len(win_rates)
        total_trades_all = sum(total_trades)

        best_idx = returns.index(max(returns))
        worst_idx = returns.index(min(returns))
        best_symbol = all_portfolios[best_idx][0]
        worst_symbol = all_portfolios[worst_idx][0]

        print(f"📈 Portfolio Count: {total_portfolios}")
        print(f"📊 Average Return: {avg_return:.2f}%")
        print(f"📉 Average Max Drawdown: {avg_max_drawdown:.2f}%")
        print(f"⚡ Average Sharpe Ratio: {avg_sharpe:.3f}")
        print(f"🎯 Average Win Rate: {avg_win_rate:.1f}%")
        print(f"🔄 Total Trades (All Symbols): {total_trades_all}")
        print(f"🏆 Best Performer: {best_symbol} ({max(returns):.2f}%)")
        print(f"📉 Worst Performer: {worst_symbol} ({min(returns):.2f}%)")

    print("=" * 80)


def run_strategy_analysis(
    strategy_name: str,
    fast_mode: bool = False,
    time_range: Optional[str] = None,
    end_date: Optional[str] = None,
    analysis_type: str = "full",
) -> Dict:
    """Run complete strategy analysis."""
    try:
        # Load config and data
        raw_config = load_strategy_config(strategy_name) or {}
        strategy_config = raw_config.copy()  # Use the full config
        strategy_config["name"] = strategy_name  # Ensure name is set

        # Validate configuration
        print(f"\n🔍 Validating configuration for '{strategy_name}'...")
        config_valid = quick_validate(strategy_name, strategy_config, auto_fix=False)
        if not config_valid:
            print("⚠️  Configuration has issues but continuing anyway...")
            print("    (Fix the config file to ensure parameters work correctly)\n")

        # Create simple strategy context using dict-like object
        from types import SimpleNamespace

        def get_parameter(key, default=None):
            if key in strategy_config:
                return strategy_config[key]
            return strategy_config.get("parameters", {}).get(key, default)

        strategy_context = SimpleNamespace(
            name=strategy_name,
            config=strategy_config,
            get_required_timeframes=lambda: ["1h"],
            get_required_columns=lambda: ["open", "high", "low", "close", "volume"],
            get_parameter=get_parameter,
        )

        data = load_data_for_strategy(strategy_context, time_range, end_date)
        _, _, primary_data = get_primary_data(data, strategy_config)

        if fast_mode:
            # Fast mode: just backtest and plot
            portfolio_results = run_full_backtest(
                data, strategy_name, strategy_config.get("parameters", {})
            )
            flattened = {
                f"{s}_{t}" if len(tfs) > 1 else s: p
                for s, tfs in portfolio_results.items()
                for t, p in tfs.items()
            }
            plot_comprehensive_analysis(flattened, strategy_config["name"])
            return {"success": True, "results": portfolio_results}

        # Get default and optimized parameters
        default_params = strategy_config.get("parameters", {})

        # Optimization (if needed for analysis type)
        optimized_params = default_params.copy()
        optimization_results = {}
        if analysis_type in ["full", "monte_carlo", "walkforward"]:
            print("\n🔧 Parameter Optimization")
            optimization_results = run_optimization(
                strategy_name, primary_data, get_optimization_grid(strategy_name)
            )
            if optimization_results.get("best_params"):
                optimized_params.update(optimization_results["best_params"])

        results = {}

        # Monte Carlo only mode
        if analysis_type == "monte_carlo":
            print("\n🎲 Monte Carlo Analysis")
            monte_carlo_results = run_monte_carlo_analysis(
                primary_data, strategy_name, optimized_params
            )
            results["monte_carlo"] = monte_carlo_results

            # Run backtest for comparison
            portfolio_results = run_full_backtest(data, strategy_name, optimized_params)
            results["optimized_portfolios"] = portfolio_results

            # Visualization
            print("\n📊 Generating Monte Carlo Plots")
            flattened = {
                f"{s}_{t}" if len(tfs) > 1 else s: p
                for s, tfs in portfolio_results.items()
                for t, p in tfs.items()
            }
            plot_comprehensive_analysis(
                flattened, strategy_config["name"], mc_results=monte_carlo_results
            )

            return {"success": True, "results": results}

        # Walk-Forward only mode
        if analysis_type == "walkforward":
            print("\n📈 Walk-Forward Analysis")

            # Create simple strategy context for walk-forward using SimpleNamespace
            from types import SimpleNamespace

            optimized_strategy = SimpleNamespace(
                name=strategy_name,
                parameters=optimized_params,
                get_required_timeframes=lambda: ["1h"],
            )
            walkforward_results = run_walkforward_analysis(
                optimized_strategy, primary_data
            )
            results["walkforward"] = walkforward_results

            # Run backtest for comparison
            portfolio_results = run_full_backtest(data, strategy_name, optimized_params)
            results["optimized_portfolios"] = portfolio_results

            # Visualization
            print("\n📊 Generating Walk-Forward Plots")
            flattened = {
                f"{s}_{t}" if len(tfs) > 1 else s: p
                for s, tfs in portfolio_results.items()
                for t, p in tfs.items()
            }
            plot_comprehensive_analysis(
                flattened, strategy_config["name"], wf_results=walkforward_results
            )

            return {"success": True, "results": results}

        # Full analysis mode
        # Default performance
        print("\n🔧 Default Strategy Performance")
        default_results = run_full_backtest(data, strategy_name, default_params)
        results["default_portfolios"] = default_results
        results["optimization"] = optimization_results

        print("\n🚀 Optimized Strategy Performance")
        optimized_results = run_full_backtest(data, strategy_name, optimized_params)
        results["optimized_portfolios"] = optimized_results

        # Robustness tests
        print("\n📈 Walk-Forward Analysis")

        # Create simple strategy context for walk-forward using SimpleNamespace
        from types import SimpleNamespace

        optimized_strategy = SimpleNamespace(
            name=strategy_name,
            parameters=optimized_params,
            get_required_timeframes=lambda: ["1h"],
        )
        walkforward_results = run_walkforward_analysis(optimized_strategy, primary_data)
        results["walkforward"] = walkforward_results

        print("\n🎲 Monte Carlo Analysis")
        monte_carlo_results = run_monte_carlo_analysis(
            primary_data, strategy_name, optimized_params
        )
        results["monte_carlo"] = monte_carlo_results

        # Visualization
        print("\n📊 Generating Analysis Plots")
        visualization_data = {
            "default_backtest": default_results,
            "full_backtest": optimized_results,
            "monte_carlo": monte_carlo_results,
            "walkforward": walkforward_results,
            "optimization": optimization_results,
        }
        results["visualizations"] = create_visualizations(
            visualization_data, strategy_config["name"]
        )

        return {"success": True, "results": results}

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_available_strategies():
    """Get list of available strategies."""
    from strategy_registry import get_available_strategies as get_strategies

    return get_strategies()


def get_user_inputs() -> Tuple[Optional[str], Optional[str], Optional[str], bool, str]:
    """Get all user inputs with simplified interface."""
    # Strategy selection
    strategies = get_available_strategies()
    if not strategies:
        print("❌ No strategies found")
        return None, None, None, False, "full"

    print("\n📊 Available Strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy}")

    while True:
        try:
            choice = input(f"\nSelect strategy (1-{len(strategies)}): ").strip()
            if not choice:
                print("❌ Please select a strategy")
                continue
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(strategies):
                strategy_name = strategies[choice_idx]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(strategies)}")
        except ValueError:
            print("❌ Please enter a valid number")

    # Simplified time range selection
    print("\n📅 Time Range:")
    print("1. 3 months")
    print("2. 6 months")
    print("3. 1 year")
    print("4. 2 years (default)")
    print("5. Full dataset")

    time_choice = input("Select (Enter for 2y): ").strip() or "4"
    time_map = {"1": "3m", "2": "6m", "3": "1y", "4": "2y", "5": None}
    time_range = time_map.get(time_choice, "2y")

    # Simplified mode selection
    print("\n⚙️ Analysis Mode:")
    print("1. Quick analysis (backtest only)")
    print("2. Full analysis (optimization + walk-forward + monte carlo)")
    print("3. Monte Carlo only")
    print("4. Walk-Forward only")

    mode_choice = input("Select (Enter for quick): ").strip() or "1"

    if mode_choice == "1":
        return strategy_name, time_range, None, True, "full"
    elif mode_choice == "3":
        return strategy_name, time_range, None, False, "monte_carlo"
    elif mode_choice == "4":
        return strategy_name, time_range, None, False, "walkforward"
    else:
        return strategy_name, time_range, None, False, "full"


def main():
    """Main entry point."""
    print("🚀 Trading Strategy Analysis Pipeline")
    print("=" * 50)

    strategy_name, time_range, end_date, fast_mode, analysis_type = get_user_inputs()
    if not strategy_name:
        return

    results = run_strategy_analysis(
        strategy_name, fast_mode, time_range, end_date, analysis_type
    )

    if results["success"]:
        print("✅ Analysis completed successfully!")
    else:
        print(f"❌ Analysis failed: {results['error']}")

    return results


def quick_test(
    strategy_name: str,
    time_range: str = "1y",
    fast_mode: bool = True,
    analysis_type: str = "full",
):
    """Quick test function."""
    mode_desc = "fast" if fast_mode else analysis_type
    print(f"🧪 Quick Test: {strategy_name} ({time_range}, {mode_desc})")
    results = run_strategy_analysis(
        strategy_name, fast_mode, time_range, analysis_type=analysis_type
    )
    print(
        "✅ Test completed!"
        if results["success"]
        else f"❌ Test failed: {results['error']}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        if len(sys.argv) < 3:
            print(
                "Usage: python main.py --quick <strategy_name> [--full|--monte-carlo|--walkforward]"
            )
            sys.exit(1)

        strategy = sys.argv[2]

        # Determine analysis type from flags
        if "--monte-carlo" in sys.argv:
            quick_test(strategy, "1y", False, "monte_carlo")
        elif "--walkforward" in sys.argv:
            quick_test(strategy, "1y", False, "walkforward")
        elif "--full" in sys.argv:
            quick_test(strategy, "1y", False, "full")
        else:
            quick_test(strategy, "1y", True, "full")
    else:
        main()
