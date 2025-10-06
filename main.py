#!/usr/bin/env python3
"""Trading System - Streamlined Implementation"""

import warnings
from typing import Dict, Any, Optional
import pandas as pd

# Core imports
from data_manager import load_data_for_strategy, load_strategy_config

# Removed base.py dependency - using simple dict for config
from strategy_registry import (
    create_portfolio,
    get_default_parameters,
    get_optimization_grid,
)
from plotter import create_visualizations, plot_comprehensive_analysis
from walk_forward import run_walkforward_analysis
from optimizer import run_optimization, run_monte_carlo_analysis

warnings.filterwarnings("ignore")


def get_primary_data(data: Dict, strategy_config: Dict) -> tuple:
    """Get primary symbol/timeframe data with fallbacks."""
    if not data:
        raise ValueError("No data provided")

    params = strategy_config.get("parameters", {})

    # Get primary symbol (case-insensitive)
    primary_symbol = params.get("primary_symbol")
    if primary_symbol:
        sym_map = {k.lower(): k for k in data.keys()}
        primary_symbol = sym_map.get(str(primary_symbol).lower(), primary_symbol)
    if primary_symbol not in data:
        primary_symbol = next(iter(data.keys()))

    # Get timeframe (case-insensitive)
    available_tfs = data[primary_symbol]
    requested_tf = params.get("primary_timeframe")
    if requested_tf:
        tf_map = {k.lower(): k for k in available_tfs.keys()}
        chosen_tf = tf_map.get(
            str(requested_tf).lower(), next(iter(available_tfs.keys()))
        )
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
        # Use the primary timeframe data
        primary_data = timeframes[timeframe]

        # Create portfolio directly
        portfolio = create_portfolio(strategy_name, primary_data, parameters)

        # Check if portfolio creation was successful
        if portfolio is None:
            print(f"❌ {symbol} {timeframe}: Portfolio creation returned None")
            return None

        # Get portfolio statistics
        try:
            stats = portfolio.stats()
            total_return = stats.get("Total Return [%]", "N/A")
            print(f"✅ {symbol} {timeframe}: Return {total_return}%")
        except Exception as stats_error:
            print(
                f"✅ {symbol} {timeframe}: Portfolio created but stats failed: {stats_error}"
            )

        return portfolio

    except Exception as e:
        print(f"❌ {symbol} {timeframe} failed: {e}")
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
            req = parameters.get("primary_timeframe", required_tfs[0])
            tf_map = {k.lower(): k for k in timeframes.keys()}
            chosen_tf = tf_map.get(str(req).lower(), next(iter(timeframes.keys())))

            portfolio = run_backtest_for_strategy(
                symbol, chosen_tf, timeframes, strategy_name, parameters
            )
            if portfolio:
                results[symbol][chosen_tf] = portfolio
        else:
            # Single timeframe: run on all available
            for timeframe in timeframes.keys():
                portfolio = run_backtest_for_strategy(
                    symbol, timeframe, timeframes, strategy_name, parameters
                )
                if portfolio:
                    results[symbol][timeframe] = portfolio

    # Calculate and display aggregated portfolio statistics
    if results:
        print_aggregated_portfolio_stats(results)
    
    return results


def print_aggregated_portfolio_stats(results: Dict):
    """Print full pf.stats() for each portfolio and aggregated statistics."""
    print("\n" + "="*80)
    print("📊 INDIVIDUAL PORTFOLIO STATISTICS (Full pf.stats())")
    print("="*80)
    
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
    print("\n" + "="*80)
    print("📊 AGGREGATED SUMMARY")
    print("="*80)
    
    # Calculate aggregated metrics from full stats
    total_portfolios = len(all_portfolios)
    returns = []
    sharpe_ratios = []
    max_drawdowns = []
    win_rates = []
    total_trades = []
    
    for symbol, timeframe, portfolio, stats in all_portfolios:
        try:
            returns.append(stats.get('Total Return [%]', 0))
            sharpe_ratios.append(stats.get('Sharpe Ratio', 0))
            max_drawdowns.append(stats.get('Max Drawdown [%]', 0))
            win_rates.append(stats.get('Win Rate [%]', 0))
            total_trades.append(stats.get('# Trades', 0))
        except:
            pass
    
    if returns:
        avg_return = sum(returns) / len(returns)
        avg_sharpe = sum(s for s in sharpe_ratios if s != 0) / max(1, len([s for s in sharpe_ratios if s != 0]))
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
    
    print("="*80)


def run_strategy_analysis(
    strategy_name: str,
    fast_mode: bool = False,
    time_range: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """Run complete strategy analysis."""
    try:
        # Load config and data
        raw_config = load_strategy_config(strategy_name) or {}
        strategy_config = raw_config.copy()  # Use the full config
        strategy_config["name"] = strategy_name  # Ensure name is set

        # Load data using simplified context
        class DataContext:
            def __init__(self, config):
                self.name = strategy_name
                self.config = config

            def get_required_timeframes(self):
                return ["1h"]

            def get_required_columns(self):
                # Return basic OHLCV columns that most strategies need
                return ["open", "high", "low", "close", "volume"]

            def get_parameter(self, key, default=None):
                # First check root level, then parameters section
                if key in self.config:
                    return self.config[key]
                return self.config.get("parameters", {}).get(key, default)

        data = load_data_for_strategy(DataContext(strategy_config), time_range, end_date)
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

        # Full analysis
        results = {}

        # Default performance
        print("\n🔧 Default Strategy Performance")
        default_results = run_full_backtest(
            data, strategy_name, strategy_config.get("parameters", {})
        )
        results["default_portfolios"] = default_results

        # Optimization
        print("\n🔧 Parameter Optimization")
        optimization_results = run_optimization(
            strategy_name, primary_data, get_optimization_grid(strategy_name)
        )
        results["optimization"] = optimization_results

        # Optimized performance
        optimized_params = strategy_config.get("parameters", {}).copy()
        if optimization_results.get("best_params"):
            optimized_params.update(optimization_results["best_params"])

        print("\n🚀 Optimized Strategy Performance")
        optimized_results = run_full_backtest(data, strategy_name, optimized_params)
        results["optimized_portfolios"] = optimized_results

        # Robustness tests
        print("\n📈 Walk-Forward Analysis")

        # Create simple strategy context for walk-forward
        class SimpleStrategy:
            def __init__(self, name, params):
                self.name = name
                self.parameters = params

            def get_required_timeframes(self):
                return ["1h"]

        optimized_strategy = SimpleStrategy(strategy_name, optimized_params)
        walkforward_results = run_walkforward_analysis(optimized_strategy, primary_data)
        results["walkforward"] = walkforward_results

        print("\n🎲 Monte Carlo Analysis")
        best_params = optimization_results.get(
            "best_params", get_default_parameters(strategy_name)
        )
        monte_carlo_results = run_monte_carlo_analysis(
            primary_data, strategy_name, best_params
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


def get_user_inputs() -> tuple[str, Optional[str], Optional[str], bool]:
    """Get all user inputs with simplified interface."""
    # Strategy selection
    strategies = get_available_strategies()
    if not strategies:
        print("❌ No strategies found")
        return None, None, None, False

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
    print("1. 3 months (default)")
    print("2. 6 months")
    print("3. 1 year")
    print("4. 2 years")
    print("5. Full dataset")

    time_choice = input("Select (Enter for 3m): ").strip() or "1"
    time_map = {"1": "3m", "2": "6m", "3": "1y", "4": "2y", "5": None}
    time_range = time_map.get(time_choice, "3m")

    # Simplified mode selection
    print("\n⚙️ Analysis Mode:")
    print("1. Quick analysis (default)")
    print("2. Full analysis (optimization + walk-forward + monte carlo)")

    mode_choice = input("Select (Enter for quick): ").strip() or "1"
    fast_mode = mode_choice == "1"

    return strategy_name, time_range, None, fast_mode


def main():
    """Main entry point."""
    print("🚀 Trading Strategy Analysis Pipeline")
    print("=" * 50)

    strategy_name, time_range, end_date, fast_mode = get_user_inputs()
    if not strategy_name:
        return

    # Ensure fast_mode is a boolean
    fast_mode = fast_mode or False

    results = run_strategy_analysis(strategy_name, fast_mode, time_range, end_date)

    if results["success"]:
        print("✅ Analysis completed successfully!")
    else:
        print(f"❌ Analysis failed: {results['error']}")

    return results


def quick_test(strategy_name: str, time_range: str = "1y", fast_mode: bool = True):
    """Quick test function."""
    print(
        f"🧪 Quick Test: {strategy_name} ({time_range}, {'fast' if fast_mode else 'full'})"
    )
    results = run_strategy_analysis(strategy_name, fast_mode, time_range)
    print(
        "✅ Test completed!"
        if results["success"]
        else f"❌ Test failed: {results['error']}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        if len(sys.argv) < 3:
            print("Usage: python main.py --quick <strategy_name> [--full]")
            sys.exit(1)
        quick_test(sys.argv[2], "1y", "--full" not in sys.argv)
    else:
        main()
