#!/usr/bin/env python3
"""
Strategy Testing Suite - Run all strategies on BTC and EURUSD with 4 years of data.
Tests each strategy and reports performance metrics in a clean table format.
"""

import warnings
from datetime import datetime
from typing import Dict

import pandas as pd

from constants import (
    STAT_MAX_DRAWDOWN,
    STAT_SHARPE_RATIO,
    STAT_TOTAL_RETURN,
    STAT_TOTAL_TRADES,
)
from data_manager import load_ohlc_csv, load_strategy_config
from strategy_registry import create_portfolio, get_available_strategies

warnings.filterwarnings("ignore")

# =============================================================================s
# CONFIGURATION CONSTANTS
# =============================================================================

# Data file paths (update these to match your data directory)
BTC_DATA_PATH = "data/BTCUSD_1h_2011-2025.csv"
EURUSD_DATA_PATH = "data/EURUSD_1H_2009-2025.csv"

# Test configuration
TEST_YEARS = 4
DEFAULT_INITIAL_CASH = 10000
DEFAULT_FEE = 0.001
DEFAULT_FREQ = "1H"

# Strategies to exclude from testing
EXCLUDED_STRATEGIES = [
    "strategy_template",
    "sol_btc_stat_arb",  # Requires multi-symbol data
]

# Table formatting
TABLE_WIDTH = 120
COLUMN_WIDTHS = {
    "strategy": 25,
    "symbol": 10,
    "status": 10,
    "return": 12,
    "sharpe": 10,
    "drawdown": 12,
    "trades": 10,
}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_test_data() -> Dict[str, pd.DataFrame]:
    """
    Load BTC and EURUSD data for testing.
    
    Returns:
        Dictionary mapping symbol name to DataFrame with 4 years of data
        
    Raises:
        FileNotFoundError: If data files cannot be loaded
    """
    data = {}
    
    # Load BTC data
    try:
        btc_df = load_ohlc_csv(BTC_DATA_PATH)
        # Get last N years of data
        data["BTC"] = btc_df.iloc[-(TEST_YEARS * 365 * 24):].copy()  # Approximate hourly bars
        print(f"✅ Loaded BTC: {len(data['BTC'])} bars from {BTC_DATA_PATH}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load BTC data from {BTC_DATA_PATH}: {e}")
    
    # Load EURUSD data
    try:
        eurusd_df = load_ohlc_csv(EURUSD_DATA_PATH)
        # Get last N years of data
        data["EURUSD"] = eurusd_df.iloc[-(TEST_YEARS * 365 * 24):].copy()  # Approximate hourly bars
        print(f"✅ Loaded EURUSD: {len(data['EURUSD'])} bars from {EURUSD_DATA_PATH}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load EURUSD data from {EURUSD_DATA_PATH}: {e}")
    
    return data


# =============================================================================
# STRATEGY TESTING
# =============================================================================


def extract_stat_value(stats, key: str, default=0.0) -> float:
    """
    Safely extract a statistic value from portfolio stats.
    
    Args:
        stats: Portfolio stats Series
        key: Stat key to extract
        default: Default value if extraction fails
        
    Returns:
        Float value of the statistic
    """
    if stats is None or key not in stats.index:
        return default
    
    value = stats[key]
    if pd.isna(value):
        return default
    
    return float(value)


def load_strategy_parameters(strategy_name: str) -> Dict:
    """
    Load strategy parameters from config file.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Dictionary of parameters with defaults
    """
    try:
        config = load_strategy_config(strategy_name)
        params = config.get("parameters", {})
    except Exception:
        params = {}
    
    # Ensure required parameters are set
    params.setdefault("initial_cash", DEFAULT_INITIAL_CASH)
    params.setdefault("fee", DEFAULT_FEE)
    params.setdefault("freq", DEFAULT_FREQ)
    
    return params


def test_strategy(strategy_name: str, data: pd.DataFrame) -> Dict:
    """
    Test a single strategy on given data.
    
    Args:
        strategy_name: Name of the strategy to test
        data: OHLCV DataFrame
        
    Returns:
        Dictionary with test results and metrics
    """
    params = load_strategy_parameters(strategy_name)
    
    portfolio = create_portfolio(strategy_name, data, params)
    
    if portfolio is None:
        raise ValueError("Portfolio creation returned None")
    
    stats = portfolio.stats()
    
    return {
        "status": "success",
        "total_return": extract_stat_value(stats, STAT_TOTAL_RETURN),
        "sharpe_ratio": extract_stat_value(stats, STAT_SHARPE_RATIO),
        "max_drawdown": extract_stat_value(stats, STAT_MAX_DRAWDOWN),
        "total_trades": int(extract_stat_value(stats, STAT_TOTAL_TRADES)),
    }


# =============================================================================
# RESULTS REPORTING
# =============================================================================


def print_results_table(results: Dict[str, Dict[str, Dict]]):
    """
    Print test results in a formatted table.
    
    Args:
        results: Nested dict of {strategy: {symbol: result}}
    """
    print("\n" + "=" * TABLE_WIDTH)
    print(f"STRATEGY PERFORMANCE SUMMARY ({TEST_YEARS} Years)")
    print("=" * TABLE_WIDTH)
    
    # Header
    header = (
        f"{'Strategy':<{COLUMN_WIDTHS['strategy']}} "
        f"{'Symbol':<{COLUMN_WIDTHS['symbol']}} "
        f"{'Status':<{COLUMN_WIDTHS['status']}} "
        f"{'Return %':<{COLUMN_WIDTHS['return']}} "
        f"{'Sharpe':<{COLUMN_WIDTHS['sharpe']}} "
        f"{'Max DD %':<{COLUMN_WIDTHS['drawdown']}} "
        f"{'Trades':<{COLUMN_WIDTHS['trades']}}"
    )
    print(header)
    print("-" * TABLE_WIDTH)
    
    # Track strategies with issues
    zero_trade_strategies = []
    
    # Results rows
    for strategy_name in sorted(results.keys()):
        for symbol in sorted(results[strategy_name].keys()):
            result = results[strategy_name][symbol]
            
            if result["status"] == "success":
                # Check for zero trades
                if result["total_trades"] == 0:
                    zero_trade_strategies.append(f"{strategy_name} on {symbol}")
                
                row = (
                    f"{strategy_name:<{COLUMN_WIDTHS['strategy']}} "
                    f"{symbol:<{COLUMN_WIDTHS['symbol']}} "
                    f"{'✅ OK':<{COLUMN_WIDTHS['status']}} "
                    f"{result['total_return']:>{COLUMN_WIDTHS['return']-2}.2f}% "
                    f"{result['sharpe_ratio']:>{COLUMN_WIDTHS['sharpe']-1}.3f} "
                    f"{result['max_drawdown']:>{COLUMN_WIDTHS['drawdown']-2}.2f}% "
                    f"{result['total_trades']:>{COLUMN_WIDTHS['trades']-1}}"
                )
            else:
                error_msg = result.get("error", "Unknown error")[:50]
                row = (
                    f"{strategy_name:<{COLUMN_WIDTHS['strategy']}} "
                    f"{symbol:<{COLUMN_WIDTHS['symbol']}} "
                    f"{'❌ FAIL':<{COLUMN_WIDTHS['status']}} "
                    f"{error_msg}"
                )
            
            print(row)
    
    print("=" * TABLE_WIDTH)
    
    # Print warnings for zero-trade strategies
    if zero_trade_strategies:
        print("\n⚠️  WARNING: The following strategies generated 0 trades:")
        for strategy_info in zero_trade_strategies:
            print(f"   • {strategy_info}")
        print("   This may indicate:")
        print("     - Strategy conditions are too strict")
        print("     - Missing 'freq' parameter in portfolio creation")
        print("     - Data frequency mismatch with strategy logic")


def calculate_summary_statistics(results: Dict[str, Dict[str, Dict]]) -> Dict:
    """
    Calculate summary statistics from test results.
    
    Args:
        results: Nested dict of test results
        
    Returns:
        Dictionary with summary statistics including ranked results
    """
    total_tests = 0
    successful_tests = 0
    
    # Collect all valid results for ranking
    valid_results = []
    
    for strategy_name, strategy_results in results.items():
        for symbol, result in strategy_results.items():
            total_tests += 1
            
            if result["status"] == "success":
                successful_tests += 1
                
                # Only include results with finite values and actual trades
                total_return = result["total_return"]
                sharpe_ratio = result["sharpe_ratio"]
                max_drawdown = result["max_drawdown"]
                total_trades = result["total_trades"]
                
                # Skip results with no trades or infinite values
                if total_trades > 0 and all(
                    pd.notna(v) and abs(v) != float("inf")
                    for v in [total_return, sharpe_ratio, max_drawdown]
                ):
                    valid_results.append({
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "total_return": total_return,
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "total_trades": total_trades,
                    })
    
    # Rank by return (worst to best)
    ranked_by_return = sorted(valid_results, key=lambda x: x["total_return"])
    
    # Rank by Sharpe (worst to best)
    ranked_by_sharpe = sorted(valid_results, key=lambda x: x["sharpe_ratio"])
    
    # Rank by drawdown (worst to best, most negative first)
    ranked_by_drawdown = sorted(valid_results, key=lambda x: x["max_drawdown"])
    
    return {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": total_tests - successful_tests,
        "valid_results_count": len(valid_results),
        "ranked_by_return": ranked_by_return,
        "ranked_by_sharpe": ranked_by_sharpe,
        "ranked_by_drawdown": ranked_by_drawdown,
    }


def print_summary_statistics(summary: Dict):
    """
    Print summary statistics with rankings.
    
    Args:
        summary: Dictionary with summary statistics
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    total = summary["total_tests"]
    successful = summary["successful_tests"]
    failed = summary["failed_tests"]
    valid = summary["valid_results_count"]
    
    print(f"Total Tests: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Valid Results (with trades): {valid}")
    
    if valid == 0:
        print("\n⚠️  No valid results to rank (all strategies had 0 trades or infinite values)")
        print("=" * 80)
        return
    
    # Top 3 by Return
    print("\n📊 TOP 3 BY RETURN (Worst to Best)")
    print("-" * 80)
    ranked_return = summary["ranked_by_return"]
    
    print("Worst Performers:")
    for i, result in enumerate(ranked_return[:3], 1):
        print(f"  {i}. {result['strategy']:20} on {result['symbol']:8} → {result['total_return']:>8.2f}%")
    
    print("\nBest Performers:")
    for i, result in enumerate(reversed(ranked_return[-3:]), 1):
        print(f"  {i}. {result['strategy']:20} on {result['symbol']:8} → {result['total_return']:>8.2f}%")
    
    # Top 3 by Sharpe
    print("\n📈 TOP 3 BY SHARPE RATIO (Worst to Best)")
    print("-" * 80)
    ranked_sharpe = summary["ranked_by_sharpe"]
    
    print("Worst Performers:")
    for i, result in enumerate(ranked_sharpe[:3], 1):
        print(f"  {i}. {result['strategy']:20} on {result['symbol']:8} → {result['sharpe_ratio']:>7.3f}")
    
    print("\nBest Performers:")
    for i, result in enumerate(reversed(ranked_sharpe[-3:]), 1):
        print(f"  {i}. {result['strategy']:20} on {result['symbol']:8} → {result['sharpe_ratio']:>7.3f}")
    
    # Top 3 by Drawdown
    print("\n🛡️  TOP 3 BY DRAWDOWN (Worst to Best)")
    print("-" * 80)
    ranked_dd = summary["ranked_by_drawdown"]
    
    print("Worst Drawdowns:")
    for i, result in enumerate(ranked_dd[:3], 1):
        print(f"  {i}. {result['strategy']:20} on {result['symbol']:8} → {result['max_drawdown']:>8.2f}%")
    
    print("\nBest Drawdowns:")
    for i, result in enumerate(reversed(ranked_dd[-3:]), 1):
        print(f"  {i}. {result['strategy']:20} on {result['symbol']:8} → {result['max_drawdown']:>8.2f}%")
    
    print("=" * 80)


# =============================================================================
# RESULTS EXPORT
# =============================================================================


def save_results_to_csv(results: Dict[str, Dict[str, Dict]]) -> str:
    """
    Save test results to CSV file.
    
    Args:
        results: Nested dict of test results
        
    Returns:
        Filename of saved CSV
    """
    rows = []
    for strategy_name, strategy_results in results.items():
        for symbol, result in strategy_results.items():
            row = {
                "strategy": strategy_name,
                "symbol": symbol,
                "status": result["status"],
                "total_return": result.get("total_return", 0),
                "sharpe_ratio": result.get("sharpe_ratio", 0),
                "max_drawdown": result.get("max_drawdown", 0),
                "total_trades": result.get("total_trades", 0),
                "error": result.get("error", ""),
            }
            rows.append(row)
    
    results_df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"strategy_test_results_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    
    return filename


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def run_all_tests() -> Dict[str, Dict[str, Dict]]:
    """
    Run all strategy tests.
    
    Returns:
        Dictionary of test results
    """
    # Load data
    print("\n📊 Loading data...")
    data = load_test_data()
    
    # Get strategies to test
    all_strategies = get_available_strategies()
    strategies = [s for s in all_strategies if s not in EXCLUDED_STRATEGIES]
    
    print(f"\n🔍 Found {len(strategies)} strategies to test")
    print(f"📈 Testing on {len(data)} symbols: {', '.join(data.keys())}")
    print(f"\nStrategies: {', '.join(strategies)}\n")
    
    # Run tests
    results = {}
    total_tests = len(strategies) * len(data)
    current_test = 0
    
    for strategy_name in strategies:
        results[strategy_name] = {}
        
        for symbol, df in data.items():
            current_test += 1
            print(f"[{current_test}/{total_tests}] Testing {strategy_name} on {symbol}...", end=" ")
            
            try:
                result = test_strategy(strategy_name, df)
                results[strategy_name][symbol] = result
                print(f"✅ Return: {result['total_return']:.2f}%, Sharpe: {result['sharpe_ratio']:.3f}")
            except Exception as e:
                results[strategy_name][symbol] = {
                    "status": "error",
                    "error": str(e)[:100],
                }
                print(f"❌ {str(e)[:50]}")
    
    return results


def main():
    """Main entry point for strategy testing suite."""
    print("=" * 80)
    print("STRATEGY TESTING SUITE")
    print(f"Testing all strategies on BTC and EURUSD ({TEST_YEARS} years)")
    print("=" * 80)
    
    # Run tests
    results = run_all_tests()
    
    # Print results
    print_results_table(results)
    
    # Print summary
    summary = calculate_summary_statistics(results)
    print_summary_statistics(summary)
    
    # Save to CSV
    filename = save_results_to_csv(results)
    print(f"\n💾 Results saved to: {filename}")


if __name__ == "__main__":
    main()
