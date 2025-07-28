#!/usr/bin/env python3
"""
Example usage of time range functionality in the trading system
"""

from trading_system import run_strategy_with_time_range

def main():
    """Demonstrate different time range usage patterns."""
    
    print("🚀 Time Range Functionality Examples")
    print("=" * 50)
    
    # Example 1: Run strategy on full dataset
    print("\n📊 Example 1: Full dataset")
    try:
        results = run_strategy_with_time_range('momentum')
        print("✅ Full dataset analysis completed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Run strategy on last 2 years
    print("\n📊 Example 2: Last 2 years")
    try:
        results = run_strategy_with_time_range('momentum', '2y')
        print("✅ 2-year analysis completed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: Run strategy on last 6 months
    print("\n📊 Example 3: Last 6 months")
    try:
        results = run_strategy_with_time_range('momentum', '6m')
        print("✅ 6-month analysis completed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Run strategy with custom end date
    print("\n📊 Example 4: 1 year ending at specific date")
    try:
        results = run_strategy_with_time_range('momentum', '1y', '2024-12-31')
        print("✅ Custom date range analysis completed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    main()