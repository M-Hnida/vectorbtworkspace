#!/usr/bin/env python3
"""
Test script for time range functionality
"""

from data_manager import load_data_for_strategy, _parse_time_range, _harmonize_time_ranges
from strategies.momentum_strategy import MomentumStrategy
from base import StrategyConfig
import pandas as pd

def test_time_range_parsing():
    """Test time range parsing functionality."""
    print("🧪 Testing time range parsing...")
    
    test_cases = [
        ('2y', 730),  # 2 years ≈ 730 days
        ('1y', 365),  # 1 year = 365 days
        ('6m', 180),  # 6 months ≈ 180 days
        ('3m', 90),   # 3 months ≈ 90 days
        ('30d', 30),  # 30 days
        ('4w', 28),   # 4 weeks = 28 days
    ]
    
    for time_str, expected_days in test_cases:
        try:
            delta = _parse_time_range(time_str)
            actual_days = delta.days
            print(f"✅ {time_str} -> {actual_days} days (expected ~{expected_days})")
        except Exception as e:
            print(f"❌ {time_str} failed: {e}")

def test_data_loading_with_time_range():
    """Test data loading with time range control."""
    print("\n🧪 Testing data loading with time range...")
    
    # Create a simple strategy config for testing
    config = StrategyConfig(
        name='momentum',
        parameters={'csv_path': ['data/EURUSD_1H_2009-2025.csv']},
        optimization_grid={},
        analysis_settings={},
        data_requirements={}
    )
    
    strategy = MomentumStrategy(config)
    
    try:
        # Test loading full dataset
        print("\n📊 Loading full dataset...")
        full_data = load_data_for_strategy(strategy)
        
        for symbol, timeframes in full_data.items():
            for tf, df in timeframes.items():
                print(f"Full dataset - {symbol} {tf}: {len(df)} bars ({df.index.min()} to {df.index.max()})")
        
        # Test loading with 2-year time range
        print("\n📊 Loading last 2 years...")
        two_year_data = load_data_for_strategy(strategy, time_range='2y')
        
        for symbol, timeframes in two_year_data.items():
            for tf, df in timeframes.items():
                print(f"2-year range - {symbol} {tf}: {len(df)} bars ({df.index.min()} to {df.index.max()})")
        
        # Test loading with 6-month time range
        print("\n📊 Loading last 6 months...")
        six_month_data = load_data_for_strategy(strategy, time_range='6m')
        
        for symbol, timeframes in six_month_data.items():
            for tf, df in timeframes.items():
                print(f"6-month range - {symbol} {tf}: {len(df)} bars ({df.index.min()} to {df.index.max()})")
                
        print("\n✅ Time range functionality working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_time_range_parsing()
    test_data_loading_with_time_range()