#!/usr/bin/env python3
"""
ORB (Opening Range Breakout) Strategy - Portfolio Direct Implementation
Multi-timeframe opening range breakout strategy.
"""

from typing import Dict
import pandas as pd
import vectorbt as vbt


def create_portfolio(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """Create ORB strategy portfolio directly."""
    if params is None:
        params = {}
        
    # Parameters
    orb_period = params.get('orb_period', 1)
    breakout_threshold = params.get('breakout_threshold', 0.005)
    
    # Ensure DatetimeIndex
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Calculate opening range (first N periods of each day)
    df['date'] = df.index.date
    
    # Group by date and calculate daily opening ranges
    daily_ranges = df.groupby('date').apply(
        lambda x: pd.Series({
            'range_high': x['high'].iloc[:orb_period].max(),
            'range_low': x['low'].iloc[:orb_period].min(),
            'range_size': x['high'].iloc[:orb_period].max() - x['low'].iloc[:orb_period].min()
        })
    )
    
    # Merge back to main dataframe
    df = df.merge(daily_ranges, left_on='date', right_index=True, how='left')
    
    # Calculate breakout signals
    range_size_threshold = df['close'] * breakout_threshold
    valid_range = df['range_size'] > range_size_threshold
    
    # Entry signals: breakout above/below range
    long_entries = (df['close'] > df['range_high']) & valid_range
    short_entries = (df['close'] < df['range_low']) & valid_range
    
    # Exit signals: return to range
    long_exits = df['close'] < df['range_low']
    short_exits = df['close'] > df['range_high']
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=data['close'],
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=10000,
        fees=0.001
    )
    
    return portfolio

