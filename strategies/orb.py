#!/usr/bin/env python3
"""
ORB (Opening Range Breakout) Strategy - Pure Functional Implementation
Multi-timeframe opening range breakout strategy.
"""

from typing import Dict, List
import pandas as pd
import pandas_ta as ta
from base import Signals, StrategyConfig


def create_orb_signals(df: pd.DataFrame, **params) -> Signals:
    """Create Opening Range Breakout signals.
    
    Args:
        df: OHLC DataFrame
        **params: Strategy parameters
            - orb_period: Number of periods for opening range (default: 1)
            - breakout_threshold: Minimum range size as % of price (default: 0.005)
            - atr_period: ATR period for volatility (default: 14)
            - atr_multiple: ATR multiple for stops (default: 2.0)
    
    Returns:
        Signals object with entries and exits
    """
    orb_period = params.get('orb_period', 1)
    breakout_threshold = params.get('breakout_threshold', 0.005)
    atr_period = params.get('atr_period', 14)
    atr_multiple = params.get('atr_multiple', 2.0)
    
    # Calculate ATR for volatility
    atr = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    
    # Calculate opening range (first N periods of each day)
    df_copy = df.copy()
    df_copy['date'] = df_copy.index.date
    
    # Group by date and calculate daily opening ranges
    daily_ranges = df_copy.groupby('date').apply(
        lambda x: pd.Series({
            'range_high': x['high'].iloc[:orb_period].max(),
            'range_low': x['low'].iloc[:orb_period].min(),
            'range_size': x['high'].iloc[:orb_period].max() - x['low'].iloc[:orb_period].min()
        })
    )
    
    # Merge back to main dataframe
    df_copy = df_copy.merge(daily_ranges, left_on='date', right_index=True, how='left')
    
    # Calculate breakout signals
    range_size_threshold = df_copy['close'] * breakout_threshold
    valid_range = df_copy['range_size'] > range_size_threshold
    
    # Entry signals: breakout above/below range
    long_entries = (df['close'] > df_copy['range_high']) & valid_range
    short_entries = (df['close'] < df_copy['range_low']) & valid_range
    
    # Exit signals: return to range or stop loss
    long_exits = df['close'] < df_copy['range_low']
    short_exits = df['close'] > df_copy['range_high']
    
    # Combine entries and exits
    entries = long_entries
    exits = long_exits
    
    return Signals(entries=entries, exits=exits, short_entries=short_entries, short_exits=short_exits)


def get_orb_required_timeframes(params: Dict) -> List[str]:
    """Get required timeframes for ORB strategy."""
    return params.get('required_timeframes', ['15m', '1h'])


def get_orb_required_columns() -> List[str]:
    """Get required columns for ORB strategy."""
    return ['open', 'high', 'low', 'close']


def generate_orb_signals(tf_data: Dict[str, pd.DataFrame], params: Dict) -> Signals:
    """Generate ORB signals from multi-timeframe data."""
    if not tf_data:
        empty_series = pd.Series(False, index=pd.Index([]))
        return Signals(empty_series, empty_series, empty_series, empty_series)
    
    # Use primary timeframe
    primary_tf = params.get('primary_timeframe', list(tf_data.keys())[0])
    if primary_tf not in tf_data:
        primary_tf = list(tf_data.keys())[0]
    
    primary_df = tf_data[primary_tf]
    return create_orb_signals(primary_df, **params)


# Pure functional approach - no classes needed
