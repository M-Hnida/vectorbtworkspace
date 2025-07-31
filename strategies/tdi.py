#!/usr/bin/env python3
"""
TDI (Traders Dynamic Index) Strategy - Pure Functional Implementation
Simplified TDI strategy with RSI-based signals.
"""

from typing import Dict, List
import pandas as pd
import pandas_ta as ta
from base import Signals


def create_tdi_signals(df: pd.DataFrame, **params) -> Signals:
    """Create TDI (Traders Dynamic Index) signals."""
    # Parameters
    rsi_period = params.get('rsi_period', 21)
    tdi_fast_period = params.get('tdi_fast_period', 2)
    tdi_slow_period = params.get('tdi_slow_period', 7)
    tdi_middle_period = params.get('tdi_middle_period', 34)
    oversold_level = params.get('oversold_level', 30)
    overbought_level = params.get('overbought_level', 70)
    
    # Calculate RSI
    rsi = ta.rsi(df['close'], length=rsi_period)
    if rsi is None or not hasattr(rsi, 'rolling'):
        rsi = pd.Series(50.0, index=df.index)
    
    # Ensure rsi is a pandas Series
    if not isinstance(rsi, pd.Series):
        rsi = pd.Series(rsi, index=df.index)
    
    # Calculate TDI components
    tdi_fast = rsi.rolling(window=tdi_fast_period, min_periods=1).mean()
    tdi_slow = rsi.rolling(window=tdi_slow_period, min_periods=1).mean()
    tdi_middle = rsi.rolling(window=tdi_middle_period, min_periods=1).mean()
    
    # Signal conditions
    above_middle = (tdi_fast > tdi_middle) & (tdi_slow > tdi_middle)
    below_middle = (tdi_fast < tdi_middle) & (tdi_slow < tdi_middle)
    
    # Crossovers
    fast_cross_up = (tdi_fast > tdi_slow) & (tdi_fast.shift(1) <= tdi_slow.shift(1))
    fast_cross_down = (tdi_fast < tdi_slow) & (tdi_fast.shift(1) >= tdi_slow.shift(1))
    
    # RSI filters
    rsi_oversold = rsi < oversold_level
    rsi_overbought = rsi > overbought_level
    
    # Generate signals
    long_entries = fast_cross_up & above_middle & ~rsi_overbought
    short_entries = fast_cross_down & below_middle & ~rsi_oversold
    long_exits = fast_cross_down | (tdi_fast < tdi_middle) | rsi_overbought
    short_exits = fast_cross_up | (tdi_fast > tdi_middle) | rsi_oversold
    
    return Signals(
        entries=long_entries.fillna(False),
        exits=long_exits.fillna(False),
        short_entries=short_entries.fillna(False),
        short_exits=short_exits.fillna(False)
    )


def get_tdi_required_timeframes(params: Dict) -> List[str]:
    """Get required timeframes for TDI strategy."""
    return params.get('required_timeframes', ['15m', '30m', '1h', '4h', '1D'])


def get_tdi_required_columns() -> List[str]:
    """Get required columns for TDI strategy."""
    return ['open', 'high', 'low', 'close']


def generate_tdi_signals(tf_data: Dict[str, pd.DataFrame], params: Dict) -> Signals:
    """Generate TDI signals from multi-timeframe data."""
    if not tf_data:
        empty_index = pd.DatetimeIndex([])
        empty_series = pd.Series(False, index=empty_index)
        return Signals(empty_series, empty_series, empty_series, empty_series)
    
    # Use primary timeframe
    primary_tf = params.get('primary_timeframe', list(tf_data.keys())[0])
    if primary_tf not in tf_data:
        primary_tf = list(tf_data.keys())[0]
    
    primary_df = tf_data[primary_tf]
    
    # Ensure the DataFrame has a DatetimeIndex
    if not isinstance(primary_df.index, pd.DatetimeIndex):
        primary_df.index = pd.to_datetime(primary_df.index)
    
    return create_tdi_signals(primary_df, **params)
