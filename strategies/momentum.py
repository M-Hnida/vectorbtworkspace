#!/usr/bin/env python3
"""
Momentum Strategy - Pure Functional Implementation
Simple momentum-based trading signals.
"""

from typing import Dict, List
import pandas as pd
import pandas_ta as ta
from base import Signals


def create_momentum_signals(df: pd.DataFrame, **params) -> Signals:
    """Create momentum trading signals."""
    # Parameters
    momentum_period = params.get("momentum_period", 10)
    signal_smoothing = params.get("signal_smoothing", 3)
    volatility_period = params.get("volatility_period", 20)
    ma_length = params.get("ma_length", 50)
    volatility_threshold = params.get("volatility_momentum_threshold", 0.01)

    # Calculate indicators
    momentum = df["close"].pct_change(momentum_period).fillna(0)
    volatility = df["close"].rolling(volatility_period).std().fillna(0)
    ma = df["close"].rolling(ma_length).mean().fillna(df["close"])

    # Generate signals
    trend_up = (df["close"] > ma).fillna(False)
    high_momentum = (momentum > volatility_threshold).fillna(False)
    sufficient_volatility = (volatility > volatility.rolling(50).mean().fillna(volatility.mean())).fillna(False)

    # Entry: uptrend + momentum + volatility
    entries = trend_up & high_momentum & sufficient_volatility
    exits = (momentum < -volatility_threshold).fillna(False)

    # Smooth signals if requested
    if signal_smoothing > 1:
        entries = (entries.rolling(signal_smoothing).sum() >= signal_smoothing).fillna(False)
        exits = (exits.rolling(signal_smoothing).sum() >= 1).fillna(False)

    # Ensure no NaN values in final signals
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # Empty short signals
    empty_short = pd.Series(False, index=entries.index)

    return Signals(entries=entries, exits=exits, short_entries=empty_short, short_exits=empty_short)


def get_momentum_required_timeframes(params: Dict) -> List[str]:
    """Get required timeframes for momentum strategy."""
    return params.get("required_timeframes", ["1h"])


def get_momentum_required_columns() -> List[str]:
    """Get required columns for momentum strategy."""
    return ["open", "high", "low", "close"]


def generate_signals(
    tf_data: Dict[str, pd.DataFrame], params: Dict
) -> Signals:
    """Generate momentum signals from multi-timeframe data."""
    if not tf_data:
        empty_index = pd.DatetimeIndex([])
        empty_series = pd.Series(False, index=empty_index)
        return Signals(empty_series, empty_series, empty_series, empty_series)

    # Use primary timeframe
    primary_tf = params.get("primary_timeframe", list(tf_data.keys())[0])
    if primary_tf not in tf_data:
        primary_tf = list(tf_data.keys())[0]

    primary_df = tf_data[primary_tf]
    
    # Ensure the DataFrame has a DatetimeIndex
    if not isinstance(primary_df.index, pd.DatetimeIndex):
        primary_df.index = pd.to_datetime(primary_df.index)
    
    return create_momentum_signals(primary_df, **params)


# Pure functional approach - no classes needed
