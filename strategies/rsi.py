#!/usr/bin/env python3
"""RSI Strategy - Simple mean reversion"""

import pandas as pd
import pandas_ta as ta
from base import Signals


def generate_signals(tf_data: dict, params: dict) -> Signals:
    """Generate RSI mean reversion signals."""
    # Get primary timeframe data
    primary_tf = list(tf_data.keys())[0]
    df = tf_data[primary_tf]
    
    # Parameters
    rsi_period = params.get('rsi_period', 14)
    oversold = params.get('oversold_level', 30)
    overbought = params.get('overbought_level', 70)
    
    # Calculate RSI
    rsi = ta.rsi(df['close'], length=rsi_period)
    if not isinstance(rsi, pd.Series):
        rsi = pd.Series(rsi, index=df.index)
    
    # Generate signals
    entries = rsi < oversold  # Buy when oversold
    exits = rsi > overbought  # Sell when overbought
    
    # No short signals for this simple strategy
    empty_short = pd.Series(False, index=entries.index)
    
    return Signals(
        entries=entries.fillna(False),
        exits=exits.fillna(False), 
        short_entries=empty_short,
        short_exits=empty_short
    )