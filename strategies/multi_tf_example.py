#!/usr/bin/env python3
"""Multi-timeframe strategy example"""

import pandas as pd
import pandas_ta as ta
from base import Signals


def generate_signals(tf_data: dict) -> Signals:
    """Multi-timeframe strategy using trend + momentum."""
    
    # Get different timeframes
    h1_data = tf_data.get('1h')
    h4_data = tf_data.get('4h') 
    
    if h1_data is None:
        # Fallback to available timeframe
        h1_data = list(tf_data.values())[0]
    
    # H4 trend filter (if available)
    if h4_data is not None:
        h4_ma = h4_data['close'].rolling(20).mean()
        trend_up = h4_data['close'] > h4_ma
        # Align to H1 timeframe
        trend_up = trend_up.reindex(h1_data.index, method='ffill')
    else:
        trend_up = pd.Series(True, index=h1_data.index)
    
    # H1 momentum signals
    rsi = ta.rsi(h1_data['close'], length=14)
    if not isinstance(rsi, pd.Series):
        rsi = pd.Series(rsi, index=h1_data.index)
    
    # Combine timeframes
    entries = (rsi < 30) & trend_up.fillna(True)
    exits = rsi > 70
    
    empty_short = pd.Series(False, index=entries.index)
    
    return Signals(
        entries=entries.fillna(False),
        exits=exits.fillna(False),
        short_entries=empty_short, 
        short_exits=empty_short
    )