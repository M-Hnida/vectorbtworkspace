#!/usr/bin/env python3
"""
TDI (Traders Dynamic Index) Strategy - Functional Implementation
Simplified TDI strategy with RSI-based signals.
"""

from typing import Dict, List
import pandas as pd
import pandas_ta as ta
from base import BaseStrategy, Signals, StrategyConfig


def create_tdi_signals(df: pd.DataFrame, **params) -> Signals:
    """Create TDI (Traders Dynamic Index) signals.
    
    Args:
        df: OHLC DataFrame
        **params: Strategy parameters
            - rsi_period: RSI calculation period (default: 21)
            - tdi_fast_period: Fast TDI line period (default: 2)
            - tdi_slow_period: Slow TDI line period (default: 7)
            - tdi_middle_period: Middle TDI line period (default: 34)
            - oversold_level: Oversold level for RSI (default: 30)
            - overbought_level: Overbought level for RSI (default: 70)
    
    Returns:
        Signals object with entries and exits
    """
    rsi_period = params.get('rsi_period', 21)
    tdi_fast_period = params.get('tdi_fast_period', 2)
    tdi_slow_period = params.get('tdi_slow_period', 7)
    tdi_middle_period = params.get('tdi_middle_period', 34)
    oversold_level = params.get('oversold_level', 30)
    overbought_level = params.get('overbought_level', 70)
    
    # Calculate RSI
    rsi = ta.rsi(df['close'], length=rsi_period)
    
    # Calculate TDI components (moving averages of RSI)
    tdi_fast = rsi.rolling(tdi_fast_period).mean()
    tdi_slow = rsi.rolling(tdi_slow_period).mean()
    tdi_middle = rsi.rolling(tdi_middle_period).mean()
    
    # TDI signals
    # Entry: fast line crosses above slow line and both above middle
    fast_above_slow = tdi_fast > tdi_slow
    above_middle = (tdi_fast > tdi_middle) & (tdi_slow > tdi_middle)
    below_middle = (tdi_fast < tdi_middle) & (tdi_slow < tdi_middle)
    
    # Crossover detection
    fast_cross_up = (tdi_fast > tdi_slow) & (tdi_fast.shift(1) <= tdi_slow.shift(1))
    fast_cross_down = (tdi_fast < tdi_slow) & (tdi_fast.shift(1) >= tdi_slow.shift(1))
    
    # RSI level filters
    rsi_oversold = rsi < oversold_level
    rsi_overbought = rsi > overbought_level
    
    # Entry and exit signals
    long_entries = fast_cross_up & above_middle & ~rsi_overbought
    short_entries = fast_cross_down & below_middle & ~rsi_oversold
    
    long_exits = fast_cross_down | (tdi_fast < tdi_middle) | rsi_overbought
    short_exits = fast_cross_up | (tdi_fast > tdi_middle) | rsi_oversold
    
    # Combine for simple long-only strategy
    entries = long_entries
    exits = long_exits
    
    return Signals(entries=entries, exits=exits, short_entries=short_entries, short_exits=short_exits)


class TDIStrategy(BaseStrategy):
    """TDI Strategy - Functional wrapper for backward compatibility."""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.signal_params = config.parameters.copy()
    
    def get_required_timeframes(self) -> List[str]:
        return self.get_parameter('required_timeframes', ['15m', '30m', '1h', '4h', '1D'])
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']
    
    def generate_signals(self, tf_data: Dict[str, pd.DataFrame]) -> Signals:
        """Generate TDI signals using functional approach."""
        if not tf_data:
            empty_series = pd.Series(False, index=pd.Index([]))
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Use primary timeframe
        primary_tf = self.signal_params.get('primary_timeframe', list(tf_data.keys())[0])
        if primary_tf not in tf_data:
            primary_tf = list(tf_data.keys())[0]
        
        primary_df = tf_data[primary_tf]
        return create_tdi_signals(primary_df, **self.signal_params)
