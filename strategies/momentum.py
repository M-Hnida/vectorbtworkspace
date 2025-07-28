#!/usr/bin/env python3
"""
Momentum Strategy - Functional Implementation
Simple momentum-based trading signals.
"""

from typing import Dict, List
import pandas as pd
import pandas_ta as ta
from base import BaseStrategy, Signals, StrategyConfig


def create_momentum_signals(df: pd.DataFrame, **params) -> Signals:
    """Create momentum trading signals.
    
    Args:
        df: OHLC DataFrame
        **params: Strategy parameters
            - momentum_period: Period for momentum calculation (default: 10)
            - signal_smoothing: Smoothing period for signals (default: 3)
            - volatility_period: Period for volatility calculation (default: 20)
            - ma_length: Moving average length for trend filter (default: 50)
            - atr_period: ATR period for volatility (default: 14)
            - atr_multiple: ATR multiple for stops (default: 2.0)
            - volatility_threshold: Minimum momentum threshold (default: 0.01)
    
    Returns:
        Signals object with entries and exits
    """
    # Parameters with defaults
    momentum_period = params.get('momentum_period', 10)
    signal_smoothing = params.get('signal_smoothing', 3)
    volatility_period = params.get('volatility_period', 20)
    ma_length = params.get('ma_length', 50)
    atr_period = params.get('atr_period', 14)
    atr_multiple = params.get('atr_multiple', 2.0)
    volatility_threshold = params.get('volatility_momentum_threshold', 0.01)
    
    # Calculate indicators
    momentum = df['close'].pct_change(momentum_period, fill_method=None)
    volatility = df['close'].rolling(volatility_period).std()
    ma = df['close'].rolling(ma_length).mean()
    atr = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    
    # Generate signals
    trend_up = df['close'] > ma
    high_momentum = momentum > volatility_threshold
    sufficient_volatility = volatility > volatility.rolling(50).mean()
    
    # Entry: uptrend + momentum + volatility
    entries = trend_up & high_momentum & sufficient_volatility
    
    # Exit: momentum reversal
    exits = momentum < -volatility_threshold
    
    # Smooth signals if requested
    if signal_smoothing > 1:
        entries = entries.rolling(signal_smoothing).sum() >= signal_smoothing
        exits = exits.rolling(signal_smoothing).sum() >= 1
    
    # Create empty short signals for strategies that don't use them
    empty_short = pd.Series(False, index=entries.index)
    
    return Signals(entries=entries, exits=exits, short_entries=empty_short, short_exits=empty_short)


class MomentumStrategy(BaseStrategy):
    """Momentum Strategy - Functional wrapper for backward compatibility."""
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.signal_params = config.parameters.copy()
    
    def get_required_timeframes(self) -> List[str]:
        return self.get_parameter('required_timeframes', ['1h'])
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']
    
    def generate_signals(self, tf_data: Dict[str, pd.DataFrame]) -> Signals:
        """Generate momentum signals using functional approach."""
        if not tf_data:
            empty_series = pd.Series(False, index=pd.Index([]))
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Use primary timeframe
        primary_tf = self.signal_params.get('primary_timeframe', list(tf_data.keys())[0])
        if primary_tf not in tf_data:
            primary_tf = list(tf_data.keys())[0]
        
        primary_df = tf_data[primary_tf]
        return create_momentum_signals(primary_df, **self.signal_params)
