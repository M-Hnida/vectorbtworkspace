#!/usr/bin/env python3
"""
LTI (Logical Trading Indicator) Strategy
Uses ATR, Moving Averages, and Bollinger Bands with multi-timeframe support.
"""

#!/usr/bin/env python3
"""
LTI (Logical Trading Indicator) Strategy
Uses ATR, Moving Averages, and Bollinger Bands with multi-timeframe support.
"""

from typing import Dict, Any, List
import pandas as pd
import pandas_ta as ta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseStrategy, Signals
from data_manager import DataManager  # Fixed import path
import numpy as np  # Required for numerical operations

class LTIStrategy(BaseStrategy):
    """Logical Trading Indicator strategy.
    
    Attributes:
        trend_period: Period for trend calculation (default: 50)
        signal_smoothing: Smoothing period for signals (default: 3)
        volatility_period: Period for volatility calculation (default: 20)
        risk_factor: Risk multiplier for position sizing (default: 1.0)
    """
    
    def __init__(self, config: dict):
        """Initialize LTI strategy with symbol from config if not provided."""
        super().__init__(config)
        self.symbol = config.get('symbol') or DataManager().defaults.get('symbol', 'EURUSD')
        self.data = {}
        
        # Strategy parameters - get from config with defaults
        self.trend_period = config.get('trend_period', 50)  # Default value
        self.signal_smoothing = config.get('signal_smoothing', 3)  # Default value
        self.volatility_period = config.get('volatility_period', 20)  # Default value
        self.risk_factor = config.get('risk_factor', 1.0)  # Default value
        self.atr_period = config.get('atr_period', 14)  # Default ATR period
        self.ma_type = config.get('ma_type', 'ema')  # Moving average type
        self.ma_length = config.get('ma_length', 50)  # Length for moving average
        self.bb_std_dev = config.get('bb_std_dev', 2)  # Standard deviation for Bollinger Bands
        self.atr_multiple = config.get('atr_multiple', 2.0)  # ATR multiple for stop loss
    
    def get_required_timeframes(self) -> List[str]:
        """Get timeframes required for strategy calculations."""
        return ['1h']  # Default required timeframe
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def generate_signals(self, tf_data: Dict[str, pd.DataFrame]) -> Signals:
        """Generate LTI trading signals with multi-timeframe support."""
        if not self.required_timeframes:
            raise ValueError("No timeframes specified in the strategy configuration.")
        # Get data for the main timeframe
        main_tf = self.required_timeframes[0]
        data = tf_data.get(main_tf, pd.DataFrame())
        
        if data.empty:
            empty_series = pd.Series(False, index=data.index)
            return Signals(empty_series, empty_series, empty_series, empty_series)
            
        # Calculate indicators
        df = self._calculate_indicators(data)
        
        # Generate signals
        entries, exits, short_entries, short_exits = self._generate_lti_signals(df)
        
        return Signals(entries, exits, short_entries, short_exits)
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate LTI indicators."""
        # Calculate ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        
        # Calculate Moving Average
        if self.ma_type.upper() == 'EMA':
            df['ma'] = ta.ema(df['close'], length=self.ma_length)
        else:
            df['ma'] = ta.sma(df['close'], length=self.ma_length)
        
        # Calculate Bollinger Bands
        bbands = ta.bbands(df['close'], length=self.ma_length, std=self.bb_std_dev)
        if bbands is not None:
            df['bb_upper'] = bbands[f'BBU_{self.ma_length}_{self.bb_std_dev}']
            df['bb_lower'] = bbands[f'BBL_{self.ma_length}_{self.bb_std_dev}']
            df['bb_middle'] = bbands[f'BBM_{self.ma_length}_{self.bb_std_dev}']
        
        return df
    
    def _generate_lti_signals(self, df: pd.DataFrame) -> tuple:
        """Generate LTI signals."""
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        short_entries = pd.Series(False, index=df.index)
        short_exits = pd.Series(False, index=df.index)
        
        # Long entry conditions
        # 1. Close above moving average
        # 2. Close above upper Bollinger Band (breakout)
        # 3. ATR confirms volatility
        long_condition1 = df['close'] > df['ma']
        long_condition2 = df['close'] > df['bb_upper']
        long_condition3 = df['atr'] > df['atr'].rolling(20).mean() * 0.5  # Minimum volatility
        
        entries = long_condition1 & long_condition2 & long_condition3
        
        # Long exit conditions
        # 1. Close below moving average
        # 2. Stop loss based on ATR
        long_stop_loss = df['close'] < (df['close'].shift(1) - df['atr'] * self.atr_multiple)
        exits = (df['close'] < df['ma']) | long_stop_loss
        
        # Short entry conditions
        # 1. Close below moving average
        # 2. Close below lower Bollinger Band (breakdown)
        # 3. ATR confirms volatility
        short_condition1 = df['close'] < df['ma']
        short_condition2 = df['close'] < df['bb_lower']
        short_condition3 = df['atr'] > df['atr'].rolling(20).mean() * 0.5  # Minimum volatility
        
        short_entries = short_condition1 & short_condition2 & short_condition3
        
        # Short exit conditions
        # 1. Close above moving average
        # 2. Stop loss based on ATR
        short_stop_loss = df['close'] > (df['close'].shift(1) + df['atr'] * self.atr_multiple)
        short_exits = (df['close'] > df['ma']) | short_stop_loss
        
        return entries, exits, short_entries, short_exits