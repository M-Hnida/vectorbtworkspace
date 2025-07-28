#!/usr/bin/env python3
"""
Momentum Strategy
Simple momentum strategy with multi-timeframe trend confirmation.
"""

from typing import Dict, List, Any
import pandas as pd
import pandas_ta as ta
from base import BaseStrategy, Signals, StrategyConfig


class MomentumStrategy(BaseStrategy):
    """Momentum trading strategy.
    
    Attributes:
        momentum_period: Period for momentum calculation (default: 10)
        signal_smoothing: Smoothing period for signals (default: 3)
        volatility_period: Period for volatility calculation (default: 20)
        risk_factor: Risk multiplier for position sizing (default: 1.0)
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize Momentum strategy with configuration."""
        super().__init__(config)
        self.data = {}
        
        # Strategy parameters - get from config with defaults
        self.momentum_period = self.get_parameter('momentum_period', 10)
        self.signal_smoothing = self.get_parameter('signal_smoothing', 3)
        self.volatility_period = self.get_parameter('volatility_period', 20)
        self.risk_factor = self.get_parameter('risk_factor', 1.0)
        self.atr_period = self.get_parameter('atr_period', 14)
        self.ma_type = self.get_parameter('ma_type', 'sma')
        self.ma_length = self.get_parameter('ma_length', 50)
        self.atr_multiple = self.get_parameter('atr_multiple', 2.0)
        self.volatility_momentum_threshold = self.get_parameter('volatility_momentum_threshold', 0.01)
    
    def get_required_timeframes(self) -> List[str]:
        """Get timeframes required for strategy calculations."""
        return self.get_parameter('required_timeframes', ['1h'])
    
    def get_required_columns(self) -> List[str]:
        """Get columns required for strategy calculations."""
        return ['open', 'high', 'low', 'close']

    def generate_signals(self, tf_data: Dict[str, pd.DataFrame]) -> Signals:
        """Generate momentum trading signals with proper error handling and validation."""
        if not tf_data:
            empty_series = pd.Series(False, index=pd.Index([]))
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Use the first available timeframe as primary
        main_tf = next(iter(tf_data)) if tf_data else '1h'
        data = tf_data.get(main_tf, pd.DataFrame())
        
        if data.empty:
            empty_series = pd.Series(False, index=data.index)
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Calculate indicators and generate signals
        df = self._calculate_indicators(data.copy())
        
        # Clean up data
        df = df.dropna()
        
        # Initialize all signals to False
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        short_entries = pd.Series(False, index=df.index)
        short_exits = pd.Series(False, index=df.index)
        
        # Only generate signals if we have enough data
        if len(df) >= self.momentum_period:
            # Calculate momentum
            df['momentum'] = df['close'].pct_change(periods=self.momentum_period)
            
            # Apply strategy logic
            entries[df['momentum'] > 0] = True  # Simple long signal when momentum positive
            exits[df['momentum'] < 0] = True     # Close long position when momentum negative
            
            short_entries[df['momentum'] < 0] = True  # Short when momentum negative
            short_exits[df['momentum'] > 0] = True    # Cover short when momentum positive
            
            # Apply smoothing
            if self.signal_smoothing > 1:
                entries = entries.rolling(self.signal_smoothing).sum() > 0
                exits = exits.rolling(self.signal_smoothing).sum() > 0
                short_entries = short_entries.rolling(self.signal_smoothing).sum() > 0
                short_exits = short_exits.rolling(self.signal_smoothing).sum() > 0
                
        # Ensure signals are aligned to original data index
        entries = entries.reindex(data.index, fill_value=False)
        exits = exits.reindex(data.index, fill_value=False)
        short_entries = short_entries.reindex(data.index, fill_value=False)
        short_exits = short_exits.reindex(data.index, fill_value=False)
        
        return Signals(entries, exits, short_entries, short_exits)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced momentum indicators with volatility normalization."""
        # Basic momentum calculations
        df['momentum_raw'] = df['close'].diff(self.momentum_period)
        
        # Volatility calculation using rolling standard deviation
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.volatility_period).std()
        
        # Normalize momentum by volatility for better signal quality
        df['momentum_vol_norm'] = df['momentum_raw'] / df['volatility']
        
        # Trend identification using simple moving average
        df['trend_sma'] = df['close'].rolling(window=self.volatility_period).mean()
        
        # ATR for volatility context
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.volatility_period)
        
        # RSI for overbought/oversold detection
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD for trend confirmation
        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
        
        return df
    