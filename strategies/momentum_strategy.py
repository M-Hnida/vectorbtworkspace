#!/usr/bin/env python3
"""
Momentum Strategy
Simple momentum strategy with multi-timeframe trend confirmation.
"""

from typing import Dict, List
import pandas as pd
import pandas_ta as ta
from base import BaseStrategy, Signals
from data_manager import DataManager


class MomentumStrategy(BaseStrategy):
    """Momentum trading strategy.
    
    Attributes:
        momentum_period: Period for momentum calculation (default: 10)
        signal_smoothing: Smoothing period for signals (default: 3)
        volatility_period: Period for volatility calculation (default: 20)
        risk_factor: Risk multiplier for position sizing (default: 1.0)
    """
    
    def __init__(self, config: dict):
        """Initialize Momentum strategy with symbol from config if not provided."""
        super().__init__(config)
        self.symbol = config.get('symbol') or DataManager().defaults.get('symbol', 'EURUSD')
        self.data = {}
        
        # Strategy parameters - get from config with defaults
        self.momentum_period = config.get('momentum_period', 10)  # Default value
        self.signal_smoothing = config.get('signal_smoothing', 3)  # Default value
        self.volatility_period = config.get('volatility_period', 20)  # Default value
        self.risk_factor = config.get('risk_factor', 1.0)  # Default value
        self.atr_period = config.get('atr_period', 14)  # Default ATR period
        self.ma_type = config.get('ma_type', 'sma')  # Moving average type
        self.ma_length = config.get('ma_length', 50)  # Length for moving average
        self.atr_multiple = config.get('atr_multiple', 2.0)  # ATR multiple for stop loss
        self.volatility_momentum_threshold = config.get('volatility_momentum_threshold', 0.01)  # Default threshold for volatility momentum
    
    def get_required_timeframes(self) -> List[str]:
        """Get timeframes required for strategy calculations."""
        return ['1h']  # Default required timeframe
    
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
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        
        return df
    
    def _generate_base_signals(self, df: pd.DataFrame) -> tuple:
        """Generate base momentum signals."""
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        short_entries = pd.Series(False, index=df.index)
        short_exits = pd.Series(False, index=df.index)
        
        # Enhanced momentum conditions
        strong_momentum_up = df['momentum'] > self.volatility_momentum_threshold
        strong_momentum_down = df['momentum'] < -self.volatility_momentum_threshold
        
        # Trend conditions
        uptrend = df['close'] > df['wma']
        downtrend = df['close'] < df['wma']
        
        # Volatility filter - avoid low volatility periods
        sufficient_volatility = df['volatility'] > df['volatility'].rolling(50).quantile(0.3)
        
        # RSI conditions for additional confirmation
        rsi_not_overbought = df['rsi'] < 70
        rsi_not_oversold = df['rsi'] > 30
        
        # Long signals: positive momentum, uptrend, sufficient volatility
        entries = strong_momentum_up & uptrend & sufficient_volatility & rsi_not_overbought
        
        # Long exits: negative momentum or downtrend
        exits = strong_momentum_down | downtrend
        
        # Short signals: negative momentum, downtrend, sufficient volatility
        short_entries = strong_momentum_down & downtrend & sufficient_volatility & rsi_not_oversold
        
        # Short exits: positive momentum or uptrend
        short_exits = strong_momentum_up | uptrend
        
        return entries, exits, short_entries, short_exits
    