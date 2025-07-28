#!/usr/bin/env python3
"""
ORB (Opening Range Breakout) Strategy
Multi-timeframe opening range breakout strategy.
Can use higher timeframe for range definition and lower timeframe for entries.
"""

import os
import sys
from typing import Dict, List

import pandas as pd
import pandas_ta as ta

from base import BaseStrategy, Signals, StrategyConfig
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy with multi-timeframe support.
    
    Attributes:
        orb_period: Number of periods to calculate the opening range (default: 1)
        breakout_threshold: Minimum range size as percentage of price to consider a breakout
        atr_multiple: Multiple of ATR to use for stop loss calculation
        required_timeframes: List of timeframes required for strategy calculations
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize ORB strategy with symbol from config if not provided."""
        super().__init__(config)
        self.symbol = self.get_parameter('symbol', 'EURUSD')
        self.data = {}
        
        # Strategy parameters - get from config with defaults
        self.orb_period = self.get_parameter('orb_period', 1)
        self.breakout_threshold = self.get_parameter('breakout_threshold', 0.005)
        self.atr_multiple = self.get_parameter('atr_multiple', 2.0)
    
    def get_required_timeframes(self) -> List[str]:
        """Get timeframes required for strategy calculations."""
        return self.get_parameter('required_timeframes', ['15m', '1h'])
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']
    
    def generate_signals(self, tf_data: Dict[str, pd.DataFrame]) -> Signals:
        """Generate ORB trading signals with multi-timeframe support."""
        if not tf_data:
            empty_series = pd.Series(False, index=pd.Index([]))
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Find the best available timeframe from the data
        available_tfs = list(tf_data.keys())
        if not available_tfs:
            empty_series = pd.Series(False, index=pd.Index([]))
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Use the first required timeframe that is present in the data, or fall back to the first available timeframe
        required_tfs = self.get_required_timeframes()
        main_tf = next((tf for tf in required_tfs if tf in available_tfs), available_tfs[0])
        
        data = tf_data.get(main_tf, pd.DataFrame())
        if data.empty:
            empty_series = pd.Series(False, index=pd.Index([]))
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Ensure data has a DateTime index
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Data index conversion to datetime failed: {e}")
        
        # Determine which timeframe to use for range calculation
        range_data = data.copy()
        entry_data = data.copy()
        
        required_tfs = self.get_required_timeframes()
        if len(required_tfs) > 1:
            range_tf = required_tfs[1]
            if range_tf in tf_data:
                range_data = tf_data[range_tf].copy()
        
        # Calculate indicators and ranges
        range_df = self._calculate_range_indicators(range_data)
        entry_df = self._calculate_entry_indicators(entry_data)
        
        # Align data if using different timeframes
        required_tfs = self.get_required_timeframes()
        if len(required_tfs) > 1:
            aligned_ranges = self._align_ranges_to_entry_timeframe(range_df, entry_df)
        else:
            aligned_ranges = range_df
        
        # Store original index for signal alignment
        original_index = data.index
        
        # Drop NaN values
        aligned_ranges = aligned_ranges.dropna()
        entry_df = entry_df.dropna()
        
        if aligned_ranges.empty or entry_df.empty:
            empty_series = pd.Series(False, index=original_index)
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Generate signals
        entries, exits, short_entries, short_exits = self._generate_orb_signals(
            aligned_ranges, entry_df
        )

        # Ensure signals are aligned to original data index
        entries = entries.reindex(original_index, fill_value=False)
        exits = exits.reindex(original_index, fill_value=False)
        short_entries = short_entries.reindex(original_index, fill_value=False)
        short_exits = short_exits.reindex(original_index, fill_value=False)
        
        return Signals(entries, exits, short_entries, short_exits)
    
    def _calculate_range_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate opening range indicators."""
        # Calculate rolling high and low for opening range
        df['range_high'] = df['high'].rolling(self.orb_period).max()
        df['range_low'] = df['low'].rolling(self.orb_period).min()
        df['range_size'] = (df['range_high'] - df['range_low']) / df['close']
        
        # Calculate range midpoint and width percentiles
        df['range_mid'] = (df['range_high'] + df['range_low']) / 2
        df['range_width_pct'] = df['range_size'].rolling(50).rank(pct=True)
        
        return df
    
    def _calculate_entry_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for entry timeframe."""
        # ATR for volatility-based filtering and stops
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Price momentum for confirmation
        df['price_change'] = df['close'].pct_change()
        df['momentum'] = df['price_change'].rolling(5).mean()
        
        # Volume analysis if available
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        else:
            df['volume_ratio'] = 1.0
        
        # Volatility context
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        return df
    
    def _align_ranges_to_entry_timeframe(self, range_df: pd.DataFrame, entry_df: pd.DataFrame) -> pd.DataFrame:
        """Align higher timeframe ranges to entry timeframe."""
        # Forward fill the range data to match entry timeframe
        range_cols = ['range_high', 'range_low', 'range_size', 'range_mid', 'range_width_pct']
        
        aligned_df = entry_df.copy()
        for col in range_cols:
            if col in range_df.columns:
                aligned_df[col] = range_df[col].reindex(entry_df.index, method='ffill')
        
        return aligned_df
    
    def _generate_orb_signals(self, range_df: pd.DataFrame, entry_df: pd.DataFrame) -> tuple:
        """Generate ORB signals with proper position tracking."""
        # Use the common index
        common_index = range_df.index.intersection(entry_df.index)
        if len(common_index) == 0:
            empty_series = pd.Series(False, index=range_df.index)
            return empty_series, empty_series, empty_series, empty_series
        
        # Align dataframes
        range_aligned = range_df.loc[common_index]
        entry_aligned = entry_df.loc[common_index]
        
        entries = pd.Series(False, index=common_index)
        exits = pd.Series(False, index=common_index)
        short_entries = pd.Series(False, index=common_index)
        short_exits = pd.Series(False, index=common_index)
        
        # Use previous bar's range for breakout detection
        prev_range_high = range_aligned['range_high'].shift(1)
        prev_range_low = range_aligned['range_low'].shift(1)
        
        # Improved breakout conditions - focus on strong breakouts
        # Long signals: close breaks above previous range high with confirmation
        long_breakout = entry_aligned['close'] > prev_range_high
        long_strong_breakout = (entry_aligned['close'] - prev_range_high) > (entry_aligned['atr'] * 0.1)  # Minimum breakout size
        long_range_filter = range_aligned['range_size'] > self.breakout_threshold  # Use original threshold
        long_momentum = entry_aligned['momentum'] > 0.0001  # Positive momentum
        long_volume = entry_aligned['volume_ratio'] > 0.8  # Good volume
        long_volatility = entry_aligned['volatility'] > 0.002  # Sufficient volatility
        
        entries = (long_breakout & long_strong_breakout & long_range_filter & 
                  long_momentum & long_volume & long_volatility)
        
        # Short signals: close breaks below previous range low with confirmation
        short_breakdown = entry_aligned['close'] < prev_range_low
        short_strong_breakdown = (prev_range_low - entry_aligned['close']) > (entry_aligned['atr'] * 0.1)  # Minimum breakdown size
        short_range_filter = range_aligned['range_size'] > self.breakout_threshold  # Use original threshold
        short_momentum = entry_aligned['momentum'] < -0.0001  # Negative momentum
        short_volume = entry_aligned['volume_ratio'] > 0.8  # Good volume
        short_volatility = entry_aligned['volatility'] > 0.002  # Sufficient volatility
        
        short_entries = (short_breakdown & short_strong_breakdown & short_range_filter & 
                        short_momentum & short_volume & short_volatility)
        
        # Position tracking for proper exit signals
        long_position = pd.Series(False, index=common_index)
        short_position = pd.Series(False, index=common_index)
        
        # Track positions and generate exits only when in position
        for i in range(1, len(common_index)):
            idx = common_index[i]
            prev_idx = common_index[i-1]
            
            # Update position status
            if entries.iloc[i]:
                long_position.iloc[i] = True
            elif long_position.iloc[i-1] and not exits.iloc[i-1]:
                long_position.iloc[i] = True
                
            if short_entries.iloc[i]:
                short_position.iloc[i] = True
            elif short_position.iloc[i-1] and not short_exits.iloc[i-1]:
                short_position.iloc[i] = True
            
            # Generate exit signals only when in position with improved logic
            if long_position.iloc[i]:
                # Take profit at 2x ATR above entry
                take_profit = entry_aligned['close'].iloc[i] > (prev_range_high.iloc[i] + entry_aligned['atr'].iloc[i] * 2.0)
                # Stop loss at range low or 1.5x ATR below entry
                stop_loss = entry_aligned['close'].iloc[i] < (prev_range_high.iloc[i] - entry_aligned['atr'].iloc[i] * self.atr_multiple)
                # Exit on negative momentum
                momentum_exit = entry_aligned['momentum'].iloc[i] < -0.001
                
                exits.iloc[i] = take_profit or stop_loss or momentum_exit
                
                if exits.iloc[i]:
                    long_position.iloc[i] = False
            
            if short_position.iloc[i]:
                # Take profit at 2x ATR below entry
                take_profit = entry_aligned['close'].iloc[i] < (prev_range_low.iloc[i] - entry_aligned['atr'].iloc[i] * 2.0)
                # Stop loss at range high or 1.5x ATR above entry
                stop_loss = entry_aligned['close'].iloc[i] > (prev_range_low.iloc[i] + entry_aligned['atr'].iloc[i] * self.atr_multiple)
                # Exit on positive momentum
                momentum_exit = entry_aligned['momentum'].iloc[i] > 0.001
                
                short_exits.iloc[i] = take_profit or stop_loss or momentum_exit
                
                if short_exits.iloc[i]:
                    short_position.iloc[i] = False
        
        return entries, exits, short_entries, short_exits
