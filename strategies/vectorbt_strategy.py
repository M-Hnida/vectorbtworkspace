#!/usr/bin/env python3
"""
VectorBT Bollinger Bands Mean Reversion Strategy
Integrated version of the vectorbt strategy for backtesting framework.
"""

from typing import Dict, List
import pandas as pd


from base import BaseStrategy, Signals, StrategyConfig


class VectorBTStrategy(BaseStrategy):
    """VectorBT Bollinger Bands Mean Reversion Strategy.
    
    This strategy implements a mean reversion approach using Bollinger Bands
    with ADX filtering and SMA support/resistance levels.
    
    Attributes:
        bbands_period: Period for Bollinger Bands calculation
        bbands_std: Standard deviation multiplier for Bollinger Bands
        adx_period: Period for ADX calculation
        adx_threshold: ADX threshold for trend strength
        adx_threshold_filter: Filter to avoid trading in directional markets
        sma_period: Period for SMA calculation
        atr_period: ATR period for volatility-based sizing
        atr_mult: Multiple of ATR used as stop distance
        risk_pct: Percentage of portfolio equity risked per entry
        max_side_exposure: Maximum exposure allowed per side
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize VectorBT strategy with configuration."""
        super().__init__(config)
        
        # Strategy parameters with defaults
        self.bbands_period = self.get_parameter('bbands_period', 20)
        self.bbands_std = self.get_parameter('bbands_std', 2.0)
        self.adx_period = self.get_parameter('adx_period', 14)
        self.adx_threshold = self.get_parameter('adx_threshold', 20)
        self.adx_threshold_filter = self.get_parameter('adx_threshold_filter', 60)
        self.sma_period = self.get_parameter('sma_period', 200)
        self.atr_period = self.get_parameter('atr_period', 14)
        self.atr_mult = self.get_parameter('atr_mult', 1.0)
        self.risk_pct = self.get_parameter('risk_pct', 0.02)
        self.max_side_exposure = self.get_parameter('max_side_exposure', 0.30)
        self.initial_cash = self.get_parameter('initial_cash', 500000)
        self.dca_size_increment = self.get_parameter('dca_size_increment', 0.01)
        self.max_dca_size = self.get_parameter('max_dca_size', 0.10)
    
    def get_required_timeframes(self) -> List[str]:
        """Get timeframes required for strategy calculations."""
        return self.get_parameter('required_timeframes', ['1h'])
    
    def get_required_columns(self) -> List[str]:
        """Get columns required for strategy calculations."""
        return ['open', 'high', 'low', 'close']

    def generate_signals(self, tf_data: Dict[str, pd.DataFrame]) -> Signals:
        """Generate VectorBT mean reversion trading signals."""
        if not tf_data:
            empty_series = pd.Series(False, index=pd.Index([]))
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Use the first available timeframe as primary
        main_tf = next(iter(tf_data)) if tf_data else '1h'
        data = tf_data.get(main_tf, pd.DataFrame()).copy()
        
        if data.empty:
            empty_series = pd.Series(False, index=data.index)
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Ensure column names are lowercase
        data.columns = [c.lower() for c in data.columns]
        
        # Validate required columns
        required_cols = ['close', 'low', 'high']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Got: {data.columns.tolist()}")
        
        # Store original index before any processing
        original_index = data.index
        
        # Calculate technical indicators
        data = self._calculate_indicators(data)
        
        # Clean up NaN values but preserve the datetime index
        data_clean = data.dropna()
        
        if data_clean.empty:
            empty_series = pd.Series(False, index=original_index)
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Generate entry/exit conditions and sizes on clean data
        signals_with_sizes = self._generate_mean_reversion_signals_with_sizing(data_clean)
        
        # Reindex signals to match original data index
        entries = signals_with_sizes['entries'].reindex(original_index, fill_value=False)
        exits = signals_with_sizes['exits'].reindex(original_index, fill_value=False)
        short_entries = signals_with_sizes['short_entries'].reindex(original_index, fill_value=False)
        short_exits = signals_with_sizes['short_exits'].reindex(original_index, fill_value=False)
        sizes = signals_with_sizes['sizes'].reindex(original_index, fill_value=0.0)
        
        return Signals(entries, exits, short_entries, short_exits, sizes)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy."""
        # Bollinger Bands
        df.ta.bbands(length=self.bbands_period, std=self.bbands_std, append=True)
        
        # ADX - Average Directional Index for trend strength
        df.ta.adx(length=self.adx_period, append=True)
        
        # SMA 200 - Support/Resistance level
        df.ta.sma(length=self.sma_period, append=True)
        
        # ATR for volatility-based sizing
        df.ta.atr(length=self.atr_period, append=True)
        
        return df

    def _generate_mean_reversion_signals_with_sizing(self, data: pd.DataFrame) -> Dict:
        """Generate mean reversion signals based on Bollinger Bands and ADX."""
        # Define column names for readability
        bbl_col = f'BBL_{self.bbands_period}_{self.bbands_std}'
        bbm_col = f'BBM_{self.bbands_period}_{self.bbands_std}'
        bbu_col = f'BBU_{self.bbands_period}_{self.bbands_std}'
        adx_col = f'ADX_{self.adx_period}'
        sma_col = f'SMA_{self.sma_period}'
        
        # Find ATR column (pandas_ta may change format)
        atr_candidates = [col for col in data.columns if col.upper().startswith('ATR')]
        if not atr_candidates:
            raise ValueError("ATR column not found after indicator calculation")
        atr_col = atr_candidates[0]
        
        # Trend conditions
        weak_trend = data[adx_col] < self.adx_threshold
        high_adx_filter = data[adx_col] >= self.adx_threshold_filter
        
        # Entry conditions
        long_initial_entries = (
            (data['close'].shift(1) < data[bbl_col].shift(1)) & 
            (data['close'] >= data[bbl_col]) & 
            (~high_adx_filter)
        )
        
        long_dca_conditions = (data['low'] <= data[bbl_col]) & (~high_adx_filter)
        
        short_initial_entries = (
            (data['close'].shift(1) > data[bbu_col].shift(1)) & 
            (data['close'] <= data[bbu_col]) & 
            (~high_adx_filter)
        )
        
        short_dca_conditions = (data['high'] >= data[bbu_col]) & (~high_adx_filter)
        
        # Initialize position tracking
        long_position = pd.Series(0, index=data.index)
        short_position = pd.Series(0, index=data.index)
        
        # Initialize signal arrays
        long_entries = pd.Series(False, index=data.index)
        short_entries = pd.Series(False, index=data.index)
        long_exits = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)
        
        # Process signals with position tracking
        for i in range(len(data)):
            # Carry forward position state
            if i > 0:
                long_position.iloc[i] = long_position.iloc[i-1]
                short_position.iloc[i] = short_position.iloc[i-1]
                
                # Prevent overlapping exposure
                if long_position.iloc[i] > 0 and short_initial_entries.iloc[i]:
                    long_exits.iloc[i] = True
                    long_position.iloc[i] = 0
                elif short_position.iloc[i] > 0 and long_initial_entries.iloc[i]:
                    short_exits.iloc[i] = True
                    short_position.iloc[i] = 0
            
            # Exit conditions
            long_exit_condition = (
                (data['close'].iloc[i] >= data[bbu_col].iloc[i]) |
                ((data['close'].iloc[i] >= data[bbm_col].iloc[i]) & weak_trend.iloc[i]) |
                high_adx_filter.iloc[i]
            )
            
            short_exit_condition = (
                (data['close'].iloc[i] <= data[bbl_col].iloc[i]) |
                ((data['close'].iloc[i] <= data[bbm_col].iloc[i]) & weak_trend.iloc[i]) |
                high_adx_filter.iloc[i]
            )
            
            # Process exits first
            if long_position.iloc[i] > 0 and long_exit_condition:
                long_exits.iloc[i] = True
                long_position.iloc[i] = 0
            
            if short_position.iloc[i] > 0 and short_exit_condition:
                short_exits.iloc[i] = True
                short_position.iloc[i] = 0
            
            # Process initial entries
            if long_position.iloc[i] == 0 and long_initial_entries.iloc[i]:
                long_entries.iloc[i] = True
                long_position.iloc[i] = 1
            
            if short_position.iloc[i] == 0 and short_initial_entries.iloc[i]:
                short_entries.iloc[i] = True
                short_position.iloc[i] = 1
            
            # Process DCA opportunities
            if (long_position.iloc[i] > 0 and not long_exits.iloc[i] and 
                long_dca_conditions.iloc[i] and not long_entries.iloc[i]):
                long_entries.iloc[i] = True
            
            if (short_position.iloc[i] > 0 and not short_exits.iloc[i] and 
                short_dca_conditions.iloc[i] and not short_entries.iloc[i]):
                short_entries.iloc[i] = True
        
        # Calculate ATR-based position sizes
        size_array = self._calculate_position_sizes(data, long_entries, short_entries)
        
        return {
            'entries': long_entries,
            'exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'sizes': size_array
        }
    
    def _calculate_position_sizes(self, data: pd.DataFrame, long_entries: pd.Series, short_entries: pd.Series) -> pd.Series:
        """Calculate ATR-based position sizes like the original vect.py"""
        # Find ATR column
        atr_candidates = [col for col in data.columns if col.upper().startswith('ATR')]
        if not atr_candidates:
            # Fallback to fixed size if no ATR
            return pd.Series(0.01, index=data.index)  # 1% fixed size
        
        atr_col = atr_candidates[0]
        size_array = pd.Series(0.0, index=data.index)
        
        # Calculate sizes for entries only
        for i in data.index:
            if long_entries.loc[i] or short_entries.loc[i]:
                close_price = data.loc[i, 'close']
                atr_val = data.loc[i, atr_col]
                
                if not pd.isna(atr_val) and atr_val > 0:
                    # ATR-based sizing: risk_pct * portfolio / (atr_mult * atr)
                    risk_value = self.risk_pct * self.initial_cash
                    size_val = risk_value / (self.atr_mult * atr_val * close_price)
                    
                    # Cap at max exposure
                    max_size = self.max_side_exposure
                    size_array.loc[i] = min(size_val, max_size)
                else:
                    # Fallback to small fixed size
                    size_array.loc[i] = 0.01
        
        return size_array