"""Opening Range Breakout (ORB) strategy implementation."""
from typing import List, Tuple
import pandas as pd
import numpy as np
import pandas_ta as ta
from core.base import BaseStrategy


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy."""
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate ORB signals."""
        # Get parameters
        orb_period = self.get_parameter('orb_period', 30)
        breakout_threshold = self.get_parameter('breakout_threshold', 0.0001)
        atr_period = self.get_parameter('atr_period', 14)
        atr_multiple = self.get_parameter('atr_multiple', 2.0)
        
        # Calculate opening range
        orb_high, orb_low = self._calculate_opening_range(data, orb_period)
        
        # Calculate ATR for stop loss
        atr_stop_size = self._calculate_atr_stop(data, atr_period, atr_multiple)
        
        # Generate signals
        entries, exits = self._generate_orb_signals(
            data, orb_high, orb_low, atr_stop_size, breakout_threshold
        )
        
        return entries, exits
    
    def get_required_columns(self) -> List[str]:
        """Return required data columns."""
        return ['open', 'high', 'low', 'close']
    
    def _calculate_opening_range(self, data: pd.DataFrame, orb_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate opening range high and low."""
        try:
            # Use rolling windows to define opening ranges
            orb_high = data['high'].rolling(window=orb_period, min_periods=orb_period).max()
            orb_low = data['low'].rolling(window=orb_period, min_periods=orb_period).min()
            
            # Shift to avoid look-ahead bias
            return orb_high.shift(1), orb_low.shift(1)
        except Exception as e:
            print(f"⚠️ Opening range calculation failed: {e}")
            return data['high'] * np.nan, data['low'] * np.nan
    
    def _calculate_atr_stop(self, data: pd.DataFrame, atr_period: int, atr_multiple: float) -> pd.Series:
        """Calculate ATR-based stop loss levels."""
        try:
            atr = ta.atr(data['high'], data['low'], data['close'], length=atr_period)
            if atr is None:
                # Fallback ATR calculation
                tr1 = data['high'] - data['low']
                tr2 = abs(data['high'] - data['close'].shift())
                tr3 = abs(data['low'] - data['close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(atr_period).mean()
            
            return atr * atr_multiple
        except Exception as e:
            print(f"⚠️ ATR calculation failed: {e}")
            return data['close'] * 0.02  # 2% fallback stop
    
    def _generate_orb_signals(self, data: pd.DataFrame, orb_high: pd.Series, orb_low: pd.Series,
                             atr_stop_size: pd.Series, breakout_threshold: float) -> Tuple[pd.Series, pd.Series]:
        """Generate ORB entry and exit signals."""
        
        # Breakout conditions
        breakout_size = orb_high - orb_low
        significant_range = breakout_size > breakout_threshold
        
        # Long signals: price breaks above opening range high
        long_breakout = (data['close'] > orb_high) & (data['close'].shift(1) <= orb_high.shift(1))
        long_entries = long_breakout & significant_range
        
        # Stop loss: price falls below opening range low or ATR stop
        long_stop_level = np.maximum(orb_low, data['close'] - atr_stop_size)
        long_exits = data['close'] < long_stop_level
        
        # Fill NaN values with False
        long_entries = long_entries.fillna(False)
        long_exits = long_exits.fillna(False)
        
        return long_entries, long_exits