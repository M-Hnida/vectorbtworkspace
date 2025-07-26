"""Opening Range Breakout (ORB) strategy implementation."""
from typing import List, Tuple
import pandas as pd
import numpy as np
import pandas_ta as ta
from core.base import BaseStrategy
from backtest import Signals

class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy."""
    
    def generate_signals(self, data: pd.DataFrame) -> Signals:
        """Generate ORB signals for both long and short positions.

        Args:
            data: Price data DataFrame

        Returns:
            Signals: Dictionary containing signal Series:
                - entries: Long entry signals
                - exits: Long exit signals
                - short_entries: Short entry signals 
                - short_exits: Short exit signals
        """

        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        orb_period = self.get_parameter('orb_period', 30)
        if len(data) < orb_period:
            raise ValueError("Insufficient data for ORB calculation")
        # endregion

        # Get parameters
        atr_period = self.get_parameter('atr_period', 14)
        atr_multiple = self.get_parameter('atr_multiple', 2.0)

        # Calculate opening range and ATR
        orb_high, orb_low = self._calculate_opening_range(data, orb_period)
        atr_stop_size = self._calculate_atr_stop(data, atr_period, atr_multiple)

        # Initialize signals and position tracking
        signals = {}
        long_position = pd.Series(False, index=data.index)
        short_position = pd.Series(False, index=data.index)
        
        # Initialize signal series
        long_entries = pd.Series(False, index=data.index)
        long_exits = pd.Series(False, index=data.index)
        short_entries = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)
        
        # Track entry prices for stop loss calculation
        long_entry_prices = pd.Series(np.nan, index=data.index)
        short_entry_prices = pd.Series(np.nan, index=data.index)
        
        # Process signals bar by bar
        for i in range(1, len(data)):
            current_orb_high = orb_high.iloc[i]
            current_orb_low = orb_low.iloc[i]
            current_atr = atr_stop_size.iloc[i]
            
            # Skip if ORB not calculated yet
            if pd.isna(current_orb_high) or pd.isna(current_orb_low):
                long_position.iloc[i] = long_position.iloc[i-1]
                short_position.iloc[i] = short_position.iloc[i-1]
                long_entry_prices.iloc[i] = long_entry_prices.iloc[i-1]
                short_entry_prices.iloc[i] = short_entry_prices.iloc[i-1]
                continue
                
            # LONG LOGIC
            if not long_position.iloc[i-1]:  # Not in long position
                if data['close'].iloc[i] > current_orb_high:
                    long_entries.iloc[i] = True
                    long_position.iloc[i] = True
                    long_entry_prices.iloc[i] = data['close'].iloc[i]
                else:
                    long_position.iloc[i] = False
                    long_entry_prices.iloc[i] = np.nan
            else:  # In long position
                stop_loss = long_entry_prices.iloc[i-1] - current_atr
                if data['low'].iloc[i] <= stop_loss:
                    long_exits.iloc[i] = True
                    long_position.iloc[i] = False
                    long_entry_prices.iloc[i] = np.nan
                else:
                    long_position.iloc[i] = True
                    long_entry_prices.iloc[i] = long_entry_prices.iloc[i-1]
            
            # SHORT LOGIC
            if not short_position.iloc[i-1]:  # Not in short position
                if data['close'].iloc[i] < current_orb_low:
                    short_entries.iloc[i] = True
                    short_position.iloc[i] = True
                    short_entry_prices.iloc[i] = data['close'].iloc[i]
                else:
                    short_position.iloc[i] = False
                    short_entry_prices.iloc[i] = np.nan
            else:  # In short position
                stop_loss = short_entry_prices.iloc[i-1] + current_atr
                if data['high'].iloc[i] >= stop_loss:
                    short_exits.iloc[i] = True
                    short_position.iloc[i] = False
                    short_entry_prices.iloc[i] = np.nan
                else:
                    short_position.iloc[i] = True
                    short_entry_prices.iloc[i] = short_entry_prices.iloc[i-1]
        
            signals = {
                'long_entries': long_entries,
                'long_exits': long_exits,
                'short_entries': short_entries,
                'short_exits': short_exits
            }
            
            return signals
        
    def get_required_columns(self) -> List[str]:
        """Return required data columns."""
        return ['open', 'high', 'low', 'close']
    
    def _calculate_opening_range(self, data: pd.DataFrame, orb_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate true opening range based on first N periods of the trading day."""
        try:
            # Ensure data has a datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Data must have a DatetimeIndex for ORB calculation.")

            data = data.copy()
            data['date'] = data.index.date

            orb_high = pd.Series(np.nan, index=data.index, dtype=float)
            orb_low = pd.Series(np.nan, index=data.index, dtype=float)

            for date in data['date'].unique():
                day_data = data[data['date'] == date]
                if len(day_data) >= orb_period:
                    # Calculate ORB from the first `orb_period` bars of the day
                    orb_range = day_data.iloc[:orb_period]
                    day_orb_high = orb_range['high'].max()
                    day_orb_low = orb_range['low'].min()

                    # Apply the calculated ORB to the rest of the day's bars
                    orb_high.loc[day_data.index[orb_period:]] = day_orb_high
                    orb_low.loc[day_data.index[orb_period:]] = day_orb_low

            return orb_high, orb_low
        except Exception as e:
            print(f"⚠️ Opening range calculation failed: {e}")
            return pd.Series(np.nan, index=data.index), pd.Series(np.nan, index=data.index)
    
    def _calculate_atr_stop(self, data: pd.DataFrame, atr_period: int, atr_multiple: float) -> pd.Series:
        """Calculate ATR-based stop loss levels with proper error handling."""
        try:
            # Try using pandas_ta first
            atr = ta.atr(data['high'], data['low'], data['close'], length=atr_period)

            # If pandas_ta fails or returns all NaNs, use manual calculation
            if atr is None or atr.isna().all():
                high_low = data['high'] - data['low']
                high_close_prev = abs(data['high'] - data['close'].shift(1))
                low_close_prev = abs(data['low'] - data['close'].shift(1))

                true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                atr = true_range.rolling(window=atr_period, min_periods=1).mean()

            return (atr * atr_multiple).fillna(0)
        except Exception as e:
            print(f"⚠️ ATR calculation failed: {e}")
            # Fallback to a percentage of the close price if ATR fails
            price_change = data['close'].pct_change().abs()
            return price_change.rolling(atr_period).mean() * data['close'] * atr_multiple
            
    def combine_signals(self, signals: dict) -> Tuple[pd.Series, pd.Series]:
        """Combine long and short signals into unified entries and exits.
        
        This method combines long and short signals into a format compatible
        with VectorBT's Portfolio.from_signals method.
        
        Args:
            signals: Dictionary containing signal series
            
        Returns:
            Tuple[pd.Series, pd.Series]: Combined entry and exit signals
        """
        entries = pd.Series(False, index=signals['long_entries'].index)
        exits = pd.Series(False, index=signals['long_exits'].index)
        
        # Combine long signals
        if 'long_entries' in signals:
            entries = entries | signals['long_entries']
            exits = exits | signals['long_exits']
            
        # Add short signals if present
        if 'short_entries' in signals:
            entries = entries | signals['short_entries']
            exits = exits | signals['short_exits']
            
        return entries, exits
    