from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import pandas_ta as ta
import vectorbt as vbt

from base import BaseStrategy, Signals

class TDIStrategy(BaseStrategy):
    """Logical Trading Indicator strategy - Adapted from MQL5 DédaleFormation.
    
    This strategy combines pivot points with TDI (Traders Dynamic Index) across multiple timeframes.
    Uses pandas_ta and vectorbt for efficient indicator calculations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LTI strategy with configuration."""
        super().__init__(config)
        self.data = {}
        
        # Pivot parameters
        self.pivot_timeframe = self.get_parameter('pivot_timeframe', '1W')
        self.pivot_number = self.get_parameter('pivot_number', 2)
        self.target_probability = self.get_parameter('target_probability', 50)
        
        # TDI parameters
        self.rsi_period = self.get_parameter('rsi_period', 21)
        self.tdi_timeframes = self.get_parameter('tdi_timeframes', ['15min', '30min', '1h', '4h', '1D'])
        self.tdi_fast_period = self.get_parameter('tdi_fast_period', 2)
        self.tdi_slow_period = self.get_parameter('tdi_slow_period', 7)
        self.tdi_middle_period = self.get_parameter('tdi_middle_period', 34)
        self.tdi_angle_min = self.get_parameter('tdi_angle_min', 20)
        self.tdi_angle_max = self.get_parameter('tdi_angle_max', 80)
        
        # Signal parameters
        self.tdi_cross_enabled = self.get_parameter('tdi_cross_enabled', [False, False, True, True, False])
        self.tdi_trend_enabled = self.get_parameter('tdi_trend_enabled', [False, True, True, True, False])
        self.tdi_angle_enabled = self.get_parameter('tdi_angle_enabled', [False, False, True, False, False])
        self.tdi_shift = self.get_parameter('tdi_shift', 1)
        
        # Risk parameters
        self.risk_factor = self.get_parameter('risk_factor', 1.0)
        self.spread_max = self.get_parameter('spread_max', 2.0)
        self.sl_distance_min = self.get_parameter('sl_distance_min', 20)
        self.tp_distance_min = self.get_parameter('tp_distance_min', 20)
    
    def get_required_timeframes(self) -> List[str]:
        """Get timeframes required for strategy calculations."""
        timeframes = self.tdi_timeframes.copy()
        if self.pivot_timeframe not in timeframes:
            timeframes.append(self.pivot_timeframe)
        return timeframes
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']

    def generate_signals(self, tf_data: Dict[str, pd.DataFrame]) -> Signals:
        """Generate LTI trading signals with multi-timeframe support."""
        # Get data for the main timeframe (first TDI timeframe)
        main_tf = self.tdi_timeframes[0]
        data = tf_data.get(main_tf, pd.DataFrame())
        
        if data.empty:
            empty_series = pd.Series(False, index=data.index)
            return Signals(empty_series, empty_series, empty_series, empty_series)
        
        # Calculate pivot points
        pivot_data = self._calculate_pivot_points(tf_data)
        
        # Calculate TDI for all timeframes
        tdi_data = self._calculate_tdi_all_timeframes(tf_data)
        
        # Generate signals
        entries, exits, short_entries, short_exits = self._generate_signals(
            data, pivot_data, tdi_data
        )
        
        return Signals(entries, exits, short_entries, short_exits)
    
    def _calculate_pivot_points(self, tf_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Calculate pivot points using pandas_ta."""
        pivot_df = tf_data.get(self.pivot_timeframe, pd.DataFrame())
        if pivot_df.empty:
            return {}
        
        # Use pandas_ta for pivot points if available, otherwise manual calculation
        try:
            # Try to use pandas_ta pivot points
            pivots = ta.pivot_points(pivot_df['high'], pivot_df['low'], pivot_df['close'])
            if pivots is not None:
                pivot_data = {
                    'pivot': pivots['PP'],
                    'supports': {},
                    'resistances': {}
                }
                
                # Extract support and resistance levels
                for i in range(1, min(self.pivot_number + 1, 4)):  # pandas_ta usually provides S1-S3, R1-R3
                    if f'S{i}' in pivots.columns:
                        pivot_data['supports'][i] = pivots[f'S{i}']
                    if f'R{i}' in pivots.columns:
                        pivot_data['resistances'][i] = pivots[f'R{i}']
                
                return pivot_data
        except:
            pass
        
        # Manual calculation as fallback
        return self._calculate_pivot_points_manual(pivot_df)
    
    def _calculate_pivot_points_manual(self, pivot_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Manual pivot point calculation."""
        # Calculate pivot point (PP)
        pp = (pivot_df['high'].shift(1) + pivot_df['low'].shift(1) + pivot_df['close'].shift(1)) / 3
        
        # Calculate support and resistance levels
        supports = {}
        resistances = {}
        
        high_prev = pivot_df['high'].shift(1)
        low_prev = pivot_df['low'].shift(1)
        
        for i in range(1, self.pivot_number + 1):
            if i == 1:
                supports[i] = 2 * pp - high_prev
                resistances[i] = 2 * pp - low_prev
            else:
                range_hl = high_prev - low_prev
                supports[i] = pp - range_hl * (i - 1)
                resistances[i] = pp + range_hl * (i - 1)
        
        return {
            'pivot': pp,
            'supports': supports,
            'resistances': resistances
        }
    
    def _calculate_tdi_all_timeframes(self, tf_data: Dict[str, pd.DataFrame]) -> Dict[int, Dict]:
        """Calculate TDI for all timeframes using pandas_ta."""
        tdi_data = {}
        
        for i, timeframe in enumerate(self.tdi_timeframes):
            df = tf_data.get(timeframe, pd.DataFrame())
            if df.empty:
                continue
            
            # Calculate RSI using pandas_ta
            rsi = ta.rsi(df['close'], length=self.rsi_period)
            
            if rsi is None or rsi.empty:
                continue
            
            # Calculate TDI moving averages using pandas_ta
            fast_ma = ta.sma(rsi, length=self.tdi_fast_period)
            slow_ma = ta.sma(rsi, length=self.tdi_slow_period)
            middle_ma = ta.sma(rsi, length=self.tdi_middle_period)
            
            # Calculate angle using vectorized operations
            angle = self._calculate_tdi_angle_vectorized(fast_ma, slow_ma)
            
            tdi_data[i] = {
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'middle_ma': middle_ma,
                'angle': angle,
                'rsi': rsi
            }
        
        return tdi_data
    
    def _calculate_tdi_angle_vectorized(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        """Calculate TDI angle using vectorized operations."""
        # Calculate angles using numpy for better performance
        fast_diff = fast_ma.diff()
        slow_diff = slow_ma.diff()
        
        fast_angle = np.arctan(fast_diff) * 180 / np.pi
        slow_angle = np.arctan(slow_diff) * 180 / np.pi
        
        # Combine angles as per MQL5 logic
        weight_factor = self.rsi_period / self.tdi_middle_period
        angle = (slow_angle + (fast_angle * weight_factor)) / (1 + weight_factor)
        
        return angle
    
    def _check_trade_signal_vectorized(self, direction: bool, tdi_data: Dict[int, Dict]) -> pd.Series:
        """Check trade signal conditions using vectorized operations.
        
        Args:
            direction: False for buy, True for sell
            tdi_data: TDI data for all timeframes
        """
        all_conditions = []
        
        for i in range(len(self.tdi_timeframes)):
            if i not in tdi_data:
                continue
            
            timeframe_conditions = []
            tdi = tdi_data[i]
            
            # Cross condition
            if i < len(self.tdi_cross_enabled) and self.tdi_cross_enabled[i]:
                fast_prev = tdi['fast_ma'].shift(self.tdi_shift + 1)
                fast_curr = tdi['fast_ma'].shift(self.tdi_shift)
                slow_prev = tdi['slow_ma'].shift(self.tdi_shift + 1)
                slow_curr = tdi['slow_ma'].shift(self.tdi_shift)
                
                if not direction:  # Buy signal
                    cross_cond = (fast_prev < slow_prev) & (fast_curr > slow_curr)
                else:  # Sell signal
                    cross_cond = (fast_prev > slow_prev) & (fast_curr < slow_curr)
                
                timeframe_conditions.append(cross_cond)
            
            # Trend condition
            if i < len(self.tdi_trend_enabled) and self.tdi_trend_enabled[i]:
                middle_prev = tdi['middle_ma'].shift(self.tdi_shift + 1)
                middle_curr = tdi['middle_ma'].shift(self.tdi_shift)
                
                if not direction:  # Buy signal
                    trend_cond = middle_prev < middle_curr
                else:  # Sell signal
                    trend_cond = middle_prev > middle_curr
                
                timeframe_conditions.append(trend_cond)
            
            # Angle condition
            if i < len(self.tdi_angle_enabled) and self.tdi_angle_enabled[i]:
                angle = tdi['angle']
                
                if not direction:  # Buy signal
                    angle_cond = (angle >= self.tdi_angle_min) & (angle <= self.tdi_angle_max)
                else:  # Sell signal
                    angle_cond = (angle <= -self.tdi_angle_min) & (angle >= -self.tdi_angle_max)
                
                timeframe_conditions.append(angle_cond)
            
            # Combine conditions for this timeframe
            if timeframe_conditions:
                tf_condition = timeframe_conditions[0]
                for cond in timeframe_conditions[1:]:
                    tf_condition = tf_condition & cond
                all_conditions.append(tf_condition)
        
        # All timeframe conditions must be true
        if all_conditions:
            final_condition = all_conditions[0]
            for cond in all_conditions[1:]:
                final_condition = final_condition & cond
            return final_condition.fillna(False)
        else:
            # Return False series if no conditions are enabled
            first_tdi = list(tdi_data.values())[0]
            return pd.Series(False, index=first_tdi['fast_ma'].index)
    
    def _get_optimal_levels_vectorized(self, direction: bool, pivot_data: Dict, 
                                     prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Get optimal SL/TP levels using vectorized operations."""
        if not pivot_data or 'supports' not in pivot_data or 'resistances' not in pivot_data:
            return pd.Series(np.nan, index=prices.index), pd.Series(np.nan, index=prices.index)
        
        supports = pivot_data['supports']
        resistances = pivot_data['resistances']
        
        # Initialize result series
        best_sl = pd.Series(np.nan, index=prices.index)
        best_tp = pd.Series(np.nan, index=prices.index)
        
        min_prob_diff = pd.Series(np.inf, index=prices.index)
        
        # Vectorized calculation for all combinations
        for i in range(1, min(self.pivot_number + 1, len(supports) + 1)):
            for j in range(1, min(self.pivot_number + 1, len(resistances) + 1)):
                if i not in supports or j not in resistances:
                    continue
                
                if not direction:  # Buy
                    tp_levels = resistances[i]
                    sl_levels = supports[j]
                    
                    # Valid conditions
                    valid = (prices > sl_levels) & (prices < tp_levels)
                    
                    tp_size = np.abs(tp_levels - prices)
                    sl_size = np.abs(prices - sl_levels)
                    
                else:  # Sell
                    tp_levels = supports[i]
                    sl_levels = resistances[j]
                    
                    # Valid conditions
                    valid = (prices < sl_levels) & (prices > tp_levels)
                    
                    tp_size = np.abs(prices - tp_levels)
                    sl_size = np.abs(sl_levels - prices)
                
                # Calculate probability
                total_size = tp_size + sl_size
                probability = np.where(total_size > 0, (tp_size / total_size) * 100, 0)
                prob_diff = np.abs(probability - self.target_probability)
                
                # Update best levels where this combination is better
                update_mask = valid & (prob_diff < min_prob_diff)
                best_sl = np.where(update_mask, sl_levels, best_sl)
                best_tp = np.where(update_mask, tp_levels, best_tp)
                min_prob_diff = np.where(update_mask, prob_diff, min_prob_diff)
        
        return pd.Series(best_sl, index=prices.index), pd.Series(best_tp, index=prices.index)
    
    def _is_trade_allowed_vectorized(self, direction: bool, sl_levels: pd.Series, 
                                   tp_levels: pd.Series, prices: pd.Series) -> pd.Series:
        """Check if trades are allowed using vectorized operations."""
        # Check for valid levels
        valid_levels = ~(sl_levels.isna() | tp_levels.isna())
        
        if not direction:  # Buy
            sl_distance_ok = np.abs(prices - sl_levels) >= self.sl_distance_min
            tp_distance_ok = np.abs(tp_levels - prices) >= self.tp_distance_min
        else:  # Sell
            sl_distance_ok = np.abs(sl_levels - prices) >= self.sl_distance_min
            tp_distance_ok = np.abs(prices - tp_levels) >= self.tp_distance_min
        
        return valid_levels & sl_distance_ok & tp_distance_ok
    
    def _generate_signals(self, data: pd.DataFrame, pivot_data: Dict, 
                         tdi_data: Dict[int, Dict]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Generate trading signals using vectorized operations."""
        
        # Check trade signals using vectorized operations
        buy_signals = self._check_trade_signal_vectorized(False, tdi_data)
        sell_signals = self._check_trade_signal_vectorized(True, tdi_data)
        
        # Get optimal levels
        buy_sl, buy_tp = self._get_optimal_levels_vectorized(False, pivot_data, data['close'])
        sell_sl, sell_tp = self._get_optimal_levels_vectorized(True, pivot_data, data['close'])
        
        # Check if trades are allowed
        buy_allowed = self._is_trade_allowed_vectorized(False, buy_sl, buy_tp, data['close'])
        sell_allowed = self._is_trade_allowed_vectorized(True, sell_sl, sell_tp, data['close'])
        
        # Final entry signals
        entries = buy_signals & buy_allowed
        short_entries = sell_signals & sell_allowed
        
        # Exit conditions based on pivot levels and basic rules
        exits = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)
        
        if pivot_data and 'pivot' in pivot_data:
            pivot_level = pivot_data['pivot']
            exits = data['close'] < pivot_level
            short_exits = data['close'] > pivot_level
        
        return entries.fillna(False), exits.fillna(False), short_entries.fillna(False), short_exits.fillna(False)