"""
Momentum Strategy with Volatility Momentum, WMA Trend Alignment, and VBT Stop Management.
Generates boolean signals for entries/exits with portfolio-level stops and targets.
"""

from typing import Dict, List, Optional
import pandas as pd
import pandas_ta as ta
from base import Signals


def create_momentum_signals(df: pd.DataFrame, **params) -> Signals:
    """Create momentum trading signals with volatility momentum and WMA trend alignment."""
    # Parameters
    vol_momentum_window = params.get("vol_momentum_window", 20)
    vol_momentum_threshold = params.get("vol_momentum_threshold", 0.0)
    vol_std_window = params.get("vol_std_window", 20)
    wma_length = params.get("wma_length", 50)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    close = df["close"]

    # Calculate volatility momentum
    returns = close.pct_change()
    vol = returns.rolling(vol_std_window).std()
    vol_momentum = vol - vol.shift(vol_momentum_window)
    
    # Fill NaN values
    vol_momentum = vol_momentum.fillna(0.0)
    strong_vol_momentum = vol_momentum > vol_momentum_threshold

    # WMA trend alignment
    price_wma = ta.wma(close, length=wma_length)
    if price_wma is None:
        price_wma = close.rolling(wma_length).mean()
    
    price_wma = price_wma.fillna(method="bfill").fillna(close)
    long_trend = close > price_wma
    short_trend = close < price_wma

    # Entry signals
    long_entries = strong_vol_momentum & long_trend
    short_entries = strong_vol_momentum & short_trend

    # Exit conditions - only for signal-based exits (stops handled by VBT)
    vol_momentum_reversal = vol_momentum <= vol_momentum_threshold
    long_exits = vol_momentum_reversal | ~long_trend
    short_exits = vol_momentum_reversal | ~short_trend

    # Ensure boolean series with proper index
    long_entries = pd.Series(long_entries, index=df.index).fillna(False).astype(bool)
    short_entries = pd.Series(short_entries, index=df.index).fillna(False).astype(bool)
    long_exits = pd.Series(long_exits, index=df.index).fillna(False).astype(bool)
    short_exits = pd.Series(short_exits, index=df.index).fillna(False).astype(bool)

    return Signals(entries=long_entries, exits=long_exits, short_entries=short_entries, short_exits=short_exits)


def get_momentum_vbt_params(df: pd.DataFrame, params: Dict) -> Dict:
    """Get VBT portfolio parameters for momentum strategy with stops and targets."""
    # ATR for dynamic stops
    atr_length = params.get("atr_length", 14)
    atr_mult = params.get("atr_mult", 2.0)
    
    # Profit target
    profit_target_percent = params.get("profit_target_percent", 0.02)  # 2%
    
    # Trailing stop
    trailing_stop_percent = params.get("trailing_stop_percent", 0.01)  # 1%
    
    # Calculate ATR
    atr = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=atr_length)
    if atr is None:
        atr = (df["high"] - df["low"]).rolling(atr_length).mean()
    atr = atr.fillna(atr.mean())
    
    # ATR-based stop loss (percentage of close price)
    atr_stop_percent = (atr * atr_mult) / df["close"]
    atr_stop_percent = atr_stop_percent.fillna(0.02)  # 2% fallback
    
    vbt_params = {
        # Stop loss: ATR-based
        'sl_stop': atr_stop_percent,
        
        # Trailing stop: percentage-based
        'sl_trail': trailing_stop_percent,
        
        # Take profit: percentage-based
        'tp_stop': profit_target_percent,
        
        # Enable stops
        'use_stops': True,
        
        # OHLC data for stop calculations
        'open': df["open"],
        'high': df["high"], 
        'low': df["low"],
    }
    
    return vbt_params


# TODO: Template for partial profit targets using adjust_tp_func_nb
def create_partial_tp_template():
    """
    Template for implementing partial profit targets with VBT.
    
    This would use:
    - allow_partial=True
    - adjust_tp_func_nb with custom Numba function
    - Multiple TP levels (e.g., 1%, 2%, 3%)
    - Partial exits at each level (e.g., 33%, 50%, 100%)
    
    Implementation steps:
    1. Define TP levels and exit percentages
    2. Create Numba-compatible adjust_tp_func_nb
    3. Track position state and current TP level
    4. Update TP dynamically based on reached levels
    5. Handle partial exits with proper sizing
    
    Example structure:
    ```python
    @nb.jit
    def adjust_tp_func_nb(c, tp_stop, *args):
        # Custom logic for multi-level TP
        # Read entry price from context
        # Calculate current TP level
        # Return updated TP and exit size
        pass
    
    vbt_params = {
        'adjust_tp_func_nb': adjust_tp_func_nb,
        'adjust_tp_args': (tp_levels, exit_percentages),
        'allow_partial': True,
        # ... other params
    }
    ```
    """
    pass


def get_momentum_required_timeframes(params: Dict) -> List[str]:
    """Get required timeframes for momentum strategy."""
    return params.get("required_timeframes", ["1h"])


def generate_signals(tf_data: Dict[str, pd.DataFrame], params: Dict) -> Signals:
    """Generate signals from multi-timeframe data."""
    if not tf_data:
        empty_index = pd.DatetimeIndex([])
        empty_series = pd.Series(False, index=empty_index)
        return Signals(empty_series, empty_series, empty_series, empty_series)

    # Use primary timeframe
    primary_tf = params.get("primary_timeframe", list(tf_data.keys())[0])
    if primary_tf not in tf_data:
        primary_tf = list(tf_data.keys())[0]

    primary_df = tf_data[primary_tf]

    # Ensure the DataFrame has a DatetimeIndex
    if not isinstance(primary_df.index, pd.DatetimeIndex):
        primary_df = primary_df.copy()
        primary_df.index = pd.to_datetime(primary_df.index)

    return create_momentum_signals(primary_df, **params)