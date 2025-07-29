"""
VectorBT Bollinger Bands Mean Reversion Strategy - Pure Functional Implementation

Clean implementation focused on signal generation and configuration.
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, List
from base import Signals, StrategyConfig

def create_bollinger_mean_reversion_signals(df: pd.DataFrame, **params) -> Signals:
    """
    Create Bollinger Bands mean reversion trading signals.

    Args:
        df: OHLC DataFrame with columns ['open', 'high', 'low', 'close']
        **params: Strategy parameters
            - bbands_period: Period for Bollinger Bands (default: 20)
            - bbands_std: Standard deviation for Bollinger Bands (default: 2.0)
            - adx_period: Period for ADX calculation (default: 14)
            - adx_threshold: ADX threshold for trend strength (default: 20)
            - adx_threshold_filter: Filter to avoid trading in directional markets (default: 60)
            - sma_period: Period for SMA calculation (default: 200)
            - atr_period: ATR window for volatility (default: 14)

    Returns:
        Signals: Trading signals with entries, exits, short_entries, short_exits
    """

    # Extract parameters with defaults
    bbands_period = params.get('bbands_period', 20)
    bbands_std = params.get('bbands_std', 2.0)
    adx_period = params.get('adx_period', 14)
    adx_threshold = params.get('adx_threshold', 20)
    adx_threshold_filter = params.get('adx_threshold_filter', 60)
    sma_period = params.get('sma_period', 200)
    atr_period = params.get('atr_period', 14)

    # ===== DATA PREPARATION =====
    data = df.copy()
    data.columns = [c.lower() for c in data.columns]

    required_cols = ['close', 'low', 'high']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Got: {data.columns.tolist()}")

    # ===== TECHNICAL INDICATORS =====
    # Add indicators using pandas_ta
    # 1. Bollinger Bands
    data.ta.bbands(length=bbands_period, std=bbands_std, append=True)

    # 2. ADX - Average Directional Index for trend strength
    data.ta.adx(length=adx_period, append=True)

    # 3. SMA - Support/Resistance level
    data.ta.sma(length=sma_period, append=True)

    # 4. ATR for volatility-based sizing
    data.ta.atr(length=atr_period, append=True)

    # Clean up NaN values
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)  # Reset index after dropping NaNs

    # Define column names for readability
    bbl_col = f'BBL_{bbands_period}_{bbands_std}'
    bbm_col = f'BBM_{bbands_period}_{bbands_std}'
    bbu_col = f'BBU_{bbands_period}_{bbands_std}'
    adx_col = f'ADX_{adx_period}'

    # ===== ENTRY/EXIT SIGNALS =====
    # Trend condition for adaptive exits
    weak_trend = data[adx_col] < adx_threshold
    # Filter condition for high ADX - avoid trading in strongly directional markets
    high_adx_filter = data[adx_col] >= adx_threshold_filter

    # Define entry conditions
    # Initial entries: Price crosses from below to above the lower band for longs
    # AND ADX is below the filter threshold
    long_initial_entries = (data['close'].shift(1) < data[bbl_col].shift(1)) & (data['close'] >= data[bbl_col]) & (~high_adx_filter)

    # DCA conditions for longs: Price touches or goes below lower band
    # AND ADX is below the filter threshold
    long_dca_conditions = (data['low'] <= data[bbl_col]) & (~high_adx_filter)

    # Initial entries: Price crosses from above to below the upper band for shorts
    # AND ADX is below the filter threshold
    short_initial_entries = (data['close'].shift(1) > data[bbu_col].shift(1)) & (data['close'] <= data[bbu_col]) & (~high_adx_filter)

    # DCA conditions for shorts: Price touches or goes above upper band
    # AND ADX is below the filter threshold
    short_dca_conditions = (data['high'] >= data[bbu_col]) & (~high_adx_filter)
    
    # Initialize arrays for position tracking
    long_position = pd.Series(0, index=data.index)  # 0 = no position, 1 = in position
    short_position = pd.Series(0, index=data.index)  # 0 = no position, 1 = in position

    # Initialize signal arrays
    long_entries = pd.Series(False, index=data.index)
    short_entries = pd.Series(False, index=data.index)
    long_exits = pd.Series(False, index=data.index)
    short_exits = pd.Series(False, index=data.index)

    # Track positions, DCA opportunities, and generate clean signals in a single pass
    for i in range(len(data)):
        # Carry forward position state from previous bar (if not first bar)
        if i > 0:
            long_position.iloc[i] = long_position.iloc[i-1]
            short_position.iloc[i] = short_position.iloc[i-1]
            # Prevent overlapping exposure – close the opposite side if a fresh initial entry appears
            if long_position.iloc[i] > 0 and short_initial_entries.iloc[i]:
                long_exits.iloc[i] = True
                long_position.iloc[i] = 0  # Close long before opening short
            elif short_position.iloc[i] > 0 and long_initial_entries.iloc[i]:
                short_exits.iloc[i] = True
                short_position.iloc[i] = 0  # Close short before opening long

        # Define exit conditions:
        # 1. Price reaching the opposite band
        # 2. Price reaching middle line if ADX is low (weak trend)
        # 3. ADX exceeding the filter threshold (strong directional market)
        long_exit_condition = (data['close'].iloc[i] >= data[bbu_col].iloc[i]) | \
                              ((data['close'].iloc[i] >= data[bbm_col].iloc[i]) & weak_trend.iloc[i]) | \
                              high_adx_filter.iloc[i]  # Exit if ADX exceeds filter threshold

        short_exit_condition = (data['close'].iloc[i] <= data[bbl_col].iloc[i]) | \
                               ((data['close'].iloc[i] <= data[bbm_col].iloc[i]) & weak_trend.iloc[i]) | \
                               high_adx_filter.iloc[i]  # Exit if ADX exceeds filter threshold

        # Process positions in priority order: exits first, then entries

        # Process exits (only if we have a position)
        if long_position.iloc[i] > 0 and long_exit_condition:
            long_exits.iloc[i] = True
            long_position.iloc[i] = 0  # Close position

        if short_position.iloc[i] > 0 and short_exit_condition:
            short_exits.iloc[i] = True
            short_position.iloc[i] = 0  # Close position

        # Process initial entries (only if we don't have a position)
        if long_position.iloc[i] == 0 and long_initial_entries.iloc[i]:
            long_entries.iloc[i] = True
            long_position.iloc[i] = 1  # Open position

        if short_position.iloc[i] == 0 and short_initial_entries.iloc[i]:
            short_entries.iloc[i] = True
            short_position.iloc[i] = 1  # Open position

        # Process DCA opportunities (only if we already have a position and no exit signal on same bar)
        if long_position.iloc[i] > 0 and not long_exits.iloc[i] and long_dca_conditions.iloc[i]:
            # Only add DCA if we didn't already have an entry on this bar
            if not long_entries.iloc[i]:
                long_entries.iloc[i] = True  # Add to position

        if short_position.iloc[i] > 0 and not short_exits.iloc[i] and short_dca_conditions.iloc[i]:
            # Only add DCA if we didn't already have an entry on this bar
            if not short_entries.iloc[i]:
                short_entries.iloc[i] = True  # Add to position

    return Signals(
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits
    )


def generate_vectorbt_signals(tf_data: Dict[str, pd.DataFrame], params: Dict) -> Signals:
    """Generate VectorBT signals from multi-timeframe data."""
    if not tf_data:
        empty_series = pd.Series(False, index=pd.Index([]))
        return Signals(empty_series, empty_series, empty_series, empty_series)

    # Use primary timeframe
    primary_tf = params.get('primary_timeframe', list(tf_data.keys())[0])
    if primary_tf not in tf_data:
        primary_tf = list(tf_data.keys())[0]

    primary_df = tf_data[primary_tf]
    return create_bollinger_mean_reversion_signals(primary_df, **params)




