"""Bollinger Bands strategy with Kalman filter for noise reduction."""

from typing import Dict
import numpy as np
import pandas as pd
import vectorbt as vbt

# Constants
DEFAULT_PERIOD = 20
DEFAULT_STD_DEV = 2
DEFAULT_PROCESS_VARIANCE = 1e-5
DEFAULT_MEASUREMENT_VARIANCE = 1e-1
DEFAULT_INIT_CASH = 10000
DEFAULT_FEES = 0.001
BAND_TOLERANCE = 0.05
INITIAL_ESTIMATE_ERROR = 1.0


def create_portfolio(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """
    Create portfolio using Bollinger Bands on Kalman filtered prices.
    
    Entry: Price touches lower band (oversold condition)
    Exit: Price touches upper band (overbought condition)
    
    Args:
        data: DataFrame with OHLCV data
        params: Strategy parameters dict
    
    Returns:
        vbt.Portfolio object
    """
    if params is None:
        params = {}
    
    # Extract parameters with defaults
    period = params.get("period", DEFAULT_PERIOD)
    std_dev = params.get("std_dev", DEFAULT_STD_DEV)
    process_variance = params.get("process_variance", DEFAULT_PROCESS_VARIANCE)
    measurement_variance = params.get("measurement_variance", DEFAULT_MEASUREMENT_VARIANCE)
    
    close = data["close"]
    
    # Apply Kalman filter to smooth prices
    filtered_prices = apply_kalman_filter(close, process_variance, measurement_variance)
    
    # Calculate Bollinger Bands on filtered prices
    middle_band = filtered_prices.rolling(window=period).mean()
    std = filtered_prices.rolling(window=period).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    
    # Calculate normalized position within bands (0 = lower, 1 = upper)
    band_width = upper_band - lower_band
    band_position = (filtered_prices - lower_band) / band_width
    
    # Generate entry/exit signals with tolerance
    entries = band_position <= BAND_TOLERANCE
    exits = band_position >= (1 - BAND_TOLERANCE)
    
    # Create portfolio with frequency specified
    return vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=DEFAULT_INIT_CASH,
        fees=DEFAULT_FEES,
        freq='1h'  # Specify frequency for Sharpe Ratio calculation (lowercase to avoid deprecation warning)
    )


def apply_kalman_filter(
    prices: pd.Series,
    process_variance: float = DEFAULT_PROCESS_VARIANCE,
    measurement_variance: float = DEFAULT_MEASUREMENT_VARIANCE
) -> pd.Series:
    """
    Apply 1D Kalman filter for price noise reduction.
    
    Uses simple Kalman filter with:
    - State: price level
    - Measurement: observed price
    
    Args:
        prices: Price series to filter
        process_variance: Process noise variance (system uncertainty)
        measurement_variance: Measurement noise variance (observation uncertainty)
    
    Returns:
        Filtered price series
    """
    if len(prices) < 2:
        return prices
    
    n = len(prices)
    filtered = np.zeros(n)
    filtered[0] = prices.iloc[0]
    estimate_error = INITIAL_ESTIMATE_ERROR
    
    for i in range(1, n):
        # Prediction step
        predicted_estimate = filtered[i - 1]
        predicted_error = estimate_error + process_variance
        
        # Update step
        kalman_gain = predicted_error / (predicted_error + measurement_variance)
        filtered[i] = predicted_estimate + kalman_gain * (prices.iloc[i] - predicted_estimate)
        estimate_error = (1 - kalman_gain) * predicted_error
    
    return pd.Series(filtered, index=prices.index)
