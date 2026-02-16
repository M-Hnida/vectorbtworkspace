#!/usr/bin/env python3
"""
Kalman MA Strategy

Trend following strategy using Kalman Filter for price smoothing 
followed by Moving Average crossover. Includes volatility targeting 
for dynamic position sizing.

Logic:
    1. Apply Kalman Filter to smooth price noise
    2. Calculate MA on filtered price
    3. Long when Kalman > MA (uptrend)
    4. Position size based on volatility targeting

Parameters:
    - kalman_q: Process noise (default: 1e-5)
    - kalman_r: Measurement noise (default: 0.001)
    - ma_period: Moving average period (default: 225)
    - target_volatility: Target annualized volatility (default: 0.25)
    - vol_window: Rolling window for vol calculation (default: 10)
"""

import pandas as pd
import vectorbt as vbt
import numpy as np
from numba import njit
from typing import Dict, Any


@njit
def kalman_filter(data: np.ndarray, q: float = 1e-5, r: float = 0.001) -> np.ndarray:
    """
    Numba-accelerated Kalman Filter for price smoothing.
    
    Args:
        data: Price array (close prices)
        q: Process noise covariance
        r: Measurement noise covariance
        
    Returns:
        Filtered price array
    """
    n = len(data)
    xhat = np.zeros(n)
    p = np.zeros(n)
    xhat_minus = np.zeros(n)
    p_minus = np.zeros(n)
    k = np.zeros(n)
    
    xhat[0] = data[0]
    p[0] = 1.0
    
    for i in range(1, n):
        # Prediction
        xhat_minus[i] = xhat[i-1]
        p_minus[i] = p[i-1] + q
        
        # Update
        k[i] = p_minus[i] / (p_minus[i] + r)
        xhat[i] = xhat_minus[i] + k[i] * (data[i] - xhat_minus[i])
        p[i] = (1 - k[i]) * p_minus[i]
        
    return xhat


def create_portfolio(
    close: pd.Series,
    params: Dict[str, Any] = None,
    **kwargs
) -> vbt.Portfolio:
    """
    Create a VectorBT Portfolio for the Kalman MA strategy.
    
    Args:
        close: Price series
        params: Strategy parameters dict
        **kwargs: Additional arguments passed to from_signals
        
    Returns:
        VectorBT Portfolio object
    """
    if params is None:
        params = {}
    
    # Parameters with defaults
    kalman_q = params.get("kalman_q", 1e-5)
    kalman_r = params.get("kalman_r", 0.001)
    ma_period = params.get("ma_period", 225)
    target_vol = params.get("target_volatility", 0.25)
    vol_window = params.get("vol_window", 10)
    
    # Calculate returns for volatility
    returns = close.pct_change()
    
    # Calculate annualization factor (assuming 4h timeframe, 24/7 markets)
    # 365 days * 6 bars/day = 2190 bars/year
    annualization = np.sqrt(2190)
    
    # Apply Kalman Filter
    kalman_price = kalman_filter(close.values, q=kalman_q, r=kalman_r)
    kalman_series = pd.Series(kalman_price, index=close.index)
    
    # Calculate MA on Kalman-filtered price
    ma = kalman_series.rolling(window=ma_period).mean()
    
    # Generate signals
    price_above_ma = kalman_series > ma
    
    long_entries = price_above_ma & ~price_above_ma.shift(1).fillna(False)
    long_exits = ~price_above_ma & price_above_ma.shift(1).fillna(False)
    
    # Volatility targeting sizing
    rolling_vol = returns.rolling(window=vol_window).std() * annualization
    vol_weights = (target_vol / rolling_vol).shift(1)
    vol_weights = vol_weights.fillna(0).replace([np.inf, -np.inf], 0).clip(upper=1.0)
    
    # Apply sizing only when in position
    sizing = vol_weights.where(price_above_ma, 0)
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=long_entries,
        exits=long_exits,
        size=sizing,
        size_type="percent",
        freq="4h",
        **kwargs
    )
    
    return portfolio
