#!/usr/bin/env python3
"""
Risk Parity Portfolio Strategy

Multi-asset portfolio construction using risk parity allocation.
Generates adaptive signals based on volatility-adjusted momentum
and allocates using inverse volatility weighting.

Logic:
    1. Generate trend signals (Z-score of price vs MA)
    2. Apply volatility regime filter
    3. Allocate using inverse volatility (risk parity)
    4. Normalize weights and apply directional bias

Parameters:
    - signal_window: Lookback for trend calculation (default: 50)
    - vol_lookback: Rolling window for volatility (default: 20)
    - vol_threshold: High vol penalty threshold multiplier (default: 2.0)
"""

import pandas as pd
import vectorbt as vbt
import numpy as np
from typing import Dict, Any


def generate_signals(close: pd.DataFrame, window: int = 50, vol_threshold: float = 2.0) -> pd.DataFrame:
    """Generate adaptive signals based on trend and volatility regime."""
    signals = pd.DataFrame(index=close.index, columns=close.columns)
    
    for col in close.columns:
        price = close[col]
        
        # Trend component: Z-score of price vs MA
        ma = price.rolling(window=window).mean()
        std = price.rolling(window=window).std()
        trend_score = (price - ma) / std
        
        # Volatility regime filter
        vol_ma = std.rolling(window=window * 2).mean()
        vol_regime = np.where(std > vol_ma * vol_threshold, 0.5, 1.0)
        
        # Combined signal
        raw_signal = trend_score * vol_regime
        signals[col] = np.clip(raw_signal, -1.0, 1.0)
        
    return signals


def allocate_weights(close: pd.DataFrame, signals: pd.DataFrame, vol_lookback: int = 20) -> pd.DataFrame:
    """Allocate portfolio weights using risk parity (inverse volatility)."""
    returns = close.pct_change()
    asset_vol = returns.rolling(vol_lookback).std()
    
    # Inverse volatility
    inv_vol = 1 / asset_vol.replace(0, np.nan)
    
    # Combine signal strength with inverse vol
    raw_weights = inv_vol * signals.abs()
    
    # Normalize to sum to 1
    total_weight = raw_weights.sum(axis=1)
    normalized = raw_weights.div(total_weight, axis=0)
    
    # Apply direction
    final_weights = normalized * np.sign(signals)
    
    return final_weights.fillna(0)


def create_portfolio(
    close: pd.DataFrame,
    params: Dict[str, Any] = None,
    **kwargs
) -> vbt.Portfolio:
    """
    Create a VectorBT Portfolio for the Risk Parity strategy.
    
    Args:
        close: DataFrame of close prices (multi-asset)
        params: Strategy parameters
        **kwargs: Additional arguments passed to from_orders
        
    Returns:
        VectorBT Portfolio object
    """
    if params is None:
        params = {}
        
    signal_window = params.get("signal_window", 50)
    vol_lookback = params.get("vol_lookback", 20)
    vol_threshold = params.get("vol_threshold", 2.0)
    
    # Generate signals and weights
    signals = generate_signals(close, signal_window, vol_threshold)
    weights = allocate_weights(close, signals, vol_lookback)
    
    # Create portfolio using from_orders for continuous rebalancing
    portfolio = vbt.Portfolio.from_orders(
        close=close,
        size=weights,
        size_type="targetpercent",
        **kwargs
    )
    
    return portfolio
