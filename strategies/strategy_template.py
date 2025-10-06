#!/usr/bin/env python3
"""Strategy Template for the Trading System"""

import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
import numpy as np
from typing import Dict


def create_portfolio(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """
    Create a portfolio using strategy signals.
    
    This is a template function that demonstrates the expected signature
    for all strategy implementations in this system.
    
    Args:
        data: DataFrame with OHLCV data (open, high, low, close, volume)
        params: Dictionary of strategy parameters from YAML config
        
    Returns:
        vbt.Portfolio: VectorBT portfolio object
    """
    if params is None:
        params = {}

    # Strategy parameters - load from config with defaults
    # Example parameters (replace with your strategy's parameters):
    example_period = params.get("example_period", 14)
    example_threshold = params.get("example_threshold", 70)
    
    # Trading parameters
    initial_cash = params.get("initial_cash", 10000)
    fee_pct = params.get("fee", 0.001)
    freq = params.get("freq", "1H")

    # Extract price data
    open = data["open"]
    high = data["high"]
    low = data["low"]
    close = data["close"]
    volume = data["volume"]
    
    # ===== INDICATOR CALCULATION =====
    # Add your technical indicators here using pandas-ta
    # Example:
    # data.ta.rsi(length=example_period, append=True)
    # rsi_col = f'RSI_{example_period}'
    # rsi = data[rsi_col]
    
    # ===== SIGNAL GENERATION =====
    # Create entry and exit signals based on your strategy logic
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    
    # Example signal logic (replace with your strategy):
    # entries = (rsi < example_threshold) & (rsi.shift(1) >= example_threshold)
    # exits = (rsi > (100 - example_threshold)) & (rsi.shift(1) <= (100 - example_threshold))
    
    # ===== PORTFOLIO CREATION =====
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fee_pct,
        freq=freq,
        direction="longonly",  # or "shortonly" or "both"
    )

    return portfolio