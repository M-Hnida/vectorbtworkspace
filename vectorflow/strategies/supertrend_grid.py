#!/usr/bin/env python3
"""Supertrend with Grid Strategy - Vectorbt Implementation"""

import pandas as pd
import numpy as np
from typing import Dict
import vectorbt as vbt
import pandas_ta as ta


def create_portfolio(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """
    Create Supertrend with Grid Strategy portfolio
    """
    if params is None:
        params = {}

    # Strategy parameters
    st_period = params.get("st_period", 10)
    st_multiplier = params.get("st_multiplier", 3.0)
    grid_levels = params.get("grid_levels", 5)
    grid_range = params.get("grid_range", 0.05)  # 5%
    initial_cash = params.get("initial_cash", 10000)
    fee_pct = params.get("fee", 0.001)
    freq = params.get("freq", "1H")

    # Extract price data
    high = data["high"]
    low = data["low"]
    close = data["close"]
    
    # Calculate Supertrend using pandas-ta
    supertrend_data = ta.supertrend(high, low, close, length=st_period, multiplier=st_multiplier)
    
    # Check if supertrend calculation was successful
    if supertrend_data is None or supertrend_data.empty:
        print(f"⚠️ Supertrend calculation failed - insufficient data or invalid parameters")
        print(f"   Data length: {len(close)}, Period: {st_period}, Multiplier: {st_multiplier}")
        return None
    
    # Extract supertrend and direction
    supertrend = supertrend_data.iloc[:, 0]
    trend_direction = supertrend_data.iloc[:, 1]
    
    # Generate Supertrend signals
    bullish_crossover = (trend_direction == 1) & (trend_direction.shift(1) == -1)
    bearish_crossover = (trend_direction == -1) & (trend_direction.shift(1) == 1)
    
    # Initialize signals
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    
    # Grid state tracking
    last_sell_price = 0.0
    grid_levels_array = np.array([])
    
    # Generate signals for all data points
    for i in range(1, len(close)):
        current_price = close.iloc[i]
        
        # Entry signal - Supertrend bullish crossover
        if bullish_crossover.iloc[i]:
            entries.iloc[i] = True
            last_sell_price = 0.0
            grid_levels_array = np.array([])
        
        # Exit signal - Supertrend bearish crossover
        elif bearish_crossover.iloc[i]:
            exits.iloc[i] = True
            # Set up grid below sell price
            last_sell_price = current_price
            step = (last_sell_price * grid_range) / grid_levels
            grid_levels_array = np.array([last_sell_price - (j * step) for j in range(1, grid_levels + 1)])
            
        # Grid re-entry - only when profitable
        elif last_sell_price > 0 and len(grid_levels_array) > 0:
            for grid_level in grid_levels_array:
                # Price crosses grid level from below AND is below sell price
                if (current_price > grid_level and close.iloc[i-1] <= grid_level and 
                    current_price < last_sell_price):
                    entries.iloc[i] = True
                    last_sell_price = 0.0
                    grid_levels_array = np.array([])
                    break
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fee_pct,
        freq=freq,
        direction="longonly",
    )

    return portfolio