#!/usr/bin/env python3
"""CMO Strategy - Chande Momentum Oscillator with Trailing Stop"""

import pandas as pd
from typing import Dict
import vectorbt as vbt


def calculate_cmo(close: pd.Series, length: int = 9) -> pd.Series:
    """Calculate Chande Momentum Oscillator."""
    # Calculate momentum (absolute price change)
    mom = abs(close - close.shift(1))

    # Calculate SMA of momentum
    sma_mom = mom.rolling(window=length).mean()

    # Calculate momentum over length periods
    mom_length = close - close.shift(length)

    # Calculate CMO
    cmo = 100 * (mom_length / (sma_mom * length))

    return cmo


def create_portfolio(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """Create CMO strategy portfolio directly."""
    if params is None:
        params = {}

    # Parameters
    length = params.get("length", 9)
    top_band = params.get("top_band", 70)
    low_band = params.get("low_band", -70)
    reverse = params.get("reverse", False)
    trail_pct = params.get("trail_pct", 0.02)

    # Calculate CMO
    cmo = calculate_cmo(data["close"], length)

    # Generate position signals based on CMO
    position = pd.Series(0, index=data.index)

    for i in range(1, len(cmo)):
        if pd.notna(cmo.iloc[i]):
            if cmo.iloc[i] > top_band:
                position.iloc[i] = 1
            elif cmo.iloc[i] <= low_band:
                position.iloc[i] = -1
            else:
                position.iloc[i] = position.iloc[i - 1]  # Hold previous position

    # Apply reverse logic if enabled
    if reverse:
        position = position * -1

    # Generate entry/exit signals
    entries = (position == 1) & (position.shift(1) != 1)
    short_entries = (position == -1) & (position.shift(1) != -1)
    
    # Simple exits when position changes
    exits = (position != 1) & (position.shift(1) == 1)
    short_exits = (position != -1) & (position.shift(1) == -1)

    # Create portfolio with VBT's native trailing stop
    portfolio = vbt.Portfolio.from_signals(
        close=data["close"],
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=10000,
        fees=0.001,
        sl_trail=trail_pct,  # VBT native trailing stop
    )

    return portfolio

