#!/usr/bin/env python3
"""
StdDev Grid Strategy
Uses Standard Deviation bands as Support/Resistance levels for a grid trading approach.
"""

import pandas as pd
import vectorbt as vbt
import pandas_ta as ta
from typing import Dict, Optional, Union


def create_portfolio(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], params: Optional[Dict] = None
) -> "vbt.Portfolio":
    """
    Create StdDev Grid Strategy portfolio.

    The strategy calculates a Moving Average and Standard Deviation.
    It creates grid levels at multiples of the Standard Deviation around the MA.

    - Buy signals are generated when price crosses below lower bands (Support).
    - Sell signals are generated when price crosses above upper bands (Resistance).
    """
    # =========================================================================
    # PARAMETER INITIALIZATION
    # =========================================================================
    if params is None:
        params = {}

    # Strategy parameters
    period = params.get("period", 20)
    std_dev_step = params.get("std_dev_step", 1.0)  # Step size in StdDevs
    grid_levels = params.get("grid_levels", 3)  # Number of levels on each side

    # Trading parameters
    initial_cash = params.get("initial_cash", 10000)
    fee_pct = params.get("fee", 0.001)
    freq = params.get("freq", "1H")

    # =========================================================================
    # DATA HANDLING
    # =========================================================================
    if isinstance(data, dict):
        primary_tf = params.get("primary_timeframe", "1h")
        if primary_tf in data:
            df = data[primary_tf]
        else:
            df = next(iter(data.values()))
    else:
        df = data

    close = df["close"]

    # =========================================================================
    # INDICATOR CALCULATION
    # =========================================================================

    # Calculate Baseline (MA) and Volatility (StdDev)
    # Using pandas-ta or vectorbt
    ma = ta.sma(close, length=period)
    if ma is None:  # Fallback if data is too short
        ma = close.rolling(window=period).mean()

    std = ta.stdev(close, length=period)
    if std is None:
        std = close.rolling(window=period).std()

    # Handle NaN values at the beginning
    ma = ma.fillna(method="bfill")
    std = std.fillna(method="bfill")

    # =========================================================================
    # SIGNAL GENERATION (GRID LOGIC)
    # =========================================================================

    # We will use a loop to simulate the grid logic statefully
    # This allows for "accumulating" positions as we go down the grid

    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)

    # To properly simulate a grid with vectorbt's from_signals, we might need to use 'accumulate=True'
    # and manage sizing, OR just use simple entry/exit signals.
    # For a robust grid, we often want to buy at Level 1, Buy again at Level 2, etc.
    # And sell Level 2 at Level 1, Sell Level 1 at MA, etc.

    # Simplified Grid Logic for this implementation:
    # - Buy if price crosses below a support band.
    # - Sell if price crosses above a resistance band.
    # - We will use 'accumulate=True' in the portfolio to allow multiple buys.

    # Pre-calculate bands to avoid doing it in the loop
    # Support levels: MA - 1*Std, MA - 2*Std...
    # Resistance levels: MA + 1*Std, MA + 2*Std...

    # We can vectorize the crossing detection

    # Initialize signal arrays
    buy_signals = pd.DataFrame(
        False, index=close.index, columns=range(1, grid_levels + 1)
    )
    sell_signals = pd.DataFrame(
        False, index=close.index, columns=range(1, grid_levels + 1)
    )

    for i in range(1, grid_levels + 1):
        # Define levels
        upper_band = ma + (i * std_dev_step * std)
        lower_band = ma - (i * std_dev_step * std)

        # Buy at Support (Lower Band)
        # Condition: Price crosses below lower band
        # Using 'low' to catch wicks, or 'close' for confirmation. Let's use 'close' for safety.
        # buy_signals[i] = (close < lower_band) & (close.shift(1) >= lower_band.shift(1))

        # Actually, for a grid, we might want to be "in" if price is below.
        # But let's stick to "entry signal" logic.
        buy_signals[i] = (close < lower_band) & (
            close.shift(1) >= lower_band
        )  # Crossover

        # Sell at Resistance (Upper Band)
        # sell_signals[i] = (close > upper_band) & (close.shift(1) <= upper_band)

        # Alternative Exit: Sell everything if we return to Mean?
        # Or Sell specific grid layers?
        # Let's implement: Buy at Supports, Sell at Resistances.
        sell_signals[i] = (close > upper_band) & (close.shift(1) <= upper_band)

    # Combine signals
    # If any level triggers a buy, we enter.
    # If any level triggers a sell, we exit (or reduce).

    # To make it a true grid (accumulate), we need to handle the signals carefully.
    # If we just OR them, we might miss simultaneous triggers (unlikely with close).

    entries = buy_signals.any(axis=1)
    exits = sell_signals.any(axis=1)

    # Optional: Add a "Take Profit at MA" rule?
    # exits = exits | ((close > ma) & (close.shift(1) <= ma))

    # =========================================================================
    # PORTFOLIO CREATION
    # =========================================================================

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fee_pct,
        freq=freq,
        direction="longonly",
        accumulate=True,  # Allow building a position (Grid behavior)
        # size=1.0 / grid_levels, # Divide capital? Or fixed amount?
        # Let's use a fixed size or amount per trade.
        # If we want to deploy 100% over 'grid_levels', size should be 1/grid_levels (approx)
        size=1.0 / (grid_levels * 2),  # Conservative sizing
        size_type="percent",
    )

    return portfolio
