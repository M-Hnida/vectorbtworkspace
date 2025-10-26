#!/usr/bin/env python3
"""RSI Strategy - Relative Strength Index with overbought/oversold levels."""

from typing import Dict, Optional
import pandas as pd
import vectorbt as vbt


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def create_portfolio(data: pd.DataFrame, params: Optional[Dict] = None) -> "vbt.Portfolio":
    """Create RSI strategy portfolio directly."""
    if params is None:
        params = {}

    # Parameters - match config file names
    rsi_period = params.get("rsi_period", 14)
    oversold_level = params.get("oversold_level", 30)
    overbought_level = params.get("overbought_level", 70)

    # Calculate RSI
    rsi = calculate_rsi(data["close"], rsi_period)

    # Generate entry/exit signals based on RSI levels
    # Buy when RSI crosses above oversold level (bullish)
    entries = (rsi > oversold_level) & (rsi.shift(1) <= oversold_level)
    
    # Sell when RSI crosses below overbought level (bearish)
    exits = (rsi < overbought_level) & (rsi.shift(1) >= overbought_level)

    # Create portfolio with frequency specified
    portfolio = vbt.Portfolio.from_signals(
        close=data["close"],
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=0.001,
        freq='1h'  # Specify frequency for Sharpe Ratio calculation (lowercase to avoid deprecation warning)
    )

    return portfolio

