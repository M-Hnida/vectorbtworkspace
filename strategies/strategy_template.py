#!/usr/bin/env python3
"""
Strategy Template for the Trading System

This template follows VectorBT best practices and integrates with the
trading system's optimization and analysis pipeline.

Key Features:
- Proper frequency handling (prevents common VectorBT errors)
- Signal validation and cleaning
- Multi-timeframe support pattern
- Pandas-ta integration examples
- Error handling
- Advanced portfolio settings
"""

import pandas as pd
import pandas_ta as ta  # noqa: F401 - Available for strategy implementations
import vectorbt as vbt
import numpy as np  # noqa: F401 - Available for strategy implementations
from typing import Dict, Union, Optional


def create_portfolio(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], params: Optional[Dict] = None
) -> "vbt.Portfolio":
    """
    Create a portfolio using strategy signals.

    This function implements the standard interface for all strategies in the system.
    It supports both single-timeframe (DataFrame) and multi-timeframe (Dict) data.

    Args:
        data: Either:
              - pd.DataFrame with OHLCV columns (single timeframe)
              - Dict[str, pd.DataFrame] with timeframe keys (multi-timeframe)
        params: Dictionary of strategy parameters from YAML config

    Returns:
        vbt.Portfolio: VectorBT portfolio object with backtest results

    Example Config (config/strategy_name.yaml):
        parameters:
          example_period: 14
          example_threshold: 70
          initial_cash: 10000
          fee: 0.001
          freq: "1H"
          primary_timeframe: "1h"

        optimization_grid:
          example_period: [10, 20, 5]  # [start, end, step]
          example_threshold: [60, 80, 10]
    """
    # =========================================================================
    # PARAMETER INITIALIZATION
    # =========================================================================
    if params is None:
        params = {}

    # Strategy-specific parameters (customize these for your strategy)
    # Uncomment and use these in your indicator calculations:
    example_period = params.get("example_period", 14)  # noqa: F841 - Template variable
    # example_threshold = params.get("example_threshold", 70)

    # Trading parameters (standard across all strategies)
    initial_cash = params.get("initial_cash", 10000)
    fee_pct = params.get("fee", 0.001)
    freq = params.get("freq", "1H")  # CRITICAL: Always set frequency

    # =========================================================================
    # DATA HANDLING (Single vs Multi-Timeframe)
    # =========================================================================

    # Check if multi-timeframe data (Dict) or single timeframe (DataFrame)
    if isinstance(data, dict):
        # Multi-timeframe strategy pattern
        primary_tf = params.get("primary_timeframe", "1h")

        # Get primary timeframe data
        if primary_tf in data:
            df = data[primary_tf]
        else:
            # Fallback to first available timeframe
            df = next(iter(data.values()))

        # Access other timeframes if needed:
        # higher_tf_data = data.get("4h", df)  # Fallback to primary if not available
    else:
        # Single timeframe strategy
        df = data

    # Validate required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract price data (uncomment as needed for your strategy)
    # open_price = df["open"]
    # high = df["high"]
    # low = df["low"]
    close = df["close"]
    # volume = df["volume"]

    # =========================================================================
    # INDICATOR CALCULATION
    # =========================================================================

    # Method 1: Using pandas-ta (recommended for most indicators)
    # df.ta.rsi(length=example_period, append=True)
    # rsi = df[f'RSI_{example_period}']

    # Method 2: Using VectorBT indicators (faster for optimization)
    # rsi_indicator = vbt.RSI.run(close, window=example_period)
    # rsi = rsi_indicator.rsi

    # Method 3: Using TA-Lib via VectorBT (for multi-timeframe)
    # rsi = vbt.talib("RSI", timeperiod=example_period).run(close, skipna=True).real

    # Example: Calculate RSI using VectorBT (uncomment to use)
    # rsi_indicator = vbt.RSI.run(close, window=example_period)
    # rsi = rsi_indicator.rsi

    # Example: Calculate Bollinger Bands
    # bb = vbt.BBANDS.run(close, window=20, alpha=2)
    # bb_upper = bb.upper
    # bb_lower = bb.lower

    # Example: Calculate Moving Averages
    # fast_ma = vbt.MA.run(close, window=10).ma
    # slow_ma = vbt.MA.run(close, window=50).ma

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    # Initialize signal arrays (all False by default)
    entries = pd.Series(False, index=close.index, dtype=bool)
    exits = pd.Series(False, index=close.index, dtype=bool)

    # Example Strategy Logic (replace with your strategy):
    # RSI Mean Reversion Example:
    # entries = (rsi < example_threshold) & (rsi.shift(1) >= example_threshold)
    # exits = (rsi > (100 - example_threshold)) & (rsi.shift(1) <= (100 - example_threshold))

    # MA Crossover Example:
    # entries = fast_ma.vbt.crossed_above(slow_ma)
    # exits = fast_ma.vbt.crossed_below(slow_ma)

    # Bollinger Band Example:
    # entries = close < bb_lower
    # exits = close > bb_upper

    # Multi-condition Example:
    # condition1 = rsi < 30
    # condition2 = close > slow_ma
    # entries = condition1 & condition2
    # exits = rsi > 70

    # =========================================================================
    # SIGNAL VALIDATION & CLEANING
    # =========================================================================

    # Ensure signals are boolean type
    entries = entries.astype(bool)
    exits = exits.astype(bool)

    # Clean signals to remove consecutive duplicates (optional but recommended)
    # entries, exits = entries.vbt.signals.clean(exits)

    # Validate signal alignment
    if len(entries) != len(close):
        raise ValueError(
            f"Signal length mismatch: entries={len(entries)}, close={len(close)}"
        )

    # Check for any signals (warn if none)
    if not entries.any():
        print("⚠️ Warning: No entry signals generated for strategy")

    # =========================================================================
    # PORTFOLIO CREATION
    # =========================================================================

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        # Capital management
        init_cash=initial_cash,
        # Position sizing (choose one approach)
        size=1.0,  # Fixed size
        size_type="percent",  # 'amount', 'percent', 'targetpercent'
        # Transaction costs
        fees=fee_pct,  # Percentage fee per trade
        # fixed_fees=1.0,            # Fixed fee per trade (optional)
        # slippage=0.001,            # Slippage percentage (optional)
        # Risk management (optional)
        # sl_stop=0.05,              # 5% stop loss
        # tp_stop=0.10,              # 10% take profit
        # sl_trail=True,             # Trailing stop loss
        # Execution settings
        direction="longonly",  # 'longonly', 'shortonly', 'both'
        conflict_mode="opposite",  # How to handle entry/exit conflicts
        accumulate=False,  # Don't accumulate positions
        # CRITICAL: Always specify frequency to avoid errors
        freq=freq,
    )

    return portfolio


# ============================================================================
# OPTIONAL: Helper Functions for Complex Strategies
# ============================================================================


def calculate_custom_indicator(close: pd.Series, window: int) -> pd.Series:
    """
    Example custom indicator calculation.

    Args:
        close: Close price series
        window: Lookback window

    Returns:
        Custom indicator values
    """
    # Example: Custom momentum indicator
    momentum = close.pct_change(window)
    normalized = (momentum - momentum.rolling(window * 2).mean()) / momentum.rolling(
        window * 2
    ).std()
    return normalized


def generate_multi_condition_signals(
    df: pd.DataFrame, conditions: Dict[str, pd.Series]
) -> tuple[pd.Series, pd.Series]:
    """
    Generate signals based on multiple conditions.

    Args:
        df: OHLCV DataFrame
        conditions: Dictionary of condition name -> boolean Series

    Returns:
        Tuple of (entries, exits)
    """
    # Combine all entry conditions with AND
    entries = pd.Series(True, index=df.index)
    for name, condition in conditions.items():
        if "entry" in name:
            entries = entries & condition

    # Combine all exit conditions with OR
    exits = pd.Series(False, index=df.index)
    for name, condition in conditions.items():
        if "exit" in name:
            exits = exits | condition

    return entries, exits


# ============================================================================
# STRATEGY METADATA (Optional but Recommended)
# ============================================================================

STRATEGY_INFO = {
    "name": "Strategy Template",
    "version": "1.0",
    "author": "Your Name",
    "description": "Template strategy demonstrating best practices",
    "required_timeframes": ["1h"],
    "required_columns": ["open", "high", "low", "close", "volume"],
    "default_parameters": {
        "example_period": 14,
        "example_threshold": 70,
        "initial_cash": 10000,
        "fee": 0.001,
        "freq": "1H",
    },
}
