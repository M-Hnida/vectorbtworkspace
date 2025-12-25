#!/usr/bin/env python3
"""
Strategy Template
=================

Copy this file to create a new strategy. Implement the create_portfolio function.

Required:
- create_portfolio(data, params) -> vbt.Portfolio

Tips:
- Use from_signals() with direction="both" for long/short (avoid size_type="percent")
- Use edge detection for entries: signal & ~signal.shift(1).fillna(False)
- Load parameters from YAML config, not hardcoded defaults
"""

import pandas as pd
import vectorbt as vbt
import pandas_ta as ta
from typing import Dict, Union, Optional


def create_portfolio(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
    params: Optional[Dict] = None
) -> "vbt.Portfolio":
    """
    Create portfolio using your strategy logic.
    
    Args:
        data: OHLCV DataFrame or dict of DataFrames (multi-timeframe)
        params: Parameters from YAML config
        
    Returns:
        vbt.Portfolio with backtest results
    """
    if params is None:
        params = {}
    
    # Handle multi-timeframe data
    if isinstance(data, dict):
        df = data.get(params.get("primary_timeframe", "1h"), next(iter(data.values()))).copy()
    else:
        df = data.copy()
    
    close = df["close"]
    
    # Example: Simple MA crossover
    fast_ma = ta.sma(close, length=params.get("fast_period", 10))
    slow_ma = ta.sma(close, length=params.get("slow_period", 30))
    
    # Entry signals (edge detection)
    long_condition = fast_ma > slow_ma
    short_condition = fast_ma < slow_ma
    
    long_entry = long_condition & ~long_condition.shift(1).fillna(False)
    short_entry = short_condition & ~short_condition.shift(1).fillna(False)
    
    # Exit on opposite signal
    long_exit = short_entry
    short_exit = long_entry
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=long_entry,
        exits=long_exit,
        short_entries=short_entry,
        short_exits=short_exit,
        init_cash=params.get("initial_cash", 10000),
        fees=params.get("fee", 0.001),
        freq=params.get("freq", "1h"),
        direction="both",
    )
    
    return portfolio
