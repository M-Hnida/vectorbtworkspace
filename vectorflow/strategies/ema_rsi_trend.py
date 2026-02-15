#!/usr/bin/env python3
"""
EMA-RSI Trend Following Strategy

Entry: On first bar when all conditions align
Exit: On trend reversal (EMA crossover)
"""

import pandas as pd
import vectorbt as vbt
import pandas_ta as ta
from typing import Dict, Union, Optional


def calculate_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Calculate technical indicators."""
    df["ema_short"] = ta.ema(df["close"], length=params.get("ema_short_period", 34))
    df["ema_long"] = ta.ema(df["close"], length=params.get("ema_long_period", 200))
    
    rsi = ta.rsi(df["close"], length=params.get("rsi_period", 14))
    df["rsi_smoothed"] = ta.ema(rsi, length=params.get("rsi_smooth_period", 5))
    
    macd_fast = params.get("macd_fast", 12)
    macd_slow = params.get("macd_slow", 26) 
    macd_signal = params.get("macd_signal", 9)
    macd = ta.macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    
    if macd is not None and f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}" in macd.columns:
        df["macd_histogram"] = macd[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"]
    else:
        df["macd_histogram"] = 0.0
    
    return df


def create_portfolio(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
    params: Optional[Dict] = None
) -> "vbt.Portfolio":
    """Create single portfolio with long and short signals."""
    if params is None:
        params = {}
    
    # Handle data
    if isinstance(data, dict):
        primary_tf = params.get("primary_timeframe", "1h")
        df = data.get(primary_tf, next(iter(data.values()))).copy()
    else:
        df = data.copy()
    
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df = calculate_indicators(df, params)
    df = df.dropna()
    close = df["close"]
    
    # Parameters
    rsi_neutral = params.get("rsi_neutral", 50)
    require_macd = params.get("require_macd_confirmation", True)
    
    # Trend conditions
    bullish_trend = df["ema_short"] > df["ema_long"]
    bearish_trend = df["ema_short"] < df["ema_long"]
    
    # RSI conditions  
    rsi_bullish = df["rsi_smoothed"] > rsi_neutral
    rsi_bearish = df["rsi_smoothed"] < rsi_neutral
    
    # MACD conditions
    if require_macd:
        macd_bull = df["macd_histogram"] > 0
        macd_bear = df["macd_histogram"] < 0
    else:
        macd_bull = pd.Series(True, index=df.index)
        macd_bear = pd.Series(True, index=df.index)
    
    # Entry conditions - all must align
    long_condition = bullish_trend & rsi_bullish & macd_bull
    short_condition = bearish_trend & rsi_bearish & macd_bear
    
    # ENTRY: Edge detection - only signal on first bar
    long_entry = long_condition & ~long_condition.shift(1).fillna(False)
    short_entry = short_condition & ~short_condition.shift(1).fillna(False)
    
    # EXIT: On EMA trend reversal
    long_exit = bearish_trend & ~bearish_trend.shift(1).fillna(False)
    short_exit = bullish_trend & ~bullish_trend.shift(1).fillna(False)
    
    print(f"[EMA-RSI] Long: {long_entry.sum()}, Short: {short_entry.sum()}")
    
    # Single portfolio with direction="both"
    # Use default size (not percent) to avoid reversal issue
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
