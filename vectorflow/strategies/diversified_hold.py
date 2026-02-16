#!/usr/bin/env python3
"""
Diversified Hold Strategy

Multi-asset buy and hold portfolio with equal weight allocation.
Designed for decorrelated instruments across asset classes:
Forex, Indices, Crypto, Commodities.

Logic:
    - Buy all assets at first available date
    - Hold indefinitely
    - Equal weight allocation across all instruments
    - Uses yfinance for data download

Parameters:
    - symbols: List of yfinance tickers (default: diversified basket)
    - period: Data download period (default: "2y")
"""

import pandas as pd
import vectorbt as vbt
import numpy as np
from typing import Dict, Any, List


def download_data(symbols: List[str], period: str = "2y") -> pd.DataFrame:
    """Download price data from yfinance."""
    try:
        import yfinance as yf
        data = yf.download(symbols, period=period, interval="1d", progress=False, auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]
            close.columns = [symbols[0]]
            
        return close.ffill().dropna(how="all").ffill().bfill()
    except Exception as e:
        raise ValueError(f"Failed to download data: {e}")


def create_portfolio(
    close: pd.DataFrame = None,
    params: Dict[str, Any] = None,
    **kwargs
) -> vbt.Portfolio:
    """
    Create a VectorBT Portfolio for the Diversified Hold strategy.
    
    Args:
        close: DataFrame of close prices (if None, downloads from yfinance)
        params: Strategy parameters including 'symbols' and 'period'
        **kwargs: Additional arguments passed to from_signals
        
    Returns:
        VectorBT Portfolio object
    """
    if params is None:
        params = {}
        
    # Default diversified basket
    default_symbols = [
        "EURUSD=X",     # Forex
        "USDJPY=X",     # Forex
        "^GSPC",        # S&P 500
        "BTC-USD",      # Bitcoin
        "GC=F",         # Gold
        "CL=F",         # Crude Oil
        "KC=F",         # Coffee
    ]
    
    symbols = params.get("symbols", default_symbols)
    period = params.get("period", "2y")
    
    # Download data if not provided
    if close is None:
        close = download_data(symbols, period)
    
    # Generate entry signals (buy once at first valid price)
    entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    for col in close.columns:
        first_valid = close[col].first_valid_index()
        if first_valid is not None:
            entries.loc[first_valid, col] = True
    
    # No exits (hold forever)
    exits = pd.DataFrame(False, index=close.index, columns=close.columns)
    
    # Equal weights
    weights = np.ones(len(close.columns)) / len(close.columns)
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        size=weights,
        size_type="targetpercent",
        cash_sharing=True,
        freq="1d",
        **kwargs
    )
    
    return portfolio
