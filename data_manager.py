#!/usr/bin/env python3
"""
Simple OHLC data loader - does one thing well.
"""

import pandas as pd
from typing import Dict, List

# Simple mappings
TIMEFRAMES = {
    '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
    '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
}

def load_ohlc_csv(file_path: str) -> pd.DataFrame:
    """Load and clean OHLC CSV data."""
    # Read file, auto-detect separator, no headers
    try:
        df = pd.read_csv(file_path, sep=None, header=None, parse_dates=[0], index_col=0, 
                         date_format='mixed', engine='python')
    except pd.errors.EmptyDataError:
        raise ValueError("File is empty or could not be parsed")
    
    # Standard column names (take what we need, ignore extras)
    columns = ['open', 'high', 'low', 'close', 'volume']
    df.columns = columns[:len(df.columns)]
    
    # Validate required columns
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Return OHLC (+ volume if available)
    available = [col for col in columns if col in df.columns]
    return df[available].dropna().sort_index()

def load_symbol_data(file_path: str, timeframes: List[str] = ['1h']) -> Dict[str, pd.DataFrame]:
    """Load symbol data for multiple timeframes."""
    base_data = load_ohlc_csv(file_path)
    
    return {
        tf: base_data
        for tf in timeframes
    }