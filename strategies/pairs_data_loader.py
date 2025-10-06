#!/usr/bin/env python3
"""Specialized data loader for pairs trading strategies"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from data_manager import _load_ccxt_data_internal


def load_pairs_data(symbols: List[str], timeframe: str = "1h", time_range: str = "6m", 
                   exchange: str = "binance") -> pd.DataFrame:
    """
    Load and align data for multiple symbols for pairs trading.
    
    Args:
        symbols: List of symbols to load (e.g., ["SOL/USDT", "BTC/USDT"])
        timeframe: Data timeframe
        time_range: Time range to load
        exchange: Exchange name
        
    Returns:
        DataFrame with aligned data for all symbols
    """
    
    # Load data for all symbols
    data = _load_ccxt_data_internal(
        exchange_name=exchange,
        symbols=symbols,
        timeframes=[timeframe],
        time_range=time_range,
        sandbox=False
    )
    
    if not data:
        raise ValueError("No data loaded from exchange")
    
    # Align all symbol data by timestamp
    aligned_data = pd.DataFrame()
    
    for symbol in symbols:
        # Clean symbol name for dict key
        clean_symbol = symbol.replace("/", "").replace(":", "")
        
        if clean_symbol in data and timeframe in data[clean_symbol]:
            symbol_data = data[clean_symbol][timeframe]
            
            # Add symbol prefix to columns
            symbol_prefix = symbol.split("/")[0].lower()  # e.g., "sol" from "SOL/USDT"
            
            for col in symbol_data.columns:
                aligned_data[f"{symbol_prefix}_{col}"] = symbol_data[col]
    
    # Drop rows with any NaN values to ensure alignment
    aligned_data = aligned_data.dropna()
    
    if aligned_data.empty:
        raise ValueError("No aligned data available after processing")
    
    print(f"✅ Loaded pairs data: {len(aligned_data)} aligned bars")
    print(f"📅 Data range: {aligned_data.index[0]} to {aligned_data.index[-1]}")
    print(f"📊 Symbols: {list(aligned_data.columns)}")
    
    return aligned_data


def create_synthetic_pairs_data(primary_data: pd.DataFrame, correlation: float = 0.7) -> pd.DataFrame:
    """
    Create synthetic pairs data for testing when only one symbol is available.
    
    Args:
        primary_data: Primary symbol data (e.g., SOL)
        correlation: Target correlation between assets
        
    Returns:
        DataFrame with both assets
    """
    np.random.seed(42)  # For reproducible results
    
    # Calculate returns of primary asset
    primary_returns = primary_data['close'].pct_change().fillna(0)
    
    # Generate correlated secondary returns
    independent_returns = np.random.normal(0, primary_returns.std(), len(primary_data))
    
    # Mix primary and independent returns to achieve target correlation
    secondary_returns = (correlation * primary_returns + 
                        np.sqrt(1 - correlation**2) * independent_returns)
    
    # Create secondary price series
    secondary_close = pd.Series(index=primary_data.index, dtype=float)
    secondary_close.iloc[0] = primary_data['close'].iloc[0] * 2  # Different price level
    
    for i in range(1, len(primary_data)):
        secondary_close.iloc[i] = secondary_close.iloc[i-1] * (1 + secondary_returns.iloc[i])
    
    # Create combined dataset
    combined_data = pd.DataFrame({
        'sol_open': primary_data['open'],
        'sol_high': primary_data['high'],
        'sol_low': primary_data['low'],
        'sol_close': primary_data['close'],
        'sol_volume': primary_data.get('volume', pd.Series(1000, index=primary_data.index)),
        
        'btc_open': secondary_close * 0.98,  # Synthetic OHLV
        'btc_high': secondary_close * 1.02,
        'btc_low': secondary_close * 0.97,
        'btc_close': secondary_close,
        'btc_volume': pd.Series(100, index=primary_data.index)
    })
    
    return combined_data


def validate_pairs_data(data: pd.DataFrame, required_symbols: List[str]) -> bool:
    """
    Validate that pairs data contains required symbols and sufficient data.
    
    Args:
        data: Pairs data DataFrame
        required_symbols: List of required symbol prefixes (e.g., ["sol", "btc"])
        
    Returns:
        True if data is valid for pairs trading
    """
    
    # Check for required columns
    for symbol in required_symbols:
        required_cols = [f"{symbol}_close", f"{symbol}_volume"]
        for col in required_cols:
            if col not in data.columns:
                print(f"❌ Missing required column: {col}")
                return False
    
    # Check data length
    if len(data) < 50:
        print(f"❌ Insufficient data: {len(data)} bars (minimum 50 required)")
        return False
    
    # Check for excessive NaN values
    nan_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
    if nan_pct > 5:
        print(f"❌ Too many NaN values: {nan_pct:.1f}% (maximum 5% allowed)")
        return False
    
    print(f"✅ Pairs data validation passed")
    return True


def calculate_pairs_statistics(data: pd.DataFrame, symbol1: str, symbol2: str) -> Dict:
    """
    Calculate key statistics for pairs trading analysis.
    
    Args:
        data: Pairs data DataFrame
        symbol1: First symbol prefix (e.g., "sol")
        symbol2: Second symbol prefix (e.g., "btc")
        
    Returns:
        Dictionary with pairs statistics
    """
    
    price1 = data[f"{symbol1}_close"]
    price2 = data[f"{symbol2}_close"]
    
    # Calculate returns
    returns1 = price1.pct_change().dropna()
    returns2 = price2.pct_change().dropna()
    
    # Price ratio
    ratio = price1 / price2
    log_ratio = np.log(ratio)
    
    # Statistics
    stats = {
        'correlation': returns1.corr(returns2),
        'price_ratio_mean': ratio.mean(),
        'price_ratio_std': ratio.std(),
        'log_ratio_mean': log_ratio.mean(),
        'log_ratio_std': log_ratio.std(),
        'ratio_min': ratio.min(),
        'ratio_max': ratio.max(),
        'volatility_ratio': returns1.std() / returns2.std(),
        'data_points': len(data),
        'date_range': f"{data.index[0]} to {data.index[-1]}"
    }
    
    # Cointegration test (simplified)
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(log_ratio.dropna())
        stats['adf_statistic'] = adf_result[0]
        stats['adf_pvalue'] = adf_result[1]
        stats['is_cointegrated'] = adf_result[1] < 0.05
    except ImportError:
        stats['adf_statistic'] = None
        stats['adf_pvalue'] = None
        stats['is_cointegrated'] = None
    
    return stats