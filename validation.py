#!/usr/bin/env python3
"""
Data Validation Module
Provides comprehensive validation functions for DataFrames, signals, and metrics.
"""

from typing import Dict, Union, List, Optional
import pandas as pd
import numpy as np
from base import Signals


def validate_ohlc_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Validate OHLC DataFrame format and data integrity.
    
    Args:
        df: DataFrame to validate
        name: Name for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if df is None:
        raise ValueError(f"{name} cannot be None")
    
    if df.empty:
        raise ValueError(f"{name} cannot be empty")
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{name} missing required columns: {missing_columns}")
    
    # Check index type
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have DatetimeIndex")
    
    # Check for NaN values
    if df[required_columns].isnull().any().any():
        raise ValueError(f"{name} contains NaN values in OHLC columns")
    
    # Check data types
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{name} column '{col}' must be numeric")
    
    # Check logical consistency (high >= low, etc.)
    if (df['high'] < df['low']).any():
        raise ValueError(f"{name} has high < low values")
    
    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        raise ValueError(f"{name} has high < open/close values")
    
    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        raise ValueError(f"{name} has low > open/close values")
    
    # Check for duplicate index values
    if df.index.duplicated().any():
        raise ValueError(f"{name} has duplicate timestamps")
    
    # Check if data is sorted
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted by datetime ascending")


def validate_signals(signals: Signals, data_index: pd.Index, name: str = "Signals") -> None:
    """Validate trading signals format and alignment.
    
    Args:
        signals: Signals object to validate
        data_index: Index to align against
        name: Name for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if signals is None:
        raise ValueError(f"{name} cannot be None")
    
    # Check all signal series exist and are not None
    signal_series = [signals.entries, signals.exits, signals.short_entries, signals.short_exits]
    series_names = ['entries', 'exits', 'short_entries', 'short_exits']
    
    for series, series_name in zip(signal_series, series_names):
        if series is None:
            raise ValueError(f"{name}.{series_name} cannot be None")
        
        if series.empty:
            raise ValueError(f"{name}.{series_name} cannot be empty")
        
        # Check data type
        if not series.dtype == bool:
            raise ValueError(f"{name}.{series_name} must be boolean dtype")
        
        # Check index type
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{name}.{series_name} must have DatetimeIndex")
    
    # Check all series have same length
    lengths = [len(series) for series in signal_series]
    if len(set(lengths)) != 1:
        raise ValueError(f"{name} series have different lengths: {dict(zip(series_names, lengths))}")
    
    # Check alignment with data index
    common_index = data_index.intersection(signals.entries.index)
    if len(common_index) == 0:
        raise ValueError(f"{name} has no common timestamps with data")
    
    # Warn if significant misalignment
    alignment_ratio = len(common_index) / len(data_index)
    if alignment_ratio < 0.5:
        print(f"⚠️ Warning: {name} only aligns with {alignment_ratio:.1%} of data timestamps")


def validate_metrics(metrics: Dict[str, Union[float, int]], name: str = "Metrics") -> None:
    """Validate metrics dictionary format and values.
    
    Args:
        metrics: Metrics dictionary to validate
        name: Name for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if metrics is None:
        raise ValueError(f"{name} cannot be None")
    
    if not isinstance(metrics, dict):
        raise ValueError(f"{name} must be a dictionary")
    
    # Check required keys
    required_keys = ['return', 'max_dd', 'sharpe', 'calmar', 'trades', 'win_rate']
    missing_keys = [key for key in required_keys if key not in metrics]
    if missing_keys:
        raise ValueError(f"{name} missing required keys: {missing_keys}")
    
    # Check data types
    for key, value in metrics.items():
        if key == 'trades':
            if not isinstance(value, int):
                raise ValueError(f"{name}['{key}'] must be int, got {type(value)}")
        else:
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name}['{key}'] must be numeric, got {type(value)}")
        
        # Check for invalid values
        if pd.isna(value):
            raise ValueError(f"{name}['{key}'] cannot be NaN")
        
        if np.isinf(value) and key not in ['win_loss_ratio']:  # Allow inf for ratios
            raise ValueError(f"{name}['{key}'] cannot be infinite")


def validate_optimization_data(data: pd.DataFrame, min_length: int = 100) -> None:
    """Validate data for optimization purposes.
    
    Args:
        data: DataFrame to validate
        min_length: Minimum required length
        
    Raises:
        ValueError: If validation fails
    """
    validate_ohlc_dataframe(data, "Optimization data")
    
    if len(data) < min_length:
        raise ValueError(f"Optimization data too short: {len(data)} < {min_length} required")
    
    # Check for sufficient price variation
    price_range = data['close'].max() - data['close'].min()
    if price_range <= 0:
        raise ValueError("Optimization data has no price variation")
    
    # Check for reasonable price values
    if (data['close'] <= 0).any():
        raise ValueError("Optimization data contains non-positive prices")


def validate_walkforward_data(data: pd.DataFrame, window_size: int, step_size: int, 
                            num_windows: int) -> None:
    """Validate data for walk-forward analysis.
    
    Args:
        data: DataFrame to validate
        window_size: Size of each window
        step_size: Step size between windows
        num_windows: Number of windows
        
    Raises:
        ValueError: If validation fails
    """
    validate_ohlc_dataframe(data, "Walk-forward data")
    
    min_required = window_size + (num_windows - 1) * step_size + step_size
    if len(data) < min_required:
        raise ValueError(
            f"Insufficient data for walk-forward analysis: "
            f"{len(data)} < {min_required} required "
            f"(window_size={window_size}, step_size={step_size}, num_windows={num_windows})"
        )


def validate_multi_timeframe_data(data: Dict[str, Dict[str, pd.DataFrame]], 
                                required_timeframes: List[str]) -> None:
    """Validate multi-timeframe data structure.
    
    Args:
        data: Multi-timeframe data dictionary
        required_timeframes: List of required timeframes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValueError("Multi-timeframe data must be a dictionary")
    
    if not data:
        raise ValueError("Multi-timeframe data cannot be empty")
    
    for symbol, timeframes in data.items():
        if not isinstance(timeframes, dict):
            raise ValueError(f"Data for symbol '{symbol}' must be a dictionary")
        
        # Check required timeframes
        missing_tfs = [tf for tf in required_timeframes if tf not in timeframes]
        if missing_tfs:
            raise ValueError(f"Symbol '{symbol}' missing required timeframes: {missing_tfs}")
        
        # Validate each DataFrame
        for tf, df in timeframes.items():
            validate_ohlc_dataframe(df, f"{symbol} {tf}")


def safe_validate(validation_func, *args, **kwargs) -> bool:
    """Safely run validation function and return success status.
    
    Args:
        validation_func: Validation function to run
        *args: Arguments for validation function
        **kwargs: Keyword arguments for validation function
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        validation_func(*args, **kwargs)
        return True
    except Exception as e:
        print(f"⚠️ Validation failed: {e}")
        return False
