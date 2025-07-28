#!/usr/bin/env python3
"""
Simple OHLC data loader - does one thing well.
"""

import os
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import yaml
from base import BaseStrategy

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
    except pd.errors.EmptyDataError as exc:
        raise ValueError("File is empty or could not be parsed") from exc
    
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

def load_strategy_config(strategy_name: str) -> Dict:
    """Load strategy configuration from YAML file."""
    config_path = f"config/{strategy_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config

def load_data_for_strategy(strategy: BaseStrategy, time_range: Optional[str] = None, 
                          end_date: Optional[Union[str, datetime]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all necessary data for a given strategy with optional time range control.
    
    Args:
        strategy: The trading strategy instance
        time_range: Time range specification (e.g., '2y', '6m', '1y', '3m')
        end_date: End date for the time range (defaults to most recent data)
    """
    required_timeframes = strategy.get_required_timeframes()
    
    # Try to get csv_path from strategy parameters first, then from config root
    csv_paths = strategy.get_parameter('csv_path', [])
    
    # If not found in parameters, check if it's in the config root level
    if not csv_paths:
        # Load the raw config to check for csv_path at root level
        # Try both the strategy name and the strategy name without "_strategy" suffix
        strategy_names_to_try = [strategy.name]
        if strategy.name.endswith('_strategy'):
            strategy_names_to_try.append(strategy.name.replace('_strategy', ''))
        
        for name in strategy_names_to_try:
            try:
                config_dict = load_strategy_config(name)
                csv_paths = config_dict.get('csv_path', [])
                if csv_paths:
                    break
            except Exception:
                continue

    if not csv_paths:
        # If no csv_path in strategy config, try to find any available data file
        # This provides fallback behavior when csv_path is not specified
        data_dir = 'data'
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                csv_paths = [os.path.join(data_dir, csv_files[0])]
            else:
                raise ValueError("No CSV files found in data directory and no csv_path specified")
        else:
            raise ValueError("No csv_path defined in strategy configuration and no data directory found")

    # Extract symbol from first file name
    symbol = csv_paths[0].split('/')[-1].split('_')[0]

    data = _load_symbol_data(csv_paths, required_timeframes, time_range, end_date)

    return {symbol: data}

def _load_symbol_data(file_paths: List[str], required_timeframes: List[str], 
                     time_range: Optional[str] = None, 
                     end_date: Optional[Union[str, datetime]] = None) -> Dict[str, pd.DataFrame]:
    """Load symbol data for specified timeframes from a list of files with time range harmonization."""
    data = {}
    all_dataframes = {}
    
    # Create mapping from timeframe to file path
    available_files = {}
    for path in file_paths:
        # Extract timeframe from filename with various patterns
        filename = os.path.basename(path)
        parts = filename.split('_')
        
        if len(parts) >= 2:
            # Handle patterns like "EURUSD_1H_2009-2025.csv" or "EURUSD_15m.csv"
            tf_part = parts[1].split('-')[0].split('.')[0]  # Remove both date range and extension
            
            # Normalize timeframe names
            tf_lower = tf_part.lower()
            
            # Store both original and normalized versions
            available_files[tf_part] = path  # Original case
            available_files[tf_lower] = path  # Lowercase
            
            # Also add common mappings
            if tf_lower.endswith('h'):
                available_files[tf_lower] = path  # 1h, 4h
            elif tf_lower.endswith('m'):
                available_files[tf_lower] = path  # 15m, 30m
            elif tf_lower.endswith('d'):
                available_files['1D'] = path  # Normalize daily to 1D

    # Load data for required timeframes
    for tf in required_timeframes:
        tf_normalized = tf.lower() if not tf.endswith('D') else tf
        
        if tf_normalized in available_files:
            try:
                all_dataframes[tf] = load_ohlc_csv(available_files[tf_normalized])
                print(f"✅ Loaded {tf} data from {available_files[tf_normalized]} ({len(all_dataframes[tf])} bars)")
            except Exception as e:
                print(f"⚠️ Failed to load {tf} data from {available_files[tf_normalized]}: {e}")
        else:
            # Try alternative mappings
            alt_mappings = {
                '1h': ['1h', '1H'],
                '4h': ['4h', '4H'], 
                '1D': ['1D', '1d'],
                '15m': ['15m', '15M'],
                '30m': ['30m', '30M'],
                # Legacy format mappings
                '15min': ['15m', '15M'],
                '30min': ['30m', '30M']
            }
            
            found = False
            if tf in alt_mappings:
                for alt_tf in alt_mappings[tf]:
                    if alt_tf in available_files:
                        try:
                            all_dataframes[tf] = load_ohlc_csv(available_files[alt_tf])
                            print(f"✅ Loaded {tf} data from {available_files[alt_tf]} ({len(all_dataframes[tf])} bars)")
                            found = True
                            break
                        except Exception as e:
                            print(f"⚠️ Failed to load {tf} data from {available_files[alt_tf]}: {e}")
            
            if not found:
                print(f"⚠️ No data file found for timeframe {tf}")
                print(f"   Available timeframes: {list(available_files.keys())}")
    
    if not all_dataframes:
        raise ValueError(f"No data loaded for any required timeframes: {required_timeframes}")
    
    # Harmonize time ranges across all loaded dataframes
    harmonized_data = _harmonize_time_ranges(all_dataframes, time_range, end_date)
    
    # Filter to only return required timeframes
    for tf in required_timeframes:
        if tf in harmonized_data:
            data[tf] = harmonized_data[tf]
    
    return data


def _parse_time_range(time_range: str) -> timedelta:
    """Parse time range string into timedelta object.
    
    Args:
        time_range: Time range string (e.g., '2y', '6m', '1y', '3m', '30d')
    
    Returns:
        timedelta object representing the time range
    """
    if not time_range:
        return None
        
    time_range = time_range.lower().strip()
    
    # Extract number and unit
    if time_range[-1] == 'y':
        years = int(time_range[:-1])
        return timedelta(days=years * 365)
    elif time_range[-1] == 'm':
        months = int(time_range[:-1])
        return timedelta(days=months * 30)  # Approximate
    elif time_range[-1] == 'd':
        days = int(time_range[:-1])
        return timedelta(days=days)
    elif time_range[-1] == 'w':
        weeks = int(time_range[:-1])
        return timedelta(weeks=weeks)
    else:
        raise ValueError(f"Unsupported time range format: {time_range}. Use format like '2y', '6m', '30d', '4w'")


def _harmonize_time_ranges(dataframes: Dict[str, pd.DataFrame], 
                          time_range: Optional[str] = None,
                          end_date: Optional[Union[str, datetime]] = None) -> Dict[str, pd.DataFrame]:
    """Harmonize time ranges across multiple dataframes.
    
    Args:
        dataframes: Dictionary of timeframe -> DataFrame
        time_range: Time range specification (e.g., '2y', '6m')
        end_date: End date for the time range
    
    Returns:
        Dictionary of harmonized dataframes with consistent time ranges
    """
    if not dataframes:
        return {}
    
    # Find the common time range across all dataframes
    earliest_start = None
    latest_end = None
    
    for tf, df in dataframes.items():
        if df.empty:
            continue
            
        df_start = df.index.min()
        df_end = df.index.max()
        
        if earliest_start is None or df_start > earliest_start:
            earliest_start = df_start
        if latest_end is None or df_end < latest_end:
            latest_end = df_end
    
    if earliest_start is None or latest_end is None:
        print("⚠️ No valid data found for harmonization")
        return dataframes
    
    # Determine the actual end date to use
    if end_date is None:
        actual_end_date = latest_end
    else:
        if isinstance(end_date, str):
            actual_end_date = pd.to_datetime(end_date)
        else:
            actual_end_date = end_date
        # Don't go beyond available data
        actual_end_date = min(actual_end_date, latest_end)
    
    # Determine the start date based on time range
    if time_range:
        time_delta = _parse_time_range(time_range)
        if time_delta:
            actual_start_date = actual_end_date - time_delta
            # Don't go before available data
            actual_start_date = max(actual_start_date, earliest_start)
        else:
            actual_start_date = earliest_start
    else:
        # Use the common overlapping period
        actual_start_date = earliest_start
    
    print(f"📅 Harmonizing data from {actual_start_date} to {actual_end_date}")
    
    # Apply the harmonized time range to all dataframes
    harmonized = {}
    for tf, df in dataframes.items():
        if df.empty:
            continue
            
        # Filter to the harmonized time range
        mask = (df.index >= actual_start_date) & (df.index <= actual_end_date)
        filtered_df = df.loc[mask].copy()
        
        if not filtered_df.empty:
            harmonized[tf] = filtered_df
            print(f"✅ Harmonized {tf}: {len(filtered_df)} bars ({filtered_df.index.min()} to {filtered_df.index.max()})")
        else:
            print(f"⚠️ No data available for {tf} in the specified time range")
    
    return harmonized