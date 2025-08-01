#!/usr/bin/env python3
"""
Simple OHLC data loader - does one thing well.
"""

import os
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import yaml

# Simple mappings
TIMEFRAMES = {
    '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
    '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
}

def load_ohlc_csv(file_path: str) -> pd.DataFrame:
    """Load and clean OHLC CSV data."""
    # First, try to detect if the file has headers by reading the first few lines
    try:
        # Read first line to check for headers
        with open(file_path, 'r',encoding='utf-8') as f:
            first_line = f.readline().strip()

        # Check if first line looks like headers (contains text like 'open', 'time', etc.)
        has_headers = any(keyword in first_line.lower() for keyword in
                         ['open', 'high', 'low', 'close', 'time', 'date', 'timestamp'])

        # Try reading with headers first if detected
        if has_headers:
            try:
                df = pd.read_csv(file_path, sep=None, header=0, parse_dates=[0], index_col=0,
                                date_format='mixed', engine='python')
            except Exception:
                # If that fails, try without parse_dates
                df = pd.read_csv(file_path, sep=None, header=0, index_col=0, engine='python')
        else:
            # Try reading without headers
            try:
                df = pd.read_csv(file_path, sep=None, header=None, parse_dates=[0], index_col=0,
                                date_format='mixed', engine='python')
            except Exception:
                # If that fails, try without parse_dates
                df = pd.read_csv(file_path, sep=None, header=None, index_col=0, engine='python')

    except pd.errors.EmptyDataError as exc:
        raise ValueError("File is empty or could not be parsed") from exc
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file {file_path}: {exc}") from exc

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notna()]
        except Exception as exc:
            raise ValueError(f"Could not convert index to datetime in {file_path}: {exc}") from exc

    # Standard column names (take what we need, ignore extras)
    columns = ['open', 'high', 'low', 'close', 'volume']

    # If we have headers, try to map them to standard names
    if has_headers and len(df.columns) > 0:
        # Create a mapping from existing columns to standard names
        column_mapping = {}
        existing_cols = [col.lower() for col in df.columns]

        for i, std_col in enumerate(columns):
            if i < len(df.columns):
                # Try to find a matching column name
                for j, existing_col in enumerate(existing_cols):
                    if std_col in existing_col or existing_col in std_col:
                        column_mapping[df.columns[j]] = std_col
                        break
                else:
                    # If no match found, use positional mapping
                    if i < len(df.columns):
                        column_mapping[df.columns[i]] = std_col

        # Apply the mapping
        df = df.rename(columns=column_mapping)
    else:
        # No headers, use positional mapping
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

def _select_timeframe(available: Dict[str, pd.DataFrame], requested: Optional[str]) -> str:
    """
    Select timeframe by order with case-insensitive match.

    Rules:
      1) If requested is provided, match ignoring case.
      2) Otherwise, return the first available key (insertion order).
    """
    if not available:
        raise ValueError("No timeframes available")

    keys = list(available.keys())
    if requested:
        req = requested.lower()
        for k in keys:
            if k.lower() == req:
                return k
    return keys[0]


def load_data_for_strategy(strategy, time_range: Optional[str] = None,
                          end_date: Optional[Union[str, datetime]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load data and return a dict: symbol -> timeframe -> DataFrame.

    Timeframe assignment is order-based, and lookups are case-insensitive per STYLE_GUIDE.
    """
    required_timeframes = strategy.get_required_timeframes()

    # Get csv_path list
    csv_paths = strategy.get_parameter('csv_path', [])
    if not csv_paths:
        try:
            config = load_strategy_config(strategy.name)
            csv_paths = config.get('csv_path', [])
        except Exception:
            csv_paths = []

    if not csv_paths:
        data_dir = 'data'
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                csv_paths = [os.path.join(data_dir, csv_files[0])]
            else:
                raise ValueError("No CSV files found in data directory and no csv_path specified")
        else:
            raise ValueError("No csv_path defined in strategy configuration and no data directory found")

    # Group files by symbol, preserving order
    symbol_files: Dict[str, List[str]] = {}
    for path in csv_paths:
        symbol = os.path.basename(path).split('_')[0]
        symbol_files.setdefault(symbol, []).append(path)

    all_symbols_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for symbol, files in symbol_files.items():
        try:
            data: Dict[str, pd.DataFrame] = {}

            # Map files to timeframes by position
            if len(files) == 1 and required_timeframes:
                timeframe = required_timeframes[0]
                data[timeframe] = load_ohlc_csv(files[0])
                print(f"✅ Loaded {timeframe} data from {files[0]} ({len(data[timeframe])} bars)")
            else:
                for idx, timeframe in enumerate(required_timeframes):
                    if idx < len(files):
                        file_path = files[idx]
                        data[timeframe] = load_ohlc_csv(file_path)
                        print(f"✅ Loaded {timeframe} data from {file_path} ({len(data[timeframe])} bars)")
                    else:
                        print(f"⚠️ No CSV file provided for timeframe {timeframe} (position {idx})")

            if not data:
                raise ValueError(f"No data loaded for {symbol}")

            # Apply time range filter or harmonization
            if len(data) > 1:
                print("📅 Harmonizing time ranges across multiple timeframes...")
                harmonized = _harmonize_time_ranges(data, time_range, end_date)
                filtered = {tf: harmonized[tf] for tf in required_timeframes if tf in harmonized}
            elif time_range is not None:
                filtered = _apply_time_range_filter(data, time_range, end_date)
                filtered = {tf: filtered[tf] for tf in required_timeframes if tf in filtered}
            else:
                filtered = {tf: data[tf] for tf in required_timeframes if tf in data}

            if not filtered:
                raise ValueError(f"No data loaded for required timeframes: {required_timeframes}")

            # Re-key filtered dict with case-insensitive resolution to respect requested order
            # This ensures downstream modules can safely pick by order and insensitive name.
            requested_primary = strategy.get_parameter('primary_timeframe', required_timeframes[0] if required_timeframes else None)
            # Ensure primary is present; if not, promote the first available
            chosen_primary = _select_timeframe(filtered, requested_primary)
            # Move chosen_primary to front while preserving order for others
            ordered_keys = [k for k in filtered.keys() if k == chosen_primary] + [k for k in filtered.keys() if k != chosen_primary]
            filtered = {k: filtered[k] for k in ordered_keys}

            all_symbols_data[symbol] = filtered

            print(f"🔍 DEBUG: Loaded {symbol} with timeframes: {list(filtered.keys())}")
            for timeframe, df in filtered.items():
                if not df.empty:
                    print(f"   {timeframe}: {len(df)} bars, range: {df.index.min()} to {df.index.max()}")

        except Exception as e:
            print(f"⚠️ Failed to load data for {symbol}: {e}")

    return all_symbols_data

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
        # Normalize timeframe key forms to accept '1h' and '1H' equivalently
        candidates = {tf}
        tfl = tf.lower()
        tfu = tf.upper()
        candidates.update({tfl, tfu})
        # Common alternates
        if tfl.endswith('h'):
            candidates.update({tfl, tfu})
        if tfl.endswith('m'):
            candidates.update({tfl, tfu})
        if tfl in ('1d', 'd1'):
            candidates.update({'1d', '1D', 'D1'})

        chosen_path = None
        for key in candidates:
            if key in available_files:
                chosen_path = available_files[key]
                break

        if chosen_path is None:
            # As a last resort, try any key that equal ignoring case
            for key in available_files.keys():
                if key.lower() == tfl:
                    chosen_path = available_files[key]
                    break

        if chosen_path is not None:
            try:
                all_dataframes[tf] = load_ohlc_csv(chosen_path)
                print(f"✅ Loaded {tf} data from {chosen_path} ({len(all_dataframes[tf])} bars)")
            except Exception as e:
                print(f"⚠️ Failed to load {tf} data from {chosen_path}: {e}")
        else:
            print(f"⚠️ No data file found for timeframe {tf}")
            print(f"   Available timeframes: {list(available_files.keys())}")
    
    if not all_dataframes:
        print(f"🔍 DEBUG: Available files: {available_files}")
        print(f"🔍 DEBUG: Required timeframes: {required_timeframes}")
        raise ValueError(f"No data loaded for any required timeframes: {required_timeframes}")
    
    print(f"🔍 DEBUG: Loaded timeframes: {list(all_dataframes.keys())}")
    
    # Apply time range filtering if needed
    if len(all_dataframes) > 1:
        print("📅 Harmonizing time ranges across multiple timeframes...")
        harmonized_data = _harmonize_time_ranges(all_dataframes, time_range, end_date)
        
        # Filter to only return required timeframes
        for tf in required_timeframes:
            if tf in harmonized_data:
                data[tf] = harmonized_data[tf]
    elif time_range is not None:
        # Use optimized single timeframe filtering
        filtered_data = _apply_time_range_filter(all_dataframes, time_range, end_date)
        
        # Filter to only return required timeframes
        for tf in required_timeframes:
            if tf in filtered_data:
                data[tf] = filtered_data[tf]
    else:
        # Single timeframe, no time filtering needed - just return the data as-is
        for tf in required_timeframes:
            if tf in all_dataframes:
                data[tf] = all_dataframes[tf]
    
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


def _apply_time_range_filter(dataframes: Dict[str, pd.DataFrame],
                           time_range: Optional[str] = None,
                           end_date: Optional[Union[str, datetime]] = None) -> Dict[str, pd.DataFrame]:
    """Apply time range filter to single or multiple dataframes.
    
    Args:
        dataframes: Dictionary of timeframe -> DataFrame
        time_range: Time range specification (e.g., '2y', '6m')
        end_date: End date for the time range
    
    Returns:
        Dictionary of filtered dataframes
    """
    if not dataframes:
        return {}
    
    # For single timeframe, use optimized filtering
    if len(dataframes) == 1:
        return _filter_single_timeframe(dataframes, time_range, end_date)
    else:
        # For multiple timeframes, use harmonization
        return _harmonize_time_ranges(dataframes, time_range, end_date)


def _filter_single_timeframe(dataframes: Dict[str, pd.DataFrame],
                            time_range: Optional[str] = None,
                            end_date: Optional[Union[str, datetime]] = None) -> Dict[str, pd.DataFrame]:
    """Apply time range filter to a single timeframe.
    
    Args:
        dataframes: Dictionary containing single timeframe -> DataFrame
        time_range: Time range specification (e.g., '2y', '6m')
        end_date: End date for the time range
    
    Returns:
        Dictionary containing filtered single timeframe
    """
    if not dataframes:
        return {}
    
    # Get the single timeframe
    tf, df = next(iter(dataframes.items()))
    
    if not time_range:
        # No time range filtering needed
        return {tf: df}
    
    # Parse time range
    time_delta = _parse_time_range(time_range)
    if not time_delta:
        return {tf: df}
    
    # Determine end date
    if end_date is None:
        actual_end_date = df.index.max()
    else:
        if isinstance(end_date, str):
            actual_end_date = pd.to_datetime(end_date)
        else:
            actual_end_date = end_date
        # Don't go beyond available data
        actual_end_date = min(actual_end_date, df.index.max())
    
    # Calculate start date
    actual_start_date = actual_end_date - time_delta
    # Don't go before available data
    actual_start_date = max(actual_start_date, df.index.min())
    
    print(f"📅 Applying time range filter: {time_range}")
    print(f"📅 Filtering data from {actual_start_date} to {actual_end_date}")
    
    # Apply the time range filter
    mask = (df.index >= actual_start_date) & (df.index <= actual_end_date)
    filtered_df = df.loc[mask].copy()
    
    if not filtered_df.empty:
        print(f"✅ Filtered {tf}: {len(filtered_df)} bars ({filtered_df.index.min()} to {filtered_df.index.max()})")
        return {tf: filtered_df}
    else:
        print(f"⚠️ No data available for {tf} in the specified time range")
        return {}


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
    
    # Debug: Log the context of harmonization
    num_timeframes = len(dataframes)
    print(f"🔍 DEBUG: _harmonize_time_ranges called with {num_timeframes} timeframe(s): {list(dataframes.keys())}")
    if time_range:
        print(f"🔍 DEBUG: Time range specified: {time_range}")
    if end_date:
        print(f"🔍 DEBUG: End date specified: {end_date}")
    
    # Find the common time range across all dataframes
    earliest_start = None
    latest_end = None
    
    for tf, df in dataframes.items():
        if df.empty:
            continue

        # Ensure the dataframe has a proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[df.index.notna()]
                dataframes[tf] = df  # Update the dataframe in the dictionary
            except Exception as e:
                print(f"⚠️ Could not convert {tf} index to datetime: {e}")
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
            # Ensure actual_end_date is a datetime object before subtraction
            if not isinstance(actual_end_date, (pd.Timestamp, datetime)):
                try:
                    actual_end_date = pd.to_datetime(actual_end_date)
                except Exception as e:
                    print(f"⚠️ Could not convert end_date to datetime: {e}")
                    actual_start_date = earliest_start
                else:
                    actual_start_date = actual_end_date - time_delta
                    # Don't go before available data
                    actual_start_date = max(actual_start_date, earliest_start)
            else:
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
    
    print(f"🔍 DEBUG: Harmonization complete. Returning {len(harmonized)} timeframe(s): {list(harmonized.keys())}")
    
    return harmonized