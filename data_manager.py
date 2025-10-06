#!/usr/bin/env python3
"""
Enhanced OHLC data loader with multiple data sources.
Supports CSV, CCXT (Binance), and Freqtrade data sources.
"""

import os
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import yaml

# Required imports
import json
import ccxt
from typing import Any

# Simple mappings
TIMEFRAMES = {
    "1m": "1T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
}


def load_ohlc_csv(
    file_path: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
    """Load and clean OHLC CSV data with optional time range filtering."""

    try:
        # Check for headers by reading first line
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        has_headers = any(
            kw in first_line.lower()
            for kw in ["open", "high", "low", "close", "time", "date", "timestamp"]
        )

        # Read CSV with appropriate header setting
        df = pd.read_csv(
            file_path,
            sep=None,
            header=0 if has_headers else None,
            parse_dates=[0],
            index_col=0,
            date_format="mixed",
            engine="python",
        )

    except (pd.errors.EmptyDataError, Exception) as exc:
        raise ValueError(f"Could not parse CSV file {file_path}: {exc}") from exc

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]

    # Standardize column names
    columns = ["open", "high", "low", "close", "volume"]

    if has_headers:
        # Smart column mapping
        mapping = {}
        existing_lower = [col.lower() for col in df.columns]

        for i, std_col in enumerate(columns):
            if i >= len(df.columns):
                break
            # Find best match or use positional
            match_idx = next(
                (
                    j
                    for j, col in enumerate(existing_lower)
                    if std_col in col or col in std_col
                ),
                i,
            )
            if match_idx < len(df.columns):
                mapping[df.columns[match_idx]] = std_col

        df = df.rename(columns=mapping)
    else:
        df.columns = columns[: len(df.columns)]

    # Validate required columns
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean and sort data
    available = [col for col in columns if col in df.columns]
    df = df[available].dropna().sort_index()

    # Apply time range filter
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]

    return df


def load_strategy_config(strategy_name: str, config_type: str = "auto") -> Dict:
    """
    Load strategy configuration from YAML or Freqtrade JSON file.

    Args:
        strategy_name: Name of the strategy
        config_type: Type of config ('auto', 'yaml', 'freqtrade')

    Returns:
        Standardized configuration dict
    """
    if config_type == "auto":
        config_type = _detect_config_type(strategy_name)

    if config_type == "yaml":
        return _load_yaml_config(strategy_name)
    elif config_type == "freqtrade":
        return _load_freqtrade_config(strategy_name)
    else:
        raise ValueError(f"Unsupported config type: {config_type}")


def _detect_config_type(strategy_name: str) -> str:
    """Auto-detect configuration file type."""
    yaml_path = f"config/{strategy_name}.yaml"
    json_path = f"config/{strategy_name}.json"

    if os.path.exists(yaml_path):
        return "yaml"
    elif os.path.exists(json_path):
        return "freqtrade"
    else:
        raise FileNotFoundError(f"No configuration file found for {strategy_name}")


def _load_yaml_config(strategy_name: str) -> Dict:
    """Load YAML configuration (existing logic)."""
    config_path = f"config/{strategy_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def _load_freqtrade_config(strategy_name: str) -> Dict:
    """Load and convert Freqtrade JSON configuration."""
    config_path = f"config/{strategy_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Freqtrade configuration file not found: {config_path}"
        )

    # Convert Freqtrade config to our standard format
    return _convert_freqtrade_config_internal(config_path)


def _select_timeframe(
    available: Dict[str, pd.DataFrame], requested: Optional[str]
) -> str:
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


def load_data_for_strategy(
    strategy,
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
    data_source: str = "auto",
    **kwargs,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load data from various sources and return dict: symbol -> timeframe -> DataFrame.

    Args:
        strategy: Strategy object with configuration
        time_range: Time range filter (e.g., '1y', '6m')
        end_date: End date for filtering
        data_source: Data source type ('auto', 'csv', 'ccxt', 'freqtrade')
        **kwargs: Additional arguments for specific data sources

    Returns:
        Dict with structure: {symbol: {timeframe: DataFrame}}
    """
    required_timeframes = strategy.get_required_timeframes()

    # Auto-detect data source if not specified
    if data_source == "auto":
        data_source = _detect_data_source(strategy)


    print(f"📊 Loading data using {data_source} source...")

    # Load data based on source type
    if data_source == "csv":
        return _load_csv_data_for_strategy(strategy, time_range, end_date)
    elif data_source == "ccxt":
        return _load_ccxt_data_for_strategy(strategy, time_range, end_date, **kwargs)
    elif data_source == "freqtrade":
        return _load_freqtrade_data_for_strategy(
            strategy, time_range, end_date, **kwargs
        )
    else:
        raise ValueError(f"Unsupported data source: {data_source}")


def _detect_data_source(strategy) -> str:
    """Auto-detect the appropriate data source based on configuration."""
    # Check for explicit data_source parameter (simple choice)
    data_source = strategy.get_parameter("data_source", "").lower()
    if data_source in ["csv", "ccxt", "freqtrade"]:
        return data_source
    
    # Fallback: Check for CCXT configuration
    ccxt_config = strategy.get_parameter("ccxt", {})
    if ccxt_config:
        return "ccxt"

    # Check for Freqtrade data directory
    freqtrade_dir = strategy.get_parameter("freqtrade_data_dir")
    if freqtrade_dir and os.path.exists(freqtrade_dir):
        return "freqtrade"

    # Default to CSV
    return "csv"


def _load_csv_data_for_strategy(
    strategy,
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load CSV data for strategy (existing logic)."""
    required_timeframes = strategy.get_required_timeframes()

    # Get csv_path list
    csv_paths = strategy.get_parameter("csv_path", [])
    if not csv_paths:
        try:
            config = load_strategy_config(strategy.name)
            csv_paths = config.get("csv_path", [])
        except Exception:
            csv_paths = []
    
    # Ensure csv_paths is always a list (handle string case)
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    if not csv_paths:
        data_dir = "data"
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
            if csv_files:
                csv_paths = [os.path.join(data_dir, csv_files[0])]
            else:
                raise ValueError(
                    "No CSV files found in data directory and no csv_path specified"
                )
        else:
            raise ValueError(
                "No csv_path defined in strategy configuration and no data directory found"
            )

    # Use existing CSV loading logic
    return _load_csv_files(csv_paths, required_timeframes, time_range, end_date)


def _load_ccxt_data_for_strategy(
    strategy,
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
    **kwargs,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load CCXT data for strategy."""
    required_timeframes = strategy.get_required_timeframes()

    # Get CCXT configuration
    ccxt_config = strategy.get_parameter("ccxt", {})

    # Extract parameters with defaults
    exchange_name = ccxt_config.get("exchange", kwargs.get("exchange", "binance"))
    symbols = ccxt_config.get("symbols", kwargs.get("symbols", ["BTC/USDT"]))
    api_key = ccxt_config.get("api_key", kwargs.get("api_key"))
    api_secret = ccxt_config.get("api_secret", kwargs.get("api_secret"))
    sandbox = ccxt_config.get("sandbox", kwargs.get("sandbox", False))
    
    # Debug: Print what symbols we're trying to load
    print(f"🔍 CCXT Config - Exchange: {exchange_name}, Symbols: {symbols}")

    return _load_ccxt_data_internal(
        exchange_name=exchange_name,
        symbols=symbols,
        timeframes=required_timeframes,
        time_range=time_range,
        end_date=end_date,
        api_key=api_key,
        api_secret=api_secret,
        sandbox=sandbox,
    )


def _load_freqtrade_data_for_strategy(
    strategy,
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
    **kwargs,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load Freqtrade data for strategy."""
    required_timeframes = strategy.get_required_timeframes()

    # Get Freqtrade configuration
    freqtrade_dir = strategy.get_parameter(
        "freqtrade_data_dir", kwargs.get("data_directory")
    )
    symbols = strategy.get_parameter("symbols", kwargs.get("symbols"))

    if not freqtrade_dir:
        raise ValueError("freqtrade_data_dir not specified in strategy configuration")

    return _load_freqtrade_data_internal(
        data_directory=freqtrade_dir,
        symbols=symbols,
        timeframes=required_timeframes,
        time_range=time_range,
    )


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
    if time_range[-1] == "y":
        years = int(time_range[:-1])
        return timedelta(days=years * 365)
    elif time_range[-1] == "m":
        months = int(time_range[:-1])
        return timedelta(days=months * 30)  # Approximate
    elif time_range[-1] == "d":
        days = int(time_range[:-1])
        return timedelta(days=days)
    elif time_range[-1] == "w":
        weeks = int(time_range[:-1])
        return timedelta(weeks=weeks)
    else:
        raise ValueError(
            f"Unsupported time range format: {time_range}. Use format like '2y', '6m', '30d', '4w'"
        )


def _apply_time_range_filter(
    dataframes: Dict[str, pd.DataFrame],
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> Dict[str, pd.DataFrame]:
    """Apply time range filter to dataframes."""
    if not dataframes or not time_range:
        return dataframes

    time_delta = _parse_time_range(time_range)
    if not time_delta:
        return dataframes

    # Find common time bounds
    all_starts = [df.index.min() for df in dataframes.values() if not df.empty]
    all_ends = [df.index.max() for df in dataframes.values() if not df.empty]

    if not all_starts or not all_ends:
        return dataframes

    # Determine actual end date
    latest_available = min(all_ends)
    if end_date:
        actual_end = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
        actual_end = min(actual_end, latest_available)
    else:
        actual_end = latest_available

    # Calculate start date
    actual_start = actual_end - time_delta
    actual_start = max(actual_start, max(all_starts))

    print(f"📅 Applying time filter: {time_range} ({actual_start} to {actual_end})")

    # Filter all dataframes
    filtered = {}
    for tf, df in dataframes.items():
        if df.empty:
            continue
        mask = (df.index >= actual_start) & (df.index <= actual_end)
        filtered_df = df.loc[mask].copy()
        if not filtered_df.empty:
            filtered[tf] = filtered_df
            print(f"✅ {tf}: {len(filtered_df)} bars")

    return filtered


def _harmonize_time_ranges(
    dataframes: Dict[str, pd.DataFrame],
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> Dict[str, pd.DataFrame]:
    """Harmonize time ranges across multiple dataframes - find common overlap."""
    if not dataframes:
        return {}

    # Ensure all dataframes have datetime index
    for tf, df in dataframes.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
            dataframes[tf] = df

    # Find common overlapping period (latest start, earliest end)
    valid_dfs = [(tf, df) for tf, df in dataframes.items() if not df.empty]
    if not valid_dfs:
        return {}

    common_start = max(df.index.min() for _, df in valid_dfs)
    common_end = min(df.index.max() for _, df in valid_dfs)

    # Apply time range constraints
    if time_range:
        time_delta = _parse_time_range(time_range)
        if time_delta:
            actual_end = pd.to_datetime(end_date) if end_date else common_end
            actual_end = min(actual_end, common_end)
            actual_start = max(common_start, actual_end - time_delta)
        else:
            actual_start, actual_end = common_start, common_end
    else:
        actual_start, actual_end = common_start, common_end

    print(f"📅 Harmonizing to common period: {actual_start} to {actual_end}")

    # Filter all dataframes to common period
    harmonized = {}
    for tf, df in dataframes.items():
        if df.empty:
            continue
        mask = (df.index >= actual_start) & (df.index <= actual_end)
        filtered_df = df.loc[mask].copy()
        if not filtered_df.empty:
            harmonized[tf] = filtered_df
            print(f"✅ {tf}: {len(filtered_df)} bars")

    return harmonized


def _harmonize_across_symbols(
    all_symbols_data: Dict[str, Dict[str, pd.DataFrame]],
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Harmonize time ranges across multiple symbols."""
    if not all_symbols_data:
        return all_symbols_data

    # Get primary timeframe data from all symbols
    primary_dfs = []
    for symbol, tf_map in all_symbols_data.items():
        if tf_map:
            primary_tf = next(iter(tf_map.keys()))
            df = tf_map.get(primary_tf)
            if df is not None and not df.empty:
                primary_dfs.append((symbol, df))

    if not primary_dfs:
        return all_symbols_data

    # Find common overlapping period across all symbols
    common_start = max(df.index.min() for _, df in primary_dfs)
    common_end = min(df.index.max() for _, df in primary_dfs)

    # Apply time range constraints
    if time_range:
        time_delta = _parse_time_range(time_range)
        if time_delta:
            actual_end = pd.to_datetime(end_date) if end_date else common_end
            actual_end = min(actual_end, common_end)
            actual_start = max(common_start, actual_end - time_delta)
        else:
            actual_start, actual_end = common_start, common_end
    else:
        actual_start, actual_end = common_start, common_end

    print(f"📅 Cross-symbol harmonization: {actual_start} to {actual_end}")

    # Apply common time range to all symbols
    harmonized = {}
    for symbol, tf_map in all_symbols_data.items():
        symbol_data = {}
        for tf, df in tf_map.items():
            if df is None or df.empty:
                continue
            mask = (df.index >= actual_start) & (df.index <= actual_end)
            filtered_df = df.loc[mask].copy()
            if not filtered_df.empty:
                symbol_data[tf] = filtered_df
        if symbol_data:
            harmonized[symbol] = symbol_data
            print(f"   {symbol}: {len(next(iter(symbol_data.values())))} bars")

    return harmonized


def _load_csv_files(
    csv_paths: List[str],
    required_timeframes: List[str],
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load CSV files and return symbol -> timeframe -> DataFrame structure."""
    # Group files by symbol, preserving order
    symbol_files: Dict[str, List[str]] = {}
    for path in csv_paths:
        symbol = os.path.basename(path).split("_")[0]
        symbol_files.setdefault(symbol, []).append(path)

    all_symbols_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for symbol, files in symbol_files.items():
        try:
            data: Dict[str, pd.DataFrame] = {}

            # Map files to timeframes by position
            if len(files) == 1 and required_timeframes:
                timeframe = required_timeframes[0]
                data[timeframe] = load_ohlc_csv(files[0])
                print(
                    f"✅ Loaded {timeframe} data from {files[0]} ({len(data[timeframe])} bars)"
                )
            else:
                for idx, timeframe in enumerate(required_timeframes):
                    if idx < len(files):
                        file_path = files[idx]
                        data[timeframe] = load_ohlc_csv(file_path)
                        print(
                            f"✅ Loaded {timeframe} data from {file_path} ({len(data[timeframe])} bars)"
                        )

            if not data:
                raise ValueError(f"No data loaded for {symbol}")

            # Apply time filtering
            if len(data) > 1:
                filtered = _harmonize_time_ranges(data, time_range, end_date)
            else:
                filtered = _apply_time_range_filter(data, time_range, end_date)

            # Keep only required timeframes
            filtered = {
                tf: filtered[tf] for tf in required_timeframes if tf in filtered
            }

            if filtered:
                all_symbols_data[symbol] = filtered

        except Exception as e:
            print(f"⚠️ Failed to load CSV data for {symbol}: {e}")

    return all_symbols_data


def _load_ccxt_data_internal(
    exchange_name: str = "binance",
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    sandbox: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load data from CCXT exchange."""
    # Default values
    if symbols is None:
        symbols = ["BTC/USDT"]
    if timeframes is None:
        timeframes = ["1h"]

    # Initialize exchange
    exchange_class = getattr(ccxt, exchange_name.lower())
    exchange = exchange_class(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": sandbox,
            "enableRateLimit": True,
        }
    )

    # Load markets
    try:
        markets = exchange.load_markets()
        print(f"✅ Connected to {exchange_name.title()}")
    except Exception as e:
        raise ConnectionError(f"Failed to connect to {exchange_name}: {e}")

    all_data = {}
    
    print(f"🔍 Processing {len(symbols)} symbols: {symbols}")

    for symbol in symbols:
        print(f"🔄 Loading data for {symbol}...")
        if symbol not in markets:
            print(f"⚠️ Symbol {symbol} not found on {exchange_name}")
            continue

        symbol_data = {}

        for timeframe in timeframes:
            try:
                # Calculate appropriate limit based on time_range (reuse existing logic)
                if time_range:
                    time_delta = _parse_time_range(time_range)
                    if time_delta:
                        # Calculate bars needed based on timeframe
                        if timeframe == "1h":
                            hours_needed = int(time_delta.total_seconds() / 3600)
                            limit = hours_needed  # Don't cap here, let multi-batch handle it
                        elif timeframe == "4h":
                            limit = int(time_delta.total_seconds() / (4 * 3600))
                        elif timeframe == "1d":
                            limit = time_delta.days
                        else:
                            limit = 1000  # Default fallback
                    else:
                        limit = 1000
                else:
                    limit = 1000  # Default when no time_range specified
                
                print(f"🔍 Fetching {limit} bars for {symbol} {timeframe} (time_range: {time_range})")
                
                # Fetch OHLCV data - use Freqtrade-like approach for large datasets
                try:
                    if limit > 1000 and time_range:
                        print(f"📊 Large dataset requested ({limit} bars for {time_range})")
                        print(f"🔄 Using multi-batch approach like Freqtrade...")
                        ohlcv_data = _fetch_historical_data_batches(exchange, symbol, timeframe, limit)
                    else:
                        ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                        
                    # Always inform about the actual data range we got
                    if ohlcv_data:
                        actual_days = len(ohlcv_data) / (24 if timeframe == "1h" else 6 if timeframe == "4h" else 1)
                        print(f"📊 Final dataset: {len(ohlcv_data)} bars (~{actual_days:.0f} days)")
                        
                except Exception as fetch_error:
                    print(f"❌ Failed to fetch with limit {limit}, trying with 1000: {fetch_error}")
                    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)

                if ohlcv_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        ohlcv_data,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)

                    # Ensure numeric types
                    numeric_cols = ["open", "high", "low", "close", "volume"]
                    df[numeric_cols] = df[numeric_cols].astype(float)

                    symbol_data[timeframe] = df
                    print(f"✅ Loaded {symbol} {timeframe}: {len(df)} bars")

            except Exception as e:
                print(f"❌ Failed to load {symbol} {timeframe}: {e}")

        if symbol_data:
            # Clean symbol name for dict key
            clean_symbol = symbol.replace("/", "")
            all_data[clean_symbol] = symbol_data

    return all_data


def _load_freqtrade_data_internal(
    data_directory: str,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    time_range: Optional[str] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load data from Freqtrade data directory."""
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Freqtrade data directory not found: {data_directory}")

    # Default values
    if symbols is None:
        symbols = _discover_freqtrade_symbols(data_directory)
    if timeframes is None:
        timeframes = ["1h"]

    all_data = {}

    for symbol in symbols:
        symbol_data = {}

        for timeframe in timeframes:
            # Freqtrade naming convention: SYMBOL_TIMEFRAME.json
            filename = f"{symbol}_{timeframe}.json"
            filepath = os.path.join(data_directory, filename)

            if os.path.exists(filepath):
                try:
                    df = _load_freqtrade_json_file(filepath)
                    if not df.empty:
                        symbol_data[timeframe] = df
                        print(
                            f"✅ Loaded Freqtrade data: {symbol} {timeframe} ({len(df)} bars)"
                        )
                except Exception as e:
                    print(f"❌ Failed to load {filepath}: {e}")

        if symbol_data:
            all_data[symbol] = symbol_data

    return all_data


def _discover_freqtrade_symbols(data_directory: str) -> List[str]:
    """Discover available symbols in Freqtrade data directory."""
    symbols = set()

    for filename in os.listdir(data_directory):
        if filename.endswith(".json"):
            # Extract symbol from filename (SYMBOL_TIMEFRAME.json)
            parts = filename.replace(".json", "").split("_")
            if len(parts) >= 2:
                symbol = "_".join(parts[:-1])  # Everything except last part (timeframe)
                symbols.add(symbol)

    return list(symbols)


def _load_freqtrade_json_file(filepath: str) -> pd.DataFrame:
    """Load Freqtrade JSON data file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Freqtrade format: [timestamp, open, high, low, close, volume]
    if len(df.columns) >= 6:
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    else:
        raise ValueError(f"Invalid Freqtrade data format in {filepath}")

    # Convert timestamp to datetime index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Ensure numeric types
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def _convert_freqtrade_config_internal(config_path: str) -> Dict[str, Any]:
    """Convert Freqtrade configuration to our standard format."""
    with open(config_path, "r", encoding="utf-8") as f:
        ft_config = json.load(f)

    # Convert to our standard format
    standard_config = {
        "parameters": {},
        "optimization_grid": {},
        "data_requirements": {
            "required_timeframes": [ft_config.get("timeframe", "1h")],
            "required_columns": ["open", "high", "low", "close", "volume"],
        },
    }

    # Map common Freqtrade parameters
    if "stake_amount" in ft_config:
        standard_config["parameters"]["initial_cash"] = ft_config["stake_amount"] * 100

    if "fee" in ft_config:
        standard_config["parameters"]["fee"] = ft_config["fee"]

    if "timeframe" in ft_config:
        standard_config["parameters"]["primary_timeframe"] = ft_config["timeframe"]

    # Map buy/sell parameters
    if "buy_params" in ft_config:
        standard_config["parameters"].update(ft_config["buy_params"])

    if "sell_params" in ft_config:
        standard_config["parameters"].update(ft_config["sell_params"])

    # Map strategy parameters if available
    if "strategy_params" in ft_config:
        standard_config["parameters"].update(ft_config["strategy_params"])

    return standard_config


# Convenience functions for easy usage
def load_binance_data(symbol="BTC/USDT", timeframe="1h", limit=1000, **kwargs):
    """Load data from Binance via CCXT."""
    return _load_ccxt_data_internal(
        exchange_name="binance", symbols=[symbol], timeframes=[timeframe], **kwargs
    )


def import_freqtrade_config(config_path: str):
    """Import Freqtrade configuration."""
    return _convert_freqtrade_config_internal(config_path)


def copy_freqtrade_config(source_path: str, target_path: str):
    """Copy and convert Freqtrade config to YAML format."""
    ft_config = import_freqtrade_config(source_path)

    with open(target_path, "w") as f:
        yaml.dump(ft_config, f, default_flow_style=False)

    print(f"✅ Converted Freqtrade config to {target_path}")
    return ft_config

# Simple utility functions for easy data source switching
def use_csv_data():
    """Simple function to load CSV data for any strategy."""
    def load_csv_for_strategy(strategy, time_range=None, end_date=None):
        return load_data_for_strategy(strategy, time_range, end_date, data_source="csv")
    return load_csv_for_strategy


def use_ccxt_data(exchange="binance", symbols=None, **kwargs):
    """Simple function to load CCXT data for any strategy."""
    if symbols is None:
        symbols = ["BTC/USDT"]
    
    def load_ccxt_for_strategy(strategy, time_range=None, end_date=None):
        return load_data_for_strategy(
            strategy, time_range, end_date, 
            data_source="ccxt", 
            exchange=exchange, 
            symbols=symbols, 
            **kwargs
        )
    return load_ccxt_for_strategy


def quick_binance_data(symbol="BTC/USDT", timeframe="1h", bars=1000):
    """Quick function to get Binance data without configuration."""
    return _load_ccxt_data_internal(
        exchange_name="binance",
        symbols=[symbol],
        timeframes=[timeframe],
        limit=bars
    )


def switch_data_source(config_file: str, new_source: str):
    """Switch data source in a YAML config file."""
    if new_source not in ["csv", "ccxt", "freqtrade"]:
        raise ValueError("Data source must be 'csv', 'ccxt', or 'freqtrade'")
    
    # Load existing config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data source
    if 'parameters' not in config:
        config['parameters'] = {}
    config['parameters']['data_source'] = new_source
    
    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Updated {config_file} to use {new_source} data source")
def _fetch_historical_data_batches(exchange, symbol: str, timeframe: str, total_limit: int):
    """
    Fetch historical data in batches like Freqtrade does.
    Makes multiple API calls to get more historical data.
    """
    import time
    
    all_data = []
    batch_size = 1000  # Binance safe limit per request
    batches_needed = min((total_limit // batch_size) + 1, 5)  # Limit to 5 batches max
    
    print(f"🔄 Fetching {batches_needed} batches of {batch_size} bars each...")
    
    # Start from the most recent data and work backwards
    since = None  # Start with most recent
    
    for batch in range(batches_needed):
        try:
            print(f"   Batch {batch + 1}/{batches_needed}...", end="")
            
            # Fetch this batch
            batch_data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
            
            if not batch_data:
                print(" No data")
                break
                
            print(f" {len(batch_data)} bars")
            
            # Add to our collection (prepend since we're going backwards)
            if batch == 0:
                all_data = batch_data
            else:
                # Remove any overlap and prepend older data
                if all_data and batch_data:
                    # Find where old data ends and new data begins
                    last_old_time = batch_data[-1][0]
                    first_new_time = all_data[0][0]
                    
                    if last_old_time < first_new_time:
                        all_data = batch_data + all_data
                    else:
                        # Remove overlap
                        non_overlapping = [bar for bar in batch_data if bar[0] < first_new_time]
                        all_data = non_overlapping + all_data
            
            # Set 'since' to the timestamp of the first bar in this batch for next iteration
            if batch_data:
                since = batch_data[0][0] - (batch_size * _get_timeframe_ms(timeframe))
            
            # Rate limiting - be nice to Binance API
            if batch < batches_needed - 1:
                time.sleep(0.1)  # 100ms delay between requests
                
        except Exception as e:
            print(f" Error: {e}")
            break
    
    print(f"✅ Collected {len(all_data)} total bars from {batches_needed} batches")
    return all_data


def _get_timeframe_ms(timeframe: str) -> int:
    """Convert timeframe to milliseconds."""
    timeframe_ms = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1d': 24 * 60 * 60 * 1000,
    }
    return timeframe_ms.get(timeframe, 60 * 60 * 1000)  # Default to 1h