#!/usr/bin/env python3
"""
Enhanced OHLC data loader with multiple data sources.
Supports CSV, CCXT (Binance), and Freqtrade data sources.
"""

import os
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import yaml
from constants import REQUIRED_OHLCV_COLUMNS, REQUIRED_MINIMUM_COLUMNS

# Time range parsing mappings
TIME_UNITS = {"y": 365, "m": 30, "d": 1, "w": 7}

# Timeframe normalization mapping
TIMEFRAME_MAPPING = {
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "15min": "15m",
    "30min": "30m",
    "1hour": "1h",
    "4hour": "4h",
    "1day": "1d",
}


def _parse_custom_date(date_str):
    """Parse custom date formats like '2020-03-13 08-PM'"""
    try:
        if "-PM" in date_str or "-AM" in date_str:
            date_part, time_part = date_str.split(" ")
            hour_str, ampm = time_part.split("-")
            hour = int(hour_str)

            if ampm == "PM" and hour != 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0

            return pd.to_datetime(f"{date_part} {hour:02d}:00:00")
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT


def load_ohlc_csv(
    file_path: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
    """Load and clean OHLC CSV data with optional time range filtering and forex gap handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        has_headers = any(
            kw in first_line.lower()
            for kw in ["open", "high", "low", "close", "time", "date", "timestamp"]
        )
        needs_custom_parsing = "-PM" in first_line or "-AM" in first_line

        if needs_custom_parsing:
            df = pd.read_csv(
                file_path, sep=None, header=0 if has_headers else None, engine="python"
            )
            df.iloc[:, 0] = df.iloc[:, 0].apply(_parse_custom_date)
            df.set_index(df.columns[0], inplace=True)
        else:
            df = pd.read_csv(
                file_path,
                sep=None,
                header=0 if has_headers else None,
                parse_dates=[0],
                index_col=0,
                date_format="mixed",
                engine="python",
            )

    except Exception as exc:
        raise ValueError(f"Could not parse CSV file {file_path}: {exc}") from exc

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]

    # Standardize column names
    columns = REQUIRED_OHLCV_COLUMNS
    if has_headers:
        mapping: Dict[str, str] = {}
        existing_lower = [col.lower() for col in df.columns]
        for i, std_col in enumerate(columns):
            if i >= len(df.columns):
                break
            match_idx = next(
                (
                    j
                    for j, col in enumerate(existing_lower)
                    if std_col in col or col in std_col
                ),
                i,
            )
            if match_idx < len(df.columns):
                mapping[str(df.columns[match_idx])] = std_col
        df = df.rename(columns=mapping)
    else:
        df.columns = columns[: len(df.columns)]

    # Validate and clean
    missing = [col for col in REQUIRED_MINIMUM_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    available = [col for col in columns if col in df.columns]
    df = df[available].dropna().sort_index()

    # Apply time range filter
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    return df


def load_strategy_config(strategy_name: str, config_type: str = "auto") -> Dict:
    """Load strategy configuration from YAML or Freqtrade JSON file."""
    if config_type == "auto":
        config_type = _detect_config_type(strategy_name)

    config_loaders = {"yaml": _load_yaml_config, "freqtrade": _load_freqtrade_config}

    if config_type not in config_loaders:
        raise ValueError(f"Unsupported config type: {config_type}")

    return config_loaders[config_type](strategy_name)


def _detect_config_type(strategy_name: str) -> str:
    """Auto-detect configuration file type."""
    for ext, config_type in [(".yaml", "yaml"), (".json", "freqtrade")]:
        if os.path.exists(f"config/{strategy_name}{ext}"):
            return config_type
    raise FileNotFoundError(f"No configuration file found for {strategy_name}")


def _load_yaml_config(strategy_name: str) -> Dict:
    """Load YAML configuration."""
    config_path = f"config/{strategy_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_freqtrade_config(strategy_name: str) -> Dict:
    """Load and convert Freqtrade JSON configuration."""
    config_path = f"config/{strategy_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Freqtrade configuration file not found: {config_path}"
        )

    return _convert_freqtrade_config_internal(config_path)


def load_data_for_strategy(
    strategy,
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
    data_source: str = "auto",
    **kwargs,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load data from various sources and return dict: symbol -> timeframe -> DataFrame."""
    if data_source == "auto":
        data_source = _detect_data_source(strategy)

    print(f"📊 Loading data using {data_source} source...")

    if data_source == "csv":
        return _load_csv_data_for_strategy(strategy, time_range, end_date)
    elif data_source == "freqtrade":
        return _load_freqtrade_data_for_strategy(
            strategy, time_range, end_date, **kwargs
        )
    elif data_source == "ccxt":
        raise ValueError(
            "CCXT data source deprecated. Use vectorbt.CCXTData.download() directly in your strategy or switch to CSV/Freqtrade data sources."
        )
    else:
        raise ValueError(
            f"Unsupported data source: {data_source}. Use 'csv' or 'freqtrade'."
        )


def _detect_data_source(strategy) -> str:
    """Auto-detect the appropriate data source based on configuration."""
    data_source = strategy.get_parameter("data_source", "").lower()
    if data_source in ["csv", "freqtrade"]:
        return data_source

    if data_source == "ccxt":
        raise ValueError(
            "CCXT data source deprecated. Use vectorbt.CCXTData.download() directly in your strategy."
        )

    freqtrade_dir = strategy.get_parameter("freqtrade_data_dir")
    if freqtrade_dir and os.path.exists(freqtrade_dir):
        return "freqtrade"

    return "csv"


def _load_csv_data_for_strategy(
    strategy,
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load CSV data for strategy."""
    required_timeframes = strategy.get_required_timeframes()
    csv_paths = strategy.get_parameter("csv_path", [])

    if not csv_paths:
        try:
            config = load_strategy_config(strategy.name)
            csv_paths = config.get("csv_path", [])
        except Exception:
            csv_paths = []

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    if not csv_paths:
        csv_paths = _find_default_csv_files()

    return _load_csv_files(csv_paths, required_timeframes, time_range, end_date)


def _find_default_csv_files() -> List[str]:
    """Find default CSV files in data directory."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        raise ValueError(
            "No csv_path defined in strategy configuration and no data directory found"
        )

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise ValueError(
            "No CSV files found in data directory and no csv_path specified"
        )

    return [os.path.join(data_dir, csv_files[0])]


def _load_freqtrade_data_for_strategy(
    strategy,
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
    **kwargs,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load Freqtrade data for strategy."""
    required_timeframes = strategy.get_required_timeframes()
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


def _parse_time_range(time_range: str) -> Optional[timedelta]:
    """Parse time range string into timedelta object."""
    if not time_range:
        return None

    time_range = time_range.lower().strip()
    unit = time_range[-1]
    value = int(time_range[:-1])

    if unit not in TIME_UNITS:
        raise ValueError(
            f"Unsupported time range format: {time_range}. Use format like '2y', '6m', '30d', '4w'"
        )

    if unit == "w":
        return timedelta(weeks=value)
    else:
        return timedelta(days=value * TIME_UNITS[unit])


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

    valid_dfs = [df for df in dataframes.values() if not df.empty]
    if not valid_dfs:
        return dataframes

    latest_available = min(df.index.max() for df in valid_dfs)
    actual_end = (
        min(pd.to_datetime(end_date), latest_available)
        if end_date
        else latest_available
    )
    actual_start = max(actual_end - time_delta, max(df.index.min() for df in valid_dfs))

    print(f"📅 Applying time filter: {time_range} ({actual_start} to {actual_end})")

    filtered = {}
    for tf, df in dataframes.items():
        if not df.empty:
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

    # Ensure datetime index and find valid dataframes
    valid_dfs = []
    for tf, df in dataframes.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
            dataframes[tf] = df
        if not df.empty:
            valid_dfs.append((tf, df))

    if not valid_dfs:
        return {}

    # Find common overlapping period
    common_start = max(df.index.min() for _, df in valid_dfs)
    common_end = min(df.index.max() for _, df in valid_dfs)

    # Apply time range constraints
    if time_range:
        time_delta = _parse_time_range(time_range)
        if time_delta:
            actual_end = min(
                pd.to_datetime(end_date) if end_date else common_end, common_end
            )
            actual_start = max(common_start, actual_end - time_delta)
        else:
            actual_start, actual_end = common_start, common_end
    else:
        actual_start, actual_end = common_start, common_end

    print(f"📅 Harmonizing to common period: {actual_start} to {actual_end}")

    # Filter all dataframes to common period
    harmonized = {}
    for tf, df in dataframes.items():
        if not df.empty:
            mask = (df.index >= actual_start) & (df.index <= actual_end)
            filtered_df = df.loc[mask].copy()
            if not filtered_df.empty:
                harmonized[tf] = filtered_df
                print(f"✅ {tf}: {len(filtered_df)} bars")

    return harmonized


def _extract_timeframe_from_filename(filename: str) -> Optional[str]:
    """Extract timeframe from filename like EURUSD_1H_2009-2025.csv -> 1h"""
    parts = os.path.basename(filename).split("_")
    if len(parts) >= 2:
        tf = parts[1].lower()
        return TIMEFRAME_MAPPING.get(tf, tf)
    return None


def _load_csv_files(
    csv_paths: List[str],
    required_timeframes: List[str],
    time_range: Optional[str] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load CSV files and return symbol -> timeframe -> DataFrame structure."""
    symbol_files: Dict[str, List[str]] = {}
    for path in csv_paths:
        symbol = os.path.basename(path).split("_")[0]
        symbol_files.setdefault(symbol, []).append(path)

    all_symbols_data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for symbol, files in symbol_files.items():
        try:
            data = _load_symbol_csv_files(files, required_timeframes)
            if not data:
                raise ValueError(f"No data loaded for {symbol}")

            # Apply time filtering
            filtered = (
                _harmonize_time_ranges(data, time_range, end_date)
                if len(data) > 1
                else _apply_time_range_filter(data, time_range, end_date)
            )

            # Keep only required timeframes
            filtered = {
                tf: filtered[tf] for tf in required_timeframes if tf in filtered
            }

            if filtered:
                all_symbols_data[symbol] = filtered

        except Exception as e:
            print(f"⚠️ Failed to load CSV data for {symbol}: {e}")

    return all_symbols_data


def _load_symbol_csv_files(
    files: List[str], required_timeframes: List[str]
) -> Dict[str, pd.DataFrame]:
    """Load CSV files for a single symbol."""
    data: Dict[str, pd.DataFrame] = {}

    # Try to extract timeframe from filename first
    for file_path in files:
        tf_from_file = _extract_timeframe_from_filename(file_path)
        if tf_from_file and tf_from_file in required_timeframes:
            data[tf_from_file] = load_ohlc_csv(file_path)
            print(
                f"✅ Loaded {tf_from_file} data from {file_path} ({len(data[tf_from_file])} bars)"
            )

    # Fallback: Map files to timeframes by position
    if not data:
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

    return data


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
def import_freqtrade_config(config_path: str) -> Dict[str, Any]:
    """Import Freqtrade configuration."""
    return _convert_freqtrade_config_internal(config_path)


def copy_freqtrade_config(source_path: str, target_path: str) -> Dict[str, Any]:
    """Copy and convert Freqtrade config to YAML format."""
    ft_config = import_freqtrade_config(source_path)

    with open(target_path, "w", encoding="utf-8") as f:
        yaml.dump(ft_config, f, default_flow_style=False)

    print(f"✅ Converted Freqtrade config to {target_path}")
    return ft_config


def switch_data_source(config_file: str, new_source: str) -> None:
    """Switch data source in a YAML config file."""
    if new_source not in ["csv", "freqtrade"]:
        raise ValueError("Data source must be 'csv' or 'freqtrade'")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "parameters" not in config:
        config["parameters"] = {}
    config["parameters"]["data_source"] = new_source

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Updated {config_file} to use {new_source} data source")
