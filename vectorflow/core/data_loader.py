#!/usr/bin/env python3
"""
Robust OHLC data loader for CSV files.
Handles various CSV formats: tabs, commas, semicolons, mixed case columns,
unsorted data, missing volume, custom date formats, etc.
"""

from datetime import datetime
from typing import Optional, Union
import pandas as pd


def load_ohlc_csv(
    file_path: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    delimiter: Optional[str] = None,
) -> pd.DataFrame:
    """Load and clean OHLC CSV data with robust error handling.

    Handles various CSV formats:
    - Any delimiter (comma, tab, semicolon, pipe, etc.)
    - Headers or no headers
    - Mixed case column names (Open, OPEN, open, etc.)
    - Unsorted timestamps
    - Missing volume column
    - Custom date formats (including "2020-03-13 08-PM")
    - Extra columns (ignored)

    Args:
        file_path: Path to CSV file
        start_date: Optional start date for filtering (inclusive)
        end_date: Optional end date for filtering (inclusive)
        delimiter: Optional delimiter override (auto-detected if None)

    Returns:
        DataFrame with DatetimeIndex and standardized columns:
        ['open', 'high', 'low', 'close', 'volume']
        (volume column optional)

    Raises:
        ValueError: If file cannot be parsed or required columns are missing
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> # Load entire file
        >>> df = load_ohlc_csv("data/BTCUSD.csv")

        >>> # Load with time range
        >>> df = load_ohlc_csv("data/BTCUSD.csv", start_date="2020-01-01", end_date="2023-12-31")

        >>> # Load tab-delimited file
        >>> df = load_ohlc_csv("data/EURUSD.txt", delimiter="\\t")
    """

    try:
        # Step 1: Read first line to detect headers and custom date formats
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        has_headers = _detect_headers(first_line)
        needs_custom_parsing = _needs_custom_date_parsing(first_line)

        # Step 2: Read CSV with appropriate settings
        if needs_custom_parsing:
            df = _read_csv_with_custom_dates(file_path, has_headers, delimiter)
        else:
            df = _read_csv_standard(file_path, has_headers, delimiter)

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file {file_path}: {exc}") from exc

    # Step 3: Ensure datetime index
    df = _ensure_datetime_index(df)

    # Step 4: Standardize column names (open, high, low, close, volume)
    df = _standardize_columns(df, has_headers)

    # Step 5: Validate required columns
    _validate_required_columns(df)

    # Step 6: Clean data (remove NaN, duplicates)
    df = _clean_dataframe(df)

    # Step 7: Sort by date
    df = df.sort_index()

    # Step 8: Apply time range filter
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if df.empty:
        raise ValueError(f"No data remains after filtering: {file_path}")

    return df


# ==================== Helper Functions ====================


def _detect_headers(first_line: str) -> bool:
    """Detect if CSV has header row."""
    keywords = ["open", "high", "low", "close", "time", "date", "timestamp", "datetime"]
    return any(kw in first_line.lower() for kw in keywords)


def _needs_custom_date_parsing(first_line: str) -> bool:
    """Detect if file uses custom date format like '2020-03-13 08-PM'."""
    return "-PM" in first_line or "-AM" in first_line


def _parse_custom_date(date_str):
    """Parse custom date formats like '2020-03-13 08-PM'."""
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


def _read_csv_with_custom_dates(
    file_path: str, has_headers: bool, delimiter: Optional[str]
) -> pd.DataFrame:
    """Read CSV with custom date parsing."""
    df = pd.read_csv(
        file_path,
        sep=delimiter if delimiter else None,
        header=0 if has_headers else None,
        engine="python",
    )

    # Apply custom date parsing to first column
    df.iloc[:, 0] = df.iloc[:, 0].apply(_parse_custom_date)
    df.set_index(df.columns[0], inplace=True)

    return df


def _read_csv_standard(
    file_path: str, has_headers: bool, delimiter: Optional[str]
) -> pd.DataFrame:
    """Read CSV with standard pandas parsing."""
    return pd.read_csv(
        file_path,
        sep=delimiter if delimiter else None,
        header=0 if has_headers else None,
        parse_dates=[0],
        index_col=0,
        date_format="mixed",
        engine="python",
    )


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index is DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]

    if df.empty:
        raise ValueError("No valid timestamps found in data")

    return df


def _standardize_columns(df: pd.DataFrame, has_headers: bool) -> pd.DataFrame:
    """Standardize column names to lowercase OHLCV."""
    if has_headers:
        # Smart column mapping - handle mixed case
        mapping = {}
        existing_lower = [col.lower().strip() for col in df.columns]
        standard_cols = ["open", "high", "low", "close", "volume"]

        for std_col in standard_cols:
            # Find best match
            match_idx = None
            for i, col in enumerate(existing_lower):
                if std_col == col or std_col in col or col in std_col:
                    match_idx = i
                    break

            if match_idx is not None and match_idx < len(df.columns):
                mapping[df.columns[match_idx]] = std_col

        df = df.rename(columns=mapping)
    else:
        # No headers - assign by position (handle variable columns)
        n_cols = len(df.columns)
        if n_cols >= 4:
            # At least OHLC
            col_names = ["open", "high", "low", "close"]
            if n_cols >= 5:
                col_names.append("volume")
            # Add any extra columns as-is
            for i in range(len(col_names), n_cols):
                col_names.append(f"col_{i}")
            df.columns = col_names
        else:
            raise ValueError(f"CSV has only {n_cols} columns, need at least 4 (OHLC)")

    return df


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that required OHLC columns exist."""
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe: remove NaN, duplicates."""
    # Keep only valid OHLCV columns
    valid_cols = ["open", "high", "low", "close", "volume"]
    available = [col for col in valid_cols if col in df.columns]
    df = df[available]

    # Remove rows with any NaN in OHLC (volume can be NaN)
    required_cols = ["open", "high", "low", "close"]
    df = df.dropna(subset=required_cols)

    # Remove duplicate timestamps (keep last)
    df = df[~df.index.duplicated(keep="last")]

    return df


# ==================== Convenience Functions ====================


def load_multiple_csvs(
    file_paths: list[str],
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
) -> dict[str, pd.DataFrame]:
    """Load multiple CSV files and return dict of DataFrames.

    Args:
        file_paths: List of CSV file paths
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering

    Returns:
        Dict mapping filename (without extension) to DataFrame

    Example:
        >>> data = load_multiple_csvs(["data/BTC.csv", "data/ETH.csv"])
        >>> btc_df = data["BTC"]
        >>> eth_df = data["ETH"]
    """
    import os

    result = {}
    for path in file_paths:
        try:
            df = load_ohlc_csv(path, start_date, end_date)
            filename = os.path.splitext(os.path.basename(path))[0]
            result[filename] = df
            print(f"✅ Loaded {filename}: {len(df)} bars")
        except Exception as e:
            print(f"⚠️  Failed to load {path}: {e}")

    return result


def preview_csv(file_path: str, n_rows: int = 5) -> None:
    """Preview CSV file structure (useful for debugging).

    Args:
        file_path: Path to CSV file
        n_rows: Number of rows to display
    """
    print(f"\n📄 Preview of: {file_path}\n")

    try:
        df = load_ohlc_csv(file_path)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}\n")
        print(df.head(n_rows))
        print("\n" + "=" * 60)
    except Exception as e:
        print(f"❌ Error: {e}")
