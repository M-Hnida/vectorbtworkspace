"""Robust data loading with multiple parsing strategies."""
import os
import re
import glob
from typing import Dict, List, Optional, Tuple, Union,Iterator
import pandas as pd
from .base import BaseDataLoader

REQUIRED_COLS = ['open', 'high', 'low', 'close']

class DataValidator:
    """Handles data validation and cleaning."""
    
    @staticmethod
    def validate_ohlc_columns(df: pd.DataFrame) -> bool:
        """Check if DataFrame has required OHLC columns."""
        return all(col in df.columns for col in REQUIRED_COLS)

    @staticmethod
    def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
        """Intelligently detect datetime column."""
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in ['time', 'date', 'datetime']):
                return col
        return df.columns[0] if len(df.columns) > 0 else None
    
    @staticmethod
    def validate_datetime_format(df, col):
        """Validate that the first value of a datetime column is a proper datetime."""
        return col in df and not df[col].empty and pd.to_datetime(df[col].iloc[0], errors='coerce') is not pd.NaT

class DataCleaner:
    """Handles data cleaning operations."""
    
    @staticmethod
    def handle_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing columns gracefully."""
        available_cols = [col for col in REQUIRED_COLS if col in df.columns]
        print(f"📊 Available OHLC columns: {available_cols}")
        if not available_cols:
            raise ValueError("No OHLC columns found in data")
        
        # If some columns are missing, derive them from available data
        if len(available_cols) < 4:
            if 'close' in available_cols and len(available_cols) == 1:
                for col in ['open', 'high', 'low']:
                    if col not in df.columns:
                        df[col] = df['close']
        
        return df
    
    @staticmethod
    def sort_and_validate_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper datetime index and sorting."""
        df = df.sort_index()
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            print("⚠️ Removing duplicate timestamps...")
            df = df[~df.index.duplicated(keep='first')]
        
        # Check for chronological order
        if not df.index.is_monotonic_increasing:
            print("⚠️ Data not in chronological order - sorting...")
            df = df.sort_index()
        
        return df


class MarketData:
    """Encapsulates multi-asset, multi-timeframe OHLC data."""

    def __init__(
        self,
        raw_data: Union[
            pd.DataFrame,
            Dict[str, pd.DataFrame],
            Dict[str, Dict[str, pd.DataFrame]]
        ],
        default_symbol: str = "SYMBOL",
        default_timeframe: str = "1h"
    ):
        """
        Args:
            raw_data: pd.DataFrame or {symbol: df} or {symbol: {timeframe: df}}
        """
        self._data: Dict[str, Dict[str, pd.DataFrame]]

        if isinstance(raw_data, pd.DataFrame):
            self._data = {default_symbol: {default_timeframe: raw_data}}

        elif isinstance(raw_data, dict):
            if all(isinstance(v, pd.DataFrame) for v in raw_data.values()):
                # It's {symbol: df}, assume default_timeframe
                self._data = {
                    symbol: {default_timeframe: df}
                    for symbol, df in raw_data.items()
                }
            elif all(
                isinstance(subdict, dict) and
                all(isinstance(df, pd.DataFrame) for df in subdict.values())
                for subdict in raw_data.values()
            ):
                # It's already nested: {symbol: {timeframe: df}}
                self._data = raw_data  # type: ignore
            else:
                raise TypeError("Invalid raw_data structure.")

    @property
    def symbols(self) -> list[str]:
        """Get list of all available symbols."""
        return list(self._data.keys())

    @property
    def timeframes(self) -> list[str]:
        """Get list of all available timeframes across all symbols."""
        return list({tf for asset in self._data.values() for tf in asset})

    def get_df(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get OHLC DataFrame for a specific symbol and timeframe."""
        return self._data.get(symbol, {}).get(timeframe, None)

    def get_data_timeframe(self, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Get all symbols' data for a given timeframe."""
        return {
            symbol: tfs[timeframe]
            for symbol, tfs in self._data.items()
            if timeframe in tfs
        }

    def get_symbol(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get data from all timeframes for a specific symbol."""
        return self._data.get(symbol, {})

    def get_dict(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Return raw nested data structure."""
        return self._data

    def __getitem__(self, symbol: str) -> Dict[str, pd.DataFrame]:
        return self._data[symbol]

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:
        return f"<MarketData: {len(self._data)} symbols, timeframes: {self.timeframes}>"

    def split(self, split_ratio: float, timeframe: str) -> Tuple['MarketData', 'MarketData']:
        """Split data into train/test sets based on time."""
        train_dict, test_dict = CSVDataLoader.train_test_split(self._data, split_ratio, timeframe)
        return MarketData(train_dict), MarketData(test_dict)

    def harmonize(self) -> 'MarketData':
        """Return new MarketData with harmonized timeframes."""
        harmonized_dict = CSVDataLoader.harmonize_timeframes(self._data)
        return MarketData(harmonized_dict)

class CSVDataLoader(BaseDataLoader):
    """Concrete implementation for CSV data loading."""

    def __init__(self, csv_directory: str = 'data', cache_enabled: bool = True, cache_dir: str = 'cache'):
        self.csv_directory = csv_directory
        self.timeframe_map = {
            '1h': '1H', '4h': '4H', '1d': '1D',
            '15m': '15M', '30m': '30M'
        }
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        if self.cache_enabled and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        """Load and process CSV data with robust error handling and caching."""
        cache_key = f"{symbol}_{interval}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)

        if self.cache_enabled and os.path.exists(cache_path):
            print(f"Read from cache: {cache_path}")
            return pd.read_pickle(cache_path)

        try:
            file_path = self._find_csv_file(symbol, interval)
            if not file_path:
                print(f"❌ No CSV file found for {symbol} ({interval})")
                return pd.DataFrame()

            print(f"📂 Loading {os.path.basename(file_path)}")

            df = self._load_csv_with_multiple_strategies(file_path)
            if df.empty:
                return df

            df = self._process_dataframe(df)

            if self.cache_enabled:
                df.to_pickle(cache_path)
                print(f"Saved to cache: {cache_path}")

            return df

        except Exception as e:
            print(f"❌ Error loading {symbol} ({interval}): {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from CSV files."""
        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        symbols = set()
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            # Extract symbol from filename patterns like BTCUSD_1H_2011-2025.csv
            match = re.match(r'^([A-Z]+)_', filename)
            if match:
                symbols.add(match.group(1))
            else:
                # Fallback: use filename without extension
                symbols.add(os.path.splitext(filename)[0])
        
        return sorted(list(symbols))
    
    def get_available_intervals(self, symbol: str) -> List[str]:
        """Get list of available intervals for a symbol."""
        pattern = os.path.join(self.csv_directory, f"{symbol}_*.csv")
        files = glob.glob(pattern)
        intervals = set()
        
        for file_path in files:
            filename = os.path.basename(file_path)
            # Extract interval from filename
            match = re.search(r'_([0-9]+[HMD])_', filename)
            if match:
                interval_code = match.group(1)
                # Convert back to standard format
                for std_interval, code in self.timeframe_map.items():
                    if code == interval_code:
                        intervals.add(std_interval)
                        break
        
        return sorted(list(intervals))
    
    def _find_csv_file(self, symbol: str, interval: str) -> Optional[str]:
        """Find CSV file with multiple pattern matching."""
        safe_symbol = re.sub(r'[\/=]', '_', symbol)
        csv_abs_path = os.path.abspath(self.csv_directory)
        tf_code = self.timeframe_map.get(interval.lower(), interval.upper())
        
        patterns = [
            f"{safe_symbol}_{tf_code}_*.csv",
            f"{safe_symbol}_{interval.lower()}_*.csv",
            f"{safe_symbol}.csv"
        ]
        
        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(csv_abs_path, pattern)))
            if matches:
                return matches[0]
        
        return None
    
    def _load_csv_with_multiple_strategies(self, file_path: str) -> pd.DataFrame:
        """Try multiple CSV parsing strategies."""
        parsing_strategies = [
            # Strategy 1: Tab-separated without headers (most common for our data)
            lambda: pd.read_csv(file_path, sep='\t', header=None, 
                              names=['datetime', 'open', 'high', 'low', 'close', 'volume']),
            # Strategy 2: Tab-separated with headers  
            lambda: pd.read_csv(file_path, sep='\t'),
            # Strategy 3: CSV with headers
            lambda: pd.read_csv(file_path),
            # Strategy 4: Comma-separated without headers
            lambda: pd.read_csv(file_path, header=None, 
                              names=['datetime', 'open', 'high', 'low', 'close', 'volume']),
            # Strategy 5: Auto-detect separator
            lambda: pd.read_csv(file_path, sep=None, engine='python'),
        ]
        
        for i, strategy in enumerate(parsing_strategies):
            try:
                df = strategy()
                # Better validation: check for multiple columns AND non-NaN data
                if not df.empty and len(df.columns) > 1 and not df.iloc[:, 1:].isna().all().all():
                    print(f"📊 Successfully parsed with strategy {i+1}")
                    return df
            except Exception as e:
                if i == len(parsing_strategies) - 1:
                    print(f"❌ All parsing strategies failed. Last error: {e}")
                continue
        
        return pd.DataFrame()
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the loaded DataFrame."""
        print(f"📊 Processing DataFrame with columns: {list(df.columns)}")
        
        # Detect and set datetime index
        datetime_col = self.validator.detect_datetime_column(df)
        if not datetime_col:
            raise ValueError("Could not detect datetime column")
        
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df.dropna(subset=[datetime_col])
        df = df.set_index(datetime_col)
        
        # Standardize column names
        df.columns = [str(c).lower() for c in df.columns]
        print(f"📊 Columns after standardization: {list(df.columns)}")
        
        # Handle missing columns
        df = self.cleaner.handle_missing_columns(df)
        print(f"📊 Columns after handling missing: {list(df.columns)}")

        # Sort and validate index
        df = self.cleaner.sort_and_validate_index(df)
        
        # Ensure required columns exist
        if not self.validator.validate_ohlc_columns(df):
            print(f"⚠️ Missing required OHLC columns. Available: {list(df.columns)}")
            print(f"⚠️ Required: {REQUIRED_COLS}")
            # Try to map common column variations
            df = self._map_column_variations(df)
        
        if not self.validator.validate_ohlc_columns(df):
            raise ValueError(f"Missing required OHLC columns. Available: {list(df.columns)}")
        
        return df[REQUIRED_COLS + [c for c in df.columns if c not in REQUIRED_COLS]]
    
    def _map_column_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map common column name variations to standard OHLC names."""
        column_mapping = {
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume',
            '0': 'open', '1': 'high', '2': 'low', '3': 'close', '4': 'volume'
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Essaye une fois le mapping positionnel si toujours pas valide
        if not self.validator.validate_ohlc_columns(df):
            cols = list(df.columns)
            if len(cols) >= 4:
                new_names = ['open', 'high', 'low', 'close']
                if len(cols) >= 5:
                    new_names.append('volume')
                mapping = {cols[i]: new_names[i] for i in range(len(new_names))}
                df = df.rename(columns=mapping)
                print(f"📊 Applied positional mapping: {mapping}")

        return df


    @staticmethod
    def train_test_split(data: Dict[str, Dict[str, pd.DataFrame]], split_ratio: float, timeframe: str) -> Tuple[Dict, Dict]:
        """Split data into train/test sets based on time."""
        symbol = next(iter(data))
        symbol_data = data[symbol][timeframe]

        split_idx = int(len(symbol_data) * split_ratio)
        split_date = symbol_data.index[split_idx]

        train_data = {}
        test_data = {}

        for symbol, timeframe_data in data.items():
            train_data[symbol] = {}
            test_data[symbol] = {}
            for tf, df in timeframe_data.items():
                train_data[symbol][tf] = df.loc[df.index < split_date].copy()
                test_data[symbol][tf] = df.loc[df.index >= split_date].copy()

        return train_data, test_data

    @staticmethod
    def harmonize_timeframes(data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Harmonize all dataframes to common time index."""
        all_dataframes = []
        for symbol_data in data.values():
            all_dataframes.extend(symbol_data.values())

        if not all_dataframes:
            return data

        common_index = None
        for df in all_dataframes:
            common_index = df.index if common_index is None else common_index.intersection(df.index)

        if common_index is None or common_index.empty:
            print("\n❌ Datasets don't share common date range - cannot synchronize.")
            return {}

        # Truncate all DataFrames to common index
        harmonized_data = {}
        for symbol in data.keys():
            harmonized_data[symbol] = {}
            for timeframe in data[symbol].keys():
                harmonized_data[symbol][timeframe] = data[symbol][timeframe].loc[common_index].copy()

        # Handle different index types for date formatting
        try:
            start_date = pd.to_datetime(common_index[0]).date()
            end_date = pd.to_datetime(common_index[-1]).date()
            print(f"\n📏 Harmonized time range for all timeframes: {start_date} → {end_date} ({len(common_index)} bars)")
        except:
            raise Exception
        
        return harmonized_data