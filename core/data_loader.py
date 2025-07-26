"""Robust data loading with multiple parsing strategies."""
import os
import re
import glob
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .base import BaseDataLoader


class DataValidator:
    """Handles data validation and cleaning."""
    
    @staticmethod
    def validate_ohlc_columns(df: pd.DataFrame) -> bool:
        """Check if DataFrame has required OHLC columns."""
        required_cols = ['open', 'high', 'low', 'close']
        return all(col in df.columns for col in required_cols)
    
    @staticmethod
    def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
        """Intelligently detect datetime column."""
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in ['time', 'date', 'datetime']):
                return col
        return df.columns[0] if len(df.columns) > 0 else None
    
    @staticmethod
    def validate_datetime_format(df: pd.DataFrame, datetime_col: str) -> bool:
        """Validate datetime format."""
        try:
            pd.to_datetime(df[datetime_col].iloc[0])
            return True
        except:
            return False


class DataCleaner:
    """Handles data cleaning operations."""
    
    @staticmethod
    def fix_inverted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """Fix inverted OHLC data where high < low."""
        df = df.copy()
        
        # Fix high/low inversions
        df['high'] = df[['high', 'low']].max(axis=1)
        df['low'] = df[['high', 'low']].min(axis=1)
        
        # Ensure open/close are within high/low bounds
        df['open'] = df[['open', 'high', 'low']].apply(
            lambda x: max(min(x['open'], x['high']), x['low']), axis=1
        )
        df['close'] = df[['close', 'high', 'low']].apply(
            lambda x: max(min(x['close'], x['high']), x['low']), axis=1
        )
        
        return df
    
    @staticmethod
    def handle_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing columns gracefully."""
        required_cols = ['open', 'high', 'low', 'close']
        available_cols = [col for col in required_cols if col in df.columns]
        
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


class CSVDataLoader(BaseDataLoader):
    """Concrete implementation for CSV data loading."""
    
    def __init__(self, csv_directory: str = 'data'):
        self.csv_directory = csv_directory
        self.timeframe_map = {
            '1h': '1H', '4h': '4H', '1d': '1D', 
            '15m': '15M', '30m': '30M'
        }
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
    
    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        """Load and process CSV data with robust error handling."""
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
        
        # Fix inverted OHLC data
        df = self.cleaner.fix_inverted_ohlc(df)
        
        # Sort and validate index
        df = self.cleaner.sort_and_validate_index(df)
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not self.validator.validate_ohlc_columns(df):
            print(f"⚠️ Missing required OHLC columns. Available: {list(df.columns)}")
            print(f"⚠️ Required: {required_cols}")
            # Try to map common column variations
            df = self._map_column_variations(df)
        
        if not self.validator.validate_ohlc_columns(df):
            raise ValueError(f"Missing required OHLC columns. Available: {list(df.columns)}")
        
        return df[required_cols + [c for c in df.columns if c not in required_cols]]
    
    def _map_column_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map common column name variations to standard OHLC names."""
        column_mapping = {
            # Common variations
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'VOLUME': 'volume',
            # Numbered columns (for headerless CSV)
            '0': 'open', '1': 'high', '2': 'low', '3': 'close', '4': 'volume'
        }
        
        # Apply mapping
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # If we still don't have OHLC columns, try positional mapping
        if not self.validator.validate_ohlc_columns(df):
            cols = list(df.columns)
            if len(cols) >= 4:
                # Assume first 4 columns are OHLC
                new_names = ['open', 'high', 'low', 'close']
                if len(cols) >= 5:
                    new_names.append('volume')
                
                # Create mapping for existing columns
                mapping = {cols[i]: new_names[i] for i in range(min(len(cols), len(new_names)))}
                df = df.rename(columns=mapping)
                print(f"📊 Applied positional mapping: {mapping}")
        
        return df