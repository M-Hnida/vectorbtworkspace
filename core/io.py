"""Data I/O operations - CSV loading, caching, data splits.
"""
import glob
import os
import re
from typing import Dict, Tuple, Optional

import pandas as pd



def train_test_split(data: Dict[str, Dict[str, pd.DataFrame]], split_ratio: float, timeframe: str) -> Tuple[Dict, Dict]:
    """Split data into train/test sets based on time.

    Args:
        data: Dictionary of market data by symbol and timeframe
        split_ratio: Ratio for train/test split (e.g., 0.7 for 70% train)
        timeframe: The main timeframe to use for determining split date

    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
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


class MarketData:
    """
    Encapsulates multi-asset, multi-timeframe OHLC data.

    Provides a clean, type-safe interface for accessing market data instead of
    nested dictionaries. Supports safe access with None returns for missing data.

    Example usage:
        data = MarketData(raw_dict_data)
        df = data.get("BTCUSDT", "1h")
        all_1h = data.get_timeframe("1h")
        print(data.symbols)  # ['BTCUSDT', 'ETHUSDT']
    """

    def __init__(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Initialize with nested dictionary data structure."""
        self._data = raw_data

    @property
    def symbols(self) -> list[str]:
        """Get list of all available symbols."""
        return list(self._data.keys())

    @property
    def timeframes(self) -> list[str]:
        """Get list of all available timeframes across all symbols."""
        return list({tf for asset in self._data.values() for tf in asset})

    def get(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get OHLC DataFrame for a specific symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')

        Returns:
            DataFrame if found, None otherwise
        """
        return self._data.get(symbol, {}).get(timeframe, None)

    def get_timeframe(self, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Get all symbols' data for a given timeframe.

        Args:
            timeframe: Timeframe to extract

        Returns:
            Dictionary mapping symbol -> DataFrame for the timeframe
        """
        return {
            symbol: tfs[timeframe]
            for symbol, tfs in self._data.items()
            if timeframe in tfs
        }

    def get_symbol(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get all timeframes for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary mapping timeframe -> DataFrame for the symbol
        """
        return self._data.get(symbol, {})

    def get_all(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Return raw nested data structure."""
        return self._data

    def __getitem__(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Allow dict-like access: data['BTCUSDT']"""
        return self._data[symbol]

    def __contains__(self, symbol: str) -> bool:
        """Support 'in' operator: 'BTCUSDT' in data"""
        return symbol in self._data

    def __len__(self) -> int:
        """Return number of symbols."""
        return len(self._data)

    def __iter__(self):
        """Iterate over symbols."""
        return iter(self._data)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<MarketData: {len(self._data)} symbols, timeframes: {self.timeframes}>"

    def split(self, split_ratio: float, timeframe: str) -> Tuple['MarketData', 'MarketData']:
        """Split data into train/test sets based on time.

        Args:
            split_ratio: Ratio for train/test split (e.g., 0.7 for 70% train)
            timeframe: The main timeframe to use for determining split date

        Returns:
            Tuple of (train_data, test_data) MarketData objects
        """
        train_dict, test_dict = train_test_split(self._data, split_ratio, timeframe)
        return MarketData(train_dict), MarketData(test_dict)

    def harmonize(self) -> 'MarketData':
        """Return new MarketData with harmonized timeframes."""
        harmonized_dict = harmonize_timeframes(self._data)
        return MarketData(harmonized_dict)


if __name__ == '__main__':
    print("This script is a module and is not meant to be run directly.")
    print("Please run the main pipeline via 'runner.py' from the project root.")
