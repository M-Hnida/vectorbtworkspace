"""
Forex Downloader Module

Downloads historical data from Dukascopy for forex instruments.
Uses the Node.js dukascopy-node library.
"""

import subprocess
import pandas as pd
from pathlib import Path


class ForexDownloader:
    """Downloads forex data from Dukascopy using Node.js script."""

    # Map symbols to Dukascopy instrument names
    DUKASCOPY_MAP = {
        "EURUSD": "eurusd",
        "GBPUSD": "gbpusd",
        "USDJPY": "usdjpy",
        "AUDUSD": "audusd",
        "USDCAD": "usdcad",
        "NZDUSD": "nzdusd",
        "USDCHF": "usdchf",
        "EURGBP": "eurgbp",
        "EURJPY": "eurjpy",
        "GBPJPY": "gbpjpy",
        "XAUUSD": "xauusd",  # Gold
        "XAGUSD": "xagusd",  # Silver
        "US500.cash": "spxusd",  # S&P 500
        "US30.cash": "djusd",  # Dow Jones
        "BTCUSD": "btcusd",
        "ETHUSD": "ethusd",
    }

    def __init__(self, data_dir: Path = None):
        """
        Initialize the forex downloader.

        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.node_script = Path(__file__).parent / "download_dukascopy.js"

    def download(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "h1",
        days: int = 730,
    ) -> Path:
        """
        Download data for a single forex instrument.

        Args:
            symbol: Symbol (e.g., "EURUSD")
            timeframe: Timeframe (m1, m5, m15, m30, h1, h4, d1)
            days: Number of days of historical data

        Returns:
            Path to the downloaded CSV file
        """
        dukascopy_symbol = self.DUKASCOPY_MAP.get(symbol)
        if not dukascopy_symbol:
            raise ValueError(f"Symbol {symbol} not found in Dukascopy mapping")

        print(f"Downloading {symbol} ({dukascopy_symbol})...")

        cmd = [
            "node",
            str(self.node_script),
            dukascopy_symbol,
            timeframe,
            str(days),
            "--save-as",
            symbol,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to download {symbol}: {result.stderr}")

        print(result.stdout)

        filename = f"{symbol}_{timeframe}.csv"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Expected file not created: {filepath}")

        return filepath

    def download_from_csv(
        self,
        csv_path: Path,
        timeframe: str = "h1",
        days: int = 730,
    ) -> dict[str, Path]:
        """
        Download data for all instruments in CSV file.

        Args:
            csv_path: Path to ftmo_symbols.csv
            timeframe: Timeframe
            days: Number of days

        Returns:
            Dict mapping symbol to downloaded file path
        """
        df = pd.read_csv(csv_path)
        downloaded = {}
        failed = []

        print("=" * 60)
        print("Downloading FTMO Instruments from Dukascopy")
        print("=" * 60)
        print(f"Timeframe: {timeframe.upper()}")
        print(f"Period: Last {days} days")
        print(f"Total symbols: {len(df)}")
        print("=" * 60 + "\n")

        for _, row in df.iterrows():
            symbol = row["Symbole"]

            if symbol not in self.DUKASCOPY_MAP:
                print(f"[SKIP] {symbol} (not available in Dukascopy)")
                failed.append(symbol)
                continue

            try:
                filepath = self.download(symbol, timeframe, days)
                downloaded[symbol] = filepath
                print(f"[OK] {symbol} downloaded successfully\n")
            except Exception as e:
                print(f"[FAIL] Failed to download {symbol}: {e}\n")
                failed.append(symbol)

        print("\n" + "=" * 60)
        print("Download Summary")
        print("=" * 60)
        print(f"[OK] Successful: {len(downloaded)}")
        print(f"[FAIL] Failed: {len(failed)}")
        if failed:
            print(f"Failed symbols: {', '.join(failed)}")
        print("=" * 60 + "\n")

        return downloaded

    def load_data(self, symbol: str, timeframe: str = "h1") -> pd.DataFrame:
        """Load downloaded data for a symbol."""
        filename = f"{symbol}_{timeframe}.csv"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = df.columns.str.lower()

        if df.index.tz is None:
            df.index = df.index.tz_localize("utc")
        else:
            df.index = df.index.tz_convert("utc")

        return df
