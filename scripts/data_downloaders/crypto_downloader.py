"""
Crypto Downloader Module

Downloads historical OHLCV data from cryptocurrency exchanges using CCXT.
Supports Binance, Bybit, OKX, and other major exchanges.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time


class CryptoDownloader:
    """Downloads historical crypto OHLCV data using CCXT."""

    TIMEFRAME_MS = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    def __init__(self, exchange_id: str = "binance", data_dir: Path = None):
        """
        Initialize the downloader.

        Args:
            exchange_id: CCXT exchange ID (e.g., 'binance', 'bybit', 'okx')
            data_dir: Directory to save downloaded data
        """
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def download(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "5m",
        days: int = 730,
    ) -> Path:
        """
        Download historical OHLCV data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe
            days: Number of days of historical data

        Returns:
            Path to the saved CSV file
        """
        print(f"Loading {self.exchange_id} markets...")
        self.exchange.load_markets()

        since_dt = datetime.now() - timedelta(days=days)
        since_ms = int(since_dt.timestamp() * 1000)
        timeframe_ms = self.TIMEFRAME_MS.get(timeframe, 5 * 60 * 1000)

        all_ohlcv = []
        current_since = since_ms

        print(f"Downloading {symbol} {timeframe} data from {since_dt.strftime('%Y-%m-%d')}...")

        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=1000
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                first_ts = datetime.fromtimestamp(ohlcv[0][0] / 1000)
                last_ts = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                print(f"  Fetched {len(ohlcv):>4} candles: {first_ts} -> {last_ts}")

                current_since = ohlcv[-1][0] + timeframe_ms

                if current_since >= int(datetime.now().timestamp() * 1000):
                    break

                time.sleep(0.1)

            except ccxt.NetworkError as e:
                print(f"Network error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            except ccxt.ExchangeError as e:
                print(f"Exchange error: {e}")
                break

        if not all_ohlcv:
            raise ValueError(f"No data fetched for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.sort_index()

        # Save to CSV
        symbol_clean = symbol.replace("/", "")
        start_year = df.index[0].year
        end_year = df.index[-1].year
        filename = f"{symbol_clean}_{timeframe}_{start_year}-{end_year}.csv"
        filepath = self.data_dir / filename

        df.to_csv(filepath)

        print(f"\nDownload Complete!")
        print(f"Symbol: {symbol}")
        print(f"Period: {df.index[0]} to {df.index[-1]}")
        print(f"Total candles: {len(df):,}")
        print(f"Saved to: {filepath}")

        return filepath

    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load previously downloaded data."""
        symbol_clean = symbol.replace("/", "")
        pattern = f"{symbol_clean}_{timeframe}_*.csv"
        files = list(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No data file found matching: {pattern}")

        filepath = max(files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)

        if df.index.tz is None:
            df.index = df.index.tz_localize("utc")

        return df
