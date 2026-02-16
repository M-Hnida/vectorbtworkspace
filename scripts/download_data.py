#!/usr/bin/env python3
"""
Data Downloader for NautTrader

Unified data download utility for both cryptocurrency and forex markets.

Usage:
    # Download crypto data (via CCXT/Binance)
    python scripts/download_data.py crypto --symbol BTC/USDT --timeframe 1h --days 30
    
    # Download forex data (via Dukascopy)
    python scripts/download_data.py forex --symbol EURUSD --timeframe h1 --days 365
    
    # Download multiple crypto pairs
    python scripts/download_data.py crypto --symbol BTC/USDT,ETH/USDT --timeframe 5m --days 7

Supported Markets:
    - Crypto: All pairs supported by CCXT (Binance by default)
    - Forex: Pairs available on Dukascopy (see ftmo_symbols.csv)
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_downloaders.crypto_downloader import CryptoDownloader
from scripts.data_downloaders.forex_downloader import ForexDownloader


def main():
    parser = argparse.ArgumentParser(
        description="Download market data for backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crypto
  python scripts/download_data.py crypto --symbol BTC/USDT --timeframe 1h --days 30
  
  # Forex  
  python scripts/download_data.py forex --symbol EURUSD --timeframe h1 --days 365
  
  # Multiple symbols
  python scripts/download_data.py crypto --symbol BTC/USDT,ETH/USDT --timeframe 5m
        """
    )
    
    subparsers = parser.add_subparsers(dest="market", help="Market type")
    
    # Crypto subcommand
    crypto_parser = subparsers.add_parser("crypto", help="Download cryptocurrency data")
    crypto_parser.add_argument(
        "--symbol", "-s",
        default="BTC/USDT",
        help="Trading pair(s), comma-separated (default: BTC/USDT)"
    )
    crypto_parser.add_argument(
        "--timeframe", "-t",
        default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Candlestick timeframe (default: 1h)"
    )
    crypto_parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Days of history to download (default: 30)"
    )
    crypto_parser.add_argument(
        "--exchange", "-e",
        default="binance",
        help="Exchange to use (default: binance)"
    )
    crypto_parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory (default: data)"
    )
    
    # Forex subcommand
    forex_parser = subparsers.add_parser("forex", help="Download forex data")
    forex_parser.add_argument(
        "--symbol", "-s",
        default="EURUSD",
        help="Currency pair (default: EURUSD)"
    )
    forex_parser.add_argument(
        "--timeframe", "-t",
        default="h1",
        choices=["m1", "m5", "m15", "m30", "h1", "h4", "d1"],
        help="Timeframe (default: h1)"
    )
    forex_parser.add_argument(
        "--days", "-d",
        type=int,
        default=365,
        help="Days of history (default: 365)"
    )
    forex_parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory (default: data)"
    )
    forex_parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Download all symbols from ftmo_symbols.csv"
    )
    
    args = parser.parse_args()
    
    if not args.market:
        parser.print_help()
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.market == "crypto":
        downloader = CryptoDownloader(
            exchange_id=args.exchange,
            data_dir=output_dir
        )
        
        symbols = [s.strip() for s in args.symbol.split(",")]
        
        for symbol in symbols:
            try:
                filepath = downloader.download(
                    symbol=symbol,
                    timeframe=args.timeframe,
                    days=args.days
                )
                print(f"[OK] Successfully downloaded {symbol} to {filepath}")
            except Exception as e:
                print(f"[FAIL] Failed to download {symbol}: {e}")
                
    elif args.market == "forex":
        if args.from_csv:
            csv_path = Path("ftmo_symbols.csv")
            if not csv_path.exists():
                print("Error: ftmo_symbols.csv not found")
                sys.exit(1)
            downloader = ForexDownloader(data_dir=output_dir)
            downloader.download_from_csv(csv_path, args.timeframe, args.days)
        else:
            downloader = ForexDownloader(data_dir=output_dir)
            try:
                filepath = downloader.download(
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    days=args.days
                )
                print(f"[OK] Successfully downloaded {args.symbol} to {filepath}")
            except Exception as e:
                print(f"[FAIL] Failed to download {args.symbol}: {e}")


if __name__ == "__main__":
    main()
