"""Data downloaders module."""

from scripts.data_downloaders.crypto_downloader import CryptoDownloader
from scripts.data_downloaders.forex_downloader import ForexDownloader

__all__ = ["CryptoDownloader", "ForexDownloader"]
