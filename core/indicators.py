"""Technical indicators computation."""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Callable
import warnings


def safe_execute(func: Callable, fallback_value=None, error_msg: str = None):
    """Centralized error handling utility for indicator calculations."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func()
    except Exception as e:
        if error_msg:
            print(f"⚠️ {error_msg}: {e}")
        return fallback_value


def get_scalar(value):
    """Safely extract a scalar from common pandas objects or scalars."""
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return np.nan
        return value.values.flatten()[0]

    if isinstance(value, pd.Series):
        if value.empty:
            return np.nan
        return value.iloc[0]

    return value


def compute_vol_momentum(returns: pd.Series, vol_window: int, vol_momentum_window: int) -> pd.Series:
    """Compute volatility momentum as difference between current and lagged volatility."""
    volatility = returns.rolling(vol_window).std()
    return volatility - volatility.shift(vol_momentum_window)


def compute_wma(close: pd.Series, window: int) -> pd.Series:
    """Compute Weighted Moving Average with fallback."""
    def _compute():
        result = ta.wma(close, length=window)
        if result is None:
            return close.rolling(window).mean()
        return result

    return safe_execute(
        _compute,
        fallback_value=close.rolling(window).mean(),
        error_msg="WMA calculation failed, using SMA"
    )


def add_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Add all technical indicators needed for the strategy."""
    df = df.copy()

    # Check if this is a momentum strategy (has volatility_momentum_window)
    if 'volatility_momentum_window' in config:
        # Momentum strategy indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(config['volatility_momentum_window']).std()
        df['volatility_momentum'] = df['volatility'] - df['volatility'].shift(config['volatility_momentum_window'])

        # WMA with error handling
        try:
            wma = ta.wma(df['close'], length=config['price_wma_window'])
            df['wma'] = wma if wma is not None else df['close'].rolling(config['price_wma_window']).mean()
        except:
            df['wma'] = df['close'].rolling(config['price_wma_window']).mean()

        # ATR for position sizing with error handling
        try:
            atr = ta.atr(df['high'], df['low'], df['close'], length=config['atr_length'])
            if atr is not None:
                df[f'ATRr_{config["atr_length"]}'] = atr
            else:
                # Fallback ATR calculation
                tr1 = df['high'] - df['low']
                tr2 = abs(df['high'] - df['close'].shift())
                tr3 = abs(df['low'] - df['close'].shift())
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df[f'ATRr_{config["atr_length"]}'] = true_range.rolling(config['atr_length']).mean()
        except Exception as e:
            raise e
    else:
        # For other strategies (like LTI), just add basic price data
        # The strategy-specific indicators will be added in the signals module
        df['returns'] = df['close'].pct_change()

    return df.dropna()


def timeframe_to_pandas_freq(timeframe: str) -> str:
    """Convert custom timeframe strings to pandas offset aliases."""
    lower = timeframe.lower()
    mapping = {
        'daily': '1D', '1d': '1D', 'd': '1D', 'day': '1D',
        'hourly': '1h', '1h': '1h', 'h': '1h',
        '30min': '30T', '15min': '15T', '5min': '5T', '1min': '1T'
    }
    return mapping.get(lower, timeframe)
