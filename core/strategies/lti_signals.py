"""Signal generation for the Logical Trading Indicator (LTI) strategy.

Translates the LTI Pine Script into Python for use with vectorbt.
"""
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import pandas_ta as ta

from ..io import MarketData


def pine_atr_trailing_stop(high: pd.Series, low: pd.Series, close: pd.Series, atr_period: int, atr_multiple: float) -> pd.Series:
    """
    Simplified ATR Trailing Stop calculation.
    Uses a more efficient approach than the original Pine Script logic.
    """
    try:
        atr_val = ta.atr(high, low, close, length=atr_period)
        if atr_val is None:
            # Fallback ATR calculation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_val = tr.rolling(atr_period).mean()

        stop_loss_val = atr_multiple * atr_val

        # Simplified trailing stop: close - ATR for long positions
        atr_ts = close - stop_loss_val

        # Apply a simple trailing logic using rolling max
        atr_ts = atr_ts.rolling(window=min(20, len(atr_ts)), min_periods=1).max()

        return atr_ts.fillna(close)  # Fill any remaining NaN with close price

    except (ValueError, IndexError) as e:
        print(f"⚠️ ATR calculation error (invalid parameters or data): {e}")
        return close * 0.95  # Simple 5% trailing stop
    except (AttributeError, KeyError) as e:
        print(f"⚠️ ATR calculation error (missing data attributes): {e}")
        return close * 0.95  # Simple 5% trailing stop
    except Exception as e:
        print(f"⚠️ Unexpected error in ATR calculation: {e}")
        return close * 0.95  # Simple 5% trailing stop


def add_lti_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add all indicators required by the LTI strategy."""

    # ATR Trailing Stop
    df['atr_ts'] = pine_atr_trailing_stop(
        df['high'], df['low'], df['close'],
        params['atr_period'], params['atr_multiple']
    )

    # Bollinger Bands
    basis_type = params.get('basis_type', 'EMA').lower()
    basis_length = params['basis_length']
    bb_std_dev = params['bb_std_dev']

    if basis_type == 'ema':
        df['bb_basis'] = ta.ema(df['close'], length=basis_length)
    else:
        df['bb_basis'] = ta.sma(df['close'], length=basis_length)

    # Bollinger Bands with error handling
    try:
        bbands = ta.bbands(df['close'], length=basis_length, std=bb_std_dev)
        if bbands is not None:
            df['bb_upper'] = bbands[f'BBU_{basis_length}_{bb_std_dev}']
            df['bb_lower'] = bbands[f'BBL_{basis_length}_{bb_std_dev}']
        else:
            # Fallback BB calculation
            sma = df['close'].rolling(basis_length).mean()
            std = df['close'].rolling(basis_length).std()
            df['bb_upper'] = sma + (std * bb_std_dev)
            df['bb_lower'] = sma - (std * bb_std_dev)
    except:
        # Fallback BB calculation
        sma = df['close'].rolling(basis_length).mean()
        std = df['close'].rolling(basis_length).std()
        df['bb_upper'] = sma + (std * bb_std_dev)
        df['bb_lower'] = sma - (std * bb_std_dev)

    # Keltner Channels with error handling
    try:
        kc = ta.kc(df['high'], df['low'], df['close'], length=basis_length)
        if kc is not None:
            df['kc_upper'] = kc[f'KCUe_{basis_length}_2']
            df['kc_lower'] = kc[f'KCLe_{basis_length}_2']
        else:
            # Fallback KC calculation
            ema = ta.ema(df['close'], length=basis_length)
            atr = ta.atr(df['high'], df['low'], df['close'], length=basis_length)
            df['kc_upper'] = ema + (2 * atr)
            df['kc_lower'] = ema - (2 * atr)
    except:
        # Fallback KC calculation
        ema = ta.ema(df['close'], length=basis_length)
        if ema is None:
            ema = df['close'].ewm(span=basis_length).mean()
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(basis_length).mean()
        df['kc_upper'] = ema + (2 * atr)
        df['kc_lower'] = ema - (2 * atr)

    df['momentum_on'] = (df['bb_lower'] < df['kc_lower']) & (df['bb_upper'] > df['kc_upper'])

    return df


def _generate_single_lti_signal(df: pd.DataFrame, params: dict) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate signals for one asset with one set of parameters."""
    df = add_lti_indicators(df.copy(), params)

    close = df['close']
    atr_ts = df['atr_ts']
    basis = df['bb_basis']

    # Crossover/Crossunder conditions
    ema = ta.ema(close, params['basis_length'])
    if ema is None:
        ema = close.ewm(span=params['basis_length']).mean()

    above_basis = (ema > atr_ts) & (ema.shift(1) <= atr_ts.shift(1))
    below_basis = (ema < atr_ts) & (ema.shift(1) >= atr_ts.shift(1))

    # Momentum filter
    use_momentum_filter = params.get('enable_consolidation_filter', True)
    momentum_ok = df['momentum_on'] if use_momentum_filter else True

    # --- Raw Signals ---
    raw_buy_signal = (close > atr_ts) & above_basis & momentum_ok
    raw_sell_signal = (close < atr_ts) & below_basis & momentum_ok

    # --- Position-based Filtering ---
    # Use a state machine to ensure signals alternate
    position = pd.Series(np.nan, index=df.index)
    position.iloc[0] = 0

    position = np.where(raw_buy_signal, 1, position)
    position = np.where(raw_sell_signal, -1, position)
    position = pd.Series(position, index=df.index).ffill()

    entries = (position == 1) & (position.shift(1) == -1)
    short_entries = (position == -1) & (position.shift(1) == 1)

    # --- Take Profit Signals ---
    long_tp = (close < df['bb_upper']) & (close.shift(1) >= df['bb_upper'].shift(1))
    short_tp = (close > df['bb_lower']) & (close.shift(1) <= df['bb_lower'].shift(1))

    # An exit is either a TP or an opposing entry signal
    exits = long_tp | short_entries
    short_exits = short_tp | entries

    return entries, exits, short_entries, short_exits


def generate_signals(
    data: MarketData,
    param_grid: List[Tuple],
    timeframe: str,
    config: Dict,
    return_params: bool = False,
) -> Tuple[Dict, Any]:
    """Generate signals for the LTI strategy across all assets and parameters."""
    param_names = ['atr_period', 'atr_multiple', 'basis_length', 'bb_std_dev']

    # Collect all signals
    entries = []
    exits = []
    short_entries = []
    short_exits = []

    for symbol in data.symbols:
        df = data.get(symbol, timeframe)
        if df is None:
            continue

        for params_tuple in param_grid:
            params = dict(zip(param_names, params_tuple))
            params.update({k: v for k, v in config.items() if k not in param_names})

            entry_signals, exit_signals, short_entry_signals, short_exit_signals = _generate_single_lti_signal(df, params)

            # Create column name: (param1, param2, param3, param4, symbol)
            column = params_tuple + (symbol,)

            entries.append(entry_signals.rename(column))
            exits.append(exit_signals.rename(column))
            short_entries.append(short_entry_signals.rename(column))
            short_exits.append(short_exit_signals.rename(column))

    # Combine into DataFrames
    if entries:
        entries_combined = pd.concat(entries, axis=1)
        exits_combined = pd.concat(exits, axis=1)
        short_entries_combined = pd.concat(short_entries, axis=1)
        short_exits_combined = pd.concat(short_exits, axis=1)

        # Set MultiIndex column names
        columns = param_names + ['symbol']
        entries_combined.columns = pd.MultiIndex.from_tuples(entries_combined.columns, names=columns)
        exits_combined.columns = pd.MultiIndex.from_tuples(exits_combined.columns, names=columns)
        short_entries_combined.columns = pd.MultiIndex.from_tuples(short_entries_combined.columns, names=columns)
        short_exits_combined.columns = pd.MultiIndex.from_tuples(short_exits_combined.columns, names=columns)
    else:
        entries_combined = exits_combined = short_entries_combined = short_exits_combined = pd.DataFrame()

    signals = {
        'entries': entries_combined,
        'exits': exits_combined,
        'short_entries': short_entries_combined,
        'short_exits': short_exits_combined,
    }

    return signals, param_names if return_params else None
