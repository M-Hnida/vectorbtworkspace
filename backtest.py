#!/usr/bin/env python3
"""Core Components - Simplified"""

import os
import warnings
from typing import Dict, List, Optional
import pandas as pd
import vectorbt as vbt
from base import Signals

warnings.filterwarnings("ignore")

DEFAULT_INIT_CASH = 50000
DEFAULT_FEES = 0.0004
DEFAULT_CONFIG_DIR = 'config'


def align_data_and_signals(df: pd.DataFrame, signals: Signals, vbt_params: Optional[Dict] = None):
    """Align data, signals, and optional vbt_params on a common index."""
    common_index = df.index.intersection(signals.entries.index)
    data = df.loc[common_index]

    if data['close'].isna().any():
        print(f"⚠️ Warning: Found {data['close'].isna().sum()} NaN values in close prices")
        data = data.dropna()
        common_index = data.index

    entries = signals.entries.loc[common_index].fillna(False)
    exits = signals.exits.loc[common_index].fillna(False)

    short_entries = signals.short_entries.loc[common_index].fillna(False) if getattr(signals, 'short_entries', None) is not None else None
    short_exits = signals.short_exits.loc[common_index].fillna(False) if getattr(signals, 'short_exits', None) is not None else None

    aligned_params: Dict = {}
    if vbt_params:
        for key, value in vbt_params.items():
            aligned_params[key] = value.loc[common_index] if isinstance(value, pd.Series) else value

    return data, entries, exits, short_entries, short_exits, aligned_params


def run_backtest(df: pd.DataFrame, signals: Signals, init_cash: int = DEFAULT_INIT_CASH,
                 fees: float = DEFAULT_FEES, vbt_params: Optional[Dict] = None) -> vbt.Portfolio:
    """Run backtest - simplified."""
    df_aligned, entries_aligned, exits_aligned, short_entries_aligned, short_exits_aligned, aligned_vbt_params = align_data_and_signals(
        df, signals, vbt_params
    )

    # Base portfolio parameters
    portfolio_params = {
        'close': df_aligned['close'],
        'init_cash': init_cash,
        'entries': entries_aligned,
        'exits': exits_aligned,
        'short_entries': short_entries_aligned,
        'short_exits': short_exits_aligned,
        'fees': fees,
        'freq': '1H'
    }

    # Merge aligned vbt params
    portfolio_params.update(aligned_vbt_params)

    # Create portfolio with all parameters
    portfolio = vbt.Portfolio.from_signals(**portfolio_params)
    return portfolio


def get_available_strategies(config_dir: str = DEFAULT_CONFIG_DIR) -> List[str]:
    """Get available strategies from config directory."""
    if not os.path.exists(config_dir):
        return []

    excluded = {'data_sources.yaml', 'global_config.yaml', 'settings.yaml'}
    strategies = []

    for filename in os.listdir(config_dir):
        if filename.endswith('.yaml') and filename not in excluded:
            strategies.append(os.path.splitext(filename)[0])

    return strategies