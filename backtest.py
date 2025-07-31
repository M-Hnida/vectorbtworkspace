#!/usr/bin/env python3
"""Core Components - Simplified"""

import os
import warnings
from typing import Dict, List
import pandas as pd
import vectorbt as vbt
import yaml
from base import Signals, StrategyConfig

warnings.filterwarnings("ignore")

DEFAULT_INIT_CASH = 50000
DEFAULT_FEES = 0.0004
DEFAULT_CONFIG_DIR = 'config'


def run_backtest(df: pd.DataFrame, signals: Signals, init_cash: int = DEFAULT_INIT_CASH, 
                 fees: float = DEFAULT_FEES) -> vbt.Portfolio:
    """Run backtest - simplified."""
    # Align data and signals
    common_index = df.index.intersection(signals.entries.index)
    df_aligned = df.loc[common_index]
    
    # Debug: Check for any None values in the data
    if df_aligned['close'].isna().any():
        print(f"⚠️ Warning: Found {df_aligned['close'].isna().sum()} NaN values in close prices")
        df_aligned = df_aligned.dropna()
        common_index = df_aligned.index
    
    # Debug: Check signals
    entries_aligned = signals.entries.loc[common_index]
    exits_aligned = signals.exits.loc[common_index]
    
    if entries_aligned.isna().any():
        print(f"⚠️ Warning: Found NaN values in entry signals")
        entries_aligned = entries_aligned.fillna(False)
    
    if exits_aligned.isna().any():
        print(f"⚠️ Warning: Found NaN values in exit signals")
        exits_aligned = exits_aligned.fillna(False)
    
    # Create portfolio with cleaned data
    portfolio = vbt.Portfolio.from_signals(
        close=df_aligned['close'],
        init_cash=init_cash,
        entries=entries_aligned,
        exits=exits_aligned,
        short_entries=signals.short_entries.loc[common_index].fillna(False) if signals.short_entries is not None else None,
        short_exits=signals.short_exits.loc[common_index].fillna(False) if signals.short_exits is not None else None,
        fees=fees,
        freq='1H'  # Default frequency
    )
    return portfolio


def load_strategy_config(strategy_name: str, config_dir: str = DEFAULT_CONFIG_DIR) -> StrategyConfig:
    """Load strategy config from YAML."""
    config_path = os.path.join(config_dir, f"{strategy_name}.yaml")
    
    if not os.path.exists(config_path):
        # Return default config if file doesn't exist
        return StrategyConfig(name=strategy_name, parameters={})
    
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file) or {}
    
    return StrategyConfig(
        name=strategy_name,
        parameters=config_data.get('parameters', {}),
        optimization_grid=config_data.get('optimization_grid', {}),
        analysis_settings=config_data.get('analysis_settings', {}),
        data_requirements=config_data.get('data_requirements', {})
    )


def get_available_strategies(config_dir: str = DEFAULT_CONFIG_DIR) -> List[str]:
    """Get available strategies from config directory."""
    if not os.path.exists(config_dir):
        return []
    
    excluded = {'data_sources.yaml', 'global_config.yaml', 'settings.yaml', 'vectorbt.yaml'}
    strategies = []
    
    for filename in os.listdir(config_dir):
        if filename.endswith('.yaml') and filename not in excluded:
            strategies.append(os.path.splitext(filename)[0])
    
    return strategies