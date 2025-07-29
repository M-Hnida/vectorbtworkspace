#!/usr/bin/env python3
"""
Core Components Module
Contains base classes and core functionality shared across modules.
"""

import os
import warnings
from typing import Dict, List

import pandas as pd
import vectorbt as vbt
import yaml

from base import Signals, StrategyConfig
from validation import validate_ohlc_dataframe, validate_signals

# Constants
DEFAULT_INIT_CASH = 50000
DEFAULT_FEES = 0.0004
DEFAULT_SLIPPAGE = 0.001
DEFAULT_CONFIG_DIR = 'config'
DEFAULT_ENCODING = 'utf-8'

warnings.filterwarnings("ignore")

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def run_backtest(df: pd.DataFrame, signals: Signals, init_cash: int = DEFAULT_INIT_CASH, 
                 fees: float = DEFAULT_FEES, slippage: float = DEFAULT_SLIPPAGE) -> vbt.Portfolio:
    """Run backtest on a single symbol and timeframe.
    
    Args:
        df: OHLC price data with datetime index
        signals: Trading signals container
        init_cash: Initial cash amount
        fees: Trading fees as decimal
        slippage: Slippage as decimal
        
    Returns:
        Portfolio object with backtest results
        
    Raises:
        ValueError: If data and signals have no common index
        RuntimeError: If backtest execution fails
    """
    _validate_backtest_inputs(df, signals)
    
    try:
        df_aligned, signals_aligned = _align_data_and_signals(df, signals)
        
        # Infer frequency from the data index
        freq = _infer_frequency(df_aligned.index)
        
        # Use custom sizes if provided, otherwise default sizing
        if hasattr(signals_aligned, 'sizes') and signals_aligned.sizes is not None:
            # Use strategy-provided sizes
            portfolio = vbt.Portfolio.from_signals(
                close=df_aligned['close'],
                init_cash=init_cash,
                entries=signals_aligned.entries,
                exits=signals_aligned.exits,
                short_entries=signals_aligned.short_entries,
                short_exits=signals_aligned.short_exits,
                size=signals_aligned.sizes,
                size_type='percent',  # Sizes are percentages of portfolio
                fees=fees,
                slippage=slippage,
                freq=freq
            )
        else:
            # Default fixed sizing
            portfolio = vbt.Portfolio.from_signals(
                close=df_aligned['close'],
                init_cash=init_cash,
                entries=signals_aligned.entries,
                exits=signals_aligned.exits,
                short_entries=signals_aligned.short_entries,
                short_exits=signals_aligned.short_exits,
                fees=fees,
                slippage=slippage,
                freq=freq
            )

        return portfolio

    except Exception as e:
        raise RuntimeError(f"Backtest failed: {str(e)}") from e


def _infer_frequency(index: pd.DatetimeIndex) -> str:
    """Infer frequency from datetime index."""
    try:
        # Try pandas infer_freq first
        freq = pd.infer_freq(index)
        if freq:
            return freq
        
        # If that fails, calculate from time differences
        if len(index) > 1:
            time_diffs = index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                most_common_diff = time_diffs.mode()
                if len(most_common_diff) > 0:
                    diff_seconds = most_common_diff.iloc[0].total_seconds()
                    if diff_seconds == 3600:  # 1 hour
                        return '1H'
                    if diff_seconds == 86400:  # 1 day
                        return '1D'
                    if diff_seconds == 900:  # 15 minutes
                        return '15T'
                    if diff_seconds == 300:  # 5 minutes
                        return '5T'
                    if diff_seconds == 60:  # 1 minute
                        return '1T'
        
        # Default fallback
        return '1H'
    except Exception:
        return '1H'


def _validate_backtest_inputs(df: pd.DataFrame, signals: Signals) -> None:
    """Validate inputs for backtest function."""
    validate_ohlc_dataframe(df, "Backtest data")
    validate_signals(signals, df.index, "Backtest signals")


def _align_data_and_signals(df: pd.DataFrame, signals: Signals) -> tuple:
    """Align data and signals to common datetime index."""
    # Convert index to datetime if needed
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if signals.entries is not None and not isinstance(signals.entries.index, pd.DatetimeIndex):
        signals.entries.index = pd.to_datetime(signals.entries.index)
    
    # Find common index
    if signals.entries is not None:
        common_index = df.index.intersection(signals.entries.index)
        if len(common_index) == 0:
            raise ValueError("No common index between data and signals - need at least one overlapping timestamp")

        # Align data
        df_aligned = df.loc[common_index]
        signals_aligned = Signals(
            entries=signals.entries.loc[common_index] if signals.entries is not None else None,
            exits=signals.exits.loc[common_index] if signals.exits is not None else None,
            short_entries=signals.short_entries.loc[common_index] if signals.short_entries is not None else None,
            short_exits=signals.short_exits.loc[common_index] if signals.short_exits is not None else None
        )
        
        return df_aligned, signals_aligned
    else:
        raise ValueError("Signals entries cannot be None")

def run_backtest_multi_symbol_timeframe(data: Dict[str, Dict[str, pd.DataFrame]],
                                       signals: Dict[str, Dict[str, Signals]],
                                       init_cash: int = DEFAULT_INIT_CASH, 
                                       fees: float = DEFAULT_FEES,
                                       slippage: float = DEFAULT_SLIPPAGE) -> Dict[str, Dict[str, vbt.Portfolio]]:
    """Run backtests on multiple symbols and timeframes.
    
    Args:
        data: Nested dict of symbol -> timeframe -> DataFrame
        signals: Nested dict of symbol -> timeframe -> Signals
        init_cash: Initial cash amount
        fees: Trading fees as decimal
        slippage: Slippage as decimal
        
    Returns:
        Nested dict of symbol -> timeframe -> Portfolio results
    """
    results = {}

    for symbol in data.keys():
        results[symbol] = {}

        if symbol not in signals:
            print(f"Warning: No signals for symbol {symbol}")
            continue

        symbol_data = data[symbol]
        symbol_signals = signals[symbol]

        for timeframe in symbol_data.keys():
            if timeframe not in symbol_signals:
                print(f"Warning: No signals for {symbol} {timeframe}")
                results[symbol][timeframe] = None
                continue

            try:
                portfolio = run_backtest(
                    symbol_data[timeframe],
                    symbol_signals[timeframe],
                    init_cash,
                    fees,
                    slippage
                )
                results[symbol][timeframe] = portfolio

            except Exception as e:
                print(f"Error in backtest for {symbol} {timeframe}: {e}")
                results[symbol][timeframe] = None

    return results


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_strategy_config(strategy_name: str, config_dir: str = DEFAULT_CONFIG_DIR) -> StrategyConfig:
    """Load strategy configuration from YAML file.

    Args:
        strategy_name: Name of the strategy configuration to load
        config_dir: Directory containing configuration files

    Returns:
        StrategyConfig object with loaded configuration

    Raises:
        ValueError: If strategy name is invalid
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not strategy_name or not strategy_name.strip():
        raise ValueError("Strategy name cannot be empty")

    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    config_path = os.path.join(config_dir, f"{strategy_name}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding=DEFAULT_ENCODING) as file:
            config_data = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}") from e

    if config_data is None:
        config_data = {}

    return StrategyConfig(
        name=strategy_name,
        parameters=config_data.get('parameters', {}),
        optimization_grid=config_data.get('optimization_grid', {}),
        analysis_settings=config_data.get('analysis_settings', {}),
        data_requirements=config_data.get('data_requirements', {})
    )

def get_available_strategies(config_dir: str = DEFAULT_CONFIG_DIR) -> List[str]:
    """Get list of available strategies from config directory.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        List of strategy names
    """
    if not os.path.exists(config_dir):
        return []

    # Exclude non-strategy config files
    excluded_files = {'data_sources.yaml', 'global_config.yaml', 'settings.yaml'}

    strategies = []
    for filename in os.listdir(config_dir):
        if filename.endswith('.yaml') and filename not in excluded_files:
            strategies.append(os.path.splitext(filename)[0])

    return strategies