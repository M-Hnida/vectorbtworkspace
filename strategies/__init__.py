#!/usr/bin/env python3
"""
Trading Strategies Package - Pure Functional Design
Each strategy provides clean functional interfaces.
"""

from .momentum import (create_momentum_signals, generate_momentum_signals, 
                      get_momentum_required_timeframes, get_momentum_required_columns)
from .orb import (create_orb_signals, generate_orb_signals,
                 get_orb_required_timeframes, get_orb_required_columns)
from .tdi import (create_tdi_signals, generate_tdi_signals,
                 get_tdi_required_timeframes, get_tdi_required_columns)
from .vectorbt import (create_bollinger_mean_reversion_signals, generate_vectorbt_signals,
                      get_vectorbt_required_timeframes, get_vectorbt_required_columns)

# Pure functional signal generators
SIGNAL_FUNCTIONS = {
    'momentum': create_momentum_signals,
    'orb': create_orb_signals,
    'tdi': create_tdi_signals,
    'vectorbt': create_bollinger_mean_reversion_signals,
}

# Multi-timeframe signal generators
STRATEGY_SIGNAL_FUNCTIONS = {
    'momentum': generate_momentum_signals,
    'orb': generate_orb_signals,
    'tdi': generate_tdi_signals,
    'vectorbt': generate_vectorbt_signals,
}

# Strategy metadata functions
TIMEFRAME_FUNCTIONS = {
    'momentum': get_momentum_required_timeframes,
    'orb': get_orb_required_timeframes,
    'tdi': get_tdi_required_timeframes,
    'vectorbt': get_vectorbt_required_timeframes,
}

COLUMN_FUNCTIONS = {
    'momentum': get_momentum_required_columns,
    'orb': get_orb_required_columns,
    'tdi': get_tdi_required_columns,
    'vectorbt': get_vectorbt_required_columns,
}


def get_signal_function(strategy_name: str):
    """Get pure functional signal generator by name."""
    strategy_key = strategy_name.lower()
    if strategy_key not in SIGNAL_FUNCTIONS:
        raise ValueError(f"Unknown signal function: {strategy_name}. Available: {list(SIGNAL_FUNCTIONS.keys())}")
    return SIGNAL_FUNCTIONS[strategy_key]


def get_strategy_signal_function(strategy_name: str):
    """Get multi-timeframe signal generator by name."""
    strategy_key = strategy_name.lower()
    if strategy_key not in STRATEGY_SIGNAL_FUNCTIONS:
        raise ValueError(f"Unknown strategy signal function: {strategy_name}. Available: {list(STRATEGY_SIGNAL_FUNCTIONS.keys())}")
    return STRATEGY_SIGNAL_FUNCTIONS[strategy_key]


def get_required_timeframes(strategy_name: str, params: dict):
    """Get required timeframes for strategy."""
    strategy_key = strategy_name.lower()
    if strategy_key not in TIMEFRAME_FUNCTIONS:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(TIMEFRAME_FUNCTIONS.keys())}")
    return TIMEFRAME_FUNCTIONS[strategy_key](params)


def get_required_columns(strategy_name: str):
    """Get required columns for strategy."""
    strategy_key = strategy_name.lower()
    if strategy_key not in COLUMN_FUNCTIONS:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(COLUMN_FUNCTIONS.keys())}")
    return COLUMN_FUNCTIONS[strategy_key]()


def list_available_strategies():
    """List all available strategies."""
    return list(SIGNAL_FUNCTIONS.keys())


def list_available_signal_functions():
    """List all available signal functions."""
    return list(SIGNAL_FUNCTIONS.keys())


# Clean functional exports
__all__ = [
    'create_momentum_signals', 'create_orb_signals', 'create_tdi_signals', 'create_bollinger_mean_reversion_signals',
    'generate_momentum_signals', 'generate_orb_signals', 'generate_tdi_signals', 'generate_vectorbt_signals',
    'get_signal_function', 'get_strategy_signal_function',
    'get_required_timeframes', 'get_required_columns',
    'list_available_strategies', 'list_available_signal_functions',
    'SIGNAL_FUNCTIONS', 'STRATEGY_SIGNAL_FUNCTIONS'
]
