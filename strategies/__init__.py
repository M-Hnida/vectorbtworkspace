#!/usr/bin/env python3
"""
Trading Strategies Package - Individual Functional Strategies
Each strategy is in its own file with both functional and class-based interfaces.
"""

from .momentum import MomentumStrategy, create_momentum_signals
from .orb import ORBStrategy, create_orb_signals
from .tdi import TDIStrategy, create_tdi_signals
from .vectorbt import VectorBTStrategy, create_bollinger_mean_reversion_signals

# Strategy registry for dynamic loading
STRATEGY_REGISTRY = {
    'momentum': MomentumStrategy,
    'orb': ORBStrategy,
    'tdi': TDIStrategy,
    'vectorbt': VectorBTStrategy,
}

# Functional signal registry
SIGNAL_FUNCTIONS = {
    'momentum': create_momentum_signals,
    'orb': create_orb_signals,
    'tdi': create_tdi_signals,
    'vectorbt': create_bollinger_mean_reversion_signals,
}


def get_strategy_class(strategy_name: str):
    """Get strategy class by name."""
    strategy_key = strategy_name.lower()
    if strategy_key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[strategy_key]


def get_signal_function(strategy_name: str):
    """Get functional signal generator by name."""
    strategy_key = strategy_name.lower()
    if strategy_key not in SIGNAL_FUNCTIONS:
        raise ValueError(f"Unknown signal function: {strategy_name}. Available: {list(SIGNAL_FUNCTIONS.keys())}")
    return SIGNAL_FUNCTIONS[strategy_key]


def list_available_strategies():
    """List all available strategies."""
    return list(STRATEGY_REGISTRY.keys())


def list_available_signal_functions():
    """List all available signal functions."""
    return list(SIGNAL_FUNCTIONS.keys())


# Backward compatibility exports
__all__ = [
    'MomentumStrategy', 'ORBStrategy', 'TDIStrategy', 'VectorBTStrategy',
    'create_momentum_signals', 'create_orb_signals', 'create_tdi_signals', 'create_bollinger_mean_reversion_signals',
    'get_strategy_class', 'get_signal_function', 'list_available_strategies', 'list_available_signal_functions',
    'STRATEGY_REGISTRY', 'SIGNAL_FUNCTIONS'
]
