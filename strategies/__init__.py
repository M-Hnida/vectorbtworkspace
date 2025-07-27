#!/usr/bin/env python3
"""
Trading Strategies Package
Contains all trading strategy implementations.
"""

from .tdi_strategy import LTIStrategy
from .momentum_strategy import MomentumStrategy
from .orb_strategy import ORBStrategy

# Strategy registry for dynamic loading
STRATEGY_REGISTRY = {
    strategy.__name__.lower().replace('strategy', ''): strategy 
    for strategy in [LTIStrategy, MomentumStrategy, ORBStrategy]
}

def get_strategy_class(strategy_name: str):
    """Get strategy class by name."""
    strategy_key = strategy_name.lower()
    if strategy_key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[strategy_key]

def list_available_strategies():
    """List all available strategies."""
    return list(STRATEGY_REGISTRY.keys())

__all__ = ['TDIStrategy', 'MomentumStrategy', 'ORBStrategy', 'get_strategy_class', 'list_available_strategies']