#!/usr/bin/env python3
"""Strategy Registry - Simple functional approach"""

from .momentum import generate_signals as momentum_signals
from .orb import generate_signals as orb_signals  
from .tdi import generate_tdi_signals as tdi_signals
from .vectorbt import generate_vectorbt_signals as vectorbt_signals
from .rsi import generate_signals as rsi_signals

STRATEGIES = {
    'momentum': momentum_signals,
    'orb': orb_signals,
    'tdi': tdi_signals,
    'vectorbt': vectorbt_signals,
    'rsi': rsi_signals
}

def get_strategy_function(name: str):
    """Get strategy signal function by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def get_available_strategies():
    """Get list of available strategy names."""
    return list(STRATEGIES.keys())