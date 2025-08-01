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

# Strategy metadata for required timeframes and columns
STRATEGY_METADATA = {
    'momentum': {
        'timeframes': ['1H'],
        'columns': ['open', 'high', 'low', 'close', 'volume']
    },
    'orb': {
        'timeframes': ['1H'],
        'columns': ['open', 'high', 'low', 'close']
    },
    'tdi': {
        'timeframes': ['1H'],
        'columns': ['open', 'high', 'low', 'close']
    },
    'vectorbt': {
        'timeframes': ['1H'],
        'columns': ['open', 'high', 'low', 'close', 'volume']
    },
    'rsi': {
        'timeframes': ['1H'],
        'columns': ['open', 'high', 'low', 'close']
    }
}

def get_strategy_function(name: str):
    """Get strategy signal function by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]

def get_strategy_signal_function(name: str):
    """Get strategy signal function by name (alias for compatibility)."""
    return get_strategy_function(name)

def get_available_strategies():
    """Get list of available strategy names."""
    return list(STRATEGIES.keys())

def list_available_strategies():
    """List available strategy names (alias for compatibility)."""
    return get_available_strategies()

def get_required_timeframes(name: str, parameters: dict = None):
    """Get required timeframes for a strategy."""
    if name not in STRATEGY_METADATA:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_METADATA.keys())}")
    return STRATEGY_METADATA[name]['timeframes']

def get_required_columns(name: str):
    """Get required columns for a strategy."""
    if name not in STRATEGY_METADATA:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_METADATA.keys())}")
    return STRATEGY_METADATA[name]['columns']

# --- New: portfolio params hook ------------------------------------------------
def get_portfolio_params(name: str, primary_data, params: dict) -> dict:
    """
    Optional hook to fetch per-strategy portfolio params (e.g., trailing stop, TP).
    Looks for strategies.<name>.get_vbt_params(primary_data, params) and returns its dict.
    If not present, returns {}.
    """
    try:
        strategy_module = __import__(f"strategies.{name}", fromlist=["*"])
        if hasattr(strategy_module, "get_vbt_params"):
            fn = getattr(strategy_module, "get_vbt_params")
            result = fn(primary_data, params)
            return result if isinstance(result, dict) else {}
    except Exception:
        pass
    return {}