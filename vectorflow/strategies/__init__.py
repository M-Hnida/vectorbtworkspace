#!/usr/bin/env python3
"""
DEPRECATED - Use strategy_registry.py instead
This file is kept for backward compatibility only.
"""

# Deprecated functions - use strategy_registry.py for new code
def get_required_timeframes(strategy_name: str, params: dict = None):
    """DEPRECATED - Use strategy_registry.py"""
    return ["1h"]

def get_strategy_function(strategy_name: str):
    """DEPRECATED - Use strategy_registry.py"""
    raise NotImplementedError("Use strategy_registry.create_portfolio instead")

def get_portfolio_params(strategy_name: str, data, params: dict = None):
    """DEPRECATED - Use strategy_registry.py"""
    return {}
# If you need a list of available strategies, maintain it explicitly or infer from config.