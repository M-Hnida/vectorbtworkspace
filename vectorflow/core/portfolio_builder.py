"""
Auto-Discovery Strategy System.
Automatically finds and loads strategies without manual registration.
"""

import os
import importlib
from typing import Dict, List, Any, Optional
from vectorflow.core.config_manager import load_strategy_config


def _discover_strategies():
    """Auto-discover all strategies by scanning the strategies folder."""
    strategies = {}
    strategies_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "strategies"
    )

    if not os.path.exists(strategies_dir):
        return strategies

    for filename in os.listdir(strategies_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            strategy_name = filename[:-3]  # Remove .py extension
            strategies[strategy_name] = f"vectorflow.strategies.{strategy_name}"

    return strategies


# Auto-discover strategies on import
_STRATEGIES = _discover_strategies()


def get_available_strategies() -> List[str]:
    """Get list of available strategy names."""
    return list(_STRATEGIES.keys())


def create_portfolio(strategy_name: str, data, params: Optional[Dict[str, Any]] = None):
    """
    Create a portfolio for any strategy.

    Args:
        strategy_name: Name of the strategy to load
        data: Data to pass to the strategy (DataFrame or Dict of DataFrames)
        params: Strategy parameters
    """
    # First try the full strategy name
    base_strategy_name = strategy_name

    # If not found, try extracting base name (remove config suffixes like _ccxt, _freqtrade)
    if base_strategy_name not in _STRATEGIES:
        base_strategy_name = strategy_name.split("_")[0]

    if base_strategy_name not in _STRATEGIES:
        available = list(_STRATEGIES.keys())
        raise ValueError(
            f"Unknown strategy: {base_strategy_name} (from {strategy_name}). Available: {available}"
        )

    module_path = _STRATEGIES[base_strategy_name]

    try:
        module = importlib.import_module(module_path)

        if not hasattr(module, "create_portfolio"):
            raise AttributeError(
                f"Strategy '{base_strategy_name}' has no 'create_portfolio' function"
            )

        portfolio_func = getattr(module, "create_portfolio")

        # If params are not provided, try to load defaults
        if params is None:
            params = get_default_parameters(strategy_name)

        return portfolio_func(data, params)

    except ImportError as e:
        raise ImportError(f"Could not import strategy module '{module_path}': {e}")
    except Exception as e:
        raise RuntimeError(f"Error creating portfolio for '{strategy_name}': {e}")


def get_optimization_grid(strategy_name: str) -> Dict[str, List]:
    """Get the optimization grid for a strategy from config file."""
    try:
        config = load_strategy_config(strategy_name)
        return config.get("optimization_grid", {})
    except Exception:
        return {}


def get_default_parameters(strategy_name: str) -> Dict[str, Any]:
    """Get default parameters for a strategy from config file."""
    try:
        config = load_strategy_config(strategy_name)
        return config.get("parameters", {})
    except Exception:
        return {}


def get_strategy_info() -> Dict[str, Dict]:
    """Get information about all discovered strategies."""
    info = {}
    for name in _STRATEGIES:
        default_params = get_default_parameters(name)
        opt_grid = get_optimization_grid(name)

        info[name] = {
            "name": name,
            "has_optimization_grid": bool(opt_grid),
            "num_parameters": len(opt_grid),
            "default_params": default_params,
        }
    return info
