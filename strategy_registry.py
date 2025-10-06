"""
Auto-Discovery Strategy System
Automatically finds and loads strategies without manual registration.
"""

import os
import importlib
import inspect
from typing import Dict, List, Any


def _discover_strategies():
    """Auto-discover all strategies by scanning the strategies folder."""
    strategies = {}
    strategies_dir = os.path.join(os.path.dirname(__file__), 'strategies')
    
    if not os.path.exists(strategies_dir):
        return strategies
    
    for filename in os.listdir(strategies_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            strategy_name = filename[:-3]  # Remove .py extension
            
            try:
                # Import the strategy module
                module = importlib.import_module(f'strategies.{strategy_name}')
                
                # Look for generic create_portfolio function
                if hasattr(module, 'create_portfolio'):
                    portfolio_func = getattr(module, 'create_portfolio')
                    
                    # Get default parameters from function signature or docstring
                    default_params = _extract_default_params(portfolio_func)
                    
                    # Get optimization grid if available
                    opt_grid = _extract_optimization_grid(module, strategy_name)
                    
                    strategies[strategy_name] = {
                        'portfolio_func': portfolio_func,
                        'default_params': default_params,
                        'optimization_grid': opt_grid
                    }
                    
            except Exception as e:
                print(f"Warning: Could not load strategy '{strategy_name}': {e}")
                continue
    
    return strategies


def _extract_default_params(func):
    """Extract default parameters from function signature or return empty dict."""
    try:
        sig = inspect.signature(func)
        params_param = sig.parameters.get('params')
        if params_param and params_param.default != inspect.Parameter.empty:
            return params_param.default
    except:
        pass
    
    # Try to call the function with None to get defaults
    try:
        # This is a bit hacky but works for our pattern
        import pandas as pd
        dummy_data = pd.DataFrame({'close': [100, 101, 102]})
        # If it worked, the function handles None properly
        return {}
    except:
        return {}


def _extract_optimization_grid(module, strategy_name):
    """Extract optimization grid from config file."""
    try:
        from data_manager import load_strategy_config
        config = load_strategy_config(strategy_name)
        if config and 'optimization_grid' in config:
            return config['optimization_grid']
    except Exception:
        pass
    
    return {}


# Auto-discover strategies on import
_STRATEGIES = _discover_strategies()


def get_available_strategies() -> List[str]:
    """Get list of available strategy names."""
    return list(_STRATEGIES.keys())


def create_portfolio(strategy_name: str, data, params: Dict[str, Any] = None):
    """Create a portfolio for any strategy."""
    # Extract base strategy name (remove config suffixes like _ccxt, _freqtrade)
    base_strategy_name = strategy_name.split('_')[0]
    
    if base_strategy_name not in _STRATEGIES:
        available = list(_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {base_strategy_name} (from {strategy_name}). Available: {available}")
    
    strategy_info = _STRATEGIES[base_strategy_name]
    portfolio_func = strategy_info['portfolio_func']
    
    if params is None:
        params = strategy_info['default_params']
    
    return portfolio_func(data, params)


def get_optimization_grid(strategy_name: str) -> Dict[str, List]:
    """Get the optimization grid for a strategy from config file first."""
    # Always try to load fresh from config file first
    try:
        from data_manager import load_strategy_config
        config = load_strategy_config(strategy_name)
        if config and 'optimization_grid' in config:
            return config['optimization_grid']
    except Exception:
        pass
    
    # Fallback to cached strategy info
    if strategy_name not in _STRATEGIES:
        return {}
    return _STRATEGIES[strategy_name]['optimization_grid']


def get_default_parameters(strategy_name: str) -> Dict[str, Any]:
    """Get default parameters for a strategy from config file first."""
    # Always try to load fresh from config file first
    try:
        from data_manager import load_strategy_config
        config = load_strategy_config(strategy_name)
        if config and 'parameters' in config:
            return config['parameters']
    except Exception:
        pass
    
    # Fallback to cached strategy info
    if strategy_name not in _STRATEGIES:
        return {}
    return _STRATEGIES[strategy_name]['default_params']


def get_strategy_info() -> Dict[str, Dict]:
    """Get information about all discovered strategies."""
    info = {}
    for name, strategy in _STRATEGIES.items():
        info[name] = {
            'name': name,
            'has_optimization_grid': bool(strategy['optimization_grid']),
            'num_parameters': len(strategy['optimization_grid']),
            'default_params': strategy['default_params']
        }
    return info