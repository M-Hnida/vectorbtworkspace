"""Signal generation for the original momentum strategy."""
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import pandas_ta as ta
from ..io import MarketData
from ..indicators import add_indicators


def _generate_single_momentum_signal(df: pd.DataFrame, params: dict) -> Tuple[pd.Series, pd.Series]:
    """Generate long-only signals for one asset with one set of parameters."""
    
    # Extract parameters
    vol_window = params['volatility_window']
    vol_mom_window = params['volatility_momentum_window']
    vol_mom_threshold = params['volatility_momentum_threshold']
    wma_window = params['higher_wma_window']

    # Calculate indicators
    df['volatility'] = df['close'].rolling(window=vol_window).std()
    df['volatility_momentum'] = df['volatility'].pct_change(periods=vol_mom_window)
    df['wma'] = ta.wma(df['close'], length=wma_window)

    # Define conditions
    price_above_wma = df['close'] > df['wma']
    volatility_increasing = df['volatility_momentum'] > vol_mom_threshold

    # Generate signals
    entries = price_above_wma & volatility_increasing
    exits = ~price_above_wma  # Exit when price crosses below WMA

    return entries, exits


def generate_signals(
    data: MarketData,
    param_grid: List[Tuple],
    timeframe: str,
    indicator_config: Dict, # Keep for compatibility, though unused here
    return_params: bool = False,
) -> Tuple[Dict, Any]:
    """
    Generate signals for the momentum strategy across all assets and parameters.
    """
    param_names = ['volatility_window', 'volatility_momentum_window', 'volatility_momentum_threshold', 'higher_wma_window']
    
    all_entries = []
    all_exits = []

    for symbol in data.symbols:
        df = data.get(symbol, timeframe)
        if df is None:
            continue
            
        # Pre-add general indicators if any are defined
        df = add_indicators(df.copy(), indicator_config)

        for p_tuple in param_grid:
            params = dict(zip(param_names, p_tuple))
            
            entries, exits = _generate_single_momentum_signal(df, params)
            
            # Name series with multi-index for vectorbt
            param_multi_index = pd.MultiIndex.from_tuples([p_tuple], names=param_names)
            
            all_entries.append(entries.rename(param_multi_index))
            all_exits.append(exits.rename(param_multi_index))

    # Concatenate signals. Columns will be (param_levels..., symbol)
    final_entries = pd.concat(all_entries, axis=1, keys=data.symbols, names=['symbol'])
    final_exits = pd.concat(all_exits, axis=1, keys=data.symbols, names=['symbol'])
    
    # Reorder levels to match optimizer expectations: (param1, param2, ..., symbol)
    final_entries = final_entries.reorder_levels(param_names + ['symbol'], axis=1)
    final_exits = final_exits.reorder_levels(param_names + ['symbol'], axis=1)
    
    signals_result = {
        'entries': final_entries,
        'exits': final_exits,
    }

    if return_params:
        return signals_result, param_names
    return signals_result, None 