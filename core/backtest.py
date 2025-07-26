from typing import Dict

import pandas as pd
import vectorbt as vbt
from core.data_loader import MarketData
from core.indicators import timeframe_to_pandas_freq
from .base import Signals


def run_backtest(
    data: MarketData,  # symbol -> timeframe -> DataFrame
    signals: Dict[str, Dict[str, Signals]],
) -> Dict[str, Dict[str, vbt.Portfolio]]:
    """
    Run vectorized backtests on all given symbols and timeframes.
    
    Args:
        data: MarketData object with structure {symbol: {timeframe: DataFrame}}
        
        portfolio_config: Portfolio configuration : fees slippage init_cash
        timeframe: Default timeframe to use
    Returns:
        Dict with structure {symbol: {timeframe: Portfolio}}
    """
    
    results = {}
    
    for symbol in data.symbols:
        results[symbol] = {}  # ✅ Initialiser le dict du symbole
        
        symbol_timeframes = data.get_symbol(symbol)
        
        for tf, df in symbol_timeframes.items():
            # Obtenir les signaux
            try:
                symbol_signals :Signals = signals[symbol][tf]
                entries = symbol_signals.entries
                exits = symbol_signals.exits
                short_entries = symbol_signals.short_entries
                short_exits = symbol_signals.short_exits
            except AttributeError as e:
                print(f"Warning: No signals found for {symbol} {tf} fk ass niga: {e}")
                continue

            try:
                portfolio = vbt.Portfolio.from_signals(
                    close=df['close'],
                    init_cash=50000,
                    entries=entries,
                    exits=exits, 
                    short_entries=short_entries,
                    short_exits=short_exits,
                    freq=timeframe_to_pandas_freq(tf),
                    fees=0.0004,
                    slippage=0.001,
                )
                
                results[symbol][tf] = portfolio
                
            except Exception as e:
                print(f"Error running backtest for {symbol} {tf}: {str(e)}")
                results[symbol][tf] = None
    
    return results
    

if __name__ == '__main__':
    print("This script is a module and is not meant to be run directly.")
    print("Please run the main pipeline via 'runner.py' from the project root.")
