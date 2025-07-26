"""Portfolio management with proper VectorBT handling."""
from typing import List, Union
import pandas as pd
import vectorbt as vbt
from .config import PortfolioConfig


class PortfolioManager:
    """Handles portfolio creation and management for single/multi-asset."""
    
    def __init__(self, symbols: Union[str, List[str]], config: PortfolioConfig):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.config = config
        self.is_multi_asset = len(self.symbols) > 1
        
    def create_portfolio(self, data: pd.DataFrame, entries: pd.Series, 
                        exits: pd.Series, **kwargs) -> vbt.Portfolio:
        """Create portfolio with proper structure handling."""
        
        # Merge config with any override kwargs
        portfolio_kwargs = {
            'init_cash': self.config.initial_cash,
            'fees': self.config.fees,
            'slippage': self.config.slippage,
            **kwargs
        }
        
        if self.is_multi_asset:
            return self._create_multi_asset_portfolio(data, entries, exits, **portfolio_kwargs)
        else:
            return self._create_single_asset_portfolio(data, entries, exits, **portfolio_kwargs)

    def _create_single_asset_portfolio(self, data: pd.Series, entries: pd.Series,
                                     exits: pd.Series, **kwargs) -> vbt.Portfolio:
        """Create single-asset portfolio with proper indexing."""
        
        # Ensure proper data structure - should be Series for single asset
        if isinstance(data, pd.DataFrame):
            print("Warning: DataFrame provided for single asset, using first column as close price.")
            if 'close' in data.columns:
                price_data = data['close']
            else:
                # Assume first column is close price
                price_data = data.iloc[:, 0]
        elif isinstance(data, pd.Series):
            price_data = data
        else:
            raise ValueError(f"Unexpected data type: {type(data)}")
        
        # Ensure entries and exits are Series, not DataFrame
        if isinstance(entries, pd.DataFrame):
            if entries.shape[1] > 0:
                entries = entries.iloc[:, 0]
            else:
                entries = pd.Series(False, index=entries.index)
        if isinstance(exits, pd.DataFrame):
            if exits.shape[1] > 0:
                exits = exits.iloc[:, 0]
            else:
                exits = pd.Series(False, index=exits.index)
        
        # Debug info (can be removed in production)
        # print(f"🔍 Portfolio creation debug:")
        # print(f"   Price data type: {type(price_data)}")
        # print(f"   Entries type: {type(entries)}, Exits type: {type(exits)}")
        
        # Create portfolio with frequency
        portfolio = vbt.Portfolio.from_signals(
            close=price_data,
            entries=entries,
            exits=exits,
            freq='1h',  # Set frequency for hourly data
            **kwargs
        )
        
        return portfolio
    
    def _create_multi_asset_portfolio(self, data: pd.DataFrame, entries: pd.DataFrame,
                                    exits: pd.DataFrame, **kwargs) -> vbt.Portfolio:
        """Create multi-asset portfolio."""
        # Handle multi-asset portfolio creation
        portfolio = vbt.Portfolio.from_signals(
            close=data,
            entries=entries,
            exits=exits,
            freq='1h',  # Set frequency for hourly data
            **kwargs
        )
        
        return portfolio
    
