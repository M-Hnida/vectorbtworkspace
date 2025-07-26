import vectorbt as vbt

from .indicators import timeframe_to_pandas_freq
from .portfolio import PortfolioManager
from .data_loader import MarketData



def run_backtest(
    data: MarketData,
    signals: dict,
    portfolio_config: dict,
    timeframe: str,
) -> vbt.Portfolio:
    """Run a vectorized backtest with the given signals and portfolio configuration."""
    
    # Create portfolio manager
    portfolio_manager = PortfolioManager(data.symbols, portfolio_config)
    
    # Process signals
    entries = signals['entries']
    exits = signals['exits']
    short_entries = signals.get('short_entries')
    short_exits = signals.get('short_exits')
    
    # Create portfolio using manager
    portfolio = portfolio_manager.create_portfolio(
        data=data.get_all(timeframe),
        entries=entries,
        exits=exits, 
        short_entries=short_entries,
        short_exits=short_exits,
        freq=timeframe_to_pandas_freq(timeframe)
    )
    
    return portfolio
    

if __name__ == '__main__':
    print("This script is a module and is not meant to be run directly.")
    print("Please run the main pipeline via 'runner.py' from the project root.")
