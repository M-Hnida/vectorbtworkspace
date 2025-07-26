import pandas as pd
import vectorbt as vbt

from core.data_loader import MarketData
from core.visualization import TradingVisualizer

data = MarketData(pd.read_csv('data/EURUSD_1D_2009-2025.csv')).get_df("EURUSD","1H")

close = data['close']

portfolio = vbt.Portfolio.from_random_signals(close=close, init_cash=100000, fees=0.0003, slippage=0.0003)
TradingVisualizer.plot_portfolio(portfolio)
