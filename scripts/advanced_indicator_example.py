# %%
"""
Advanced Example: Recreating the 2BB Strategy Plot from the docs
Shows how to use add_indicator and add_bbands to create complex visualizations
"""

import pandas as pd
import vectorbt as vbt
import numpy as np
from vectorflow.visualization import add_indicator, add_bbands, add_signals, remove_date_gaps

vbt.settings.set_theme("dark")
vbt.settings['plotting']['layout']['width'] = 1280

# %%
# ===== SAMPLE DATA SETUP =====
# Note: Replace this with your actual data loading
# For this example, we'll create some sample data

# Load your data (example)
# data = vbt.YFData.download("QQQ", start="2019-01-01", end="2020-12-31", timeframe="4h").get()

# For demonstration, using sample data
print("Setting up sample data...")

# You would load your actual data here
# For now, this is a placeholder - replace with your data source

# %%
# ===== COMPUTE INDICATORS =====

# Example: Computing BBands on Price
# bb_price = vbt.indicators.factory.talib('BBANDS').run(
#     data['Close'], 
#     timeperiod=20,
#     nbdevup=2,
#     nbdevdn=2
# )

# Example: Computing RSI
# rsi = vbt.indicators.factory.talib('RSI').run(
#     data['Close'],
#     timeperiod=14
# ).rsi

# Example: Computing BBands on RSI
# bb_rsi = vbt.indicators.factory.talib('BBANDS').run(
#     rsi,
#     timeperiod=14,
#     nbdevup=2,
#     nbdevdn=2
# )

# %%
# ===== CREATE STRATEGY SIGNALS =====

# Example entry/exit logic (replace with your actual strategy)
# entries = (data['Close'] > bb_price.lowerband) & (rsi < bb_rsi.lowerband)
# exits = (data['Close'] < bb_price.upperband) | (rsi > bb_rsi.upperband)

# %%
# ===== RUN BACKTEST =====

# portfolio = vbt.Portfolio.from_signals(
#     close=data['Close'],
#     entries=entries,
#     exits=exits,
#     init_cash=10000,
#     fees=0.001,
#     freq='4h'
# )

# %%
# ===== CREATE ENHANCED VISUALIZATION =====

# METHOD 1: Using portfolio.plot() with custom subplots
def create_2bb_strategy_plot_v1(portfolio, data, bb_price, rsi, bb_rsi, entries, exits):
    """
    Version 1: Define subplots in portfolio.plot(), then add indicators
    This is the recommended approach
    """
    
    # Create portfolio plot with predefined subplots
    fig = portfolio.plot(
        subplots=['orders', 'trade_pnl'],
        subplot_settings={
            'orders': {
                'close_trace_kwargs': {'visible': True}
            }
        }
    )
    
    # Row 1: Orders subplot with price + BBands
    fig = add_bbands(
        fig, 
        bb_price, 
        row=1,
        upper_name='BB Price Upper',
        middle_name='BB Price Middle',
        lower_name='BB Price Lower',
        line_style=dict(color='white', width=1, dash='dot')
    )
    
    # Row 2: Trade P&L (already created by portfolio.plot)
    
    # We need to manually add RSI subplot since portfolio.plot doesn't have it
    # This requires knowing the subplot structure
    
    # For now, let's show a simpler version
    return fig


# METHOD 2: Start fresh and build everything
def create_2bb_strategy_plot_v2(portfolio, data, bb_price, rsi, bb_rsi, entries, exits, 
                                slice_lower=None, slice_upper=None):
    """
    Version 2: Build plot from scratch using vbt.make_subplots
    This gives you full control
    """
    
    # Slice data if needed
    if slice_lower and slice_upper:
        data_slice = data[slice_lower:slice_upper]
        bb_price_slice = bb_price[slice_lower:slice_upper]
        rsi_slice = rsi[slice_lower:slice_upper]
        bb_rsi_slice = bb_rsi[slice_lower:slice_upper]
        entries_slice = entries[slice_lower:slice_upper]
        exits_slice = exits[slice_lower:slice_upper]
        pf_slice = portfolio[slice_lower:slice_upper]
    else:
        data_slice = data
        bb_price_slice = bb_price
        rsi_slice = rsi
        bb_rsi_slice = bb_rsi
        entries_slice = entries
        exits_slice = exits
        pf_slice = portfolio
    
    # Create subplots: 2 rows (70/30 split)
    fig = vbt.make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=['Price with BBands', 'RSI with BBands']
    )
    
    # Plot OHLCV in row 1
    ohlc_data = data_slice[['Open', 'High', 'Low', 'Close']]
    ohlc_data.vbt.ohlcv.plot(
        plot_type='Candlestick',
        add_trace_kwargs=dict(row=1, col=1),
        fig=fig,
        show_volume=False
    )
    
    # Add BBands to price (row 1)
    fig = add_bbands(
        fig,
        bb_price_slice,
        row=1,
        line_style=dict(color='white', width=1, dash='dot')
    )
    
    # Add RSI line (row 2)
    fig = add_indicator(
        fig,
        rsi_slice,
        row=2,
        name='RSI',
        trace_kwargs=dict(
            line=dict(color='yellow', width=2),
            connectgaps=True
        )
    )
    
    # Add BBands to RSI (row 2)
    fig = add_bbands(
        fig,
        bb_rsi_slice,
        row=2,
        limits=(25, 75),  # Reference lines for oversold/overbought
        line_style=dict(color='white', width=1, dash='dot')
    )
    
    # Add entry/exit signals on RSI (row 2)
    fig = add_signals(
        fig,
        entries_slice,
        exits_slice,
        base_data=rsi_slice,
        row=2,
        entry_name='Long Entry',
        exit_name='Long Exit',
        entry_color='limegreen',
        exit_color='red'
    )
    
    # Add trade signals on price chart (row 1)
    pf_slice.plot_trade_signals(
        add_trace_kwargs=dict(row=1, col=1),
        fig=fig,
        plot_close=False,
        plot_positions='lines'
    )
    
    # Add P/L boxes (row 1)
    pf_slice.trades.direction_long.plot(
        add_trace_kwargs=dict(row=1, col=1),
        fig=fig,
        plot_close=False,
        plot_markers=False
    )
    
    # Remove gaps
    fig = remove_date_gaps(fig, data_slice)
    
    # Styling
    fig.update_layout(
        title_text='2BB Strategy: Price & RSI with Bollinger Bands',
        height=960,
        width=1280,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        showlegend=True
    )
    
    return fig


# %%
# ===== SIMPLE EXAMPLE (Works with existing nasdaqma.py) =====

def simple_example_from_nasdaqma():
    """
    Simple example that works with your existing nasdaqma.py
    Just add these lines after creating the portfolio
    """
    
    # Assuming you have:
    # - portfolio (from vbt.Portfolio.from_signals)
    # - data (your OHLCV data)
    # - data['close_kalman'] (Kalman filtered close)
    # - data['SMA_225'] (your moving average)
    
    # Traditional portfolio plot
    # fig = portfolio.plot()
    
    # Enhanced with indicators
    # fig = portfolio.plot(subplots=['orders', 'trade_pnl'])
    # fig = add_indicator(fig, data['close_kalman'], row=1, name='Kalman Filter', 
    #                    trace_kwargs=dict(line=dict(color='cyan', width=1.5)))
    # fig = add_indicator(fig, data['SMA_225'], row=1, name='SMA 225',
    #                    trace_kwargs=dict(line=dict(color='orange', width=2)))
    # fig = remove_date_gaps(fig, data)
    # fig.update_layout(title_text="NasdaqMA Strategy - Enhanced")
    # fig.show()
    
    print("Add the code above to your nasdaqma.py file!")


# %%
# ===== USAGE INSTRUCTIONS =====

print("""
To use these plotting utilities:

1. SIMPLE (Add to existing portfolio plot):
   ```python
   fig = portfolio.plot()
   fig = add_indicator(fig, your_indicator, row=1, name='Indicator Name')
   fig.show()
   ```

2. INTERMEDIATE (Multiple indicators):
   ```python
   fig = portfolio.plot(subplots=['orders', 'trade_pnl'])
   fig = add_indicator(fig, sma, row=1, name='SMA')
   fig = add_bbands(fig, bb_price, row=1)
   fig = remove_date_gaps(fig, data)
   fig.show()
   ```

3. ADVANCED (Full custom subplots):
   Use create_2bb_strategy_plot_v2() function above as a template

For a working example, see: scripts/nasdaqma_with_indicators.py
""")
