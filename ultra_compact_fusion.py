"""
Ultra-compact portfolio fusion using advanced Python techniques.
Maximally efficient code without compression - leverages comprehensions, unpacking, and functional patterns.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt
from functools import partial

def create_ultra_fused_plot(portfolios: dict, strategy_name: str = "Strategy") -> go.Figure:
    """Ultra-compact fused plot using advanced Python techniques."""
    if not portfolios:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Portfolio Value', 'Orders & Price', 'Cumulative Returns', 'Drawdown'],
        vertical_spacing=0.1, horizontal_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Add price data from first portfolio
    first_pf = next(iter(portfolios.values()))
    if (price_data := getattr(first_pf, 'close', None)) is not None:
        fig.add_trace(go.Scatter(
            x=price_data.index, y=price_data.values, mode='lines',
            name='Price', line=dict(color='gray', width=1), opacity=0.7
        ), row=1, col=2)
    
    # Process all portfolios using enumerate and tuple unpacking
    for idx, (name, pf) in enumerate(portfolios.items()):
        color = colors[idx % len(colors)]
        
        try:
            # Unpack all data in one line
            value, orders, returns, drawdown = pf.value(), pf.orders.records_readable, pf.returns(), pf.drawdown()
            
            # Create traces using list comprehension and conditional expressions
            traces = [
                # Portfolio Value
                (go.Scatter(x=value.index, y=value.values, mode='lines', name=name, 
                           line=dict(color=color, width=2)), 1, 1),
                
                # Cumulative Returns  
                (go.Scatter(x=(cum_ret := (1 + returns).cumprod() * 100).index, y=cum_ret.values, 
                           mode='lines', line=dict(color=color, width=2), showlegend=False), 2, 1),
                
                # Drawdown with RGBA conversion
                (go.Scatter(x=drawdown.index, y=drawdown.values, mode='lines', fill='tonexty',
                           line=dict(color=color, width=1), 
                           fillcolor=f'rgba({",".join(map(str, [int(color[i:i+2], 16) for i in (1,3,5)]))},0.3)',
                           showlegend=False), 2, 2)
            ]
            
            # Add all traces at once
            [fig.add_trace(trace, row=row, col=col) for trace, row, col in traces]
            
            # Add order markers efficiently
            if len(orders) > 0:
                order_config = {'Buy': ('triangle-up', 'green'), 'Sell': ('triangle-down', 'red')}
                for side, (symbol, marker_color) in order_config.items():
                    if len(filtered := orders[orders['Side'] == side]) > 0:
                        fig.add_trace(go.Scatter(
                            x=filtered['Timestamp'], y=filtered['Price'], mode='markers',
                            marker=dict(symbol=symbol, color=marker_color, size=10),
                            name=f'{name} {side}', showlegend=(idx==0)
                        ), row=1, col=2)
            
        except Exception as e:
            print(f"⚠️ Failed to process {name}: {e}")
    
    # Configure layout and axes using dictionary unpacking
    fig.update_layout(**{
        'title': f"📊 {strategy_name} - Ultra Fused Analysis",
        'template': 'plotly_dark', 'height': 600, 'width': 1000, 'showlegend': True
    })
    
    # Batch axis updates using zip and enumerate
    labels = [("Value", "Price"), ("Cumulative Returns (%)", "Drawdown (%)")]
    [fig.update_yaxes(title_text=label, row=row, col=col) 
     for row, (left, right) in enumerate(labels, 1) 
     for col, label in enumerate([left, right], 1)]
    
    return fig

def quick_test():
    """Test with sample data using generator expressions."""
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
    np.random.seed(42)
    
    # Create portfolios using dictionary comprehension
    portfolios = {
        name: vbt.Portfolio.from_orders(
            price := pd.Series(100 + np.cumsum(np.random.normal(0, 1, len(dates))), index=dates),
            pd.Series(np.random.choice([0, 0.1, -0.1], len(dates), p=[0.8, 0.1, 0.1]), index=dates),
            init_cash=10000, freq='D'
        ) for i, name in enumerate(['Conservative', 'Aggressive'])
    }
    
    # Create and show plot
    if fig := create_ultra_fused_plot(portfolios, "Ultra Test"):
        fig.show()
        print("✅ Ultra-compact fusion plot created!")
    else:
        print("❌ Failed to create plot")

if __name__ == "__main__":
    quick_test()