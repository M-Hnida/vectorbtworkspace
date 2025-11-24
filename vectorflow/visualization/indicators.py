"""
Simple utilities to add indicators to portfolio plots.
No magic, no overhead - just explicit indicator additions.

Usage:
    fig = portfolio.plot()
    fig = add_indicator(fig, rsi_data, subplot='new', name='RSI')
    fig = add_indicator(fig, sma_data, row=1, name='SMA')
    fig.show()
"""

from typing import Optional, Union, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import vectorbt as vbt


def add_indicator(
    fig: go.Figure,
    data: Union[pd.Series, pd.DataFrame, Any],
    subplot: bool = False,
    row: Optional[int] = None,
    name: Optional[str] = None,
    trace_kwargs: Optional[Dict[str, Any]] = None,
    **plot_kwargs
) -> go.Figure:
    """
    Add an indicator to an existing portfolio plot figure.
    This is just a wrapper for vbt.plot() to make life a bit easier. 
    Args:
        fig: Plotly figure (from portfolio.plot())
        data: Indicator data - pd.Series, pd.DataFrame, or vbt indicator object
        subplot: Boolean (add to first subplot) or 'new' (create new subplot)
        row: Specific row number to add to (overrides subplot param)
        name: Legend name for the indicator
        trace_kwargs: Dict of kwargs for the trace styling
        **plot_kwargs: Additional kwargs passed to vbt.plot()
    
    Returns:
        Modified figure
    
    Examples:
        >>> fig = portfolio.plot()
        >>> fig = add_indicator(fig, rsi, subplot='new', name='RSI')
        >>> fig = add_indicator(fig, sma200, row=1, name='SMA 200')
    """
    if trace_kwargs is None:
        trace_kwargs = {}
    
    # Determine target row
    if row is not None:
        target_row = row
    elif subplot:
        # Create new subplot
        current_rows = len(fig._grid_ref)
        fig = _add_subplot_row(fig)
        target_row = current_rows + 1
    else:
        target_row = 1
    
    # Add trace kwargs for subplot positioning
    add_trace_kwargs = {'row': target_row, 'col': 1}
    
    # Set name if provided
    if name and isinstance(data, pd.Series):
        data = data.rename(name)
    
    # Handle different data types - check pandas first since they also have .plot()
    if isinstance(data, (pd.Series, pd.DataFrame)):
        # Standard pandas series/dataframe - use VectorBT accessor
        data.vbt.plot(
            add_trace_kwargs=add_trace_kwargs,
            trace_kwargs=trace_kwargs,
            fig=fig,
            **plot_kwargs
        ).update_layout(width=None, height=None)
    elif hasattr(data, 'plot') and hasattr(data, 'wrapper'):
        # VectorBT indicator with .plot() method (e.g., BBands)
        # VectorBT indicators have a 'wrapper' attribute
        data.plot(
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            trace_kwargs=trace_kwargs,
            **plot_kwargs
        ).update_layout(width=None, height=None)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Expected pd.Series, pd.DataFrame, or VectorBT indicator.")
    
    return fig

def remove_date_gaps(
    fig: go.Figure,
    data: Union[pd.DataFrame, pd.Series, pd.DatetimeIndex]
) -> go.Figure:
    """
    Remove gaps from missing dates (weekends, holidays) in the plot.
    
    Args:
        fig: Plotly figure
        data: DataFrame, Series, or DatetimeIndex to detect gaps from
    
    Returns:
        Modified figure with rangebreaks applied
    
    Example:
        >>> fig = portfolio.plot()
        >>> fig = add_indicator(fig, sma, name='SMA')
        >>> fig = remove_date_gaps(fig, price_data)
        >>> fig.show()
    """
    # Get datetime index
    if isinstance(data, pd.DatetimeIndex):
        dt_index = data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        dt_index = data.index
    else:
        raise ValueError("data must be DataFrame, Series, or DatetimeIndex")
    
    # Find gaps - dates that have NaN values or are missing
    if isinstance(data, (pd.DataFrame, pd.Series)):
        # Get close column or first column
        if isinstance(data, pd.DataFrame):
            close_col = 'close' if 'close' in data.columns else data.columns[0]
            valid_dates = data[close_col].dropna().index.to_list()
        else:
            valid_dates = data.dropna().index.to_list()
        
        all_dates = dt_index.to_list()
        dt_breaks = [d for d in all_dates if d not in valid_dates]
    else:
        dt_breaks = []
    
    # Apply rangebreaks
    if dt_breaks:
        fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])
    
    return fig


def _add_subplot_row(fig: go.Figure) -> go.Figure:
    """
    Internal helper to add a new subplot row to an existing figure.
    
    Note: This is a simplified approach. For complex layouts, it's better
    to define all subplots upfront in portfolio.plot().
    """
    # This is tricky with plotly - we'd need to recreate the figure
    # For now, we'll raise an error suggesting to use row numbers instead
    raise NotImplementedError(
        "Creating new subplots dynamically is not supported. "
        "Please specify the row number directly, or define all subplots "
        "upfront using portfolio.plot(subplots=[...])"
    )
