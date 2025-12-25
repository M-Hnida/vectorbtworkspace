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
import plotly.graph_objects as go


def add_indicator(
    fig: go.Figure,
    data: Union[pd.Series, pd.DataFrame, Any],
    subplot: bool = False,
    row: Optional[int] = None,
    name: Optional[str] = None,
    trace_kwargs: Optional[Dict[str, Any]] = None,
    **plot_kwargs,
) -> go.Figure:
    """Add an indicator to an existing portfolio plot figure."""
    if trace_kwargs is None:
        trace_kwargs = {}

    target_row = row if row is not None else (len(fig._grid_ref) + 1 if subplot else 1)
    add_trace_kwargs = {"row": target_row, "col": 1}

    if name and isinstance(data, pd.Series):
        data = data.rename(name)

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data.vbt.plot(
            add_trace_kwargs=add_trace_kwargs,
            trace_kwargs=trace_kwargs,
            fig=fig,
            **plot_kwargs,
        ).update_layout(width=None, height=None)
    elif hasattr(data, "plot") and hasattr(data, "wrapper"):
        data.plot(
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            trace_kwargs=trace_kwargs,
            **plot_kwargs,
        ).update_layout(width=None, height=None)
    else:
        raise ValueError(
            f"Unsupported data type: {type(data)}. Expected pd.Series, pd.DataFrame, or VectorBT indicator."
        )

    return fig


def add_trade_signals(
    portfolio: Any,
    fig: go.Figure,
    line_width: int = 2,
    profit_color: str = "rgba(0, 255, 0, 0.6)",
    loss_color: str = "rgba(255, 0, 0, 0.6)",
    show_markers: bool = False,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    **kwargs,
) -> go.Figure:
    """
    Add profit/loss colored connector lines between trade entry/exit points.

    Green lines for profitable trades, red lines for losses.

    Args:
        portfolio: VectorBT Portfolio object
        fig: Existing Plotly figure
        line_width: Width of connector lines (default: 2)
        profit_color: Color for profitable trades (default: green)
        loss_color: Color for losing trades (default: red)
        show_markers: Show entry/exit markers (default: False)
        start_date: Optional start date filter
        end_date: Optional end date filter

    Example:
        >>> fig = portfolio.plot()
        >>> fig = add_trade_signals(portfolio, fig)
    """
    wrapper, close_prices, trades = (
        portfolio.wrapper,
        portfolio.close,
        portfolio.trades.records,
    )

    if len(trades) == 0:
        return fig

    # Convert to DataFrame
    trades_df = trades.to_pd() if hasattr(trades, "to_pd") else pd.DataFrame(trades)

    # Filter by date
    if start_date or end_date:
        mask = pd.Series(True, index=trades_df.index)
        if start_date:
            mask &= wrapper.index[trades_df["entry_idx"]] >= pd.Timestamp(start_date)
        if end_date:
            mask &= wrapper.index[trades_df["exit_idx"]] <= pd.Timestamp(end_date)
        trades_df = trades_df[mask]

        if len(trades_df) == 0:
            return fig

    # Plot trades
    entry_dates, entry_prices, exit_dates, exit_prices = [], [], [], []

    for idx in range(len(trades_df)):
        trade = trades_df.iloc[idx]
        entry_idx, exit_idx = int(trade["entry_idx"]), int(trade["exit_idx"])

        # Get prices (handle DataFrame or Series)
        if isinstance(close_prices, pd.DataFrame):
            col_idx = int(trade.get("col", 0))
            entry_price, exit_price = (
                close_prices.iloc[entry_idx, col_idx],
                close_prices.iloc[exit_idx, col_idx],
            )
        else:
            entry_price, exit_price = (
                close_prices.iloc[entry_idx],
                close_prices.iloc[exit_idx],
            )

        entry_dates.append(wrapper.index[entry_idx])
        exit_dates.append(wrapper.index[exit_idx])
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)

        # Color by PnL
        pnl = trade.get("pnl", exit_price - entry_price)
        color = profit_color if pnl > 0 else loss_color

        # Add line
        fig.add_trace(
            go.Scatter(
                x=[wrapper.index[entry_idx], wrapper.index[exit_idx]],
                y=[entry_price, exit_price],
                mode="lines",
                line=dict(color=color, width=line_width),
                showlegend=False,
                hoverinfo="skip",
                **kwargs,
            ),
            row=1,
            col=1,
        )

    # Optional markers
    if show_markers:
        fig.add_trace(
            go.Scatter(
                x=entry_dates,
                y=entry_prices,
                mode="markers",
                marker=dict(color="green", size=8, symbol="triangle-up"),
                name="Entry",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exit_dates,
                y=exit_prices,
                mode="markers",
                marker=dict(color="red", size=8, symbol="triangle-down"),
                name="Exit",
            ),
            row=1,
            col=1,
        )

    return fig
