#!/usr/bin/env python3
"""Bollinger Bands Mean Reversion Strategy with ADX Filter"""

import pandas as pd
import pandas_ta as ta  # This registers the .ta accessor
import vectorbt as vbt
import numpy as np
import plotly.graph_objects as go
from typing import Dict
from collections import namedtuple

# Configure vectorbt for better visualization
vbt.settings.set_theme("dark")
vbt.settings["plotting"]["layout"]["template"] = "plotly_dark"
vbt.settings["plotting"]["layout"]["width"] = 1200
vbt.settings["plotting"]["layout"]["height"] = 200

# Simple signals container
Signals = namedtuple(
    "Signals",
    ["entries", "exits", "short_entries", "short_exits"],
    defaults=[None, None],
)


def calculate_indicators(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Calculate technical indicators for the strategy."""
    df = data.copy()

    # Ensure lowercase column names
    df.columns = [c.lower() for c in df.columns]

    # Parameters
    bbands_period = params.get("bbands_period", 20)
    bbands_std = params.get("bbands_std", 2.0)
    adx_period = params.get("adx_period", 14)
    sma_period = params.get("sma_period", 200)
    atr_period = params.get("atr_period", 14)

    # Add indicators using pandas_ta
    df.ta.bbands(length=bbands_period, std=bbands_std, append=True)
    df.ta.adx(length=adx_period, append=True)
    df.ta.sma(length=sma_period, append=True)
    df.ta.atr(length=atr_period, append=True)

    # Clean up NaN values
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def generate_signals(data: pd.DataFrame, params: Dict) -> Signals:
    """Generate Bollinger Bands mean reversion signals with ADX filter."""

    # Parameters
    bbands_period = params.get("bbands_period", 20)
    bbands_std = params.get("bbands_std", 2.0)
    adx_period = params.get("adx_period", 14)
    adx_threshold = params.get("adx_threshold", 25)
    adx_threshold_filter = params.get("adx_threshold_filter", 60)

    # Define column names
    bbl_col = f"BBL_{bbands_period}_{bbands_std}"
    bbm_col = f"BBM_{bbands_period}_{bbands_std}"
    bbu_col = f"BBU_{bbands_period}_{bbands_std}"
    adx_col = f"ADX_{adx_period}"

    # Trend conditions
    weak_trend = data[adx_col] < adx_threshold
    high_adx_filter = data[adx_col] >= adx_threshold_filter

    # Entry conditions
    long_initial_entries = (
        (data["close"].shift(1) < data[bbl_col].shift(1))
        & (data["close"] >= data[bbl_col])
        & (~high_adx_filter)
    )

    long_dca_conditions = (data["low"] <= data[bbl_col]) & (~high_adx_filter)

    short_initial_entries = (
        (data["close"].shift(1) > data[bbu_col].shift(1))
        & (data["close"] <= data[bbu_col])
        & (~high_adx_filter)
    )

    short_dca_conditions = (data["high"] >= data[bbu_col]) & (~high_adx_filter)

    # Initialize tracking arrays
    long_position = pd.Series(0, index=data.index)
    short_position = pd.Series(0, index=data.index)
    long_entries = pd.Series(False, index=data.index)
    short_entries = pd.Series(False, index=data.index)
    long_exits = pd.Series(False, index=data.index)
    short_exits = pd.Series(False, index=data.index)

    # Process signals
    for i in range(len(data)):
        if i > 0:
            long_position.iloc[i] = long_position.iloc[i - 1]
            short_position.iloc[i] = short_position.iloc[i - 1]

            # Prevent overlapping exposure
            if long_position.iloc[i] > 0 and short_initial_entries.iloc[i]:
                long_exits.iloc[i] = True
                long_position.iloc[i] = 0
            elif short_position.iloc[i] > 0 and long_initial_entries.iloc[i]:
                short_exits.iloc[i] = True
                short_position.iloc[i] = 0

        # Exit conditions
        long_exit_condition = (
            (data["close"].iloc[i] >= data[bbu_col].iloc[i])
            | ((data["close"].iloc[i] >= data[bbm_col].iloc[i]) & weak_trend.iloc[i])
            | high_adx_filter.iloc[i]
        )

        short_exit_condition = (
            (data["close"].iloc[i] <= data[bbl_col].iloc[i])
            | ((data["close"].iloc[i] <= data[bbm_col].iloc[i]) & weak_trend.iloc[i])
            | high_adx_filter.iloc[i]
        )

        # Process exits
        if long_position.iloc[i] > 0 and long_exit_condition:
            long_exits.iloc[i] = True
            long_position.iloc[i] = 0

        if short_position.iloc[i] > 0 and short_exit_condition:
            short_exits.iloc[i] = True
            short_position.iloc[i] = 0

        # Process entries
        if long_position.iloc[i] == 0 and long_initial_entries.iloc[i]:
            long_entries.iloc[i] = True
            long_position.iloc[i] = 1

        if short_position.iloc[i] == 0 and short_initial_entries.iloc[i]:
            short_entries.iloc[i] = True
            short_position.iloc[i] = 1

        # Process DCA
        if (
            long_position.iloc[i] > 0
            and not long_exits.iloc[i]
            and long_dca_conditions.iloc[i]
            and not long_entries.iloc[i]
        ):
            long_entries.iloc[i] = True

        if (
            short_position.iloc[i] > 0
            and not short_exits.iloc[i]
            and short_dca_conditions.iloc[i]
            and not short_entries.iloc[i]
        ):
            short_entries.iloc[i] = True

    return Signals(
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )


def calculate_atr_based_size_vectorized(
    entries,
    exits,
    close,
    atr,
    initial_cash,
    risk_pct,
    atr_mult,
    dca_increment,
    max_dca_size,
    max_exposure_pct,
):
    """Fully vectorized calculation of position sizes based on ATR and risk management"""

    # Calculate risk and base size
    risk_value = risk_pct * initial_cash

    # Calculate position size directly in dollar value
    stop_distance_pct = np.where(
        (atr > 0) & ~np.isnan(atr) & (close > 0),
        (atr_mult * atr) / close,  # Stop distance as percentage of price
        0.01,  # Default 1% stop if ATR is invalid
    )

    # Size in dollar value directly
    base_size_value = np.where(
        stop_distance_pct > 0,
        risk_value / stop_distance_pct,  # Risk amount / stop distance %
        0,
    )

    # Vectorize DCA counter
    trade_id = exits.cumsum().shift(1).fillna(0)
    entries_only = entries.where(entries, 0).astype(int)
    dca_count = entries_only.groupby(trade_id).cumsum()

    # Apply DCA scaling logic
    dca_multiplier = 1 + (dca_count - 1) * dca_increment
    scaled_size_value = base_size_value * dca_multiplier

    # Apply caps
    max_dca_value = max_dca_size * initial_cash
    max_exposure_value = max_exposure_pct * initial_cash
    absolute_max = 0.05 * initial_cash

    final_size = scaled_size_value.clip(upper=max_dca_value)
    final_size = final_size.clip(upper=max_exposure_value)
    final_size = final_size.clip(upper=absolute_max)

    # Ensure size is 0 where there's no entry
    final_size[~entries] = 0

    return final_size


def create_vectorbt_portfolio(data: pd.DataFrame, params: Dict = None) -> vbt.Portfolio:
    """Create VectorBT portfolio for Bollinger Bands mean reversion strategy."""

    if params is None:
        params = {}

    # Default parameters
    default_params = {
        "bbands_period": 20,
        "bbands_std": 2.0,
        "adx_period": 14,
        "adx_threshold": 25,
        "adx_threshold_filter": 60,
        "sma_period": 200,
        "atr_period": 14,
        "atr_mult": 1.0,
        "risk_pct": 0.02,
        "dca_size_increment": 0.01,
        "max_dca_size": 0.10,
        "max_side_exposure": 0.30,
        "initial_cash": 10000,
        "fee": 0.001,
    }

    # Merge with provided params
    final_params = {**default_params, **params}

    # Calculate indicators
    data_with_indicators = calculate_indicators(data, final_params)

    # Generate signals
    signals = generate_signals(data_with_indicators, final_params)

    # Find ATR column
    atr_candidates = [
        col for col in data_with_indicators.columns if col.upper().startswith("ATR")
    ]
    if not atr_candidates:
        raise ValueError("ATR column not found after indicator calculation")
    atr_col = atr_candidates[0]

    # Calculate sizes
    long_size = calculate_atr_based_size_vectorized(
        signals.entries,
        signals.exits,
        data_with_indicators["close"],
        data_with_indicators[atr_col],
        final_params["initial_cash"],
        final_params["risk_pct"],
        final_params["atr_mult"],
        final_params["dca_size_increment"],
        final_params["max_dca_size"],
        final_params["max_side_exposure"],
    )

    short_size = calculate_atr_based_size_vectorized(
        signals.short_entries,
        signals.short_exits,
        data_with_indicators["close"],
        data_with_indicators[atr_col],
        final_params["initial_cash"],
        final_params["risk_pct"],
        final_params["atr_mult"],
        final_params["dca_size_increment"],
        final_params["max_dca_size"],
        final_params["max_side_exposure"],
    )

    # Combine sizes
    size_array = np.where(signals.entries, long_size, 0) + np.where(
        signals.short_entries, short_size, 0
    )

    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=data_with_indicators["close"],
        entries=signals.entries,
        exits=signals.exits,
        short_entries=signals.short_entries,
        short_exits=signals.short_exits,
        size=size_array,
        size_type="value",
        sl_stop=data_with_indicators[atr_col]* final_params["atr_mult"]/ data_with_indicators["close"],
        init_cash=final_params["initial_cash"],
        fees=final_params["fee"],
        accumulate=True,
    )

    return portfolio


def create_strategy_dashboard(portfolio, data, params):
    """
    Advanced dashboard with vectorbt plotting features for Bollinger Bands strategy.
    """
    try:
        import plotly.graph_objects as go

        print(f"\n{'=' * 60}")
        print(f"📊 BOLLINGER BANDS MEAN REVERSION DASHBOARD")
        print(f"{'=' * 60}")

        # Start with portfolio plotting
        fig = portfolio.plot(make_subplots_kwargs={"vertical_spacing": 0.05})

        # Add OHLC data with advanced styling
        fig = data.vbt.ohlcv.plot(
            plot_type="candlestick",
            fig=fig,
            show_volume=False,
            xaxis_rangeslider_visible=False,
        )

        # Add Bollinger Bands indicators
        bbands_period = params.get("bbands_period", 20)
        bbands_std = params.get("bbands_std", 2.0)
        sma_period = params.get("sma_period", 200)

        bbl_col = f"BBL_{bbands_period}_{bbands_std}"
        bbm_col = f"BBM_{bbands_period}_{bbands_std}"
        bbu_col = f"BBU_{bbands_period}_{bbands_std}"
        sma_col = f"SMA_{sma_period}"

        if all(col in data.columns for col in [bbl_col, bbm_col, bbu_col, sma_col]):
            colors = ["yellow", "orange", "cyan", "magenta"]
            indicator_names = ["Lower BB", "Middle BB", "Upper BB", "SMA 200"]
            indicator_cols = [bbl_col, bbm_col, bbu_col, sma_col]

            for i, (col, name) in enumerate(zip(indicator_cols, indicator_names)):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode="lines",
                        name=name,
                        line=dict(color=colors[i], width=1.5),
                        opacity=0.8,
                    ),
                    row=1,
                    col=1,
                )

        # Advanced layout configuration
        fig.update_layout(
            height=1400,
            width=None,
            title="📊 Bollinger Bands Mean Reversion Strategy Analysis",
        )

        fig.show()

        # Print strategy statistics
        stats = portfolio.stats()
        print(f"\n📈 Strategy Performance:")
        print(f"Total Return: {stats.get('Total Return [%]', 'N/A'):.2f}%")
        print(f"Max Drawdown: {stats.get('Max Drawdown [%]', 'N/A'):.2f}%")
        print(f"Win Rate: {stats.get('Win Rate [%]', 'N/A'):.2f}%")
        print(f"Total Trades: {len(portfolio.trades)}")

    except Exception as e:
        print(f"⚠️ Dashboard creation failed: {e}")


# Backward compatibility alias
create_bbands_portfolio = create_vectorbt_portfolio
