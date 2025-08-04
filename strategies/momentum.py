"""Simple momentum strategy based on price rate of change and moving average."""

from typing import Dict, List
import pandas as pd
import pandas_ta as ta
from collections import namedtuple

Signals = namedtuple(
    "Signals",
    ["entries", "exits", "short_entries", "short_exits"],
    defaults=[None, None],
)


def create_momentum_signals(df: pd.DataFrame, **params) -> Signals:
    """Create momentum signals using ROC and SMA."""
    # Parameters
    roc_period = params.get("roc_period", 10)
    roc_threshold = params.get("roc_threshold", 0.02)  # 2%
    sma_period = params.get("sma_period", 20)

    close = df["close"]

    # Rate of change momentum
    roc = close.pct_change(roc_period)
    strong_momentum = abs(roc) > roc_threshold

    # Trend filter
    sma = close.rolling(sma_period).mean()
    uptrend = close > sma
    downtrend = close < sma

    # Entry signals
    long_entries = strong_momentum & (roc > 0) & uptrend
    short_entries = strong_momentum & (roc < 0) & downtrend

    # Exit signals
    long_exits = (roc <= 0) | ~uptrend
    short_exits = (roc >= 0) | ~downtrend

    # Convert to boolean series
    long_entries = pd.Series(long_entries, index=df.index).fillna(False).astype(bool)
    short_entries = pd.Series(short_entries, index=df.index).fillna(False).astype(bool)
    long_exits = pd.Series(long_exits, index=df.index).fillna(False).astype(bool)
    short_exits = pd.Series(short_exits, index=df.index).fillna(False).astype(bool)

    return Signals(
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )


def get_momentum_vbt_params(df: pd.DataFrame, params: Dict) -> Dict:
    """Get VBT parameters with simple stops."""
    stop_loss = params.get("stop_loss", 0.02)  # 2%
    take_profit = params.get("take_profit", 0.04)  # 4%

    return {
        "sl_stop": stop_loss,
        "tp_stop": take_profit,
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
    }


def get_momentum_required_timeframes(params: Dict) -> List[str]:
    """Required timeframes."""
    return params.get("required_timeframes", ["1h"])


def generate_signals(tf_data: Dict[str, pd.DataFrame], params: Dict) -> Signals:
    """Generate signals from timeframe data."""
    if not tf_data:
        empty_index = pd.DatetimeIndex([])
        empty_series = pd.Series(False, index=empty_index)
        return Signals(empty_series, empty_series, empty_series, empty_series)

    primary_tf = list(tf_data.keys())[0]
    return create_momentum_signals(tf_data[primary_tf], **params)


def create_momentum_portfolio(
    data: pd.DataFrame, params: Dict = None
) -> "vbt.Portfolio":
    """Create VBT portfolio."""
    import vectorbt as vbt

    if params is None:
        params = {}

    signals = create_momentum_signals(data, **params)
    vbt_params = get_momentum_vbt_params(data, params)

    return vbt.Portfolio.from_signals(
        close=data["close"],
        entries=signals.entries,
        exits=signals.exits,
        short_entries=signals.short_entries,
        short_exits=signals.short_exits,
        init_cash=10000,
        fees=0.001,
        **vbt_params,
    )
