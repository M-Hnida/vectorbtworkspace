"""Simple momentum strategy based on price rate of change and moving average."""

from typing import Dict
import pandas as pd
import vectorbt as vbt


def create_portfolio(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """Create momentum strategy portfolio directly."""
    if params is None:
        params = {}

    # Parameters
    roc_period = params.get("roc_period", 10)
    roc_threshold = params.get("roc_threshold", 0.02)  # 2%
    sma_period = params.get("sma_period", 20)
    stop_loss = params.get("stop_loss", 0.02)  # 2%
    take_profit = params.get("take_profit", 0.04)  # 4%

    close = data["close"]

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
    long_entries = pd.Series(long_entries, index=data.index).fillna(False).astype(bool)
    short_entries = pd.Series(short_entries, index=data.index).fillna(False).astype(bool)
    long_exits = pd.Series(long_exits, index=data.index).fillna(False).astype(bool)
    short_exits = pd.Series(short_exits, index=data.index).fillna(False).astype(bool)

    return vbt.Portfolio.from_signals(
        close=close,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=10000,
        fees=0.001,
        sl_stop=stop_loss,
        tp_stop=take_profit,
        open=data["open"],
        high=data["high"],
        low=data["low"],
    )



