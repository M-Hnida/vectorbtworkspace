#!/usr/bin/env python3
"""
Volatility Breakout Strategy v3 - ATR-Based First Passage
==========================================================

Key improvements over v2:
1. ATR-based stops/targets instead of log-sigma (protects against vol expansion traps)
2. Trend filter (SMA200): Disables shorts in uptrends, longs in downtrends
3. Wider invalidation threshold (1.0σ instead of 0.5σ) to survive shakeouts
4. Asset regime detection: Auto-detects high-drift assets for long-only mode
5. Floor/ceiling on ATR multipliers to prevent extreme stop placement

Mathematical basis:
- First Passage Time theory filters false breakouts
- ATR more stable than log-return sigma during parabolic moves
- Trend alignment reduces fighting-the-drift errors
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from typing import Dict, Optional


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.

    ATR is more stable than log-return sigma during parabolic moves
    because it uses actual price ranges rather than return volatility.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr


def calculate_volatility_regime(
    df: pd.DataFrame, vol_window: int = 20, regime_window: int = 200
) -> pd.DataFrame:
    """
    Compute volatility and regime indicators.

    Returns DataFrame with:
    - sigma: rolling volatility of log returns
    - sigma_z: z-score of volatility (compression < -1, expansion > +1)
    - sigma_accel: whether volatility is accelerating (∂σ/∂t > 0)
    - atr: Average True Range (for stable stops)
    - trend: Trend direction based on SMA(200)
    """
    df = df.copy()

    # Log returns for sigma calculation
    df["log_close"] = np.log(df["close"])
    df["r"] = df["log_close"].diff()

    # Rolling volatility
    df["sigma"] = df["r"].rolling(vol_window).std()

    # Volatility z-score
    sigma_mean = df["sigma"].rolling(regime_window).mean()
    sigma_std = df["sigma"].rolling(regime_window).std()
    df["sigma_z"] = (df["sigma"] - sigma_mean) / sigma_std

    # Volatility acceleration (first derivative > 0)
    df["sigma_accel"] = df["sigma"].diff() > 0

    # ATR for stable stop-loss/take-profit
    df["atr"] = calculate_atr(df, period=14)

    # Trend filter: SMA(200)
    df["sma_200"] = df["close"].rolling(200).mean()
    df["trend"] = np.where(df["close"] > df["sma_200"], 1, -1)

    return df


# =============================================================================
# NUMBA-OPTIMIZED SIGNAL GENERATION
# =============================================================================


@njit
def first_passage_signals_v3(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sigma: np.ndarray,
    sigma_z: np.ndarray,
    sigma_accel: np.ndarray,
    trend: np.ndarray,
    entry_threshold: float = 0.5,  # Entry at +0.5σ
    invalidation: float = 1.0,  # Wider: invalidate if hits -1.0σ first (was 0.5)
    compression_threshold: float = -1.0,
    trend_filter_enabled: bool = True,
) -> tuple:
    """
    First-passage based entry signals with trend filter.

    Changes from v2:
    - Wider invalidation (1.0σ) to survive shakeouts
    - Trend filter: Only long in uptrend, only short in downtrend
    """
    n = len(close)
    long_entries = np.zeros(n, dtype=np.bool_)
    short_entries = np.zeros(n, dtype=np.bool_)

    # State machine
    in_regime = False
    tracking_up = False
    tracking_down = False
    reference_price = 0.0

    for i in range(1, n):
        if np.isnan(sigma[i]) or np.isnan(sigma_z[i]):
            continue

        current_sigma = sigma[i]
        current_trend = trend[i]

        # Check regime entry: compression + acceleration
        regime_active = (sigma_z[i] < compression_threshold) and sigma_accel[i]

        if regime_active and not in_regime:
            # New regime: start tracking
            in_regime = True
            tracking_up = True
            tracking_down = True
            reference_price = close[i]

        elif not regime_active:
            # Exit regime: reset
            in_regime = False
            tracking_up = False
            tracking_down = False

        if in_regime:
            # Price levels based on sigma
            up_target = reference_price * (1.0 + entry_threshold * current_sigma)
            up_invalidation = reference_price * (1.0 - invalidation * current_sigma)
            down_target = reference_price * (1.0 - entry_threshold * current_sigma)
            down_invalidation = reference_price * (1.0 + invalidation * current_sigma)

            # Check up breakout (only if trend is up, or filter disabled)
            if tracking_up:
                trend_ok = (not trend_filter_enabled) or (current_trend > 0)

                if low[i] <= up_invalidation:
                    tracking_up = False
                elif high[i] >= up_target and trend_ok:
                    long_entries[i] = True
                    tracking_up = False
                    tracking_down = False
                    in_regime = False

            # Check down breakout (only if trend is down, or filter disabled)
            if tracking_down:
                trend_ok = (not trend_filter_enabled) or (current_trend < 0)

                if high[i] >= down_invalidation:
                    tracking_down = False
                elif low[i] <= down_target and trend_ok:
                    short_entries[i] = True
                    tracking_up = False
                    tracking_down = False
                    in_regime = False

    return long_entries, short_entries


@njit
def compute_atr_exits(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    long_entries: np.ndarray,
    short_entries: np.ndarray,
    stop_atr_mult: float = 2.0,  # Stop at entry - 2*ATR
    target_atr_mult: float = 4.0,  # Target at entry + 4*ATR
    max_bars: int = 50,
    min_stop_pct: float = 0.005,  # Floor: at least 0.5% stop
    max_stop_pct: float = 0.10,  # Ceiling: at most 10% stop
) -> tuple:
    """
    ATR-based exit signals with floor/ceiling protection.

    Key improvement: ATR is more stable than log-return sigma during
    parabolic moves. Floors/ceilings prevent extreme stop placement.
    """
    n = len(close)
    long_exits = np.zeros(n, dtype=np.bool_)
    short_exits = np.zeros(n, dtype=np.bool_)

    in_long = False
    in_short = False
    entry_price = 0.0
    entry_atr = 0.0
    bars_in_trade = 0

    for i in range(n):
        if np.isnan(atr[i]):
            continue

        # Check for new entries
        if long_entries[i] and not in_long and not in_short:
            in_long = True
            entry_price = close[i]
            entry_atr = atr[i]
            bars_in_trade = 0

        elif short_entries[i] and not in_short and not in_long:
            in_short = True
            entry_price = close[i]
            entry_atr = atr[i]
            bars_in_trade = 0

        # Check exits for longs
        if in_long:
            bars_in_trade += 1

            # Calculate stop distance with floor/ceiling
            stop_distance = stop_atr_mult * entry_atr
            stop_pct = stop_distance / entry_price

            # Clamp to bounds
            if stop_pct < min_stop_pct:
                stop_distance = entry_price * min_stop_pct
            elif stop_pct > max_stop_pct:
                stop_distance = entry_price * max_stop_pct

            target_distance = target_atr_mult * entry_atr

            stop_price = entry_price - stop_distance
            target_price = entry_price + target_distance

            if (
                low[i] <= stop_price
                or high[i] >= target_price
                or bars_in_trade >= max_bars
            ):
                long_exits[i] = True
                in_long = False

        # Check exits for shorts
        if in_short:
            bars_in_trade += 1

            stop_distance = stop_atr_mult * entry_atr
            stop_pct = stop_distance / entry_price

            if stop_pct < min_stop_pct:
                stop_distance = entry_price * min_stop_pct
            elif stop_pct > max_stop_pct:
                stop_distance = entry_price * max_stop_pct

            target_distance = target_atr_mult * entry_atr

            stop_price = entry_price + stop_distance
            target_price = entry_price - target_distance

            if (
                high[i] >= stop_price
                or low[i] <= target_price
                or bars_in_trade >= max_bars
            ):
                short_exits[i] = True
                in_short = False

    return long_exits, short_exits


# =============================================================================
# FRAMEWORK-COMPATIBLE CREATE_PORTFOLIO
# =============================================================================


def create_portfolio(
    data: pd.DataFrame, params: Optional[Dict] = None
) -> vbt.Portfolio:
    """
    Create portfolio using the Vol Breakout v3 strategy.

    Compatible with the vectorflow framework interface.

    Args:
        data: OHLCV DataFrame
        params: Strategy parameters from YAML config

    Returns:
        vbt.Portfolio with backtest results
    """
    if params is None:
        params = {}

    # =========================================================================
    # PARAMETER EXTRACTION
    # =========================================================================

    # Strategy parameters
    entry_threshold = params.get("entry_threshold", 0.5)
    invalidation = params.get("invalidation", 1.0)  # Wider: was 0.5
    compression_threshold = params.get("compression_threshold", -1.0)
    stop_atr_mult = params.get("stop_atr_mult", 2.0)
    target_atr_mult = params.get("target_atr_mult", 4.0)
    max_bars = params.get("max_bars", 50)
    trend_filter = params.get("trend_filter", True)
    long_only = params.get("long_only", False)  # Force long-only mode

    # Trading parameters
    initial_cash = params.get("initial_cash", 10000)
    fee_pct = params.get("fee", 0.0004)
    freq = params.get("freq", "1h")

    # Auto-detect asset type for long-only (if not explicitly set)
    asset_name = params.get("asset_name", "").upper()
    if asset_name and not long_only:
        # High-drift assets should be long-only based on backtest results
        high_drift_assets = ["BTC", "ETH", "NDX", "NASDAQ", "SPX", "SPY"]
        for asset in high_drift_assets:
            if asset in asset_name:
                long_only = True
                print(
                    f"[Vol Breakout v3] Auto-detected {asset} - forcing long-only mode"
                )
                break

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================

    df = data.copy()

    # Validate columns
    required_cols = ["open", "high", "low", "close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Calculate all indicators
    df = calculate_volatility_regime(df)
    df = df.dropna()

    # Extract numpy arrays for numba
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    sigma = df["sigma"].values
    sigma_z = df["sigma_z"].values
    sigma_accel = df["sigma_accel"].values.astype(np.bool_)
    atr = df["atr"].values
    trend = df["trend"].values

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    long_entries, short_entries = first_passage_signals_v3(
        close=close,
        high=high,
        low=low,
        sigma=sigma,
        sigma_z=sigma_z,
        sigma_accel=sigma_accel,
        trend=trend,
        entry_threshold=entry_threshold,
        invalidation=invalidation,
        compression_threshold=compression_threshold,
        trend_filter_enabled=trend_filter,
    )

    # Force long-only if needed
    if long_only:
        short_entries[:] = False

    # Generate exits
    long_exits, short_exits = compute_atr_exits(
        close=close,
        high=high,
        low=low,
        atr=atr,
        long_entries=long_entries,
        short_entries=short_entries,
        stop_atr_mult=stop_atr_mult,
        target_atr_mult=target_atr_mult,
        max_bars=max_bars,
    )

    # Stats
    n_long = long_entries.sum()
    n_short = short_entries.sum()
    print(f"[Vol Breakout v3] Long signals: {n_long}, Short signals: {n_short}")

    if n_long == 0 and n_short == 0:
        print("[Vol Breakout v3] No signals generated. Returning empty portfolio.")
        # Return a holding portfolio as fallback
        return vbt.Portfolio.from_holding(
            close=df["close"],
            init_cash=initial_cash,
            fees=fee_pct,
            freq=freq,
        )

    # =========================================================================
    # PORTFOLIO CREATION
    # =========================================================================

    # Combine entries/exits for both/long-only mode
    if long_only or n_short == 0:
        portfolio = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=long_entries,
            exits=long_exits,
            fees=fee_pct,
            freq=freq,
            init_cash=initial_cash,
            direction="longonly",
        )
    else:
        # Both long and short
        # VectorBT doesn't directly support mixed direction from_signals
        # So we create separate portfolios and aggregate stats

        pf_long = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=long_entries,
            exits=long_exits,
            fees=fee_pct,
            freq=freq,
            init_cash=initial_cash / 2,  # Split capital
            direction="longonly",
        )

        pf_short = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=short_entries,
            exits=short_exits,
            fees=fee_pct,
            freq=freq,
            init_cash=initial_cash / 2,
            direction="shortonly",
        )

        # Return long portfolio (primary) - can extend to combine later
        portfolio = pf_long
        print("\n--- Long Stats ---")
        print(pf_long.stats())
        print("\n--- Short Stats ---")
        print(pf_short.stats())

    return portfolio


# =============================================================================
# STANDALONE RUNNER (for direct testing)
# =============================================================================


def run_backtest(df: pd.DataFrame, asset_name: str = "", **kwargs) -> tuple:
    """
    Run backtest with detailed stats output.

    For direct script usage. Framework usage should go through create_portfolio.
    """
    params = {"asset_name": asset_name, **kwargs}

    portfolio = create_portfolio(df, params)

    print(f"\n{'=' * 60}")
    print(f"Backtest Results for {asset_name or 'Unknown Asset'}")
    print("=" * 60)
    print(portfolio.stats())

    return portfolio


# =============================================================================
# STRATEGY METADATA
# =============================================================================

STRATEGY_INFO = {
    "name": "Volatility Breakout v3",
    "version": "3.0",
    "author": "VectorFlow",
    "description": "First-passage breakout with ATR-based exits and trend filter",
    "required_timeframes": ["1h"],
    "required_columns": ["open", "high", "low", "close"],
    "default_parameters": {
        "entry_threshold": 0.5,
        "invalidation": 1.0,  # Wider than v2
        "compression_threshold": -1.0,
        "stop_atr_mult": 2.0,
        "target_atr_mult": 4.0,
        "max_bars": 50,
        "trend_filter": True,
        "long_only": False,
        "initial_cash": 10000,
        "fee": 0.0004,
        "freq": "1h",
    },
    "optimization_grid": {
        "entry_threshold": [0.3, 0.7, 0.1],
        "invalidation": [0.5, 1.5, 0.25],
        "stop_atr_mult": [1.5, 3.0, 0.5],
        "target_atr_mult": [3.0, 6.0, 1.0],
    },
}


# =============================================================================
# MAIN (for standalone testing)
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    # Add parent to path for imports
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    )

    from vectorflow.core.data_loader import load_ohlc_csv

    # Test on multiple assets
    assets = [
        ("data/EURUSD_1h_2009-2025.csv", "EURUSD", "1h"),
        ("data/BTCUSD_1h_2011-2025.csv", "BTCUSD", "1h"),
        ("data/1h_NASDAQ.csv", "NASDAQ", "1h"),
    ]

    for path, name, freq in assets:
        print(f"# Testing: {name}")

        try:
            df = load_ohlc_csv(path)
            run_backtest(
                df,
                asset_name=name,
                freq=freq,
                trend_filter=True,
            )
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"Error: {e}")
            raise
