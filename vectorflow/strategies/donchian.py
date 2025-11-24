#!/usr/bin/env python3
"""Donchian ATR Trend Strategy - Vectorbt Implementation"""

import pandas as pd
import numpy as np
from typing import Dict
import vectorbt as vbt
import pandas_ta as ta


def calculate_donchian(data, period):
    """Calculate Donchian Channel"""
    donchian_high = data['high'].rolling(period).max()
    donchian_low = data['low'].rolling(period).min()
    return donchian_high.shift(1), donchian_low.shift(1)  # Exclure la bougie courante


def calculate_atr_normalized(data, atr_period):
    """Calculate ATR and normalized ATR"""
    atr = ta.atr(data['high'], data['low'], data['close'], length=atr_period)
    atr_normalized = atr / data['close']
    return atr, atr_normalized


def calculate_indicators(data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
    """Calculate all technical indicators for the strategy"""
    if params is None:
        params = {}

    # Default parameters
    donchian_period = params.get("donchian_period", 32)
    atr_period = params.get("atr_period", 13)
    atr_sma_period = params.get("atr_sma_period", 16)
    ema_short_period = params.get("ema_short_period", 50)
    sma_long_period = params.get("sma_long_period", 204)
    adx_period = params.get("adx_period", 12)
    adx_threshold = params.get("adx_threshold", 15)
    volume_ma_period = params.get("volume_ma_period", 19)
    atr_normalized_threshold = params.get("atr_normalized_threshold", 0.012)

    # Donchian Channel
    donchian_upper, donchian_lower = calculate_donchian(data, donchian_period)
    data['donchian_upper'] = donchian_upper
    data['donchian_lower'] = donchian_lower

    # ATR and normalized ATR
    atr, atr_normalized = calculate_atr_normalized(data, atr_period)
    data['atr'] = atr
    data['atr_normalized'] = atr_normalized

    # ATR SMA
    data['atr_sma'] = data['atr'].rolling(atr_sma_period).mean()

    # ADX
    data['adx'] = ta.adx(data['high'], data['low'], data['close'], length=adx_period)['ADX_{}'.format(adx_period)]

    # EMA short
    data['ema_short'] = ta.ema(data['close'], length=ema_short_period)

    # Volume MA
    data['volume_ma'] = ta.sma(data['volume'], length=volume_ma_period)

    # SMA long on 4h timeframe (simulated by resampling)
    try:
        data_4h = data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        sma_long_4h = ta.sma(data_4h['close'], length=sma_long_period)
        
        # Handle None case (insufficient data for SMA)
        if sma_long_4h is not None:
            # Forward fill to align with original timeframe
            data['sma_long_4h'] = sma_long_4h.reindex(data.index, method='ffill')
        else:
            # Fallback: use same timeframe SMA if 4H resample fails
            data['sma_long_4h'] = ta.sma(data['close'], length=sma_long_period)
    except Exception as e:
        # If resampling fails (e.g., insufficient data), use 1h SMA as fallback
        data['sma_long_4h'] = ta.sma(data['close'], length=sma_long_period)

    # Threshold values
    data['adx_threshold'] = adx_threshold
    data['atr_normalized_threshold'] = atr_normalized_threshold

    return data


def calculate_dynamic_stop_loss(data: pd.DataFrame, entry_price: float, position: int, 
                               atr_sl_multiplier: float, params: Dict = None) -> float:
    """
    Calculate dynamic stop-loss based on Donchian Channel and ATR
    """
    if params is None:
        params = {}
    
    donchian_period = params.get("donchian_period", 32)
    
    # Get last candle data
    last_candle = data.iloc[-1]
    atr = last_candle['atr']
    
    if position > 0:  # Long position
        # Initial stop = entry - ATR_SL_multiplier * ATR
        initial_sl = entry_price - (atr_sl_multiplier * atr)
        
        # Trailing dynamic stop towards Donchian lower band
        donchian_trailing = last_candle['donchian_lower']
        
        # Use maximum between initial stop and trailing Donchian
        # This allows stop to move up with the market
        new_stop = max(initial_sl, donchian_trailing)
        
        # Calculate stop percentage
        stop_percentage = (new_stop - entry_price) / entry_price
        
        # Limit stop to -10% maximum to avoid too tight stops
        return max(stop_percentage, -0.10)
    
    elif position < 0:  # Short position
        # Initial stop = entry + ATR_SL_multiplier * ATR
        initial_sl = entry_price + (atr_sl_multiplier * atr)
        
        # Trailing dynamic stop towards Donchian upper band
        donchian_trailing = last_candle['donchian_upper']
        
        # Use minimum between initial stop and trailing Donchian
        new_stop = min(initial_sl, donchian_trailing)
        
        # Calculate stop percentage
        stop_percentage = (entry_price - new_stop) / entry_price
        
        # Limit stop to -10% maximum to avoid too tight stops
        return max(stop_percentage, -0.10)
    
    return 0.0  # No stop for flat position


def apply_position_sizing(data: pd.DataFrame, params: Dict = None) -> pd.Series:
    """
    Position sizing based on ATR volatility
    Higher ATR = smaller position size
    """
    if params is None:
        params = {}
    
    # Get parameters
    initial_cash = params.get("initial_cash", 10000)
    max_position_size = params.get("max_position_size", 0.2)  # 20% of portfolio
    
    # Calculate ATR normalized (volatility relative)
    atr_normalized = data['atr'] / data['close']
    
    # Reduce position size if volatility is high
    # Higher volatility = smaller position
    volatility_multiplier = 1.0 / (1.0 + atr_normalized * 10)  # Gradual reduction
    
    # Calculate position size as percentage of portfolio
    position_size = max_position_size * volatility_multiplier
    
    return position_size.fillna(max_position_size)


def apply_leverage_management(data: pd.DataFrame, params: Dict = None) -> pd.Series:
    """
    Leverage management based on volatility
    Reduce leverage if volatility is high
    """
    if params is None:
        params = {}
    
    # Get parameters
    atr_normalized_threshold = params.get("atr_normalized_threshold", 0.012)
    max_leverage = params.get("max_leverage", 3.0)
    
    # Calculate ATR normalized
    atr_normalized = data['atr'] / data['close']
    
    # Initialize leverage series
    leverage = pd.Series(1.0, index=data.index)
    
    # Reduce leverage if volatility is high
    high_volatility = atr_normalized > atr_normalized_threshold * 2
    medium_volatility = (atr_normalized > atr_normalized_threshold * 1.5) & ~high_volatility
    
    # Apply leverage rules
    leverage[high_volatility] = 1.0  # No leverage for high volatility
    leverage[medium_volatility] = 2.0  # Max 2x leverage for medium volatility
    leverage[~high_volatility & ~medium_volatility] = max_leverage  # Max 3x leverage for normal volatility
    
    return leverage


def calculate_dynamic_stop_loss_series(data: pd.DataFrame, params: Dict = None) -> tuple:
    """
    Calculate dynamic stop-loss series based on Donchian Channel and ATR
    Returns stop price levels for each candle
    """
    if params is None:
        params = {}
    
    atr_sl_multiplier = params.get("atr_sl_multiplier", 5.1)
    
    # Calculate dynamic stops for long positions
    long_stop_initial = data['close'] - (atr_sl_multiplier * data['atr'])
    long_stop_trailing = data['donchian_lower']
    stop_long = pd.concat([long_stop_initial, long_stop_trailing], axis=1).max(axis=1)
    stop_long = stop_long.clip(lower=data['close'] * 0.9)  # Limit to -10% max
    
    # Calculate dynamic stops for short positions
    short_stop_initial = data['close'] + (atr_sl_multiplier * data['atr'])
    short_stop_trailing = data['donchian_upper']
    stop_short = pd.concat([short_stop_initial, short_stop_trailing], axis=1).min(axis=1)
    stop_short = stop_short.clip(upper=data['close'] * 1.1)  # Limit to +10% max
    
    return stop_long, stop_short




def create_portfolio_vectorized(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """
    Donchian ATR Trend Strategy - Fully Vectorized Implementation
    
    This strategy combines:
    - Donchian Channel breakouts
    - ATR-based volatility filtering
    - ADX for trend strength
    - Multi-timeframe trend filtering
    - Dynamic stop-loss and take-profit
    - Position sizing based on volatility
    - Leverage management
    """
    if params is None:
        params = {}

    # Strategy parameters
    init_cash = params.get("initial_cash", 10000)
    fees = params.get("fee", 0.001)
    atr_sl_multiplier = params.get("atr_sl_multiplier", 5.1)
    atr_tp_multiplier = params.get("atr_tp_multiplier", 9.5)
    partial_tp_atr_multiplier = params.get("partial_tp_atr_multiplier", 6.9)
    max_position_size = params.get("max_position_size", 0.2)
    
    # Calculate indicators
    data = calculate_indicators(data, params)
    
    # Apply position sizing
    position_sizes = apply_position_sizing(data, params)
    
    # Apply leverage management
    leverage = apply_leverage_management(data, params)
    
    # Calculate dynamic stop-loss series
    stop_long, stop_short = calculate_dynamic_stop_loss_series(data, params)
    
    # Entry conditions
    
    # Common filters
    has_minimum_volatility = data['atr_normalized'] > data['atr_normalized_threshold']
    has_trend = data['adx'] > data['adx_threshold']
    has_volume = data['volume'] > data['volume_ma']
    atr_filter = data['atr'] > data['atr_sma']
    
    # Trend filter
    trend_filter = (
        (data['ema_short'] > data['sma_long_4h']) &
        (data['close'] > data['ema_short'])
    )
    
    # News hours filter (avoid important news hours) - Vectorized version
    current_hour = data.index.hour
    avoid_news_hours = ~pd.Series(current_hour, index=data.index).isin([8, 9, 13, 14, 15])
    
    # Long entry conditions
    long_condition = (
        (data['close'] > data['donchian_upper']) &  # Breakout Donchian
        trend_filter &
        has_minimum_volatility &
        has_trend &
        has_volume &
        atr_filter &
        avoid_news_hours
    )
    
    # Short entry conditions
    short_condition = (
        (data['close'] < data['donchian_lower']) &  # Breakdown Donchian
        (data['ema_short'] < data['sma_long_4h']) &
        (data['close'] < data['ema_short']) &
        has_minimum_volatility &
        has_trend &
        has_volume &
        atr_filter &
        avoid_news_hours
    )
    
    # Additional confirmation using confirm_trade_entry logic - Vectorized version
    atr_normalized_threshold = params.get("atr_normalized_threshold", 0.012)
    adx_threshold = params.get("adx_threshold", 15)
    
    # Vectorized confirmation conditions
    has_minimum_volatility = data['atr_normalized'] >= atr_normalized_threshold
    has_trend_strength = data['adx'] >= adx_threshold
    has_sufficient_volume = data['volume'] > data['volume_ma']
    
    # Combine all confirmation conditions
    confirm_condition = has_minimum_volatility & has_trend_strength & has_sufficient_volume
    
    # Final entry signals
    enter_long = long_condition & confirm_condition
    enter_short = short_condition & confirm_condition
    
    # Calculate take-profit levels
    tp_long_full = data['close'] + (atr_tp_multiplier * data['atr'])
    tp_long_partial = data['close'] + (partial_tp_atr_multiplier * data['atr'])
    tp_short_full = data['close'] - (atr_tp_multiplier * data['atr'])
    tp_short_partial = data['close'] - (partial_tp_atr_multiplier * data['atr'])
    
    # Simple exit conditions for now - will be enhanced with custom stops
    # For vectorbt, we'll use custom stop and take-profit levels
    exit_long = pd.Series(False, index=data.index)
    exit_short = pd.Series(False, index=data.index)
    
    # Debug info
    total_long = enter_long.sum()
    total_short = enter_short.sum()
    total_trades = total_long + total_short
    years = len(data) / (365 * 24)
    trades_per_year = total_trades / years if years > 0 else 0
    
    # Calculate order sizes based on position sizing and leverage
    order_sizes_long = position_sizes * leverage * init_cash / data['close']
    order_sizes_short = -position_sizes * leverage * init_cash / data['close']
    
    # Create order arrays for from_orders
    orders = []
    order_sizes_list = []
    order_prices = []
    order_dates = []
    
    # Long entries
    long_entry_indices = enter_long[enter_long].index
    for idx in long_entry_indices:
        # Entry order
        orders.append('buy')
        order_sizes_list.append(order_sizes_long.loc[idx])
        order_prices.append(data['close'].loc[idx])
        order_dates.append(idx)
        
        # Add stop loss order (50% of position)
        orders.append('sell')
        order_sizes_list.append(-order_sizes_long.loc[idx] * 0.5)  # Close 50% of position
        order_prices.append(stop_long.loc[idx])
        order_dates.append(idx)
        
        # Add partial take profit order (30% of position)
        orders.append('sell')
        order_sizes_list.append(-order_sizes_long.loc[idx] * 0.3)  # Close 30% of position
        order_prices.append(tp_long_partial.loc[idx])
        order_dates.append(idx)
        
        # Add full take profit order (20% of position)
        orders.append('sell')
        order_sizes_list.append(-order_sizes_long.loc[idx] * 0.2)  # Close 20% of position
        order_prices.append(tp_long_full.loc[idx])
        order_dates.append(idx)
    
    # Short entries
    short_entry_indices = enter_short[enter_short].index
    for idx in short_entry_indices:
        # Entry order
        orders.append('sell')
        order_sizes_list.append(order_sizes_short.loc[idx])
        order_prices.append(data['close'].loc[idx])
        order_dates.append(idx)
        
        # Add stop loss order (50% of position)
        orders.append('buy')
        order_sizes_list.append(-order_sizes_short.loc[idx] * 0.5)  # Close 50% of position
        order_prices.append(stop_short.loc[idx])
        order_dates.append(idx)
        
        # Add partial take profit order (30% of position)
        orders.append('buy')
        order_sizes_list.append(-order_sizes_short.loc[idx] * 0.3)  # Close 30% of position
        order_prices.append(tp_short_partial.loc[idx])
        order_dates.append(idx)
        
        # Add full take profit order (20% of position)
        orders.append('buy')
        order_sizes_list.append(-order_sizes_short.loc[idx] * 0.2)  # Close 20% of position
        order_prices.append(tp_short_full.loc[idx])
        order_dates.append(idx)
    
    # Convert lists to pandas Series with proper indexing
    if len(orders) > 0:
        orders_series = pd.Series(orders, index=pd.Index(order_dates, name='date'))
        sizes_series = pd.Series(order_sizes_list, index=pd.Index(order_dates, name='date'))
        prices_series = pd.Series(order_prices, index=pd.Index(order_dates, name='date'))
    else:
        orders_series = pd.Series([], dtype='object')
        sizes_series = pd.Series([], dtype='float64')
        prices_series = pd.Series([], dtype='float64')
    
    # Create portfolio with custom stops using from_signals
    primary_timeframe = params.get("primary_timeframe", "1H")
    
    # Create portfolio using from_signals with SL and TP as percentages
    # Convert stop levels to percentages and ensure they are positive
    sl_long = (data['close'] - stop_long) / data['close']
    sl_long = sl_long.clip(lower=0)  # Ensure stop loss is positive
    
    sl_short = (stop_short - data['close']) / data['close']
    sl_short = sl_short.clip(lower=0)  # Ensure stop loss is positive
    
    tp_long_pct = (tp_long_full - data['close']) / data['close']
    tp_long_pct = tp_long_pct.clip(lower=0)  # Ensure take profit is positive
    
    tp_short_pct = (data['close'] - tp_short_full) / data['close']
    tp_short_pct = tp_short_pct.clip(lower=0)  # Ensure take profit is positive
    
    # Create portfolio using from_signals with SL and TP
    portfolio = vbt.Portfolio.from_signals(
        close=data['close'],
        entries=enter_long,
        exits=exit_long,
        short_entries=enter_short,
        short_exits=exit_short,
        sl_stop=sl_long,
        tp_stop=tp_long_pct,
        init_cash=init_cash,
        fees=fees,
        freq=primary_timeframe,
    )
    
    return portfolio


def confirm_trade_entry(data: pd.DataFrame, index: int, params: Dict = None) -> bool:
    """
    Confirmer l'entrée uniquement si tous les critères sont réunis
    Similaire à la fonction confirm_trade_entry de FreqTrade
    """
    if params is None:
        params = {}
    
    try:
        # Récupérer le seuil ATR normalisé
        atr_normalized_threshold = params.get("atr_normalized_threshold", 0.012)
        adx_threshold = params.get("adx_threshold", 15)
        
        # Vérifier que la volatilité est suffisante
        if data['atr_normalized'].iloc[index] < atr_normalized_threshold:
            return False
        
        # Vérifier qu'il y a une tendance
        if data['adx'].iloc[index] < adx_threshold:
            return False
        
        # Vérifier le volume
        if data['volume'].iloc[index] <= data['volume_ma'].iloc[index]:
            return False
        
        return True
        
    except Exception:
        return False


def generate_sample_data(start_date='2023-01-01', end_date='2024-01-01', initial_price=100, volatility=0.02):
    """Génère des données de test pour la stratégie"""
    np.random.seed(42)  # Pour la reproductibilité
    
    # Créer un index temporel
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    n_points = len(dates)
    
    # Générer un prix de base avec une tendance et une volatilité
    returns = np.random.normal(0, volatility, n_points)
    prices = initial_price * (1 + returns).cumprod()
    
    # Générer OHLCV
    base_prices = pd.Series(prices, index=dates)
    
    # Simuler des bougies réalistes
    data = pd.DataFrame(index=dates)
    data['close'] = base_prices
    
    # Générer OHLC avec des variations réalistes
    data['open'] = data['close'].shift(1).fillna(initial_price)
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, n_points))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, n_points))
    data['volume'] = np.random.uniform(1000, 10000, n_points)
    
    # S'assurer que high >= low >= open/close
    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    
    return data


def create_portfolio(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """Create portfolio using the vectorized implementation"""
    return create_portfolio_vectorized(data, params)


def run_backtest(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """Run backtest with the strategy"""
    if params is None:
        params = {}
    
    # Set default parameters
    default_params = {
        "donchian_period": 32,
        "atr_period": 13,
        "atr_sma_period": 16,
        "ema_short_period": 50,
        "sma_long_period": 204,
        "adx_period": 12,
        "adx_threshold": 15,
        "volume_ma_period": 19,
        "atr_normalized_threshold": 0.012,
        "atr_sl_multiplier": 5.1,
        "atr_tp_multiplier": 9.5,
        "partial_tp_atr_multiplier": 6.9,
        "initial_cash": 10000,
        "fee": 0.001,
        "max_position_size": 0.2,
        "max_leverage": 3.0,
        "primary_timeframe": "1H"
    }
    
    # Merge with user provided params
    default_params.update(params)
    
    # Run strategy
    portfolio = create_portfolio(data, default_params)
    
    return portfolio


# This strategy is intended to be run from main.py
# Example usage:
# from strategies.donchian import create_portfolio_vectorized
# portfolio = create_portfolio_vectorized(data, params)



