#!/usr/bin/env python3
"""SOL/BTC Statistical Arbitrage Strategy - VectorBT Implementation"""

import pandas as pd
import numpy as np
from typing import Dict
import vectorbt as vbt


def calculate_spread_indicators(sol_data: pd.DataFrame, btc_data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Calculate statistical arbitrage indicators between SOL and BTC."""
    
    # Parameters
    lookback_period = params.get("lookback_period", 20)
    zscore_entry = params.get("zscore_entry", 2.0)
    zscore_exit = params.get("zscore_exit", 0.5)
    correlation_threshold = params.get("correlation_threshold", 0.7)
    
    # Align data by timestamp
    aligned_data = pd.DataFrame({
        'sol_close': sol_data['close'],
        'btc_close': btc_data['close']
    }).dropna()
    
    if len(aligned_data) < lookback_period:
        raise ValueError(f"Insufficient data: {len(aligned_data)} < {lookback_period}")
    
    # Calculate log prices for better statistical properties
    aligned_data['sol_log'] = np.log(aligned_data['sol_close'])
    aligned_data['btc_log'] = np.log(aligned_data['btc_close'])
    
    # Calculate price ratio (SOL/BTC)
    aligned_data['price_ratio'] = aligned_data['sol_close'] / aligned_data['btc_close']
    aligned_data['log_ratio'] = aligned_data['sol_log'] - aligned_data['btc_log']
    
    # Rolling statistics for mean reversion
    aligned_data['ratio_mean'] = aligned_data['log_ratio'].rolling(lookback_period).mean()
    aligned_data['ratio_std'] = aligned_data['log_ratio'].rolling(lookback_period).std()
    
    # Z-score of the spread
    aligned_data['zscore'] = (aligned_data['log_ratio'] - aligned_data['ratio_mean']) / aligned_data['ratio_std']
    
    # Rolling correlation
    aligned_data['correlation'] = aligned_data['sol_log'].rolling(lookback_period).corr(aligned_data['btc_log'])
    
    # Volatility measures
    aligned_data['sol_volatility'] = aligned_data['sol_log'].rolling(lookback_period).std()
    aligned_data['btc_volatility'] = aligned_data['btc_log'].rolling(lookback_period).std()
    
    # Bollinger Bands for the ratio
    aligned_data['ratio_upper'] = aligned_data['ratio_mean'] + (2 * aligned_data['ratio_std'])
    aligned_data['ratio_lower'] = aligned_data['ratio_mean'] - (2 * aligned_data['ratio_std'])
    
    # Store thresholds for reference
    aligned_data['zscore_entry'] = zscore_entry
    aligned_data['zscore_exit'] = zscore_exit
    aligned_data['correlation_threshold'] = correlation_threshold
    
    return aligned_data


def generate_stat_arb_signals(data: pd.DataFrame, params: Dict) -> tuple:
    """Generate statistical arbitrage trading signals."""
    
    # Parameters
    zscore_entry = params.get("zscore_entry", 2.0)
    zscore_exit = params.get("zscore_exit", 0.5)
    correlation_threshold = params.get("correlation_threshold", 0.7)
    min_holding_period = params.get("min_holding_period", 5)
    
    # Initialize signals
    long_sol_entries = pd.Series(False, index=data.index)
    long_sol_exits = pd.Series(False, index=data.index)
    short_sol_entries = pd.Series(False, index=data.index)
    short_sol_exits = pd.Series(False, index=data.index)
    
    # Track positions
    sol_position = 0  # 1 = long SOL/short BTC, -1 = short SOL/long BTC, 0 = flat
    entry_time = None
    
    for i in range(len(data)):
        current_zscore = data['zscore'].iloc[i]
        current_correlation = data['correlation'].iloc[i]
        
        # Skip if we don't have enough data or correlation is too low
        if pd.isna(current_zscore) or pd.isna(current_correlation):
            continue
            
        if abs(current_correlation) < correlation_threshold:
            continue
        
        current_time = data.index[i]
        
        # Entry conditions
        if sol_position == 0:  # No position
            if current_zscore > zscore_entry:
                # SOL is overvalued relative to BTC -> Short SOL, Long BTC
                short_sol_entries.iloc[i] = True
                sol_position = -1
                entry_time = current_time
                
            elif current_zscore < -zscore_entry:
                # SOL is undervalued relative to BTC -> Long SOL, Short BTC
                long_sol_entries.iloc[i] = True
                sol_position = 1
                entry_time = current_time
        
        # Exit conditions
        elif sol_position != 0:
            # Check minimum holding period
            if entry_time and (current_time - entry_time).total_seconds() / 3600 < min_holding_period:
                continue
            
            # Mean reversion exit
            if sol_position == 1 and current_zscore > -zscore_exit:
                # Close long SOL position
                long_sol_exits.iloc[i] = True
                sol_position = 0
                entry_time = None
                
            elif sol_position == -1 and current_zscore < zscore_exit:
                # Close short SOL position
                short_sol_exits.iloc[i] = True
                sol_position = 0
                entry_time = None
            
            # Stop loss on correlation breakdown
            elif abs(current_correlation) < correlation_threshold * 0.5:
                if sol_position == 1:
                    long_sol_exits.iloc[i] = True
                else:
                    short_sol_exits.iloc[i] = True
                sol_position = 0
                entry_time = None
    
    return long_sol_entries, long_sol_exits, short_sol_entries, short_sol_exits


def create_portfolio(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """Create SOL/BTC Statistical Arbitrage portfolio."""
    if params is None:
        params = {}
    
    # Parameters
    initial_cash = params.get("initial_cash", 10000)
    fee = params.get("fee", 0.001)
    freq = params.get("freq", "1h")
    
    # Import pairs data loader
    try:
        from strategies.pairs_data_loader import create_synthetic_pairs_data, validate_pairs_data
    except ImportError:
        # Fallback if pairs_data_loader is not available
        pass
    
    # Check if we have the required columns for both assets
    required_cols = ['sol_close', 'btc_close']
    if not all(col in data.columns for col in required_cols):
        # If we don't have both assets, create synthetic data for demonstration
        if 'close' in data.columns:
            print("📊 Creating synthetic BTC data correlated with SOL for demonstration")
            combined_data = create_synthetic_pairs_data(data, correlation=0.75)
        else:
            raise ValueError("Data must contain either 'close' column or both 'sol_close' and 'btc_close' columns")
    else:
        combined_data = data
        
    # Validate pairs data
    if not validate_pairs_data(combined_data, ['sol', 'btc']):
        raise ValueError("Invalid pairs data for statistical arbitrage")
    
    # Calculate statistical arbitrage indicators
    stat_data = calculate_spread_indicators(
        pd.DataFrame({'close': combined_data['sol_close']}),
        pd.DataFrame({'close': combined_data['btc_close']}),
        params
    )
    
    # Generate trading signals
    long_entries, long_exits, short_entries, short_exits = generate_stat_arb_signals(stat_data, params)
    
    # Combine all entries and exits
    all_entries = long_entries | short_entries
    all_exits = long_exits | short_exits
    
    # Position sizing based on volatility
    volatility_lookback = params.get("volatility_lookback", 20)
    sol_vol = stat_data['sol_volatility'].rolling(volatility_lookback).mean()
    base_size = initial_cash * 0.5  # Use 50% of capital
    
    # Adjust size based on volatility (lower vol = larger size)
    vol_adj_factor = 0.02 / sol_vol.clip(lower=0.01)  # Target 2% volatility
    position_size = (base_size * vol_adj_factor).clip(upper=initial_cash * 0.8)
    
    # Create size array
    sizes = pd.Series(0.0, index=stat_data.index)
    sizes[all_entries] = position_size[all_entries]
    
    # Create size array for long positions only
    long_sizes = pd.Series(0.0, index=stat_data.index)
    long_sizes[long_entries] = position_size[long_entries]
    
    # For simplicity, create a long-only portfolio representing the statistical arbitrage
    # In practice, this would be combined with a BTC hedge
    portfolio = vbt.Portfolio.from_signals(
        close=stat_data['sol_close'],
        entries=long_entries,
        exits=long_exits,
        size=long_sizes,
        size_type="value",
        init_cash=initial_cash,
        fees=fee,
        freq=freq,
        accumulate=False
    )
    
    return portfolio


def calculate_pair_correlation(sol_data: pd.DataFrame, btc_data: pd.DataFrame, window: int = 30) -> pd.Series:
    """Calculate rolling correlation between SOL and BTC."""
    sol_returns = sol_data['close'].pct_change()
    btc_returns = btc_data['close'].pct_change()
    return sol_returns.rolling(window).corr(btc_returns)


def calculate_spread_metrics(data: pd.DataFrame) -> Dict:
    """Calculate key spread metrics for analysis."""
    if 'zscore' not in data.columns:
        return {}
    
    zscore = data['zscore'].dropna()
    correlation = data['correlation'].dropna()
    
    return {
        'mean_zscore': zscore.mean(),
        'std_zscore': zscore.std(),
        'max_zscore': zscore.max(),
        'min_zscore': zscore.min(),
        'mean_correlation': correlation.mean(),
        'min_correlation': correlation.min(),
        'zscore_reversions': len(zscore[(zscore.abs() > 2) & (zscore.shift(1).abs() < 1)]),
        'high_correlation_pct': (correlation.abs() > 0.7).mean() * 100
    }