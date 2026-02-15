#!/usr/bin/env python3
"""Grid Trading Strategy - Vectorbt Implementation"""

import pandas as pd
import numpy as np
from typing import Dict
import vectorbt as vbt


def compute_grid_levels(center_price, k, n, m):
    """Calculate geometric grid levels"""
    n, m = int(n), int(m)
    levels = [center_price * (k**i) for i in range(-(n - m), m + 1)]
    return np.array(sorted(levels))


def generate_grid_signals(data, grid_k, grid_n, initial_capital, fee_pct):
    """Generate dynamic grid trading signals"""
    n = len(data)
    entries = pd.Series(False, index=data.index)
    exits = pd.Series(False, index=data.index)
    grid_resets = pd.Series(False, index=data.index)

    # Initialize
    num_above_m = int(grid_n) // 2
    start_price = data["close"].iloc[0]

    # Wallet state initialization
    usdt_balance = initial_capital * (grid_n - num_above_m) / grid_n
    crypto_balance = (initial_capital * num_above_m / grid_n) / start_price
    cumulative_profit = 0
    input_money = initial_capital

    grid_levels = compute_grid_levels(start_price, grid_k, grid_n, num_above_m)
    current_level_index = int(grid_n - num_above_m)

    for i in range(1, n):
        current_price = data["close"].iloc[i]
        high = data["high"].iloc[i]
        low = data["low"].iloc[i]

        # Check for grid reset conditions
        if high > grid_levels[-1]:
            # Upper limit breach - sell all crypto
            if crypto_balance > 0:
                usdt_gain = crypto_balance * high * (1 - fee_pct)
                usdt_balance += usdt_gain
                crypto_balance = 0

            # Calculate profit and reset
            arbitrage_profit = usdt_balance - input_money
            cumulative_profit += arbitrage_profit

            # Reset wallet
            input_money = initial_capital + cumulative_profit
            usdt_balance = input_money * (grid_n - num_above_m) / grid_n
            crypto_balance = (input_money * num_above_m / grid_n) / high

            # Reset grid
            grid_levels = compute_grid_levels(high, grid_k, grid_n, num_above_m)
            current_level_index = int(grid_n - num_above_m)
            grid_resets.iloc[i] = True
            continue

        elif low < grid_levels[0]:
            # Lower limit breach - hold crypto, use profit for new grid
            input_money = initial_capital + cumulative_profit
            usdt_balance = input_money * (grid_n - num_above_m) / grid_n
            crypto_balance += (input_money * num_above_m / grid_n) / low

            # Reset grid
            grid_levels = compute_grid_levels(low, grid_k, grid_n, num_above_m)
            current_level_index = int(grid_n - num_above_m)
            grid_resets.iloc[i] = True
            continue

        # Check for level crossings
        for level_idx in range(len(grid_levels)):
            if (
                current_price > grid_levels[level_idx]
                and level_idx > current_level_index
            ):
                # Up cross - sell signal
                if crypto_balance > 0:
                    g_i = grid_levels[level_idx] / grid_levels[0]  # Geometric factor
                    sell_amount = crypto_balance / g_i
                    usdt_gain = sell_amount * current_price * (1 - fee_pct)
                    usdt_balance += usdt_gain
                    crypto_balance -= sell_amount
                    exits.iloc[i] = True
                    current_level_index = level_idx
                    break

            elif (
                current_price < grid_levels[level_idx]
                and level_idx < current_level_index
            ):
                # Down cross - buy signal
                if usdt_balance > 0:
                    g_j = grid_levels[-1] / grid_levels[level_idx]  # Geometric factor
                    buy_amount_usdt = usdt_balance / g_j
                    crypto_gain = (buy_amount_usdt * (1 - fee_pct)) / current_price
                    crypto_balance += crypto_gain
                    usdt_balance -= buy_amount_usdt
                    entries.iloc[i] = True
                    current_level_index = level_idx
                    break

    return entries, exits, grid_resets, cumulative_profit


def create_portfolio(data: pd.DataFrame, params: Dict = None) -> "vbt.Portfolio":
    """Create Grid Trading Strategy portfolio."""
    if params is None:
        params = {}

    # Parameters
    grid_k = params.get("grid_k", 1.02)
    grid_n = params.get("grid_n", 15)
    initial_cash = params.get("initial_cash", 10000)
    fee_pct = params.get("fee", 0.0008)
    freq = params.get("freq", "1H")

    # Generate grid signals
    entries, exits, grid_resets, cumulative_profit = generate_grid_signals(
        data, grid_k, grid_n, initial_cash, fee_pct
    )

    # Dynamic position sizing
    sizes = pd.Series(0.0, index=data.index)
    base_size = initial_cash / grid_n

    for i, (entry, exit) in enumerate(zip(entries, exits)):
        if entry or exit:
            grid_factor = 1.0 + (i % int(grid_n)) * 0.1
            sizes.iloc[i] = base_size * grid_factor

    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=data["close"],
        entries=entries,
        exits=exits,
        size=sizes,
        size_type="value",
        init_cash=initial_cash,
        fees=fee_pct,
        freq=freq,
        accumulate=True,
        direction="both",
    )

    return portfolio
