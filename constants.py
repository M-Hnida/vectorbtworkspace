#!/usr/bin/env python3
"""
Application-wide constants for the trading system.
Only contains constants used across multiple modules.

Organization:
- Domain-specific constants are in their respective modules
- Application-wide constants are here
- Runtime config should be in .env or config files
"""

# =============================================================================
# PORTFOLIO CONFIGURATION (used across multiple modules)
# =============================================================================
INIT_CASH = 10000
FEES = 0.001

# =============================================================================
# OPTIMIZATION CONFIGURATION (used by optimizer.py and main.py)
# =============================================================================
MAX_PARAM_COMBINATIONS = 50

# =============================================================================
# MONTE CARLO CONFIGURATION (used by optimizer.py and main.py)
# =============================================================================
MONTE_CARLO_SIMULATIONS = 500
MONTE_CARLO_BATCH_SIZE = 128

# =============================================================================
# WALK-FORWARD CONFIGURATION (used by walk_forward.py and main.py)
# =============================================================================
TRAIN_WINDOW_DAYS = 730  # 2 years
TEST_WINDOW_DAYS = 180   # 6 months
MAX_WINDOWS = 10

# =============================================================================
# DATA CONFIGURATION (used by data_manager.py and strategies)
# =============================================================================

# Timeframe Mappings (milliseconds) - shared across data loading
TIMEFRAME_MS = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '30m': 30 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
}

# Default Timeframes
DEFAULT_TIMEFRAMES = ['1h']

# Required OHLCV Columns
REQUIRED_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
REQUIRED_MINIMUM_COLUMNS = ['open', 'high', 'low', 'close']

# =============================================================================
# VECTORBT STATS KEYS (standardized across all modules)
# =============================================================================
STAT_TOTAL_RETURN = "Total Return [%]"
STAT_SHARPE_RATIO = "Sharpe Ratio"
STAT_MAX_DRAWDOWN = "Max Drawdown [%]"
STAT_WIN_RATE = "Win Rate [%]"
STAT_TOTAL_TRADES = "Total Trades"
