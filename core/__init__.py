"""Core modules for the momentum strategy."""

from . import backtest
from . import indicators
from . import io
from . import metrics
from . import monte_carlo
from . import optimizer
# from . import plotting  # Removed - now using visualization module

from . import walkforward

# Make modules available for direct import
__all__ = [
    'backtest',
    'indicators', 
    'io',
    'metrics',
    'monte_carlo',
    'optimizer',
    # 'plotting',  # Removed
    'walkforward'
]
