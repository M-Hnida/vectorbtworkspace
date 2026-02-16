"""
VectorFlow - Advanced Quantitative Trading Strategy Framework

A powerful backtesting and optimization framework built on VectorBT,
featuring multiple trading strategies, Monte Carlo analysis,
walk-forward validation, and interactive visualizations.
"""

# Version
__version__ = "0.1.0"

# Core functionality exports
from vectorflow.validation.path_randomization import run_path_randomization_mc
from vectorflow.core.portfolio_builder import (
    create_portfolio,
    get_available_strategies,
    get_default_parameters,
    get_optimization_grid,
)
from vectorflow.core.config_manager import load_strategy_config
from vectorflow.core.data_loader import load_ohlc_csv

# Analysis exports
from vectorflow.optimization.grid_search import run_optimization
from vectorflow.optimization.param_monte_carlo import run_monte_carlo_analysis
from vectorflow.validation.walk_forward import run_walkforward_analysis
from vectorflow.visualization.plotters import plot_comprehensive_analysis

__all__ = [
    # Version
    "__version__",
    # Core
    "create_portfolio",
    "get_available_strategies",
    "get_default_parameters",
    "get_optimization_grid",
    "load_strategy_config",
    "load_ohlc_csv",
    # Analysis
    "run_optimization",
    "run_monte_carlo_analysis",
    "run_path_randomization_mc",
    "run_walkforward_analysis",
    "plot_comprehensive_analysis",
]
