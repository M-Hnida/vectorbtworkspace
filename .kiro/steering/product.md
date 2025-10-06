# Trading Strategy Analysis System

A comprehensive Python-based trading strategy backtesting and analysis framework that supports multiple data sources and provides advanced analytics.

## Core Features

- **Multi-Strategy Support**: Automated strategy discovery and execution (CHOP, RSI, Momentum, ORB, TDI, VectorBT)
- **Multiple Data Sources**: CSV files, CCXT exchanges (Binance), and Freqtrade data formats
- **Advanced Analytics**: Parameter optimization, walk-forward analysis, Monte Carlo simulations
- **Visualization**: Comprehensive plotting with Plotly for performance analysis
- **Configuration-Driven**: YAML and JSON configuration files for strategy parameters

## Key Components

- **Strategy Registry**: Auto-discovery system that finds and loads strategies from the `strategies/` folder
- **Data Manager**: Unified data loading from CSV, CCXT exchanges, or Freqtrade formats
- **Optimizer**: Grid search parameter optimization with composite scoring
- **Walk-Forward Analysis**: Time-series cross-validation for strategy robustness
- **Plotter**: Interactive visualizations for strategy performance and analysis

## Target Users

Quantitative traders, researchers, and developers who need a flexible framework for backtesting trading strategies with professional-grade analytics and multiple data source support.