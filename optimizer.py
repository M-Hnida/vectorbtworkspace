#!/usr/bin/env python3
"""
Parameter Optimization Module
Handles grid search optimization and walk-forward analysis for trading strategies.
"""

from typing import Dict, Any, List, Optional, Union
from itertools import product
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

from core_components import run_backtest
from metrics import calc_metrics
from base import BaseStrategy, StrategyConfig, Signals
import vectorbt as vbt

# Configure logging
logger = logging.getLogger(__name__)


def _optimize_single_combination(args):
    """Global function for parallel optimization of a single parameter combination."""
    try:
        data, strategy_class, config, params, combination_id = args
        
        # Create strategy with parameters
        new_config = copy.deepcopy(config)
        new_config.parameters.update(params)
        temp_strategy = strategy_class(new_config)
        
        # Generate signals
        required_tfs = temp_strategy.get_required_timeframes()
        main_tf = required_tfs[0] if required_tfs else '1h'
        tf_data = {main_tf: data}
        signals = temp_strategy.generate_signals(tf_data)
        
        # Run backtest
        portfolio = run_backtest(data, signals)
        metrics = calc_metrics(portfolio)
        
        return OptimizationResult(
            portfolio=portfolio,  # Will be replaced with final portfolio for best result
            metrics=metrics,
            sharpe_ratio=float(metrics.get('sharpe', -999)),
            total_return=metrics.get('total_return', -999),
            max_drawdown=metrics.get('max_drawdown', 999),
            profit_factor=metrics.get('profit_factor', 0),
            param_combination=params,
            combination_id=combination_id
        )
        
    except Exception as e:
        # Return None on any error - will be handled by the optimizer
        return None

@dataclass
class OptimizationConfig:
    """Configuration class for optimization parameters."""
    # Grid search parameters
    split_ratio: float = 0.7
    verbose: bool = True
    max_workers: int = min(4, mp.cpu_count())  # Limit parallel workers
    enable_parallel: bool = True  # Enable/disable parallel processing
    early_stopping: bool = True
    early_stopping_patience: int = 10  # Stop if no improvement for N combinations
    
    # Walk forward analysis parameters
    window_size: int = 504  # Minimum 2 years of daily data (252 * 2)
    step_size: int = 63     # Quarter step (3 months)
    num_windows: int = 5
    
    # Monte Carlo analysis parameters
    num_mc_runs: int = 100
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.split_ratio < 1:
            raise ValueError("split_ratio must be between 0 and 1")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.num_windows <= 0:
            raise ValueError("num_windows must be positive")
        if self.num_mc_runs <= 0:
            raise ValueError("num_mc_runs must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")


@dataclass
class OptimizationResult:
    """Data contract for optimization results."""
    portfolio: vbt.Portfolio
    metrics: Dict[str, Union[float, int]]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    profit_factor: float
    param_combination: Optional[Dict[str, Any]] = None
    combination_id: Optional[int] = None


@dataclass
class WalkForwardResult:
    """Data contract for walk-forward analysis results."""
    windows: List[Dict[str, Any]]
    summary: Dict[str, float]
    efficiency: float


@dataclass
class MonteCarloResult:
    """Data contract for Monte Carlo analysis results."""
    base_metrics: Dict[str, Union[float, int]]
    simulations: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    num_successful_runs: int


class ParameterOptimizer:
    """Handles parameter optimization using grid search with parallel processing."""
    
    def __init__(self, strategy: BaseStrategy, config: StrategyConfig, 
                 opt_config: Optional[OptimizationConfig] = None):
        self.strategy = strategy
        self.config = config
        self.optimization_grid = config.optimization_grid
        self.opt_config = opt_config or OptimizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._best_sharpe = -999  # Track best sharpe for early stopping
        
    def optimize(self, data: pd.DataFrame) -> OptimizationResult:
        """Run grid search optimization on training data."""
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
            
        if self.opt_config.verbose:
            self.logger.info("Running Grid Search Optimization")
        
        # Split data for training
        split_idx = int(len(data) * self.opt_config.split_ratio)
        train_data = data.iloc[:split_idx].copy()
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        
        if not param_combinations:
            if self.opt_config.verbose:
                self.logger.warning("No optimization parameters defined, using default parameters")
            return self._run_single_optimization(train_data, self.strategy.parameters)
        
        if self.opt_config.verbose:
            self.logger.info(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Use parallel processing for optimization with fallback to sequential
        if (len(param_combinations) > 1 and 
            self.opt_config.max_workers > 1 and 
            self.opt_config.enable_parallel):
            try:
                results = self._optimize_parallel(train_data, param_combinations)
            except Exception as e:
                if self.opt_config.verbose:
                    self.logger.warning(f"Parallel optimization failed ({e}), falling back to sequential")
                results = self._optimize_sequential(train_data, param_combinations)
        else:
            results = self._optimize_sequential(train_data, param_combinations)
        
        if not results:
            raise ValueError("No successful parameter combinations found")
        
        # Find best parameters
        best_result = self._select_best_parameters(results)
        
        if self.opt_config.verbose:
            self.logger.info("Optimal parameters found:")
            for param, value in best_result.param_combination.items():
                self.logger.info(f"• {param.replace('_', ' ').title()}: {value}")
        
        return best_result
    
    def _optimize_parallel(self, data: pd.DataFrame, param_combinations: List[Dict[str, Any]]) -> List[OptimizationResult]:
        """Run optimization using parallel processing."""
        results = []
        
        # Prepare arguments for parallel processing
        args_list = [
            (data, self.strategy.__class__, self.config, params, i)
            for i, params in enumerate(param_combinations)
        ]
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=self.opt_config.max_workers) as executor:
            # Submit all jobs
            future_to_id = {
                executor.submit(_optimize_single_combination, args): i 
                for i, args in enumerate(args_list)
            }
            
            completed = 0
            no_improvement_count = 0
            
            # Process completed futures
            for future in as_completed(future_to_id):
                completed += 1
                result = future.result()
                
                if result is not None:
                    results.append(result)
                    
                    # Early stopping check
                    if self.opt_config.early_stopping and result.sharpe_ratio > self._best_sharpe:
                        self._best_sharpe = result.sharpe_ratio
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    # Stop early if no improvement
                    if (self.opt_config.early_stopping and 
                        no_improvement_count >= self.opt_config.early_stopping_patience and 
                        len(results) >= 5):  # Minimum 5 results
                        if self.opt_config.verbose:
                            self.logger.info(f"Early stopping after {completed} combinations (no improvement)")
                        break
                
                # Progress reporting
                if self.opt_config.verbose and completed % 5 == 0:
                    self.logger.info(f"Completed {completed}/{len(param_combinations)} combinations")
        
        return results
    
    def _optimize_sequential(self, data: pd.DataFrame, param_combinations: List[Dict[str, Any]]) -> List[OptimizationResult]:
        """Run optimization sequentially with early stopping."""
        results = []
        no_improvement_count = 0
        
        for i, params in enumerate(param_combinations):
            try:
                result = self._run_single_optimization(data, params)
                result.param_combination = params
                result.combination_id = i
                results.append(result)
                
                # Early stopping check
                if self.opt_config.early_stopping and result.sharpe_ratio > self._best_sharpe:
                    self._best_sharpe = result.sharpe_ratio
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Stop early if no improvement
                if (self.opt_config.early_stopping and 
                    no_improvement_count >= self.opt_config.early_stopping_patience and 
                    len(results) >= 5):
                    if self.opt_config.verbose:
                        self.logger.info(f"Early stopping after {i + 1} combinations (no improvement)")
                    break
                
                # Progress reporting
                if self.opt_config.verbose and len(param_combinations) > 10 and (i + 1) % 5 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(param_combinations)} combinations")
                    
            except Exception as e:
                if self.opt_config.verbose:
                    self.logger.warning(f"Error with parameter combination {i}: {e}")
                continue
        
        return results
    
    def _create_strategy_with_params(self, params: Dict[str, Any]) -> BaseStrategy:
        """Create a new strategy instance with updated parameters."""
        # Create a deep copy of the config and update parameters
        new_config = copy.deepcopy(self.config)
        new_config.parameters.update(params)
        # Create new strategy instance with updated config
        return self.strategy.__class__(new_config)
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from optimization grid."""
        if not self.optimization_grid:
            return []
        
        param_names = list(self.optimization_grid.keys())
        param_values = []
        
        for name in param_names:
            param_config = self.optimization_grid[name]
            
            # Handle both range/step and discrete values formats
            if isinstance(param_config, dict) and 'min' in param_config and 'max' in param_config:
                step = param_config.get('step', 1)
                values = np.arange(
                    param_config['min'], 
                    param_config['max'] + step/2,  # Add small epsilon to include max
                    step
                ).tolist()
            elif isinstance(param_config, list):
                values = param_config
            else:
                values = [param_config]
                
            param_values.append(values)
        
        combinations = []
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _run_single_optimization(self, data: pd.DataFrame, params: Dict[str, Any]) -> OptimizationResult:
        """Run optimization for a single parameter combination."""
        # Create a new strategy instance with the given parameters to avoid mutating state
        temp_strategy = self._create_strategy_with_params(params)
        
        try:
            # The strategy expects a dictionary of dataframes, keyed by timeframe.
            # For optimization, we use the first required timeframe as the key.
            required_tfs = temp_strategy.get_required_timeframes()
            main_tf = required_tfs[0] if required_tfs else '1h'
            tf_data = {main_tf: data}
            signals: Signals = temp_strategy.generate_signals(tf_data)
            
            # Ensure signals are properly formatted for vectorbt
            if isinstance(signals, pd.DataFrame):
                signals = signals.astype(bool)
            
            portfolio: vbt.Portfolio = run_backtest(data, signals)
            metrics: Dict[str, Union[float, int]] = calc_metrics(portfolio)
            
            return OptimizationResult(
                portfolio=portfolio,  # Will be replaced with final portfolio for best result
                metrics=metrics,
                sharpe_ratio=float(metrics.get('sharpe', -999)),
                total_return=metrics.get('total_return', -999),
                max_drawdown=metrics.get('max_drawdown', 999),
                profit_factor=metrics.get('profit_factor', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in parameter combination {params}: {str(e)}")
            raise
    
    def _select_best_parameters(self, results: List[OptimizationResult]) -> OptimizationResult:
        """Select best parameters using composite scoring."""
        if not results:
            raise ValueError("No results to select from")
        
        # Multi-objective optimization: Sharpe ratio (70%) + Return/Drawdown ratio (30%)
        for result in results:
            sharpe_score = result.sharpe_ratio if result.sharpe_ratio > -999 else 0
            return_dd_ratio = (result.total_return / max(abs(result.max_drawdown), 1)) if result.max_drawdown != 999 else 0
            result.composite_score = 0.7 * sharpe_score + 0.3 * return_dd_ratio
        
        # Sort by composite score
        sorted_results = sorted(results, key=lambda x: getattr(x, 'composite_score', x.sharpe_ratio), reverse=True)
        best = sorted_results[0]
        
        if self.opt_config.verbose:
            self.logger.info(f"Best Sharpe Ratio: {best.sharpe_ratio:.3f}")
            self.logger.info(f"Best Total Return: {best.total_return:.2f}%")
            self.logger.info(f"Best Max Drawdown: {best.max_drawdown:.2f}%")
            self.logger.info(f"Tested {len(results)} combinations")
        
        return best


class WalkForwardAnalysis:
    """Implements walk-forward analysis for strategy validation."""
    
    def __init__(self, strategy: BaseStrategy, config: StrategyConfig, 
                 opt_config: Optional[OptimizationConfig] = None):
        self.strategy = strategy
        self.config = config
        self.opt_config = opt_config or OptimizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def run_analysis(self, data: pd.DataFrame) -> WalkForwardResult:
        """Run walk-forward analysis."""
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
            
        self.logger.info(f"Running walk-forward analysis with {self.opt_config.num_windows} windows")
        
        if len(data) < self.opt_config.window_size * 2:
            error_msg = f"Insufficient data for walk-forward analysis. Need at least {self.opt_config.window_size * 2} bars"
            self.logger.warning(error_msg)
            raise ValueError(error_msg)
        
        results: List[Dict[str, Any]] = []
        total_length = len(data)
        max_windows = (total_length - self.opt_config.window_size) // self.opt_config.step_size
        actual_windows = min(self.opt_config.num_windows, max_windows)
        
        self.logger.info(f"Analyzing {actual_windows} walk-forward windows...")
        
        for i in range(actual_windows):
            start_idx = i * self.opt_config.step_size
            train_end_idx = start_idx + self.opt_config.window_size
            test_end_idx = min(train_end_idx + self.opt_config.step_size, total_length)
            
            train_data = data.iloc[start_idx:train_end_idx].copy()
            test_data = data.iloc[train_end_idx:test_end_idx].copy()
            
            if len(test_data) < 10:
                continue
                
            try:
                window_result = self._analyze_window(train_data, test_data, i + 1)
                results.append(window_result)
                
            except Exception as e:
                self.logger.error(f"Error in window {i + 1}: {e}")
                continue
        
        if not results:
            raise ValueError("No successful walk-forward windows")
        
        summary: Dict[str, float] = self._aggregate_results(results)
        efficiency: float = self._calculate_efficiency(results)
        
        return WalkForwardResult(
            windows=results,
            summary=summary,
            efficiency=efficiency
        )
    
    def _analyze_window(self, train_data: pd.DataFrame, test_data: pd.DataFrame, window_num: int) -> Dict[str, Any]:
        """Analyze a single walk-forward window."""
        self.logger.info(f"Window {window_num}: Train={len(train_data)} bars, Test={len(test_data)} bars")
        
        optimizer = ParameterOptimizer(self.strategy, self.config, self.opt_config)
        optimization_result = optimizer.optimize(train_data)
        
        # Create a new strategy instance with the optimal parameters to avoid mutating state
        optimal_params = optimization_result.param_combination
        temp_strategy = optimizer._create_strategy_with_params(optimal_params)
        
        try:
            required_tfs = temp_strategy.get_required_timeframes()
            main_tf = required_tfs[0] if required_tfs else '1h'
            test_tf_data = {main_tf: test_data}
            test_signals: Signals = temp_strategy.generate_signals(test_tf_data)
            test_portfolio: vbt.Portfolio = run_backtest(test_data, test_signals)
            test_metrics: Dict[str, Union[float, int]] = calc_metrics(test_portfolio)
            
            train_tf_data = {main_tf: train_data}
            train_signals: Signals = temp_strategy.generate_signals(train_tf_data)
            train_portfolio: vbt.Portfolio = run_backtest(train_data, train_signals)
            train_metrics: Dict[str, Union[float, int]] = calc_metrics(train_portfolio)
            
            return {
                "window": window_num,
                "train_period": f"{train_data.index[0]} to {train_data.index[-1]}",
                "test_period": f"{test_data.index[0]} to {test_data.index[-1]}",
                "optimal_params": optimal_params,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "train_portfolio": train_portfolio,
                "test_portfolio": test_portfolio
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating optimal parameters in window {window_num}: {e}")
            raise
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate walk-forward results."""
        train_returns = [r['train_metrics'].get('return', 0) for r in results]
        test_returns = [r['test_metrics'].get('return', 0) for r in results]
        train_sharpes = [r['train_metrics'].get('sharpe', 0) for r in results]
        test_sharpes = [r['test_metrics'].get('sharpe', 0) for r in results]
        
        return {
            "avg_train_return": float(np.mean(train_returns)),
            "avg_test_return": float(np.mean(test_returns)),
            "avg_train_sharpe": float(np.mean(train_sharpes)),
            "avg_test_sharpe": float(np.mean(test_sharpes)),
            "return_consistency": float(np.std(test_returns)),
            "sharpe_consistency": float(np.std(test_sharpes))
        }
    
    def _calculate_efficiency(self, results: List[Dict[str, Any]]) -> float:
        """Calculate walk-forward efficiency."""
        if not results:
            return 0.0
        
        train_returns = [r['train_metrics'].get('return', 0) for r in results]
        test_returns = [r['test_metrics'].get('return', 0) for r in results]
        
        avg_train_return = np.mean(train_returns)
        avg_test_return = np.mean(test_returns)
        
        if avg_train_return <= 0:
            return 0.0
        
        efficiency = avg_test_return / avg_train_return
        
        self.logger.info(f"Walk-Forward Efficiency: {efficiency:.2%}")
        self.logger.info(f"Average Train Return: {avg_train_return:.2f}%")
        self.logger.info(f"Average Test Return: {avg_test_return:.2f}%")
        
        return float(efficiency)


class MonteCarloAnalysis:
    """Implements Monte Carlo analysis for strategy validation."""
    
    def __init__(self, strategy: BaseStrategy, opt_config: Optional[OptimizationConfig] = None):
        self.strategy = strategy
        self.opt_config = opt_config or OptimizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def run_analysis(self, data: pd.DataFrame) -> MonteCarloResult:
        """Run Monte Carlo analysis with return randomization."""
        # Validate input data
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
            
        self.logger.info(f"Running Monte Carlo analysis with {self.opt_config.num_mc_runs} simulations")
        
        required_tfs = self.strategy.get_required_timeframes()
        main_tf = required_tfs[0] if required_tfs else '1h'
        tf_data = {main_tf: data}
        base_signals: Signals = self.strategy.generate_signals(tf_data)
        base_portfolio: vbt.Portfolio = run_backtest(data, base_signals)
        if base_portfolio is None or base_portfolio.returns().empty:
            error_msg = "Cannot run Monte Carlo analysis on empty portfolio"
            self.logger.warning(error_msg)
            raise ValueError(error_msg)
        base_metrics: Dict[str, Union[float, int]] = calc_metrics(base_portfolio)
        
        # Get returns for bootstrap resampling
        returns: pd.Series = base_portfolio.returns()
        if returns.empty or len(returns) < 10:
            error_msg = "Insufficient returns data for Monte Carlo analysis"
            self.logger.warning(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Base strategy has {len(returns)} return observations")
        
        # Run Monte Carlo simulations
        mc_results: List[Dict[str, Any]] = []
        
        for run in range(self.opt_config.num_mc_runs):
            try:
                # Bootstrap resample returns
                resampled_returns: pd.Series = returns.sample(n=len(returns), replace=True)
                
                # Calculate metrics from resampled returns
                total_return = (1 + resampled_returns).prod() - 1
                sharpe = resampled_returns.mean() / resampled_returns.std() * np.sqrt(252) if resampled_returns.std() > 0 else 0
                max_dd = (resampled_returns.cumsum().cummax() - resampled_returns.cumsum()).max()
                
                mc_results.append({
                    "run": run + 1,
                    "metrics": {
                        "return": float(total_return * 100),
                        "sharpe": float(sharpe),
                        "max_dd": float(max_dd * 100)
                    }
                })
                
                if (run + 1) % 20 == 0:
                    self.logger.info(f"Completed {run + 1}/{self.opt_config.num_mc_runs} simulations")
                    
            except Exception as e:
                self.logger.warning(f"Simulation {run + 1} failed: {e}")
                continue
        
        if not mc_results:
            raise ValueError("All Monte Carlo simulations failed")
        
        analysis: Dict[str, Any] = self._analyze_monte_carlo_results(mc_results, base_metrics)
        
        return MonteCarloResult(
            base_metrics=base_metrics,
            simulations=mc_results,
            analysis=analysis,
            num_successful_runs=len(mc_results)
        )
    
    def _analyze_monte_carlo_results(self, results: List[Dict[str, Any]], base_metrics: Dict[str, Union[float, int]]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""
        returns = [r['metrics'].get('return', 0) for r in results]
        sharpes = [r['metrics'].get('sharpe', 0) for r in results]
        
        base_return = base_metrics.get('return', 0)
        base_sharpe = base_metrics.get('sharpe', 0)
        
        return_percentiles = np.percentile(returns, [5, 25, 50, 75, 95])
        sharpe_percentiles = np.percentile(sharpes, [5, 25, 50, 75, 95])
        
        analysis = {
            "return_stats": {
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
                "percentiles": return_percentiles.tolist()
            },
            "sharpe_stats": {
                "mean": float(np.mean(sharpes)),
                "std": float(np.std(sharpes)),
                "percentiles": sharpe_percentiles.tolist()
            },
            "base_vs_simulations": {
                "return_rank": float(sum(1 for v in returns if v <= base_return) / len(returns)),
                "sharpe_rank": float(sum(1 for v in sharpes if v <= base_sharpe) / len(sharpes))
            }
        }
        
        self.logger.info("Monte Carlo Analysis Summary:")
        self.logger.info(f"Return - Mean: {analysis['return_stats']['mean']:.2f}%, Std: {analysis['return_stats']['std']:.2f}%")
        self.logger.info(f"Sharpe - Mean: {analysis['sharpe_stats']['mean']:.3f}, Std: {analysis['sharpe_stats']['std']:.3f}")
        self.logger.info(f"Base Strategy Return Rank: {analysis['base_vs_simulations']['return_rank']:.1%}")
        self.logger.info(f"Base Strategy Sharpe Rank: {analysis['base_vs_simulations']['sharpe_rank']:.1%}")
        
        return analysis