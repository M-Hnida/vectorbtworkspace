#!/usr/bin/env python3
"""
Simplified Functional Parameter Optimizer
Works directly with signal functions with maximum flexibility.
"""

import time
import warnings
from typing import Dict, Any, List, Optional, NamedTuple, Callable
import numpy as np
import pandas as pd
import vectorbt as vbt
from itertools import product
from scipy import stats

# Import strategy functions
from strategies import get_strategy_signal_function
from backtest import run_backtest
from data_manager import load_strategy_config

warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'split_ratio': 0.7,
    'init_cash': 10000,
    'fees': 0.001,
    'verbose': False,
    'max_test_combinations': 20,  # Maximum combinations to test for speed
    'monte_carlo_simulations': 1000,  # Number of Monte Carlo simulations
    'monte_carlo_batch_size': 128,    # Batch size for MC
    'random_seed': None               # Optional RNG seed for reproducibility
}

DEFAULT_PARAM_GRIDS = {}

# =============================================================================
# CORE CLASSES
# =============================================================================

def _extract_metrics(portfolio: vbt.Portfolio) -> Dict[str, float]:
    """Return consistent metrics from a vbt.Portfolio."""
    try:
        stats = portfolio.stats()
        return {
            'sharpe_ratio': float(stats.get('Sharpe Ratio', 0.0)),
            'total_return': float(stats.get('Total Return [%]', 0.0)),
            'max_drawdown': float(stats.get('Max Drawdown [%]', 0.0)),
        }
    except Exception:
       raise Exception("Failed to extract metrics from portfolio. Ensure it is a valid vbt.Portfolio object.")

def _composite_score(metrics: Dict[str, float]) -> float:
    """Weighted selection score."""
    sr = metrics.get('sharpe_ratio', 0.0)
    ret = metrics.get('total_return', 0.0)
    dd = abs(metrics.get('max_drawdown', 0.0))
    return 0.7 * sr + 0.3 * (ret / max(dd, 1.0))

class OptimizationResult(NamedTuple):
    best_portfolio: vbt.Portfolio
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_results: pd.DataFrame
    execution_time: float

class Optimizer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
    
    def optimize(self, market_data: pd.DataFrame, signal_function: Callable,
                 param_grid: Dict[str, List]) -> OptimizationResult:
        """Main optimization entry point."""
        data = self._prepare_data(market_data)
        split_idx = int(len(data) * self.config['split_ratio'])
        train_data = data.iloc[:split_idx]
        return self._run_grid(train_data, signal_function, param_grid)
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for optimization."""
        return data.asfreq('1H').ffill().dropna()
    
    def _run_grid(self, data: pd.DataFrame, signal_function: Callable,
                  param_grid: Dict[str, List]) -> OptimizationResult:
        """Unified optimization engine with best-effort vectorization."""
        start_time = time.time()
        param_names = list(param_grid.keys())
        combos = list(product(*param_grid.values()))
        if not combos:
            raise ValueError("Empty parameter grid")
        
        # Attempt vectorized stacking first
        try:
            entries_list, exits_list = [], []
            for combo in combos:
                params = dict(zip(param_names, combo))
                sig = signal_function(data, **params)
                entries_list.append(sig.entries.reindex(data.index, fill_value=False))
                exits_list.append(sig.exits.reindex(data.index, fill_value=False))
            
            entries_df = pd.concat(entries_list, axis=1, keys=combos)
            exits_df = pd.concat(exits_list, axis=1, keys=combos)
            portfolios = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries_df,
                exits=exits_df,
                init_cash=self.config['init_cash'],
                fees=self.config['fees']
            )
            
            # Extract metrics column-wise (one per combo), supporting both property and method styles
            def _get_metric(obj, attr: str):
                val = getattr(obj, attr, None)
                if val is None:
                    return None
                try:
                    return val() if callable(val) else val
                except Exception:
                    return None

            sr_raw = _get_metric(portfolios, 'sharpe_ratio')
            tr_raw = _get_metric(portfolios, 'total_return')

            # Drawdowns accessor may be an attribute with its own metrics
            dd_acc = getattr(portfolios, 'drawdowns', None)
            dd_raw = None
            if dd_acc is not None:
                dd_raw = _get_metric(dd_acc, 'max_drawdown')

            # Coerce to Series aligned with vectorized columns; default to zeros if missing
            def _as_series(x, scale: float = 1.0):
                if isinstance(x, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
                    try:
                        s = pd.Series(x)
                        s.index = entries_df.columns
                        return s.astype(float) * scale
                    except Exception:
                        pass
                try:
                    return pd.Series([float(x) * scale] * len(entries_df.columns), index=entries_df.columns)
                except Exception:
                    return pd.Series(0.0, index=entries_df.columns)

            sr = _as_series(sr_raw, 1.0)
            tr = _as_series(tr_raw, 100.0)  # convert to percentage if raw is fractional
            dd = _as_series(dd_raw, 100.0) if dd_raw is not None else pd.Series(0.0, index=entries_df.columns)
            
            # Normalize to flat index of combos
            def _to_series(x):
                if isinstance(x, pd.Series):
                    try:
                        x.index = pd.Index(combos)
                        return x
                    except Exception:
                        pass
                return pd.Series([float(x)] * len(combos), index=pd.Index(combos))
            
            results_df = pd.DataFrame({
                'sharpe_ratio': _to_series(sr),
                'total_return': _to_series(tr),
                'max_drawdown': _to_series(dd)
            })
            results_df['composite_score'] = results_df.apply(_composite_score, axis=1)
            best_key = results_df['composite_score'].idxmax()
            best_params = dict(zip(param_names, best_key))
            try:
                best_portfolio = portfolios[best_key]
            except Exception:
                col = list(results_df.index).index(best_key)
                best_portfolio = portfolios.iloc[:, col]
            
            return OptimizationResult(
                best_portfolio=best_portfolio,
                best_params=best_params,
                best_metrics=results_df.loc[best_key].to_dict(),
                all_results=results_df.reset_index().rename(columns={'index': 'params'}),
                execution_time=time.time() - start_time
            )
        except Exception as e:
            if self.config.get('verbose'):
                print(f"Vectorized path failed, sequential fallback: {e}")
        
        # Sequential fallback
        rows = []
        best_params = None
        best_score = -float('inf')
        best_portfolio = None
        
        for combo in combos:
            params = dict(zip(param_names, combo))
            try:
                sig = signal_function(data, **params)
                pf = vbt.Portfolio.from_signals(
                    close=data['close'],
                    entries=sig.entries.reindex(data.index, fill_value=False),
                    exits=sig.exits.reindex(data.index, fill_value=False),
                    init_cash=self.config['init_cash'],
                    fees=self.config['fees']
                )
                metrics = _extract_metrics(pf)
                score = _composite_score(metrics)
                rows.append({**params, **metrics, 'composite_score': score})
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_portfolio = pf
            except Exception:
                continue
        
        if not rows:
            raise ValueError("No successful parameter combinations")
        
        results_df = pd.DataFrame(rows).sort_values('composite_score', ascending=False, ignore_index=True)
        return OptimizationResult(
            best_portfolio=best_portfolio,
            best_params=best_params,
            best_metrics={k: results_df.loc[0, k] for k in ['sharpe_ratio', 'total_return', 'max_drawdown', 'composite_score'] if k in results_df.columns},
            all_results=results_df,
            execution_time=time.time() - start_time
        )

# =============================================================================
# MONTE CARLO ANALYSIS
# =============================================================================

def run_optimization(strategy, strategy_config, data: pd.DataFrame) -> Dict[str, Any]:
    """Run parameter optimization."""
    try:
        
        if not hasattr(strategy_config, 'optimization_grid') or not strategy_config.optimization_grid:
            print("⚠️ No optimization grid found, using default parameters")
            return {'best_params': strategy.parameters}

        # Get the signal function
        signal_func = get_strategy_signal_function(strategy.name)
        param_grid = strategy_config.optimization_grid
        print(f"🎯 Optimizing {strategy.name} with {len(param_grid)} parameters")

        # Simple grid search implementation
        best_params = strategy.parameters.copy()
        best_score = -999999

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"📊 Testing {len(combinations)} parameter combinations...")
        
        max_combinations = 10  # Default for standalone function
        for combo in combinations[:max_combinations]:  # Limit for testing
            test_params = strategy.parameters.copy()
            for name, value in zip(param_names, combo):
                test_params[name] = value

            # Generate signals with test parameters
            signals = signal_func({strategy.get_required_timeframes()[0]: data}, test_params)
            
            # Quick backtest
            portfolio = run_backtest(data, signals)
            
            # Calculate score (using Sharpe ratio)
            try:
                stats = portfolio.stats()
                score = float(stats.get('Sharpe Ratio', -999))
                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
            except Exception:
                continue

        # Update strategy with best parameters
        strategy.parameters.update(best_params)
        print(f"✅ Best parameters found: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'tested_combinations': len(combinations)
        }
        
    except Exception as e:
        print(f"⚠️ Optimization failed: {e}")
        return {'error': str(e), 'best_params': strategy.parameters}


def _get_strategy_equity_curve(strategy, data: pd.DataFrame):
    """Helper function to get strategy equity curve, reducing nesting."""
    try:
        signal_func = get_strategy_signal_function(strategy.name)
        signals = signal_func({strategy.get_required_timeframes()[0]: data}, strategy.parameters)
        portfolio = run_backtest(data, signals)
        return portfolio.value().tolist()
    except Exception:
        return None
def _get_actual_strategy_return(strategy, data: pd.DataFrame, actual_return: Optional[float] = None):
    """Helper function to get actual strategy return, reducing nesting."""
    if strategy is not None:
        try:
            signal_func = get_strategy_signal_function(strategy.name)
            signals = signal_func({strategy.get_required_timeframes()[0]: data}, strategy.parameters)
            portfolio = run_backtest(data, signals)
            actual_stats = portfolio.stats()
            return float(actual_stats.get('Total Return [%]', 0))
        except Exception:
            return 0.0
    return actual_return


def _load_and_expand_param_grid(strategy):
    """Helper function to load and expand parameter grid, reducing nesting."""
    try:
        config = load_strategy_config(strategy.name if strategy else 'rsi')
        param_grid = config.get('optimization_grid', {})
        
        # Expand parameter ranges for better Monte Carlo sampling
        expanded_grid = {}
        for param_name, param_values in param_grid.items():
            if len(param_values) >= 2:
                min_val, max_val = min(param_values), max(param_values)
                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        expanded_grid[param_name] = list(range(min_val, max_val + 1))
                    else:
                        expanded_grid[param_name] = [min_val + (max_val - min_val) * i / 9 for i in range(10)]
                else:
                    expanded_grid[param_name] = param_values
            else:
                expanded_grid[param_name] = param_values
        return expanded_grid
        
    except:
        # Fallback parameter ranges for RSI
        return {
            'rsi_period': list(range(10, 29)),
            'oversold_level': list(range(20, 36)),
            'overbought_level': list(range(65, 81))
        }


def run_monte_carlo_analysis(data: pd.DataFrame, strategy=None, actual_return: float = None) -> Dict[str, Any]:
    """Run Monte Carlo parameter sensitivity analysis with path matrix, batching, and robust stats."""
    try:
        t0 = time.time()
        n_bars = len(data)
        print(f"🎲 Monte Carlo parameter analysis on {n_bars} bars")
        # Guard: empty or too small data
        if n_bars < 10:
            return {'error': 'insufficient_data', 'summary': f'Not enough bars for MC: {n_bars}'}

        actual_return = _get_actual_strategy_return(strategy, data, actual_return)
        param_grid = _load_and_expand_param_grid(strategy)
        if not isinstance(param_grid, dict) or len(param_grid) == 0:
            return {'error': 'empty_param_grid', 'summary': 'No parameters to sample for MC'}

        # Configurable knobs
        num_simulations = int(DEFAULT_CONFIG.get('monte_carlo_simulations', 1000))
        if num_simulations <= 0:
            return {'error': 'zero_simulations', 'summary': 'monte_carlo_simulations must be > 0'}

        batch_size = int(DEFAULT_CONFIG.get('monte_carlo_batch_size', 128))
        batch_size = max(1, batch_size)
        seed = DEFAULT_CONFIG.get('random_seed', None)
        if seed is not None and np.isfinite(seed):
            np.random.seed(int(seed))

        # Prepare collectors
        simulations: List[Dict[str, Any]] = []
        final_returns: List[float] = []
        path_matrix_list: List[np.ndarray] = []
        success = 0
        failures = 0

        # Strategy hooks
        signal_func = get_strategy_signal_function(strategy.name if strategy else 'rsi')
        tf_key = strategy.get_required_timeframes()[0] if strategy else '1H'
        param_names = list(param_grid.keys())

        print(f"   Running {num_simulations} Monte Carlo simulations in batches of {batch_size}...")

        def sample_params() -> Dict[str, Any]:
            rnd: Dict[str, Any] = {}
            for pname, pvalues in param_grid.items():
                try:
                    if len(pvalues) >= 2 and all(isinstance(v, (int, float)) for v in pvalues):
                        lo, hi = min(pvalues), max(pvalues)
                        if isinstance(lo, int) and isinstance(hi, int):
                            rnd[pname] = int(np.random.randint(lo, hi + 1))
                        else:
                            rnd[pname] = float(np.random.uniform(lo, hi))
                    else:
                        rnd[pname] = np.random.choice(pvalues)
                except Exception:
                    rnd[pname] = pvalues[0] if isinstance(pvalues, list) and pvalues else None
            return rnd

        # Iterate batches
        for start in range(0, num_simulations, batch_size):
            end = min(start + batch_size, num_simulations)
            batch_params = [sample_params() for _ in range(start, end)]

            for i, rnd_params in enumerate(batch_params, start=start):
                try:
                    signals = signal_func({tf_key: data}, rnd_params)
                    portfolio = run_backtest(data, signals)
                    metrics = _extract_metrics(portfolio)
                    total_return_pct = float(metrics.get('total_return', np.nan))

                    # Skip invalid returns
                    if not np.isfinite(total_return_pct):
                        failures += 1
                        continue

                    eq_series = portfolio.value()
                    if eq_series is None or len(eq_series) == 0 or eq_series.isna().all():
                        failures += 1
                        continue

                    # Normalize to % change from start (vectorized)
                    s0 = float(eq_series.iloc[0])
                    if not np.isfinite(s0) or s0 == 0:
                        failures += 1
                        continue

                    norm = (eq_series.values / s0 - 1.0) * 100.0
                    # Guard against Inf/NaN in path
                    mask = np.isfinite(norm)
                    if not mask.any():
                        failures += 1
                        continue
                    norm = norm[mask]
                    path_matrix_list.append(norm.astype(np.float32))  # store as float32 to save memory

                    # Collect sim record
                    sim_record = {
                        'simulation': i + 1,
                        'total_return': total_return_pct,
                        'parameters': rnd_params.copy(),
                        'param1': rnd_params.get(param_names[0], 0) if param_names else 0,
                        'param2': rnd_params.get(param_names[1], 0) if len(param_names) > 1 else 0,
                        'equity_curve': None  # do not store full curve here to avoid duplication; paths in path_matrix
                    }
                    simulations.append(sim_record)
                    final_returns.append(total_return_pct)
                    success += 1
                except Exception:
                    failures += 1
                    # do not synthesize fake returns; just count failure

            if (start // max(batch_size,1)) % 5 == 0:
                print(f"   ... progress: {min(end, num_simulations)}/{num_simulations} sims, success={success}, failures={failures}")

        # Harmonize path lengths by padding or truncating to min/max length
        lengths = [len(p) for p in path_matrix_list]
        if len(lengths) == 0:
            print("⚠️ No successful simulations; returning empty Monte Carlo result")
            return {'error': 'no_successful_simulations', 'simulations': [], 'statistics': {}, 'summary': 'No MC sims succeeded'}

        target_len = int(np.percentile(lengths, 10))
        target_len = max(10, min(target_len, n_bars))
        def fit_length(arr: np.ndarray) -> np.ndarray:
            if len(arr) >= target_len:
                return arr[:target_len]
            # pad with last value to maintain shape
            if len(arr) == 0:
                return np.zeros((target_len,), dtype=np.float32)
            pad_val = arr[-1]
            pad = np.full((target_len - len(arr),), pad_val, dtype=np.float32)
            return np.concatenate([arr, pad], axis=0)

        fitted = [fit_length(p) for p in path_matrix_list]
        path_matrix = np.stack(fitted, axis=1)  # shape (T, N)

        # Robust distribution stats from final_returns
        returns_arr = np.asarray(final_returns, dtype=np.float64)
        finite_mask = np.isfinite(returns_arr)
        returns_arr = returns_arr[finite_mask]
        valid_count = int(returns_arr.size)

        if valid_count == 0:
            stats_out = {'mean_return': np.nan, 'std_return': np.nan, 'min_return': np.nan, 'max_return': np.nan,
                         'percentile_5': np.nan, 'percentile_95': np.nan, 'count': 0}
            percentile = p_value = is_significant = None
        else:
            stats_out = {
                'mean_return': float(np.mean(returns_arr)),
                'std_return': float(np.std(returns_arr, ddof=0)),
                'min_return': float(np.min(returns_arr)),
                'max_return': float(np.max(returns_arr)),
                'percentile_5': float(np.percentile(returns_arr, 5)),
                'percentile_95': float(np.percentile(returns_arr, 95)),
                'count': valid_count
            }
            if actual_return is not None and np.isfinite(actual_return):
                percentile = float(stats.percentileofscore(returns_arr, actual_return))
                p_value = float(min(percentile, 100.0 - percentile) / 100.0)
                is_significant = bool(p_value < 0.05)
            else:
                percentile = p_value = is_significant = None

        # Logging summary
        dt = time.time() - t0
        print(f"   MC done in {dt:.2f}s: success={success}, failures={failures}, valid_returns={valid_count}, path_matrix={path_matrix.shape}")

        strategy_equity_curve = _get_strategy_equity_curve(strategy, data) if strategy else None

        statistics = {
            **stats_out,
            'actual_return': actual_return,
            'percentile_rank': percentile,
            'p_value': p_value,
            'is_significant': is_significant,
            'path_matrix_shape': tuple(path_matrix.shape),
            'success_count': success,
            'failure_count': failures,
            'duration_sec': dt,
        }

        return {
            'simulations': simulations,  # lightweight per-sim metadata
            'statistics': statistics,
            'path_matrix': path_matrix,  # numpy array (T, N), normalized %
            'summary': f"Completed {success} / {num_simulations} Monte Carlo simulations",
            'significance_test': {
                'actual_return': actual_return,
                'percentile_rank': percentile,
                'p_value': p_value,
                'is_significant': is_significant,
                'interpretation': (
                    f"Strategy performance is {'significant' if is_significant else 'not significant'} vs random"
                    if is_significant is not None else "No actual return provided"
                )
            }
        }
    except Exception as e:
        print(f"⚠️ Monte Carlo analysis failed: {e}")
        return {'error': str(e)}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def _adapt_tf_to_df(tf_signal_fn: Callable, primary_tf: str = '1H') -> Callable:
    """Adapter to call a multi-timeframe signal function with a single-DF signature."""
    def df_fn(df: pd.DataFrame, **params):
        tf_data = {primary_tf: df}
        # Registry functions expect (tf_data, params_dict)
        return tf_signal_fn(tf_data, params)
    return df_fn

def optimize_strategy(market_data: pd.DataFrame, strategy_name: str,
                     param_grid: Optional[Dict] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Simple grid search optimization using Optimizer when a param_grid is provided."""
    try:
        if not param_grid:
            print("⚠️ No optimization grid found, using default parameters")
            return {'best_params': {}, 'best_score': 0}

        print(f"🎯 Optimizing {strategy_name} with {len(param_grid)} parameters")

        # Prefer using the Optimizer with an adapter for consistency/performance
        try:
            tf_signal_fn = get_strategy_signal_function(strategy_name)
            df_signal_fn = _adapt_tf_to_df(tf_signal_fn, primary_tf='1H')

            opt = Optimizer(config=config)
            opt_res = opt.optimize(market_data, df_signal_fn, param_grid)

            print(f"✅ Optimization complete. Best composite: {opt_res.best_metrics.get('composite_score', float('nan')):.3f}")
            return {
                'best_params': opt_res.best_params,
                'best_score': float(opt_res.best_metrics.get('sharpe_ratio', 0.0)),
                'tested_combinations': int(len(list(product(*param_grid.values()))))
            }
        except Exception as e:
            if config and config.get('verbose'):
                print(f"⚠️ Optimizer path failed, falling back to simple loop: {e}")

        # Fallback: original simple loop (TF-dict call)
        signal_function = get_strategy_signal_function(strategy_name)
        best_params = {}
        best_score = -999999

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        max_combinations = config.get('max_test_combinations', 20) if config else 20
        print(f"📊 Testing {min(len(combinations), max_combinations)} parameter combinations...")

        for _, combo in enumerate(combinations[:max_combinations]):  # Limit for speed
            test_params = dict(zip(param_names, combo))
            try:
                tf_data = {'1H': market_data}
                signals = signal_function(tf_data, test_params)
                portfolio = run_backtest(market_data, signals)

                metrics = _extract_metrics(portfolio)
                score = _composite_score(metrics)

                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                    if config and config.get('verbose'):
                        print(f"  ✅ New best: score={score:.3f} metrics={metrics} params={test_params}")
            except Exception:
                continue

        if best_params:
            print(f"✅ Optimization complete. Best Sharpe: {best_score:.3f}")
            return {
                'best_params': best_params,
                'best_score': best_score,
                'tested_combinations': min(len(combinations), max_combinations)
            }
        else:
            print("⚠️ No successful parameter combinations found")
            return {'best_params': {}, 'best_score': 0}

    except Exception as e:
        print(f"⚠️ Optimization failed: {e}")
        return {'error': str(e), 'best_params': {}}

def compare_strategies(data: pd.DataFrame, strategies: List[str]) -> pd.DataFrame:
    """Compare multiple strategies."""
    results = []
    for strategy in strategies:
        try:
            result = optimize_strategy(data, strategy)
            # Handle different return structures
            if 'error' in result:
                results.append({'strategy': strategy, 'error': str(result['error'])})
            else:
                results.append({
                    'strategy': strategy,
                    'sharpe': result.get('best_score', 0),
                    'return': result.get('best_params', {}),
                    'drawdown': 0,  # Not available in optimize_strategy return
                    'time': 0  # Not available in optimize_strategy return
                })
        except Exception as e:
            results.append({'strategy': strategy, 'error': str(e)})
    
    return pd.DataFrame(results).sort_values('sharpe', ascending=False, na_position='last')

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='1H')
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    
    data = pd.DataFrame({
        'open': close * (1 + np.random.randn(len(dates)) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(len(dates))) * 0.002),
        'low': close * (1 - np.abs(np.random.randn(len(dates))) * 0.002),
        'close': close
    }, index=dates)
    
    # Load param grid from config if available
    try:
        cfg = load_strategy_config('vectorbt')
        param_grid = cfg.get('optimization_grid', {})
    except Exception:
        param_grid = {}

    # Optimize single strategy
    result = optimize_strategy(data, 'vectorbt', param_grid=param_grid, config={'max_test_combinations': 20, 'verbose': True})
    print(f"Best Sharpe: {result.get('best_score', 0):.3f}")

    # Monte Carlo analysis
    mc_results = run_monte_carlo_analysis(data, None, result.get('best_score', 0))
    print(f"MC Mean Return: {mc_results.get('statistics', {}).get('mean_return', 0):.3f}")

    # Compare strategies
    comparison = compare_strategies(data, ['vectorbt', 'momentum'])
    print(comparison)