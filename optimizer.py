#!/usr/bin/env python3
"""
Simple Parameter Optimizer
"""

from itertools import product
from typing import Dict, Any, List, Optional, NamedTuple
import time
import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats

# Simple config
CONFIG = {
    "init_cash": 10000,
    "fees": 0.001,
    "max_combinations": 50,
    "monte_carlo_simulations": 500,
}

# =============================================================================
# CORE CLASSES
# =============================================================================


def _extract_metrics(portfolio: "vbt.Portfolio") -> Dict[str, float]:
    """Return consistent metrics from a vbt.Portfolio."""
    try:
        stats = portfolio.stats()
        return {
            "sharpe_ratio": float(stats.get("Sharpe Ratio", 0.0)),
            "total_return": float(stats.get("Total Return [%]", 0.0)),
            "max_drawdown": float(stats.get("Max Drawdown [%]", 0.0)),
        }
    except Exception:
        raise Exception(
            "Failed to extract metrics from portfolio. Ensure it is a valid vbt.Portfolio object."
        )


def _composite_score(metrics: Dict[str, float]) -> float:
    """Weighted selection score."""
    sr = metrics.get("sharpe_ratio", 0.0)
    ret = metrics.get("total_return", 0.0)
    dd = abs(metrics.get("max_drawdown", 0.0))
    return 0.7 * sr + 0.3 * (ret / max(dd, 1.0))


class OptimizationResult(NamedTuple):
    best_portfolio: "vbt.Portfolio"
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_results: pd.DataFrame
    execution_time: float


class Optimizer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {**CONFIG, **(config or {})}

    def optimize(
        self, strategy_name: str, market_data: pd.DataFrame, param_grid: Dict[str, List]
    ) -> OptimizationResult:
        """Main optimization entry point."""
        data = self._prepare_data(market_data)
        split_idx = int(len(data) * self.config.get("split_ratio", 0.8))
        train_data = data.iloc[:split_idx]
        return self._run_grid(strategy_name, train_data, param_grid)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for optimization."""
        return data.asfreq("1H").ffill().dropna()

    def _run_grid(
        self, strategy_name: str, data: pd.DataFrame, param_grid: Dict[str, List]
    ) -> OptimizationResult:
        """Unified optimization engine using portfolio-direct approach."""
        from strategy_registry import create_portfolio

        start_time = time.time()
        param_names = list(param_grid.keys())
        combos = list(product(*param_grid.values()))
        if not combos:
            raise ValueError("Empty parameter grid")

        # Sequential approach using create_portfolio directly
        rows = []
        best_params = None
        best_score = -float("inf")
        best_portfolio = None

        for combo in combos:
            params = dict(zip(param_names, combo))
            try:
                pf = create_portfolio(strategy_name, data, params)
                metrics = _extract_metrics(pf)
                score = _composite_score(metrics)
                rows.append({**params, **metrics, "composite_score": score})
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_portfolio = pf
            except Exception:
                continue

        if not rows:
            raise ValueError("No successful parameter combinations")

        results_df = pd.DataFrame(rows).sort_values(
            "composite_score", ascending=False, ignore_index=True
        )
        return OptimizationResult(
            best_portfolio=best_portfolio,
            best_params=best_params,
            best_metrics={
                k: results_df.loc[0, k]
                for k in [
                    "sharpe_ratio",
                    "total_return",
                    "max_drawdown",
                    "composite_score",
                ]
                if k in results_df.columns
            },
            all_results=results_df,
            execution_time=time.time() - start_time,
        )


# =============================================================================
# MONTE CARLO ANALYSIS
# =============================================================================


def run_optimization(
    strategy_name: str, data: pd.DataFrame, param_grid: Dict = None
) -> Dict[str, Any]:
    """Run parameter optimization with direct portfolio creation."""
    from strategy_registry import (
        create_portfolio,
        get_optimization_grid,
        get_default_parameters,
    )

    try:
        # Get optimization grid
        if param_grid is None:
            param_grid = get_optimization_grid(strategy_name)

        if not param_grid:
            print("⚠️ No optimization grid found, using default parameters")
            return {"best_params": get_default_parameters(strategy_name)}

        print(f"🎯 Optimizing {strategy_name} with {len(param_grid)} parameters")

        # Use create_portfolio function directly

        # Simple grid search implementation
        default_params = get_default_parameters(strategy_name)
        best_params = default_params.copy() if default_params else {}
        best_score = -999999
        best_portfolio = None

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"📊 Testing {len(combinations)} parameter combinations...")

        max_combinations = 20
        for combo in combinations[:max_combinations]:
            test_params = best_params.copy()
            for name, value in zip(param_names, combo):
                test_params[name] = value

            # Create portfolio directly with test parameters
            try:
                portfolio = create_portfolio(strategy_name, data, test_params)
                stats = portfolio.stats()
                score = float(stats.get("Sharpe Ratio", -999))

                if score > best_score:
                    best_score = score
                    best_params = test_params.copy()
                    best_portfolio = portfolio

            except Exception as e:
                print(f"Failed combo {test_params}: {e}")
                continue

        if best_portfolio is not None:
            print(f"✅ Best parameters found: {best_params} (Sharpe: {best_score:.3f})")
        else:
            print("⚠️ No successful parameter combinations found")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_portfolio": best_portfolio,
            "tested_combinations": min(len(combinations), max_combinations),
        }

    except Exception as e:
        print(f"⚠️ Optimization failed: {e}")
        return {"error": str(e), "best_params": get_default_parameters(strategy_name)}


def _get_strategy_equity_curve(strategy_name: str, data: pd.DataFrame, params: Dict):
    """Helper function to get strategy equity curve."""
    from strategy_registry import create_portfolio

    try:
        portfolio = create_portfolio(strategy_name, data, params)
        return portfolio.value().tolist()
    except Exception:
        return None


def _get_actual_strategy_return(
    strategy_name: str,
    data: pd.DataFrame,
    params: Dict,
    actual_return: Optional[float] = None,
):
    """Helper function to get actual strategy return."""
    if strategy_name is not None:
        try:
            from strategy_registry import create_portfolio

            portfolio = create_portfolio(strategy_name, data, params)
            actual_stats = portfolio.stats()
            return float(actual_stats.get("Total Return [%]", 0))
        except Exception:
            return 0.0
    return actual_return


def _load_and_expand_param_grid(strategy_name: str):
    """Helper function to load and expand parameter grid."""
    from strategy_registry import get_optimization_grid

    try:
        param_grid = get_optimization_grid(strategy_name)

        # Expand parameter ranges for better Monte Carlo sampling
        expanded_grid = {}
        for param_name, param_values in param_grid.items():
            if len(param_values) >= 2:
                min_val, max_val = min(param_values), max(param_values)
                if isinstance(min_val, (int, float)) and isinstance(
                    max_val, (int, float)
                ):
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        expanded_grid[param_name] = list(range(min_val, max_val + 1))
                    else:
                        # Special handling for threshold parameters - keep them small
                        if "threshold" in param_name.lower() and min_val == 0.0:
                            expanded_grid[param_name] = [
                                0.0,
                                0.01,
                                0.02,
                                0.05,
                                0.1,
                                max_val,
                            ]
                        else:
                            expanded_grid[param_name] = [
                                min_val + (max_val - min_val) * i / 9 for i in range(10)
                            ]
                else:
                    expanded_grid[param_name] = param_values
            else:
                expanded_grid[param_name] = param_values
        return expanded_grid

    except:
        # Fallback parameter ranges
        return {
            "vol_momentum_window": list(range(15, 26)),
            "vol_momentum_threshold": [0.0, 0.01, 0.02, 0.05],
            "wma_length": list(range(30, 71, 10)),
        }


def run_monte_carlo_analysis(
    data: pd.DataFrame,
    strategy_name: str = None,
    params: Dict = None,
    actual_return: float = None,
) -> Dict[str, Any]:
    """Run Monte Carlo parameter sensitivity analysis with path matrix, batching, and robust stats."""
    try:
        t0 = time.time()
        n_bars = len(data)
        print(f"🎲 Monte Carlo parameter analysis on {n_bars} bars")
        # Guard: empty or too small data
        if n_bars < 10:
            return {
                "error": "insufficient_data",
                "summary": f"Not enough bars for MC: {n_bars}",
            }

        from strategy_registry import get_default_parameters

        if params is None:
            params = get_default_parameters(strategy_name or "momentum")

        actual_return = _get_actual_strategy_return(
            strategy_name, data, params, actual_return
        )
        param_grid = _load_and_expand_param_grid(strategy_name or "momentum")
        if not isinstance(param_grid, dict) or len(param_grid) == 0:
            return {
                "error": "empty_param_grid",
                "summary": "No parameters to sample for MC",
            }

        # Configurable knobs
        num_simulations = int(CONFIG.get("monte_carlo_simulations", 500))
        if num_simulations <= 0:
            return {
                "error": "zero_simulations",
                "summary": "monte_carlo_simulations must be > 0",
            }

        batch_size = int(CONFIG.get("monte_carlo_batch_size", 128))
        batch_size = max(1, batch_size)
        seed = CONFIG.get("random_seed", None)
        if seed is not None and np.isfinite(seed):
            np.random.seed(int(seed))

        # Prepare collectors
        simulations: List[Dict[str, Any]] = []
        final_returns: List[float] = []
        path_matrix_list: List[np.ndarray] = []
        success = 0
        failures = 0

        # Strategy hooks
        from strategy_registry import create_portfolio

        param_names = list(param_grid.keys())

        print(
            f"   Running {num_simulations} Monte Carlo simulations in batches of {batch_size}..."
        )

        def sample_params() -> Dict[str, Any]:
            rnd: Dict[str, Any] = {}
            for pname, pvalues in param_grid.items():
                try:
                    if len(pvalues) >= 2 and all(
                        isinstance(v, (int, float)) for v in pvalues
                    ):
                        lo, hi = min(pvalues), max(pvalues)
                        if isinstance(lo, int) and isinstance(hi, int):
                            rnd[pname] = int(np.random.randint(lo, hi + 1))
                        else:
                            rnd[pname] = float(np.random.uniform(lo, hi))
                    else:
                        rnd[pname] = np.random.choice(pvalues)
                except Exception:
                    rnd[pname] = (
                        pvalues[0] if isinstance(pvalues, list) and pvalues else None
                    )
            return rnd

        # Iterate batches
        for start in range(0, num_simulations, batch_size):
            end = min(start + batch_size, num_simulations)
            batch_params = [sample_params() for _ in range(start, end)]

            for sim_idx, rnd_params in enumerate(batch_params, start=start):
                try:
                    portfolio = create_portfolio(
                        strategy_name or "momentum", data, rnd_params
                    )
                    metrics = _extract_metrics(portfolio)
                    total_return_pct = float(metrics.get("total_return", np.nan))

                    # Skip invalid returns
                    if not np.isfinite(total_return_pct):
                        failures += 1
                        continue

                    eq_series = portfolio.value()
                    if (
                        eq_series is None
                        or len(eq_series) == 0
                        or eq_series.isna().all()
                    ):
                        failures += 1
                        continue

                    # Normalize to % change from start (vectorized)
                    s0 = float(eq_series.iloc[0])
                    if not np.isfinite(s0) or s0 == 0:
                        failures += 1
                        continue

                    norm = (eq_series.values / s0 - 1.0) * 100.0

                    # Handle NaN/Inf values by forward filling instead of removing them
                    # This preserves the time series length
                    if not np.isfinite(norm).any():
                        failures += 1
                        continue

                    # Forward fill NaN/Inf values to maintain time series integrity
                    norm_clean = np.copy(norm)
                    finite_mask = np.isfinite(norm_clean)

                    # If first value is not finite, find first finite value
                    if not finite_mask[0]:
                        first_finite_idx = np.where(finite_mask)[0]
                        if len(first_finite_idx) == 0:
                            failures += 1
                            continue
                        norm_clean[: first_finite_idx[0]] = 0.0  # Start at 0% return

                    # Forward fill any remaining NaN/Inf values
                    for j in range(1, len(norm_clean)):
                        if not np.isfinite(norm_clean[j]):
                            norm_clean[j] = norm_clean[j - 1]

                    path_matrix_list.append(
                        norm_clean.astype(np.float32)
                    )  # store as float32 to save memory

                    # Collect sim record
                    sim_record = {
                        "simulation": sim_idx + 1,
                        "total_return": total_return_pct,
                        "parameters": rnd_params.copy(),
                        "param1": rnd_params.get(param_names[0], 0)
                        if param_names
                        else 0,
                        "param2": rnd_params.get(param_names[1], 0)
                        if len(param_names) > 1
                        else 0,
                        "equity_curve": None,  # do not store full curve here to avoid duplication; paths in path_matrix
                    }
                    simulations.append(sim_record)
                    final_returns.append(total_return_pct)
                    success += 1
                except Exception:
                    failures += 1
                    # do not synthesize fake returns; just count failure

            if (start // max(batch_size, 1)) % 5 == 0:
                print(
                    f"   ... progress: {min(end, num_simulations)}/{num_simulations} sims, success={success}, failures={failures}"
                )

        # Harmonize path lengths by padding or truncating to min/max length
        lengths = [len(p) for p in path_matrix_list]
        if len(lengths) == 0:
            print("⚠️ No successful simulations; returning empty Monte Carlo result")
            return {
                "error": "no_successful_simulations",
                "simulations": [],
                "statistics": {},
                "summary": "No MC sims succeeded",
            }

        # Use a more robust target length - use the most common length or median
        target_len = int(np.median(lengths))
        target_len = max(10, min(target_len, n_bars))

        print(
            f"   Debug: Path lengths - min: {min(lengths)}, max: {max(lengths)}, median: {np.median(lengths)}, target: {target_len}"
        )

        def fit_length(arr: np.ndarray) -> np.ndarray:
            if len(arr) == 0:
                print(
                    "   Warning: Empty array in path_matrix_list, this should not happen"
                )
                return np.zeros((target_len,), dtype=np.float32)

            if len(arr) >= target_len:
                return arr[:target_len]

            # Pad with last value to maintain shape
            pad_val = arr[-1] if len(arr) > 0 else 0.0
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
            stats_out = {
                "mean_return": np.nan,
                "std_return": np.nan,
                "min_return": np.nan,
                "max_return": np.nan,
                "percentile_5": np.nan,
                "percentile_95": np.nan,
                "count": 0,
            }
            percentile = p_value = is_significant = None
        else:
            stats_out = {
                "mean_return": float(np.mean(returns_arr)),
                "std_return": float(np.std(returns_arr, ddof=0)),
                "min_return": float(np.min(returns_arr)),
                "max_return": float(np.max(returns_arr)),
                "percentile_5": float(np.percentile(returns_arr, 5)),
                "percentile_95": float(np.percentile(returns_arr, 95)),
                "count": valid_count,
            }
            if actual_return is not None and np.isfinite(actual_return):
                percentile = float(stats.percentileofscore(returns_arr, actual_return))
                p_value = float(min(percentile, 100.0 - percentile) / 100.0)
                is_significant = bool(p_value < 0.05)
            else:
                percentile = p_value = is_significant = None

        # Logging summary
        dt = time.time() - t0
        print(
            f"   MC done in {dt:.2f}s: success={success}, failures={failures}, valid_returns={valid_count}, path_matrix={path_matrix.shape}"
        )

        strategy_equity_curve = (
            _get_strategy_equity_curve(strategy_name, data, params)
            if strategy_name
            else None
        )

        statistics = {
            **stats_out,
            "actual_return": actual_return,
            "percentile_rank": percentile,
            "p_value": p_value,
            "is_significant": is_significant,
            "path_matrix_shape": tuple(path_matrix.shape),
            "success_count": success,
            "failure_count": failures,
            "duration_sec": dt,
        }

        return {
            "simulations": simulations,  # lightweight per-sim metadata
            "statistics": statistics,
            "path_matrix": path_matrix,  # numpy array (T, N), normalized %
            "summary": f"Completed {success} / {num_simulations} Monte Carlo simulations",
            "significance_test": {
                "actual_return": actual_return,
                "percentile_rank": percentile,
                "p_value": p_value,
                "is_significant": is_significant,
                "interpretation": (
                    f"Strategy performance is {'significant' if is_significant else 'not significant'} vs random"
                    if is_significant is not None
                    else "No actual return provided"
                ),
            },
        }
    except Exception as e:
        print(f"⚠️ Monte Carlo analysis failed: {e}")
        return {"error": str(e)}
