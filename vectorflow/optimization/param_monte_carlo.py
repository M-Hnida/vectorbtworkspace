#!/usr/bin/env python3
"""Monte Carlo parameter sensitivity analysis."""

from typing import Dict, Any, List, Optional
import time
import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy import stats

from vectorflow.core.constants import (
    MONTE_CARLO_SIMULATIONS,
    MONTE_CARLO_BATCH_SIZE,
)
from vectorflow.optimization.grid_search import (
    expand_parameter_grid,
    extract_portfolio_metrics,
    get_strategy_return,
)

# Monte Carlo parameter sampling configuration
MONTE_CARLO_THRESHOLD_SAMPLES = [0.0, 0.01, 0.02, 0.05, 0.1]
MONTE_CARLO_FINE_GRID_POINTS = 10


def expand_grid_for_monte_carlo(param_grid: Dict) -> Dict[str, List]:
    """Expand parameter grid for Monte Carlo sampling with finer granularity."""
    expanded = expand_parameter_grid(param_grid)

    monte_carlo_grid = {}
    for param_name, param_values in expanded.items():
        if len(param_values) >= 2:
            min_val, max_val = min(param_values), max(param_values)
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                if isinstance(min_val, int) and isinstance(max_val, int):
                    monte_carlo_grid[param_name] = list(range(min_val, max_val + 1))
                else:
                    if "threshold" in param_name.lower() and min_val == 0.0:
                        monte_carlo_grid[param_name] = MONTE_CARLO_THRESHOLD_SAMPLES + [
                            max_val
                        ]
                    else:
                        monte_carlo_grid[param_name] = [
                            min_val
                            + (max_val - min_val)
                            * i
                            / (MONTE_CARLO_FINE_GRID_POINTS - 1)
                            for i in range(MONTE_CARLO_FINE_GRID_POINTS)
                        ]
            else:
                monte_carlo_grid[param_name] = param_values
        else:
            monte_carlo_grid[param_name] = param_values

    return monte_carlo_grid


def run_monte_carlo_analysis(
    data,
    strategy_name: Optional[str] = None,
    params: Optional[Dict] = None,
    actual_return: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run Monte Carlo parameter sensitivity analysis.

    Args:
        data: Either pd.DataFrame (single TF) or Dict[str, pd.DataFrame] (multi TF)
        strategy_name: Name of the strategy
        params: Optional parameters
        actual_return: Optional actual return for comparison

    This is a standard Monte Carlo simulation that:
    1. Randomly samples parameters from the optimization grid
    2. Runs the strategy with each parameter set
    3. Collects equity curves and final returns
    4. Analyzes the distribution of outcomes

    The goal is to understand parameter sensitivity and robustness:
    - If the strategy performs well across many random parameters, it's robust
    - If it only works with specific parameters, it's overfit
    """
    start_time = time.time()

    # Handle both single and multi-timeframe data
    num_bars = len(data[list(data.keys())[0]]) if isinstance(data, dict) else len(data)

    print(
        f"🎲 Monte Carlo parameter sensitivity analysis ({MONTE_CARLO_SIMULATIONS} simulations)"
    )

    if num_bars < 10:
        raise ValueError(f"Insufficient data for Monte Carlo: {num_bars} bars")

    from vectorflow.core.portfolio_builder import (
        get_default_parameters,
        get_optimization_grid,
        create_portfolio,
    )

    if not strategy_name:
        raise ValueError("strategy_name is required for Monte Carlo analysis")

    if params is None:
        default_params = get_default_parameters(strategy_name)
        params = default_params if default_params else {}

    if actual_return is None and params:
        actual_return = get_strategy_return(strategy_name, data, params)

    param_grid = get_optimization_grid(strategy_name)
    if not param_grid or not isinstance(param_grid, dict):
        raise ValueError("No parameter grid found for Monte Carlo analysis")

    expanded_grid = expand_grid_for_monte_carlo(param_grid)

    # Debug: Print parameter ranges
    print("   Parameter ranges for Monte Carlo:")
    for param_name, param_values in expanded_grid.items():
        if len(param_values) >= 2:
            print(
                f"      {param_name}: [{min(param_values)}, {max(param_values)}] ({len(param_values)} values)"
            )
        else:
            print(f"      {param_name}: {param_values}")

    simulations = []
    final_returns = []
    path_matrix_list = []
    success_count = 0
    failure_count = 0

    def sample_random_params() -> Dict[str, Any]:
        """Sample random parameters uniformly from the expanded grid."""
        sampled = {}
        for param_name, param_values in expanded_grid.items():
            if len(param_values) >= 2 and all(
                isinstance(v, (int, float)) for v in param_values
            ):
                min_val, max_val = min(param_values), max(param_values)
                sampled[param_name] = (
                    int(np.random.randint(min_val, max_val + 1))
                    if isinstance(min_val, int) and isinstance(max_val, int)
                    else float(np.random.uniform(min_val, max_val))
                )
            else:
                sampled[param_name] = np.random.choice(param_values)
        return sampled

    # Debug: Sample a few params to verify variation
    if MONTE_CARLO_SIMULATIONS >= 5:
        print("   Sample parameter sets:")
        for i in range(min(5, MONTE_CARLO_SIMULATIONS)):
            sample = sample_random_params()
            print(f"      Sample {i + 1}: {sample}")

    for batch_start in range(0, MONTE_CARLO_SIMULATIONS, MONTE_CARLO_BATCH_SIZE):
        batch_end = min(batch_start + MONTE_CARLO_BATCH_SIZE, MONTE_CARLO_SIMULATIONS)

        for sim_index in range(batch_start, batch_end):
            random_params = sample_random_params()

            try:
                portfolio = create_portfolio(strategy_name, data, random_params)  # type: ignore[arg-type]

                if portfolio is None:
                    failure_count += 1
                    if failure_count <= 3:
                        print(
                            f"   ❌ Sim {sim_index + 1}: create_portfolio returned None"
                        )
                        print(f"      Params: {random_params}")
                    continue

                metrics = extract_portfolio_metrics(portfolio)
                total_return = float(metrics.get("total_return", np.nan))

                if not np.isfinite(total_return):
                    failure_count += 1
                    if failure_count <= 3:
                        print(
                            f"   ❌ Sim {sim_index + 1}: total_return is {total_return}"
                        )
                        print(f"      Params: {random_params}")
                    continue

                equity_series = portfolio.value()
                if (
                    equity_series is None
                    or len(equity_series) == 0
                    or equity_series.isna().all()
                ):
                    failure_count += 1
                    continue

                initial_value = float(equity_series.iloc[0])
                if not np.isfinite(initial_value) or initial_value == 0:
                    failure_count += 1
                    continue

                # Normalize equity curve to percentage returns
                normalized_returns = (
                    equity_series.values / initial_value - 1.0
                ) * 100.0

                if not np.isfinite(normalized_returns).any():
                    failure_count += 1
                    continue

                # Clean up NaN/Inf values
                clean_returns = np.copy(normalized_returns)
                finite_mask = np.isfinite(clean_returns)

                if not finite_mask[0]:
                    first_finite = np.where(finite_mask)[0]
                    if len(first_finite) == 0:
                        failure_count += 1
                        continue
                    clean_returns[: first_finite[0]] = 0.0

                # Forward fill any remaining NaN values
                for i in range(1, len(clean_returns)):
                    if not np.isfinite(clean_returns[i]):
                        clean_returns[i] = clean_returns[i - 1]

                path_matrix_list.append(clean_returns.astype(np.float32))

                sim_record = {
                    "simulation": sim_index + 1,
                    "total_return": total_return,
                    "parameters": random_params.copy(),
                }
                simulations.append(sim_record)
                final_returns.append(total_return)
                success_count += 1

            except Exception:
                failure_count += 1

        # Progress reporting every 5 batches
        if (batch_start // MONTE_CARLO_BATCH_SIZE) % 5 == 0 and batch_start > 0:
            print(
                f"   Progress: {batch_end}/{MONTE_CARLO_SIMULATIONS} ({success_count} successful)"
            )

    if not path_matrix_list:
        raise ValueError("No successful Monte Carlo simulations")

    # Harmonize path lengths
    lengths = [len(p) for p in path_matrix_list]
    target_length = int(np.median(lengths))
    target_length = max(10, min(target_length, num_bars))

    def normalize_path_length(path: np.ndarray) -> np.ndarray:
        """Normalize path to target length by truncating or padding."""
        if len(path) >= target_length:
            return path[:target_length]
        pad_value = path[-1] if len(path) > 0 else 0.0
        padding = np.full((target_length - len(path),), pad_value, dtype=np.float32)
        return np.concatenate([path, padding])

    normalized_paths = [normalize_path_length(p) for p in path_matrix_list]
    path_matrix = np.stack(normalized_paths, axis=1)

    # Calculate statistics
    returns_array = np.array(final_returns, dtype=np.float64)
    finite_returns = returns_array[np.isfinite(returns_array)]

    if len(finite_returns) == 0:
        raise ValueError("No valid returns in Monte Carlo simulations")

    statistics = {
        "mean_return": float(np.mean(finite_returns)),
        "std_return": float(np.std(finite_returns, ddof=0)),
        "min_return": float(np.min(finite_returns)),
        "max_return": float(np.max(finite_returns)),
        "percentile_5": float(np.percentile(finite_returns, 5)),
        "percentile_95": float(np.percentile(finite_returns, 95)),
        "count": len(finite_returns),
        "actual_return": actual_return,
        "success_count": success_count,
        "failure_count": failure_count,
        "duration_sec": time.time() - start_time,
    }

    # Significance test
    if actual_return is not None and np.isfinite(actual_return):
        percentile_rank = float(stats.percentileofscore(finite_returns, actual_return))
        p_value = float(min(percentile_rank, 100.0 - percentile_rank) / 100.0)
        statistics.update(
            {
                "percentile_rank": percentile_rank,
                "p_value": p_value,
                "is_significant": p_value < 0.05,
            }
        )

    print(
        f"   Completed in {statistics['duration_sec']:.2f}s: {success_count} successful, {failure_count} failed"
    )

    return {
        "simulations": simulations,
        "statistics": statistics,
        "path_matrix": path_matrix,
        "summary": f"Completed {success_count} of {MONTE_CARLO_SIMULATIONS} Monte Carlo simulations",
    }
