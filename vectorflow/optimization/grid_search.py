#!/usr/bin/env python3
"""Parameter optimization using grid search."""

from itertools import product
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import vectorbt as vbt

from vectorflow.core.constants import MAX_PARAM_COMBINATIONS


def expand_parameter_grid(param_grid: Dict[str, Any]) -> Dict[str, List]:
    """Expand parameter grid from [start, end, step] format to list of values."""
    expanded = {}

    for param_name, param_config in param_grid.items():
        if not isinstance(param_config, list) or len(param_config) != 3:
            raise ValueError(
                f"Parameter '{param_name}' must be [start, end, step] format. Got: {param_config}"
            )

        start, end, step = param_config

        if not all(isinstance(x, (int, float)) for x in [start, end, step]):
            raise ValueError(
                f"Parameter '{param_name}' values must be numeric. Got: {param_config}"
            )

        if step <= 0:
            raise ValueError(f"Parameter '{param_name}' step must be positive")
        if start > end:
            raise ValueError(
                f"Parameter '{param_name}' start ({start}) must be <= end ({end})"
            )

        # Generate range
        if isinstance(start, int) and isinstance(end, int) and isinstance(step, int):
            expanded[param_name] = list(range(start, end + 1, step))
        else:
            values = []
            current = start
            while current <= end:
                values.append(current)
                current += step
            expanded[param_name] = values

    return expanded


def extract_portfolio_metrics(portfolio: "vbt.Portfolio") -> Dict[str, float]:
    """Extract key metrics from portfolio with safe null handling."""
    try:
        stats = portfolio.stats()
        if stats is None:
            raise Exception("Portfolio stats are None")

        def safe_get(key: str, default: float = 0.0) -> float:
            try:
                if key not in stats.index:
                    return default
                value = stats[key]
                if isinstance(value, (int, float)):
                    return float(value) if np.isfinite(value) else default
                if hasattr(value, "item"):
                    numeric_val = float(value.item())
                    return numeric_val if np.isfinite(numeric_val) else default
                return default
            except (KeyError, TypeError, ValueError, AttributeError):
                return default

        return {
            "sharpe_ratio": safe_get("Sharpe Ratio"),
            "total_return": safe_get("Total Return [%]"),
            "max_drawdown": safe_get("Max Drawdown [%]"),
        }
    except Exception as e:
        raise Exception(f"Failed to extract portfolio metrics: {e}")


def run_optimization(
    strategy_name: str, data, param_grid: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Run parameter optimization using grid search.

    Args:
        strategy_name: Name of the strategy
        data: Either pd.DataFrame (single TF) or Dict[str, pd.DataFrame] (multi TF)
        param_grid: Optional parameter grid override
    """
    from vectorflow.core.portfolio_builder import (
        create_portfolio,
        get_optimization_grid,
        get_default_parameters,
    )

    if param_grid is None:
        param_grid = get_optimization_grid(strategy_name)

    if not param_grid:
        print("⚠️ No optimization grid found, using default parameters")
        return {"best_params": get_default_parameters(strategy_name)}

    print(f"🎯 Optimizing {strategy_name} with {len(param_grid)} parameters")

    # Expand parameter grid (supports start/end/step format)
    expanded_grid = expand_parameter_grid(param_grid)

    # Get default parameters as base
    default_params = get_default_parameters(strategy_name)
    best_params = default_params.copy() if default_params else {}
    best_score = -float("inf")
    best_portfolio = None

    # Generate parameter combinations
    param_names = list(expanded_grid.keys())
    param_values = list(expanded_grid.values())
    combinations = list(product(*param_values))

    total_combinations = len(combinations)
    test_limit = min(total_combinations, MAX_PARAM_COMBINATIONS)

    print(f"📊 Testing {test_limit} of {total_combinations} parameter combinations...")
    if combinations:
        print(f"   First combo to test: {dict(zip(param_names, combinations[0]))}")

    success_count = 0
    failure_count = 0

    for idx, combo in enumerate(combinations[:test_limit]):
        test_params = best_params.copy()
        for name, value in zip(param_names, combo):
            test_params[name] = value

        try:
            portfolio = create_portfolio(strategy_name, data, test_params)

            if portfolio is None:
                failure_count += 1
                if failure_count <= 3:
                    print(f"   ⚠️ Combo {idx + 1}: create_portfolio returned None")
                    print(f"      Params: {test_params}")
                continue

            stats = portfolio.stats()
            if stats is None:
                failure_count += 1
                continue

            try:
                sharpe = (
                    stats["Sharpe Ratio"] if "Sharpe Ratio" in stats.index else None
                )
                if sharpe is None or pd.isna(sharpe) or not np.isfinite(float(sharpe)):
                    failure_count += 1
                    if failure_count <= 3:
                        print(f"   ⚠️ Combo {idx + 1}: Invalid Sharpe Ratio")
                        print(f"      Params: {test_params}")
                    continue
                score = float(sharpe)
            except (KeyError, TypeError, ValueError):
                failure_count += 1
                continue

            if score > best_score:
                best_score = score
                best_params = test_params.copy()
                best_portfolio = portfolio

            success_count += 1

        except Exception as e:
            failure_count += 1
            if failure_count <= 3:
                print(f"   ❌ Combo {idx + 1} CRASHED: {test_params}")
                print(f"      Error: {type(e).__name__}: {str(e)[:200]}")
                import traceback

                print(f"      Traceback: {traceback.format_exc()[:500]}")
            continue

    if best_portfolio is not None:
        print(
            f"✅ Best parameters found (Sharpe: {best_score:.3f}) - {success_count}/{test_limit} succeeded"
        )
    else:
        print(
            f"⚠️ No successful parameter combinations found ({failure_count}/{test_limit} failed)"
        )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_portfolio": best_portfolio,
        "tested_combinations": test_limit,
    }


def get_strategy_return(strategy_name: str, data, params: Dict) -> float:
    """Get actual strategy return with safe null handling. Accepts DataFrame or Dict."""
    from vectorflow.core.portfolio_builder import create_portfolio

    portfolio = create_portfolio(strategy_name, data, params)
    if portfolio is None:
        return 0.0

    stats = portfolio.stats()
    if stats is None:
        return 0.0

    try:
        value = stats["Total Return [%]"] if "Total Return [%]" in stats.index else 0.0
        return float(value) if pd.notna(value) and np.isfinite(float(value)) else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0
