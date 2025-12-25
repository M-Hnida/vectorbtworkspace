#!/usr/bin/env python3
"""
Configuration validation to ensure strategy configs match strategy code.
Prevents silent failures where parameters are ignored.
"""

from typing import Dict, List, Set, Optional
import inspect
import logging
import re

# Configure logging
logger = logging.getLogger(__name__)

# Constants
SEPARATOR_WIDTH = 60
PARAM_GET_PATTERN = r'params\.get\(["\']([^"\']+)["\']'
NON_STRATEGY_PARAMS = {
    "primary_timeframe",
    "primary_symbol",
    "data_source",
    "csv_path",
    "initial_cash",
    "fee",
}


def get_strategy_expected_params(strategy_name: str) -> Optional[Set[str]]:
    """
    Extract expected parameter names from strategy code.
    Looks at params.get() calls in create_portfolio function.
    """
    try:
        from vectorflow.core.portfolio_builder import _STRATEGIES
        import importlib

        if strategy_name not in _STRATEGIES:
            logger.debug(f"Strategy '{strategy_name}' not found in registry")
            return None

        module_path = _STRATEGIES[strategy_name]
        module = importlib.import_module(module_path)

        if not hasattr(module, "create_portfolio"):
            return None

        portfolio_func = getattr(module, "create_portfolio")

        # Get source code
        source = inspect.getsource(portfolio_func)

        # Find all params.get("param_name") calls
        matches = re.findall(PARAM_GET_PATTERN, source)

        return set(matches) if matches else None

    except (ImportError, AttributeError, ValueError, OSError) as e:
        logger.debug(
            f"Could not extract parameters for strategy '{strategy_name}': {e}"
        )
        return None


def validate_strategy_config(strategy_name: str, config: Dict) -> Dict[str, List[str]]:
    """
    Validate that config parameters match what the strategy expects.

    Returns:
        Dict with 'errors', 'warnings', and 'info' lists
    """
    result = {"errors": [], "warnings": [], "info": []}

    # Get expected parameters from strategy code
    expected_params = get_strategy_expected_params(strategy_name)

    if expected_params is None:
        # Parameter extraction failed - configs use defaults from strategy
        return result

    # Get configured parameters
    config_params = set(config.get("parameters", {}).keys())
    optimization_params = set(config.get("optimization_grid", {}).keys())

    # Check for mismatches
    unused_config = config_params - expected_params
    unused_optimization = optimization_params - expected_params
    missing_in_config = expected_params - config_params

    # Report issues
    if unused_config:
        result["warnings"].append(
            f"Parameters in config but not used by strategy: {sorted(unused_config)}"
        )

    if unused_optimization:
        result["errors"].append(
            f"Optimization grid parameters not used by strategy: {sorted(unused_optimization)}"
        )
        result["errors"].append(
            "This means Monte Carlo and optimization will have NO EFFECT!"
        )

    if missing_in_config:
        result["info"].append(
            f"Strategy expects these parameters (using defaults): {sorted(missing_in_config)}"
        )

    # Check if optimization params are in config params
    optimization_not_in_config = optimization_params - config_params
    if optimization_not_in_config:
        result["warnings"].append(
            f"Optimization grid has parameters not in config: {sorted(optimization_not_in_config)}"
        )

    return result


def print_validation_results(
    strategy_name: str, validation: Dict[str, List[str]]
) -> bool:
    """
    Print validation results in a readable format.

    Returns:
        True if validation passed (no errors), False otherwise
    """
    has_errors = len(validation["errors"]) > 0
    has_warnings = len(validation["warnings"]) > 0
    has_info = len(validation["info"]) > 0

    if not (has_errors or has_warnings or has_info):
        print(f"✅ Config validation passed for '{strategy_name}'")
        return True

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print(f"Config Validation Results: {strategy_name}")
    print(f"{'=' * SEPARATOR_WIDTH}")

    if validation["errors"]:
        print("\n❌ ERRORS (must fix):")
        for error in validation["errors"]:
            print(f"   • {error}")

    if validation["warnings"]:
        print("\n⚠️  WARNINGS (should review):")
        for warning in validation["warnings"]:
            print(f"   • {warning}")

    if validation["info"]:
        print("\n💡 INFO:")
        for info in validation["info"]:
            print(f"   • {info}")

    print(f"{'=' * SEPARATOR_WIDTH}\n")

    return not has_errors


def validate_and_fix_config(strategy_name: str, config: Dict) -> Dict:
    """
    Validate config and attempt to auto-fix common issues.

    Returns:
        Fixed config (or original if no fixes possible)
    """
    validation = validate_strategy_config(strategy_name, config)

    # Auto-fix: Remove unused optimization parameters
    if validation["errors"]:
        expected_params = get_strategy_expected_params(strategy_name)
        if expected_params and "optimization_grid" in config:
            original_grid = config["optimization_grid"].copy()
            fixed_grid = {
                k: v for k, v in original_grid.items() if k in expected_params
            }

            if fixed_grid != original_grid:
                removed = set(original_grid.keys()) - set(fixed_grid.keys())
                print(
                    f"🔧 Auto-fix: Removed unused optimization params: {sorted(removed)}"
                )
                config["optimization_grid"] = fixed_grid

    return config


def suggest_parameter_mapping(
    strategy_name: str, config: Dict
) -> Optional[Dict[str, str]]:
    """
    Suggest parameter name mappings when there's a mismatch.

    Returns:
        Dict mapping config names to strategy names, or None
    """
    expected_params = get_strategy_expected_params(strategy_name)
    if not expected_params:
        return None

    config_params = set(config.get("parameters", {}).keys())
    optimization_params = set(config.get("optimization_grid", {}).keys())
    all_config_params = config_params | optimization_params

    # Remove common non-strategy params
    all_config_params = all_config_params - NON_STRATEGY_PARAMS

    # If there's a mismatch, try to suggest mappings
    if all_config_params != expected_params:
        suggestions = {}

        # Simple heuristics for common mappings
        mapping_hints = {
            "rsi_period": ["period", "length", "window"],
            "period": ["rsi_period", "length", "window"],
            "length": ["period", "rsi_period", "window"],
            "oversold_level": ["oversold", "lower_band", "low_band"],
            "overbought_level": ["overbought", "upper_band", "top_band"],
            "oversold": ["oversold_level", "lower_band", "low_band"],
            "overbought": ["overbought_level", "upper_band", "top_band"],
        }

        for config_param in all_config_params:
            if config_param not in expected_params:
                # Try to find a match
                if config_param in mapping_hints:
                    for hint in mapping_hints[config_param]:
                        if hint in expected_params:
                            suggestions[config_param] = hint
                            break

        return suggestions if suggestions else None

    return None


# Quick validation function for use in main pipeline
def quick_validate(strategy_name: str, config: Dict, auto_fix: bool = False) -> bool:
    """
    Quick validation with optional auto-fix.

    Returns:
        True if validation passed, False if critical errors
    """
    if auto_fix:
        config = validate_and_fix_config(strategy_name, config)

    validation = validate_strategy_config(strategy_name, config)

    # Print results
    passed = print_validation_results(strategy_name, validation)

    # Print suggestions if available
    if not passed:
        suggestions = suggest_parameter_mapping(strategy_name, config)
        if suggestions:
            print("💡 Suggested parameter name mappings:")
            for config_param, suggested_param in suggestions.items():
                print(f"   '{config_param}' → '{suggested_param}'")
            print()

    return passed
