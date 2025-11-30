#!/usr/bin/env python3
"""
Strategy configuration management.
Handles loading and validating YAML and JSON configuration files.
"""

import os
import json
from typing import Dict, Any
import yaml


def load_strategy_config(strategy_name: str, config_type: str = "auto") -> Dict:
    """Load strategy configuration from YAML or JSON file.

    Args:
        strategy_name: Name of the strategy (matches config filename)
        config_type: 'auto', 'yaml', or 'json'

    Returns:
        Dict containing strategy configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config type is unsupported
    """
    if config_type == "auto":
        config_type = _detect_config_type(strategy_name)

    config_loaders = {"yaml": _load_yaml_config, "json": _load_json_config}

    if config_type not in config_loaders:
        raise ValueError(
            f"Unsupported config type: {config_type}. Use 'yaml' or 'json'."
        )

    return config_loaders[config_type](strategy_name)


def _detect_config_type(strategy_name: str) -> str:
    """Auto-detect configuration file type based on file extension.

    Args:
        strategy_name: Strategy name to search for

    Returns:
        'yaml' or 'json'

    Raises:
        FileNotFoundError: If no config file found
    """
    for ext, config_type in [(".yaml", "yaml"), (".yml", "yaml"), (".json", "json")]:
        if os.path.exists(f"config/{strategy_name}{ext}"):
            return config_type
    raise FileNotFoundError(
        f"No configuration file found for '{strategy_name}' in config/ directory"
    )


def _load_yaml_config(strategy_name: str) -> Dict:
    """Load YAML configuration file.

    Args:
        strategy_name: Strategy name

    Returns:
        Parsed YAML configuration

    Raises:
        FileNotFoundError: If YAML file doesn't exist
    """
    # Try both .yaml and .yml extensions
    for ext in [".yaml", ".yml"]:
        config_path = f"config/{strategy_name}{ext}"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)

    raise FileNotFoundError(
        f"YAML configuration file not found: config/{strategy_name}.yaml"
    )


def _load_json_config(strategy_name: str) -> Dict:
    """Load JSON configuration file.

    Args:
        strategy_name: Strategy name

    Returns:
        Parsed JSON configuration

    Raises:
        FileNotFoundError: If JSON file doesn't exist
    """
    config_path = f"config/{strategy_name}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"JSON configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_config(strategy_name: str, config: Dict, format: str = "yaml") -> None:
    """Save configuration to file.

    Args:
        strategy_name: Strategy name (determines filename)
        config: Configuration dictionary to save
        format: 'yaml' or 'json'

    Raises:
        ValueError: If format is not supported
    """
    if format not in ["yaml", "json"]:
        raise ValueError("Format must be 'yaml' or 'json'")

    ext = ".yaml" if format == "yaml" else ".json"
    config_path = f"config/{strategy_name}{ext}"

    with open(config_path, "w", encoding="utf-8") as f:
        if format == "yaml":
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(config, f, indent=2)

    print(f"✅ Saved configuration to {config_path}")


def update_config_parameter(strategy_name: str, parameter: str, value: Any) -> None:
    """Update a single parameter in a strategy config file.

    Args:
        strategy_name: Strategy name
        parameter: Parameter name (supports nested keys with dots, e.g., 'parameters.fees')
        value: New value for the parameter
    """
    config = load_strategy_config(strategy_name)

    # Handle nested parameters (e.g., "parameters.fees")
    keys = parameter.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value

    # Save back to original format
    config_type = _detect_config_type(strategy_name)
    save_config(strategy_name, config, format=config_type)

    print(f"✅ Updated {parameter} = {value} in {strategy_name} config")
