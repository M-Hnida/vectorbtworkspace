"""Configuration management for the trading framework."""
from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional

import yaml

@dataclass
class SystemConfig:
    """System-wide configuration."""
    data_path: str = 'data'
    log_level: str = 'INFO'
    cache_enabled: bool = True
    max_workers: int = 4


@dataclass
class MarketDataConfig:
    """Market data requirements configuration."""
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)

@dataclass
class StrategyConfig:
    """Strategy-specific configuration."""
    name: str
    class_name: Optional[str] = None  # Explicit class name if different from convention
    parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_grid: Dict[str, List[Any]] = field(default_factory=dict)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    def get_class_name(self) -> str:
        """Get the strategy class name using the configured name or auto-generate it."""
        if self.class_name:
            return self.class_name
        # Par défaut, utilise le CamelCase
        return ''.join(word.capitalize() for word in self.name.split('_')) + 'Strategy'

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.system_config: Optional[SystemConfig] = None
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        
    def load_config(self, strategy_name: str) -> StrategyConfig:
        """Load configuration for a specific strategy."""
        config_path = os.path.join(self.config_path, f'{strategy_name}.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # Load system config if present
        if 'system' in yaml_config:
            self.system_config = self._load_system_config(yaml_config['system'])
        else:
            self.system_config = SystemConfig()
        
        # Load strategy config
        strategy_config = self._load_strategy_config(strategy_name, yaml_config)
        self.strategy_configs[strategy_name] = strategy_config
        
        return strategy_config
    
    def _load_system_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Load system configuration with validation."""
        return SystemConfig(
            data_path=config_dict.get('data_path', 'data'),
            log_level=config_dict.get('log_level', 'INFO'),
            cache_enabled=config_dict.get('cache_enabled', True),
            max_workers=config_dict.get('max_workers', 4)
        )
    
    def _load_strategy_config(self, name: str, config_dict: Dict[str, Any]) -> StrategyConfig:
        """Load strategy configuration."""
        # Load portfolio config
        
        # Load market data config
        market_data_dict = config_dict.get('market_data', {})
        market_data_config = MarketDataConfig(
            symbols=market_data_dict.get('symbols', []),
            timeframes=market_data_dict.get('timeframes', [])
        )
        
        # Load strategy config
        return StrategyConfig(
            name=name,
            class_name=config_dict.get('class_name'),
            parameters=config_dict.get('indicators', {}),  # Using indicators as parameters
            optimization_grid=config_dict.get('optimization', {}),
            market_data=market_data_config
        )