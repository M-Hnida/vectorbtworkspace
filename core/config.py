"""Configuration management for the trading framework."""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
import os


@dataclass
class SystemConfig:
    """System-wide configuration."""
    data_path: str = 'data'
    output_path: str = 'output'
    log_level: str = 'INFO'
    cache_enabled: bool = True
    max_workers: int = 4


@dataclass
class PortfolioConfig:
    """Portfolio configuration."""
    initial_cash: float = 100000
    fees: float = 0.001
    slippage: float = 0.001
    risk_per_trade: float = 0.02


@dataclass
class StrategyConfig:
    """Strategy-specific configuration."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_grid: Dict[str, List[Any]] = field(default_factory=dict)
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.system_config: Optional[SystemConfig] = None
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        
    def load_config(self, strategy_name: str) -> StrategyConfig:
        """Load configuration for a specific strategy."""
        config_file = os.path.join('config', f'{strategy_name}.yaml')
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Load system config if present
        if 'system' in raw_config:
            self.system_config = self._load_system_config(raw_config['system'])
        else:
            self.system_config = SystemConfig()
        
        # Load strategy config
        strategy_config = self._load_strategy_config(strategy_name, raw_config)
        self.strategy_configs[strategy_name] = strategy_config
        
        return strategy_config
    
    def _load_system_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Load system configuration with validation."""
        return SystemConfig(
            data_path=config_dict.get('data_path', 'data'),
            output_path=config_dict.get('output_path', 'output'),
            log_level=config_dict.get('log_level', 'INFO'),
            cache_enabled=config_dict.get('cache_enabled', True),
            max_workers=config_dict.get('max_workers', 4)
        )
    
    def _load_strategy_config(self, name: str, config_dict: Dict[str, Any]) -> StrategyConfig:
        """Load strategy configuration."""
        # Load portfolio config
        portfolio_dict = config_dict.get('portfolio', {})
        portfolio_config = PortfolioConfig(
            initial_cash=portfolio_dict.get('cash', 100000),
            fees=portfolio_dict.get('fees', 0.001),
            slippage=portfolio_dict.get('slippage', 0.001),
            risk_per_trade=portfolio_dict.get('risk_pct', 0.02)
        )
        
        # Load strategy config
        return StrategyConfig(
            name=name,
            parameters=config_dict.get('indicators', {}),  # Legacy compatibility
            optimization_grid=config_dict.get('optimization', {}),
            data_requirements=config_dict.get('market_data', {}),
            portfolio=portfolio_config
        )
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        if self.system_config is None:
            self.system_config = SystemConfig()
        return self.system_config