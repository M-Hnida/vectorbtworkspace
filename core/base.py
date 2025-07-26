"""Base classes for the trading framework."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import vectorbt as vbt


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = self._validate_config(config)
        self.parameters = self.config.get('parameters', {})
        self.name = self.config.get('name', self.__class__.__name__)
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate entry and exit signals.
        
        Args:
            data: OHLC data with datetime index
            
        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return list of required data columns."""
        pass
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy-specific configuration."""
        required_keys = ['parameters']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        return config
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter value with optional default."""
        return self.parameters.get(key, default)


class BaseDataLoader(ABC):
    """Abstract base class for data loading strategies."""
    
    @abstractmethod
    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        """Load data for given symbol and interval.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            interval: Time interval (e.g., '1h', '4h')
            
        Returns:
            DataFrame with OHLC data and datetime index
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
    
    @abstractmethod
    def get_available_intervals(self, symbol: str) -> List[str]:
        """Get list of available intervals for a symbol."""
        pass