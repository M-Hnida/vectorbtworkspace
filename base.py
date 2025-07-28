import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    M1, M5, M15, H1, H4, D1 = "1m", "5m", "15m", "1h", "4h", "1d"


@dataclass()
class Signals:
    """Trading signals container."""
    entries: Optional[pd.Series] = None
    exits: Optional[pd.Series] = None
    short_entries: Optional[pd.Series] = None
    short_exits: Optional[pd.Series] = None
    sizes: Optional[pd.Series] = None  # Optional position sizes

    def __post_init__(self):
        """
        Validate that at least one signal series is provided and that all provided signal series
        have the same index.

        Raises:
            ValueError: If no signal series is provided or if the indexes of the provided series differ.
        """
        # Collect non-None series
        series_list = [s for s in [self.entries, self.exits, self.short_entries, self.short_exits] if s is not None]
        if not series_list:
            raise ValueError("At least one signal series must be provided")
        first_index = series_list[0].index
        for series in series_list[1:]:
            if not series.index.equals(first_index):
                raise ValueError("All provided signal series must have the same index")
        
        # Validate sizes if provided
        if self.sizes is not None and not self.sizes.index.equals(first_index):
            raise ValueError("Sizes series must have the same index as signal series")



@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    parameters: Dict[str, Any]
    optimization_grid: Dict[str, List[Any]] = None
    analysis_settings: Dict[str, Any] = None
    data_requirements: Dict[str, Any] = None
    required_columns: List[str] = None

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Strategy name cannot be empty")
        if self.optimization_grid is None:
            self.optimization_grid = {}
        elif not isinstance(self.optimization_grid, dict):
            raise TypeError("optimization_grid must be a dictionary")
        if self.analysis_settings is None:
            self.analysis_settings = {}
        if self.data_requirements is None:
            self.data_requirements = {}
        if self.required_columns is None:
            self.required_columns = ['open', 'high', 'low', 'close']


class StrategyError(Exception):
    """Strategy-specific exception."""
    pass


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.{config.name}")

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def parameters(self) -> Dict[str, Any]:
        return self.config.parameters

    @abstractmethod
    def generate_signals(self, tf_data: Dict[str, pd.DataFrame]) -> Signals:
        """Generate trading signals."""
        pass

    def get_required_columns(self) -> List[str]:
        return self.config.required_columns

    def get_parameter(self, key: str, default: Any = None) -> Any:
        return self.parameters.get(key, default)

    def get_required_timeframes(self) -> List[str]:
        return ['1h']  # Default timeframe

    def validate_data(self, tf_data: Dict[str, pd.DataFrame]) -> None:
        """Basic data validation."""
        if not tf_data:
            raise StrategyError("Empty data provided")

        for tf, df in tf_data.items():
            if df.empty:
                raise StrategyError(f"Empty dataframe for {tf}")

            missing = set(self.get_required_columns()) - set(df.columns)
            if missing:
                raise StrategyError(f"Missing columns in {tf}: {missing}")

    def execute(self, tf_data: Dict[str, pd.DataFrame]) -> Signals:
        """Execute strategy with validation."""
        self.validate_data(tf_data)
        self._last_tf_data = tf_data  # For multi-timeframe check

        try:
            return self.generate_signals(tf_data)
        except Exception as e:
            raise StrategyError(f"Signal generation failed: {e}") from e
