import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

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
        have the same index. Also ensures all signals are properly boolean.

        Raises:
            ValueError: If no signal series is provided or if the indexes of the provided series differ.
        """
        # Collect non-None series and convert to boolean if needed
        series_list = []
        for attr_name in ['entries', 'exits', 'short_entries', 'short_exits']:
            series = getattr(self, attr_name)
            if series is not None:
                # Convert to boolean if it's a boolean expression result
                if not pd.api.types.is_bool_dtype(series):
                    try:
                        # Handle numeric 0/1 values or boolean expressions
                        series = series.astype(bool)
                        setattr(self, attr_name, series)
                    except :
                        raise ValueError(f"{attr_name} must be boolean or boolean-convertible")
                series_list.append(series)

        if not series_list:
            raise ValueError("At least one signal series must be provided")

        # Check index alignment
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

