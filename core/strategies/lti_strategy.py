"""LTI (Logical Trading Indicator) strategy implementation."""
from typing import Dict, Any
import pandas as pd
from core.base import BaseStrategy
from core.strategies.lti_signals import generate_signals


class LTIStrategy(BaseStrategy):
    """Logical Trading Indicator strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration."""
        super().__init__(config)
        self.name = "LTI"
        self.description = "Logical Trading Indicator strategy using multiple timeframe analysis"
        
        # Default parameter grid if not specified in config
        self.default_parameters = {
            'lookback': [10, 20, 30],
            'ma_fast': [5, 10, 15],
            'ma_slow': [20, 30, 40],
            'rsi_period': [14, 21, 28]
        }
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, pd.DataFrame]:
        """Generate entry/exit signals for the LTI strategy."""
        param_grid = kwargs.get('param_grid', self._get_default_param_grid())
        timeframe = kwargs.get('timeframe', self.default_timeframe)
        
        print(f"\n📊 Generating LTI signals with {len(param_grid)} parameter combinations")
        print(f"   Timeframe: {timeframe}")
        
        signals, _ = generate_signals(
            data=data,
            param_grid=param_grid,
            timeframe=timeframe,
            indicator_config=kwargs.get('indicator_config', {})
        )
        return signals
