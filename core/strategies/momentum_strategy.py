"""Momentum strategy implementation."""
from typing import Dict, Any
import pandas as pd
import vectorbt as vbt
from core.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using volatility and WMA."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration."""
        super().__init__(config)
        self.name = "Momentum"
        self.description = "Volatility-based momentum strategy using WMA crossover"
        
        # Default parameter grid if not specified in config
        self.default_parameters = {
            'volatility_window': [10, 20, 30],
            'volatility_momentum_window': [5, 10, 15],
            'volatility_momentum_threshold': [0.1, 0.2, 0.3],
            'higher_wma_window': [50, 100, 200]
        }
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate entry/exit signals for the momentum strategy."""
        # Feature Generation
        volatility = data['close'].rolling(window=self.parameters['volatility_window']).std()
        volatility_momentum = volatility.pct_change(periods=self.parameters['volatility_momentum_window'])
        higher_wma = vbt.ta.wma(data['close'], window=self.parameters['higher_wma_window'])

        # Signal Generation
        entries = (volatility_momentum > self.parameters['volatility_momentum_threshold']) & (data['close'] > higher_wma.iloc[-1])
        exits = (volatility_momentum < -self.parameters['volatility_momentum_threshold']) | (data['close'] < higher_wma.iloc[-1])

        return {'entries': entries, 'exits': exits}
