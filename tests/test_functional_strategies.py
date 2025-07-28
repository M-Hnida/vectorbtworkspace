#!/usr/bin/env python3
"""
Test for Functional Strategies
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies import list_available_strategies, get_strategy_class


class TestFunctionalStrategies(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        # Generate realistic OHLC data
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        high_prices = close_prices + np.random.uniform(0, 0.5, 100)
        low_prices = close_prices - np.random.uniform(0, 0.5, 100)
        open_prices = close_prices + np.random.uniform(-0.25, 0.25, 100)
        
        self.test_data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_strategy_registry(self):
        """Test that all strategies are available."""
        strategies = list_available_strategies()
        expected_strategies = ['momentum', 'orb', 'tdi', 'vectorbt']
        
        for strategy in expected_strategies:
            self.assertIn(strategy, strategies)

    def test_strategy_loading(self):
        """Test that strategies can be loaded."""
        strategies = list_available_strategies()
        
        for strategy_name in strategies:
            strategy_class = get_strategy_class(strategy_name)
            self.assertIsNotNone(strategy_class)

    def test_functional_signal_generation(self):
        """Test that functional signal generation works."""
        from strategies.momentum import create_momentum_signals
        
        signals = create_momentum_signals(self.test_data, momentum_period=5)
        
        # Check that signals object has required attributes
        self.assertTrue(hasattr(signals, 'entries'))
        self.assertTrue(hasattr(signals, 'exits'))
        self.assertTrue(hasattr(signals, 'short_entries'))
        self.assertTrue(hasattr(signals, 'short_exits'))
        
        # Check that signals are pandas Series
        self.assertIsInstance(signals.entries, pd.Series)
        self.assertIsInstance(signals.exits, pd.Series)
        self.assertIsInstance(signals.short_entries, pd.Series)
        self.assertIsInstance(signals.short_exits, pd.Series)


if __name__ == '__main__':
    unittest.main()
