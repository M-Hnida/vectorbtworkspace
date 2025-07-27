#!/usr/bin/env python3
"""
Test for ORB Strategy
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import StrategyConfig
from strategies.orb_strategy import ORBStrategy


class TestORBStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test data and strategy."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
        }, index=dates)
        
        # Create config
        config = StrategyConfig(
            name="orb_strategy",
            parameters={
                'orb_period': 1,
                'breakout_threshold': 0.005,
                'atr_multiple': 2.0,
                'required_timeframes': ['1h']
            }
        )
        
        # Initialize strategy
        self.strategy = ORBStrategy(config)
        
    def test_strategy_initialization(self):
        """Test that strategy initializes correctly."""
        self.assertEqual(self.strategy.name, "orb_strategy")
        self.assertEqual(self.strategy.orb_period, 1)
        self.assertEqual(self.strategy.breakout_threshold, 0.005)
        
    def test_generate_signals(self):
        """Test that strategy generates signals correctly."""
        tf_data = {'1h': self.sample_data}
        signals = self.strategy.generate_signals(tf_data)
        
        # Check that signals are generated
        self.assertIsNotNone(signals.entries)
        self.assertIsNotNone(signals.exits)
        self.assertIsNotNone(signals.short_entries)
        self.assertIsNotNone(signals.short_exits)
        
        # Check that all signals have the same index
        self.assertTrue(signals.entries.index.equals(signals.exits.index))
        self.assertTrue(signals.entries.index.equals(signals.short_entries.index))
        self.assertTrue(signals.entries.index.equals(signals.short_exits.index))
        
    def test_required_timeframes(self):
        """Test that strategy returns correct timeframes."""
        timeframes = self.strategy.get_required_timeframes()
        self.assertIn('1h', timeframes)
        
    def test_required_columns(self):
        """Test that strategy returns correct columns."""
        columns = self.strategy.get_required_columns()
        expected_columns = ['open', 'high', 'low', 'close']
        for col in expected_columns:
            self.assertIn(col, columns)


if __name__ == '__main__':
    unittest.main()