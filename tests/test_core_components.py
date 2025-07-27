import unittest
import pandas as pd
import sys
import os
# Add project root to Python path for individual test execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import Signals
class TestCoreComponents(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.data = pd.DataFrame({
            'open': [100, 102, 101, 103],
            'high': [105, 108, 107, 109],
            'low': [95, 100, 99, 101],
            'close': [102, 105, 104, 107]
        })
        
    def test_backtest_signals(self):
        """Test that backtesting signals generates expected results."""
        # Create dummy signals
        entries = pd.Series([True, False, True, False], index=self.data.index)
        exits = pd.Series([False, True, False, True], index=self.data.index)
        
        signals = Signals(entries=entries, exits=exits)
        
        # Check if signals are generated correctly
        self.assertTrue(signals.entries.equals(entries))
        self.assertTrue(signals.exits.equals(exits))
if __name__ == '__main__':
    unittest.main()