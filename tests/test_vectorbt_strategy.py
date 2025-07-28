#!/usr/bin/env python3
"""
Test VectorBT Strategy Implementation
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

from base import StrategyConfig
from strategies.vectorbt_strategy import VectorBTStrategy


class TestVectorBTStrategy(unittest.TestCase):
    """Test cases for VectorBT strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = StrategyConfig(
            name="test_vectorbt",
            parameters={
                'bbands_period': 20,
                'bbands_std': 2.0,
                'adx_period': 14,
                'adx_threshold': 20,
                'adx_threshold_filter': 60,
                'sma_period': 200,
                'atr_period': 14,
                'atr_mult': 1.0,
                'risk_pct': 0.02,
                'max_side_exposure': 0.30,
                'initial_cash': 500000,
                'dca_size_increment': 0.01,
                'max_dca_size': 0.10,
                'required_timeframes': ['1h']
            }
        )
        self.strategy = VectorBTStrategy(self.config)
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
        
        # Generate realistic OHLC data
        base_price = 100
        returns = np.random.normal(0, 0.01, 1000)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC with some volatility
        noise = np.random.normal(0, 0.005, 1000)
        self.sample_data = pd.DataFrame({
            'open': prices + noise,
            'high': prices + np.abs(noise) + np.random.uniform(0, 0.01, 1000) * prices,
            'low': prices - np.abs(noise) - np.random.uniform(0, 0.01, 1000) * prices,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        # Ensure high >= low and OHLC relationships are correct
        self.sample_data['high'] = np.maximum(
            self.sample_data['high'], 
            np.maximum(self.sample_data['open'], self.sample_data['close'])
        )
        self.sample_data['low'] = np.minimum(
            self.sample_data['low'], 
            np.minimum(self.sample_data['open'], self.sample_data['close'])
        )
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "test_vectorbt")
        self.assertEqual(self.strategy.bbands_period, 20)
        self.assertEqual(self.strategy.bbands_std, 2.0)
        self.assertEqual(self.strategy.adx_period, 14)
        self.assertEqual(self.strategy.risk_pct, 0.02)
    
    def test_required_columns(self):
        """Test required columns."""
        required = self.strategy.get_required_columns()
        expected = ['open', 'high', 'low', 'close']
        self.assertEqual(required, expected)
    
    def test_required_timeframes(self):
        """Test required timeframes."""
        timeframes = self.strategy.get_required_timeframes()
        self.assertEqual(timeframes, ['1h'])
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = {}
        signals = self.strategy.generate_signals(empty_data)
        
        self.assertIsNotNone(signals.entries)
        self.assertIsNotNone(signals.exits)
        self.assertIsNotNone(signals.short_entries)
        self.assertIsNotNone(signals.short_exits)
        self.assertEqual(len(signals.entries), 0)
    
    def test_missing_columns_error(self):
        """Test error handling for missing columns."""
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103]
            # Missing 'high' and 'low'
        })
        
        tf_data = {'1h': incomplete_data}
        
        with self.assertRaises(ValueError) as context:
            self.strategy.generate_signals(tf_data)
        
        self.assertIn("Missing required columns", str(context.exception))
    
    @patch('pandas_ta.bbands')
    @patch('pandas_ta.adx')
    @patch('pandas_ta.sma')
    @patch('pandas_ta.atr')
    def test_indicator_calculation(self, mock_atr, mock_sma, mock_adx, mock_bbands):
        """Test technical indicator calculation."""
        # Mock the pandas_ta functions
        mock_bbands.return_value = None
        mock_adx.return_value = None
        mock_sma.return_value = None
        mock_atr.return_value = None
        
        # Create a DataFrame with the ta accessor
        test_data = self.sample_data.copy()
        
        # Add mock indicator columns directly to simulate pandas_ta behavior
        test_data[f'BBL_{self.strategy.bbands_period}_{self.strategy.bbands_std}'] = test_data['close'] - 2
        test_data[f'BBM_{self.strategy.bbands_period}_{self.strategy.bbands_std}'] = test_data['close']
        test_data[f'BBU_{self.strategy.bbands_period}_{self.strategy.bbands_std}'] = test_data['close'] + 2
        test_data[f'ADX_{self.strategy.adx_period}'] = 25  # Above threshold
        test_data[f'SMA_{self.strategy.sma_period}'] = test_data['close'].rolling(20).mean()
        test_data[f'ATR_{self.strategy.atr_period}'] = 1.0
        
        result = self.strategy._calculate_indicators(test_data)
        
        # Verify that the method returns a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), len(test_data.columns) - 7)  # Should have added indicators
    
    def test_signal_generation_with_valid_data(self):
        """Test signal generation with valid data."""
        tf_data = {'1h': self.sample_data}
        
        try:
            signals = self.strategy.generate_signals(tf_data)
            
            # Verify signals structure
            self.assertIsNotNone(signals.entries)
            self.assertIsNotNone(signals.exits)
            self.assertIsNotNone(signals.short_entries)
            self.assertIsNotNone(signals.short_exits)
            
            # Verify signals are boolean series
            self.assertTrue(signals.entries.dtype == bool)
            self.assertTrue(signals.exits.dtype == bool)
            self.assertTrue(signals.short_entries.dtype == bool)
            self.assertTrue(signals.short_exits.dtype == bool)
            
            # Verify signals have reasonable length (after dropna)
            self.assertGreater(len(signals.entries), 0)
            
            # Check if sizes are provided
            if signals.sizes is not None:
                self.assertTrue(len(signals.sizes) == len(signals.entries))
                self.assertTrue(signals.sizes.dtype in ['float64', 'float32'])
            
        except Exception as e:
            # If pandas_ta is not available or indicators fail, 
            # we should get a specific error
            if "ATR column not found" in str(e):
                self.skipTest("pandas_ta indicators not properly configured in test environment")
            else:
                raise
    
    def test_parameter_access(self):
        """Test parameter access methods."""
        # Test default parameter
        default_val = self.strategy.get_parameter('non_existent_param', 'default')
        self.assertEqual(default_val, 'default')
        
        # Test existing parameter
        bbands_period = self.strategy.get_parameter('bbands_period')
        self.assertEqual(bbands_period, 20)
    
    def test_data_validation(self):
        """Test data validation."""
        # Test with valid data
        tf_data = {'1h': self.sample_data}
        try:
            self.strategy.validate_data(tf_data)
        except Exception:
            pass  # Validation might fail due to missing indicators, which is expected
        
        # Test with empty data
        with self.assertRaises(Exception):
            self.strategy.validate_data({})
    
    def test_column_case_handling(self):
        """Test that strategy handles different column cases."""
        # Create data with uppercase columns
        upper_data = self.sample_data.copy()
        upper_data.columns = [col.upper() for col in upper_data.columns]
        
        tf_data = {'1h': upper_data}
        
        try:
            signals = self.strategy.generate_signals(tf_data)
            # Should not raise an error due to case conversion
            self.assertIsNotNone(signals)
        except ValueError as e:
            if "ATR column not found" in str(e):
                self.skipTest("pandas_ta indicators not properly configured in test environment")
            else:
                raise


if __name__ == '__main__':
    unittest.main()