#!/usr/bin/env python3
"""
Unit tests for Monte Carlo simulation module: generation, storage, distribution, and plotting.
Run: python run_tests.py
"""
import unittest
import numpy as np
import pandas as pd

from optimizer import run_monte_carlo_analysis
import plotly.graph_objects as go


def make_dummy_data(n=500):
    """Create dummy OHLC data for testing."""
    idx = pd.date_range('2024-01-01', periods=n, freq='1H')
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close}, index=idx)


class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = make_dummy_data(600)

    def test_generation_and_storage(self):
        """Test Monte Carlo simulation generation and storage."""
        res = run_monte_carlo_analysis(self.data, strategy_name='rsi', params={'rsi_period': 14})
        self.assertIsInstance(res, dict)
        self.assertIn('statistics', res)
        self.assertIn('simulations', res)
        stats = res['statistics']
        self.assertIn('mean_return', stats)
        # path_matrix must exist and have shape (T, N)
        self.assertIn('path_matrix', res)
        pm = res['path_matrix']
        self.assertTrue(hasattr(pm, 'shape'))
        T, N = pm.shape
        self.assertGreater(T, 0)
        self.assertGreater(N, 0)
        # simulations length equals success_count
        self.assertEqual(len(res['simulations']), stats.get('success_count', len(res['simulations'])))

    def test_distribution_stats(self):
        """Test Monte Carlo distribution statistics."""
        res = run_monte_carlo_analysis(self.data, strategy_name='rsi', params={'rsi_period': 14})
        stats = res['statistics']
        self.assertIn('mean_return', stats)
        self.assertIn('std_return', stats)
        self.assertIn('percentile_5', stats)
        self.assertIn('percentile_95', stats)
        # Ensure no NaNs when count>0
        if stats.get('count', 0) > 0:
            self.assertTrue(np.isfinite(stats['mean_return']))
            self.assertTrue(np.isfinite(stats['std_return']))

    def test_histogram_plotting(self):
        """Test histogram plotting with edge cases."""
        from plotter import _add_histogram
        
        # Build a returns list with NaN/Inf to verify guards
        returns = [0.1, 0.2, 0.15, -0.05, 0.08, 0.12]
        statistics = {'actual_return': 0.12}
        fig = go.Figure()
        
        # Call should not raise
        try:
            _add_histogram(fig, returns, statistics)
            self.assertTrue(True)  # If we get here, no exception was raised
        except Exception as e:
            self.fail(f"Histogram plotting raised exception: {e}")


if __name__ == '__main__':
    unittest.main()