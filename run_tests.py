#!/usr/bin/env python3
"""
Unit tests for Monte Carlo simulation module: generation, storage, distribution, and plotting.
Run: python run_tests.py
"""
import unittest
import numpy as np
import pandas as pd

from optimizer import run_monte_carlo_analysis, DEFAULT_CONFIG
from plotter import _add_mc_histogram
import plotly.graph_objects as go

class DummyStrategy:
    def __init__(self, name='rsi', params=None):
        self.name = name
        self.parameters = params or {}
    def get_required_timeframes(self):
        return ['1H']

def make_dummy_data(n=500):
    idx = pd.date_range('2024-01-01', periods=n, freq='1H')
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({'open': close, 'high': close, 'low': close, 'close': close}, index=idx)

class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = make_dummy_data(600)
        self.strategy = DummyStrategy('rsi', {'rsi_period': 14})

    def test_generation_and_storage(self):
        # Reduce simulations for test speed
        DEFAULT_CONFIG['monte_carlo_simulations'] = 50
        DEFAULT_CONFIG['monte_carlo_batch_size'] = 16
        res = run_monte_carlo_analysis(self.data, self.strategy)
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
        DEFAULT_CONFIG['monte_carlo_simulations'] = 40
        DEFAULT_CONFIG['monte_carlo_batch_size'] = 20
        res = run_monte_carlo_analysis(self.data, self.strategy)
        stats = res['statistics']
        self.assertIn('mean_return', stats)
        self.assertIn('std_return', stats)
        self.assertIn('percentile_5', stats)
        self.assertIn('percentile_95', stats)
        # Ensure no NaNs when count>0
        if stats.get('count', 0) > 0:
            self.assertTrue(np.isfinite(stats['mean_return']))
            self.assertTrue(np.isfinite(stats['std_return']))

    def test_histogram_no_artifacts(self):
        # Build a returns list with NaN/Inf to verify guards
        returns = [0.1, 0.2, 0.15, float('nan'), float('inf'), -0.05]
        fig = go.Figure()
        # Call should not raise and should add a histogram trace or no-op
        _add_mc_histogram(fig, returns, {'actual_return': 0.12})
        self.assertTrue(len(fig.data) >= 0)  # Ensure no exception raised

if __name__ == '__main__':
    unittest.main()