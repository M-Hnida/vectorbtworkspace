import sys
import os
# Add project root to Python path for individual test execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from optimizer import ParameterOptimizer, WalkForwardAnalysis, MonteCarloAnalysis
from base import BaseStrategy, StrategyConfig, Signals

class MockStrategy(BaseStrategy):
        """Mock strategy for testing."""
        
        def __init__(self, config, required_timeframes=['1h']):
            super().__init__(config)
            self.required_timeframes = required_timeframes
        
        def generate_signals(self, tf_data):
            """Generate mock signals."""
            primary_tf = self.required_timeframes[0]
            df = tf_data[primary_tf]
            length = len(df)
            entries = pd.Series([False] * length, index=df.index)
            exits = pd.Series([False] * length, index=df.index)
            short_entries = pd.Series([False] * length, index=df.index)
            short_exits = pd.Series([False] * length, index=df.index)
            return Signals(entries=entries, exits=exits, 
                         short_entries=short_entries, short_exits=short_exits)
        

class TestOptimizationComponents(unittest.TestCase):
    """Test suite for optimization components."""
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple strategy config for testing
        self.strategy_config = StrategyConfig(
            name='test_strategy',
            parameters={'param1': 10, 'param2': 20},
            optimization_grid={'param1': [5, 10, 15], 'param2': [15, 20, 25]},
            analysis_settings={'monte_carlo_runs': 10}
        )
        
        # Sample data for testing
        np.random.seed(42)  # For reproducible tests
        self.data = pd.DataFrame({
            'open': np.random.rand(1200) * 100 + 50,
            'high': np.random.rand(1200) * 100 + 60,
            'low': np.random.rand(1200) * 100 + 40,
            'close': np.random.rand(1200) * 100 + 50
        }, index=pd.to_datetime(np.arange(1200), unit='D'))
        
        # Simple strategy implementation for testing
        class SimpleStrategy(BaseStrategy):
            """Simple test strategy."""
            
            def __init__(self, config, required_timeframes=['1h']):
                super().__init__(config)
                self.required_timeframes = required_timeframes
            
            def generate_signals(self, tf_data):
                """Generate simple random signals for testing."""
                primary_tf = self.required_timeframes[0]
                df = tf_data[primary_tf]
                length = len(df)
                # Generate simple buy/sell signals
                entries = pd.Series([True if i % 10 == 0 else False for i in range(length)], index=df.index)
                exits = pd.Series([True if i % 10 == 5 else False for i in range(length)], index=df.index)
                short_entries = pd.Series([False] * length, index=df.index)
                short_exits = pd.Series([False] * length, index=df.index)
                
                return Signals(entries=entries, exits=exits, 
                             short_entries=short_entries, short_exits=short_exits)

        self.strategy = SimpleStrategy(self.strategy_config)
        
        # Initialize components for testing
        self.monte_carlo = MonteCarloAnalysis(self.strategy)
        self.optimizer = ParameterOptimizer(self.strategy, self.strategy_config)
        self.wfa = WalkForwardAnalysis(self.strategy, self.strategy_config)

    def test_monte_carlo_initialization(self):
        """Test MonteCarloAnalysis initialization."""
        self.assertIsNotNone(self.monte_carlo)
        self.assertEqual(self.monte_carlo.strategy.config.name, 'test_strategy')
        
    def test_parameter_optimizer_initialization(self):
        """Test ParameterOptimizer initialization."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.strategy.config.name, 'test_strategy')
        
    def test_walk_forward_analysis_initialization(self):
        """Test WalkForwardAnalysis initialization."""
        self.assertIsNotNone(self.wfa)
        self.assertEqual(self.wfa.strategy.config.name, 'test_strategy')
        
    def test_optimization_runs(self):
        """Test basic optimization runs."""
        # Test Monte Carlo analysis
        mc_results = self.monte_carlo.run_analysis(data=self.data)
        self.assertIn('base_metrics', mc_results)
        self.assertIn('simulations', mc_results)
        self.assertIsInstance(mc_results['base_metrics'], dict)
        self.assertIsInstance(mc_results['simulations'], list)
        
        # Test parameter optimization
        opt_results = self.optimizer.optimize(data=self.data)
        self.assertIn('param_combination', opt_results)
        self.assertIn('sharpe_ratio', opt_results)
        self.assertIsInstance(opt_results['param_combination'], dict)
        self.assertIsInstance(opt_results['sharpe_ratio'], float)
        
        # Test walk-forward analysis
        wfa_results = self.wfa.run_analysis(data=self.data)
        self.assertIn('windows', wfa_results)
        self.assertIn('summary', wfa_results)
        self.assertIsInstance(wfa_results['windows'], list)
        self.assertIsInstance(wfa_results['summary'], dict)


class TestParameterOptimizer(unittest.TestCase):
    """Test suite for ParameterOptimizer class."""
    
    def test_optimizer_creation(self):
        """Test optimizer creation with valid parameters."""
        config = StrategyConfig(name='test', parameters={})
        strategy = MockStrategy(config)
        optimizer = ParameterOptimizer(strategy, config)
        self.assertIsNotNone(optimizer)


class TestWalkForwardAnalysis(unittest.TestCase):
    """Test suite for WalkForwardAnalysis class."""
    
    def test_wfa_creation(self):
        """Test walk-forward analysis creation."""
        config = StrategyConfig(name='test', parameters={})
        strategy = MockStrategy(config)
        wfa = WalkForwardAnalysis(strategy, config)
        self.assertIsNotNone(wfa)


class TestMonteCarloAnalysis(unittest.TestCase):
    """Test suite for MonteCarloAnalysis class."""
    

    def test_monte_carlo_creation(self):
        """Test Monte Carlo analysis creation."""
        config = StrategyConfig(name='test', parameters={})
        strategy = MockStrategy(config)
        mc = MonteCarloAnalysis(strategy)
        self.assertIsNotNone(mc)


if __name__ == '__main__':
    unittest.main()
