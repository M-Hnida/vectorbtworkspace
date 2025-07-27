
import sys
import unittest
import io
import sys
from pathlib import Path
import vectorbt as vbt
   
import pandas as pd
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))

from plotter import TradingVisualizer  # This assumes plotter.py contains TradingVisualizer class
from data_manager import load_ohlc_csv  # Changed from load_and_clean_csv to match actual implementation

# Load sample data for testing
data = load_ohlc_csv('data/EURUSD_1D_2009-2025.csv')


class TestPlotter(unittest.TestCase):
    def setUp(self):
        # Create sample test data
        self.dates = pd.date_range(start='2025-01-01', periods=100, freq='1D')
        self.data = pd.DataFrame({
            'open': np.random.rand(100) * 100,
            'high': np.random.rand(100) * 100,
            'low': np.random.rand(100) * 100,
            'close': np.random.rand(100) * 100
        }, index=self.dates)
        
        # Initialize the visualizer
        self.visualizer = TradingVisualizer()
        
        # Create a test portfolio using vectorbt
        self.portfolio = vbt.Portfolio.from_random_signals(
            self.data['close'],
            prob=0.5,
            freq='1D'
        )

    def test_visualizer_initialization(self):
        """Test that TradingVisualizer can be properly initialized"""
        self.assertIsInstance(self.visualizer, TradingVisualizer)

    def test_comprehensive_analysis_plot(self):
        """Test that comprehensive analysis plot can be generated without errors"""
        try:
            self.visualizer.plot_comprehensive_analysis(self.portfolio, "Test Strategy")
        except Exception as e:
            self.fail(f"plot_comprehensive_analysis raised an exception: {str(e)}")

    def test_portfolio_plot(self):
        """Test that individual portfolio plot can be generated without errors"""
        result = self.visualizer.plot_portfolio(self.portfolio, "Single Portfolio Test")
        self.assertIsInstance(result, dict)
        self.assertTrue(result['success'], "Portfolio plot should succeed")

    def test_multiple_portfolios_analysis(self):
        """Test comprehensive analysis with multiple portfolios"""
        # Create a second portfolio with different parameters
        portfolio2 = vbt.Portfolio.from_random_signals(
            self.data['close'],
            prob=0.3,  # Different probability for comparison
            freq='1D'
        )
        
        # Test with dict of portfolios
        portfolios = {
            "Strategy 1": self.portfolio,
            "Strategy 2": portfolio2
        }
        result = self.visualizer.plot_comprehensive_analysis(portfolios, "Multiple Portfolios Test")
        self.assertTrue(result['success'], "Multiple portfolios analysis should succeed")
        self.assertIn("plots_created", result)

    def test_failed_portfolio_plot(self):
        """Test that plotting handles errors gracefully"""
        # Test with invalid portfolio
        result = self.visualizer.plot_portfolio(None, "Invalid Portfolio")
        self.assertFalse(result.get('success', True), "Should fail with invalid portfolio")
        self.assertIn('error', result)

    def test_monte_carlo_analysis(self):
        """Test Monte Carlo analysis plotting"""
        # Create mock Monte Carlo results
        mc_results = {
            'analysis': True,
            'base_metrics': {
                'total_return': 15.5,
                'sharpe_ratio': 1.2
            },
            'simulations': [
                {'metrics': {'return': 12.5, 'sharpe': 1.1}},
                {'metrics': {'return': 15.5, 'sharpe': 1.2}},
                {'metrics': {'return': 18.5, 'sharpe': 1.3}}
            ]
        }
        
        # Test Monte Carlo plotting through comprehensive analysis
        result = self.visualizer.plot_comprehensive_analysis(
            self.portfolio,
            "Monte Carlo Test",
            mc_results=mc_results
        )
        self.assertTrue(result['success'])
        self.assertIn("monte_carlo", result['plots_created'])

    def test_walkforward_analysis(self):
        """Test Walk-forward analysis plotting"""
        # Create mock walk-forward results
        wf_results = {
            'windows': [
                {
                    'window': 1,
                    'train_metrics': {'total_return': 10.0, 'sharpe_ratio': 1.1},
                    'test_metrics': {'total_return': 8.0, 'sharpe_ratio': 0.9}
                },
                {
                    'window': 2,
                    'train_metrics': {'total_return': 12.0, 'sharpe_ratio': 1.2},
                    'test_metrics': {'total_return': 9.0, 'sharpe_ratio': 1.0}
                }
            ]
        }
        
        # Test Walk-forward plotting through comprehensive analysis
        result = self.visualizer.plot_comprehensive_analysis(
            self.portfolio,
            "Walk-forward Test",
            wf_results=wf_results
        )
        self.assertTrue(result['success'])
        self.assertIn("walk_forward", result['plots_created'])

    def test_portfolio_data_validation(self):
        """Test that the visualizer handles invalid portfolio input gracefully"""
        # Redirect stdout to capture the warning message

        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test with invalid portfolio
        self.visualizer.plot_comprehensive_analysis(None, "Invalid Portfolio")
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Check if warning message was printed
        self.assertIn("⚠️ No portfolios provided", captured_output.getvalue())


if __name__ == '__main__':
    unittest.main()
