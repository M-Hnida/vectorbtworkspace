
import sys
import unittest
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import vectorbt as vbt
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from plotter import TradingVisualizer, create_performance_plots


class TestPlotter(unittest.TestCase):
    def setUp(self):
        # Create sample test data with proper OHLC structure
        self.dates = pd.date_range(start='2025-01-01', periods=100, freq='1D')
        np.random.seed(42)  # For reproducible tests

        # Generate realistic OHLC data
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        prices = base_price * np.cumprod(1 + returns)

        self.data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 100))),
            'close': prices
        }, index=self.dates)

        # Ensure OHLC consistency
        self.data['high'] = self.data[['open', 'high', 'close']].max(axis=1)
        self.data['low'] = self.data[['open', 'low', 'close']].min(axis=1)

        # Initialize the visualizer
        self.visualizer = TradingVisualizer()

        # Create test portfolios using VectorBT
        self.portfolio = vbt.Portfolio.from_random_signals(
            self.data['close'],
            prob=0.1,  # Lower probability for more realistic signals
            freq='1D',
            seed=42
        )

    def test_visualizer_initialization(self):
        """Test that TradingVisualizer can be properly initialized with VectorBT settings"""
        self.assertIsInstance(self.visualizer, TradingVisualizer)
        self.assertIsInstance(self.visualizer.dark_colors, list)
        self.assertEqual(len(self.visualizer.dark_colors), 6)

    @patch('plotly.graph_objects.Figure.show')
    def test_comprehensive_analysis_plot(self, mock_show):
        """Test that comprehensive analysis plot uses VectorBT native functionality"""
        result = self.visualizer.plot_comprehensive_analysis(
            {"Test": self.portfolio},
            "VectorBT Test Strategy"
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        mock_show.assert_called()

    @patch('plotly.graph_objects.Figure.show')
    def test_vectorbt_portfolio_plotting(self, mock_show):
        """Test VectorBT native portfolio plotting functionality"""
        portfolios = {"Strategy": self.portfolio}

        # Test that VectorBT's native plot method is called
        with patch.object(self.portfolio, 'plot') as mock_portfolio_plot:
            mock_portfolio_plot.return_value = MagicMock()

            result = self.visualizer.plot_comprehensive_analysis(portfolios, "VectorBT Portfolio Test")
            mock_portfolio_plot.assert_called_once()
            self.assertTrue(result.get('success', False))

    @patch('plotly.graph_objects.Figure.show')
    def test_vectorbt_multiple_portfolios_analysis(self, mock_show):
        """Test VectorBT native multiple portfolio comparison"""
        # Create a second portfolio with different parameters
        portfolio2 = vbt.Portfolio.from_random_signals(
            self.data['close'],
            prob=0.05,  # Different probability for comparison
            freq='1D',
            seed=123
        )

        # Test with dict of portfolios
        portfolios = {
            "Strategy 1": self.portfolio,
            "Strategy 2": portfolio2
        }

        result = self.visualizer.plot_comprehensive_analysis(portfolios, "VectorBT Multiple Portfolios")
        self.assertTrue(result.get('success', False))

        # Verify show was called for plotting
        mock_show.assert_called()

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_error_handling(self, mock_stdout):
        """Test that plotting handles errors gracefully with VectorBT"""
        # Test with empty portfolios dict
        result = self.visualizer.plot_comprehensive_analysis({}, "Empty Portfolio Test")
        self.assertFalse(result.get('success', True))

        # Test with None input
        result = self.visualizer.plot_comprehensive_analysis(None, "None Portfolio Test")
        self.assertFalse(result.get('success', True))

    @patch('plotly.graph_objects.Figure.show')
    def test_vectorbt_monte_carlo_analysis(self, mock_show):
        """Test VectorBT native Monte Carlo analysis plotting"""
        # Create mock Monte Carlo results with rolling Sharpe data
        mc_results = {
            'analysis': True,
            'base_metrics': {
                'total_return': 15.5,
                'sharpe_ratio': 1.2
            },
            'simulations': [
                {
                    'metrics': {'return': 12.5, 'sharpe': 1.1},
                    'rolling_sharpe': np.random.normal(1.0, 0.2, 50)
                },
                {
                    'metrics': {'return': 15.5, 'sharpe': 1.2},
                    'rolling_sharpe': np.random.normal(1.2, 0.15, 50)
                },
                {
                    'metrics': {'return': 18.5, 'sharpe': 1.3},
                    'rolling_sharpe': np.random.normal(1.3, 0.1, 50)
                }
            ]
        }

        # Test VectorBT Monte Carlo plotting
        result = self.visualizer.plot_comprehensive_analysis(
            {"Test": self.portfolio},
            "VectorBT Monte Carlo Test",
            mc_results=mc_results
        )

        self.assertTrue(result.get('success', False))
        # Verify show was called for plotting
        mock_show.assert_called()

    @patch('plotly.graph_objects.Figure.show')
    def test_vectorbt_walkforward_analysis(self, mock_show):
        """Test VectorBT native Walk-forward analysis plotting"""
        # Create mock walk-forward results with enhanced data
        wf_results = {
            'windows': [
                {
                    'window': 1,
                    'train_metrics': {'total_return': 10.0, 'sharpe_ratio': 1.1},
                    'test_metrics': {'total_return': 8.0, 'sharpe_ratio': 0.9},
                    'asset_results': {
                        'AAPL': {'train_return': 12.0, 'test_return': 9.0},
                        'MSFT': {'train_return': 8.0, 'test_return': 7.0}
                    }
                },
                {
                    'window': 2,
                    'train_metrics': {'total_return': 12.0, 'sharpe_ratio': 1.2},
                    'test_metrics': {'total_return': 9.0, 'sharpe_ratio': 1.0},
                    'asset_results': {
                        'AAPL': {'train_return': 14.0, 'test_return': 11.0},
                        'MSFT': {'train_return': 10.0, 'test_return': 7.5}
                    }
                }
            ]
        }

        # Test VectorBT Walk-forward plotting
        result = self.visualizer.plot_comprehensive_analysis(
            {"Test": self.portfolio},
            "VectorBT Walk-forward Test",
            wf_results=wf_results
        )

        self.assertTrue(result.get('success', False))
        # Verify show was called for plotting
        mock_show.assert_called()

    @patch('plotly.graph_objects.Figure.show')
    def test_vectorbt_rolling_metrics(self, mock_show):
        """Test VectorBT native rolling metrics functionality"""
        # Test rolling metrics through portfolio plotting
        portfolios = {"Test": self.portfolio}
        result = self.visualizer.plot_comprehensive_analysis(portfolios, "Rolling Metrics Test")

        self.assertTrue(result.get('success', False))
        # Verify show was called for plotting
        mock_show.assert_called()

    @patch('plotly.graph_objects.Figure.show')
    def test_vectorbt_drawdown_analysis(self, mock_show):
        """Test VectorBT native drawdown analysis"""
        portfolios = {"Test": self.portfolio}

        # Test that drawdown plot method is called
        with patch.object(self.portfolio, 'plot_drawdowns') as mock_plot_drawdowns:
            mock_plot_drawdowns.return_value = MagicMock()

            result = self.visualizer.plot_comprehensive_analysis(portfolios, "Drawdown Test")
            mock_plot_drawdowns.assert_called_once()
            self.assertTrue(result.get('success', False))

    def test_vectorbt_settings_configuration(self):
        """Test that VectorBT settings are properly configured"""
        # Test that VectorBT is properly imported and configured
        self.assertTrue(hasattr(vbt, 'settings'))
        self.assertTrue(hasattr(vbt, 'plotting'))

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_portfolio_data_validation(self, mock_stdout):
        """Test that the visualizer handles invalid portfolio input gracefully"""
        # Test with invalid portfolio
        result = self.visualizer.plot_comprehensive_analysis(None, "Invalid Portfolio")
        self.assertFalse(result.get('success', True))

        # Test with empty dict
        result = self.visualizer.plot_comprehensive_analysis({}, "Empty Portfolio")
        self.assertFalse(result.get('success', True))

    def test_create_performance_plots_function(self):
        """Test the module-level create_performance_plots function"""
        portfolios = {"Test": self.portfolio}

        with patch.object(TradingVisualizer, 'plot_comprehensive_analysis') as mock_method:
            mock_method.return_value = {"success": True}

            result = create_performance_plots(portfolios, "Function Test")
            mock_method.assert_called_once_with(portfolios, "Function Test")
            self.assertTrue(result.get('success', False))


if __name__ == '__main__':
    unittest.main()
