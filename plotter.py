"""
Visualization Module
Handles all plotting and visualization for trading strategy analysis.
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt
# Set VectorBT dark theme
vbt.settings.set_theme("dark")

def create_performance_plots(portfolios: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """Create performance plots for multiple portfolios."""
    visualizer = TradingVisualizer()
    return visualizer.create_multi_portfolio_dashboard(portfolios, strategy_name)

class TradingVisualizer:
    """Streamlined visualization leveraging VectorBT's built-in plotting."""

    def __init__(self):
        self.dark_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

    def plot_comprehensive_analysis(self, portfolios, strategy_name: str = "Trading Strategy",
                                  mc_results: Optional[Dict[str, Any]] = None,
                                  wf_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Plot everything: portfolios, Monte Carlo, and walk-forward analysis."""

        results = {"success": True, "plots_created": []}

        try:
            # Handle single portfolio or dict of portfolios
            if portfolios is None:
                print("⚠️ No portfolios provided")
                return {"success": False, "error": "No portfolios provided"}

            # Convert single portfolio to dict format
            if not isinstance(portfolios, dict):
                portfolios = {"Portfolio": portfolios}

            # 1. Main portfolio analysis
            if portfolios:
                if len(portfolios) == 1:
                    # Single portfolio - show detailed analysis
                    print("📊 Plotting single portfolio detailed analysis...")
                    portfolio_name, portfolio = next(iter(portfolios.items()))
                    self.plot_portfolio(portfolio, f"{strategy_name} - {portfolio_name}")
                    results["plots_created"].append("detailed_analysis")
                else:
                    # Multiple portfolios - show comparison dashboard
                    print("📊 Plotting multi-portfolio comparison...")
                    self.create_multi_portfolio_dashboard(portfolios, strategy_name)
                    results["plots_created"].append("portfolio_dashboard")

            # 2. Monte Carlo analysis
            if mc_results and 'error' not in mc_results:
                print("🎲 Plotting Monte Carlo analysis...")
                mc_result = self._plot_monte_carlo(mc_results)
                if mc_result.get("success"):
                    results["plots_created"].append("monte_carlo")

            # 3. Walk-forward analysis
            if wf_results and 'error' not in wf_results:
                print("🚶 Plotting walk-forward analysis...")
                wf_result = self._plot_walkforward(wf_results)
                if wf_result.get("success"):
                    results["plots_created"].append("walk_forward")

            # No summary statistics - removed for simplification

            print(f"✅ Comprehensive analysis completed. Created {len(results['plots_created'])} plots.")
            return results

        except Exception as e:
            print(f"⚠️ Comprehensive analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def create_multi_portfolio_dashboard(self, portfolios: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
        """Create dashboard for multiple portfolios."""
        print("📊 Creating multi-portfolio visualization dashboard...")

        try:
            if not portfolios:
                print("⚠️ No portfolios provided for visualization")
                return {"error": "No portfolios"}

            # Plot each portfolio
            for name, portfolio in portfolios.items():
                try:
                    print(f"📈 Creating plot for {name}...")
                    fig = portfolio.plot()
                    fig.update_layout(
                        title=f"📊 {strategy_name} Strategy - {name} Performance",
                        height=None,
                        width=None,
                        showlegend=True,
                        template='plotly_dark'
                    )

                    fig.show()

                except Exception as e:
                    print(f"⚠️ Failed to plot {name}: {e}")
                    continue

            # Create comparison plot if multiple portfolios
            if len(portfolios) > 1:
                self._create_comparison_plot(portfolios, strategy_name)

            print("✅ Dashboard creation completed successfully")
            return {"success": True}

        except Exception as e:
            print(f"⚠️ Dashboard creation failed: {e}")
            return {"error": str(e)}

    def _create_comparison_plot(self, portfolios: Dict[str, Any], strategy_name: str):
        """Create comparison plot for multiple portfolios."""
        try:
            print("📊 Creating portfolio comparison plot...")

            fig = go.Figure()

            for name, portfolio in portfolios.items():
                try:
                    equity = portfolio.value()
                    fig.add_trace(go.Scatter(
                        x=equity.index,
                        y=equity.values,
                        mode='lines',
                        name=name,
                        line={"width": 2}
                    ))
                except Exception as e:
                    print(f"⚠️ Failed to add {name} to comparison: {e}")
                    continue

            fig.update_layout(
                title=f"📊 {strategy_name} Strategy - Portfolio Comparison",
                xaxis_title="Date",
                yaxis_title="Portfolio Value",
                height=None,
                width=None,
                template='plotly_dark',
                showlegend=True
            )

            fig.show()

        except Exception as e:
            print(f"⚠️ Comparison plot failed: {e}")

    def plot_portfolio(self, portfolio, title: str) -> Dict[str, Any]:
        """Plot detailed portfolio analysis with comprehensive stats."""
        try:
            print(f"📈 Creating comprehensive analysis for {title}...")

            # Main portfolio plot with all subplots
            fig = portfolio.plot()
            fig.update_layout(
                title=f"📊 {title} - Complete Performance Analysis",
                height=None,
                width=None,
                template='plotly_dark',
                showlegend=True
            )
            fig.show()

            return {"success": True}

        except Exception as e:
            print(f"⚠️ Detailed portfolio plot failed: {e}")
            return {"success": False, "error": str(e)}

    def _plot_monte_carlo(self, mc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Plot Monte Carlo results."""
        try:
            if 'analysis' not in mc_results:
                return {"success": False, "reason": "no_analysis"}

            base_metrics = mc_results.get('base_metrics', {})

            returns_data = [sim['metrics']['return'] for sim in mc_results.get('simulations', [])]
            sharpes_data = [sim['metrics']['sharpe'] for sim in mc_results.get('simulations', [])]

            if not returns_data or not sharpes_data:
                return {"success": False, "reason": "no_data"}

            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Return Distribution (%)', 'Sharpe Ratio Distribution')
            )

            # Add vertical line for base return
            base_return = base_metrics.get('total_return', 0)
            fig.add_vline(
                x=base_return,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Base: {base_return:.2f}%",
                col=1
            )

            # Sharpe histogram
            fig.add_trace(
                go.Histogram(
                    x=sharpes_data,
                    nbinsx=20,
                    name='Sharpe',
                    marker_color=self.dark_colors[1],
                    opacity=0.7
                ),
                row=1, col=2
            )

            # Add vertical line for base sharpe
            base_sharpe = base_metrics.get('sharpe_ratio', 0)
            fig.add_vline(
                x=base_sharpe,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Base: {base_sharpe:.3f}",
                col=2
            )

            fig.update_layout(
                height=None,
                width=None,
                title_text="🎲 Monte Carlo Analysis Results",
                template='plotly_dark',
                showlegend=True
            )
            fig.show()

            return {"success": True}

        except Exception as e:
            print(f"⚠️ Monte Carlo plot failed: {e}")
            return {"success": False, "error": str(e)}

    def _plot_walkforward(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Plot walk-forward analysis results."""
        try:
            if 'windows' not in wf_results:
                return {"success": False, "reason": "no_windows"}

            windows = wf_results['windows']
            if not windows:
                return {"success": False, "reason": "empty_windows"}

            # Extract data from windows
            window_nums = [w['window'] for w in windows]
            train_returns = [w['train_metrics'].get('total_return', 0) for w in windows]
            test_returns = [w['test_metrics'].get('total_return', 0) for w in windows]
            train_sharpes = [w['train_metrics'].get('sharpe_ratio', 0) for w in windows]
            test_sharpes = [w['test_metrics'].get('sharpe_ratio', 0) for w in windows]

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Returns by Window (%)',
                    'Sharpe Ratio by Window',
                    'Train vs Test Returns',
                    'Train vs Test Sharpe'
                )
            )

            # Returns over windows
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=train_returns,
                    mode='lines+markers',
                    name='Train Returns',
                    line={"color": self.dark_colors[0]}
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=test_returns,
                    mode='lines+markers',
                    name='Test Returns',
                    line={"color": self.dark_colors[1]}
                ),
                row=1, col=1
            )

            # Sharpe over windows
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=train_sharpes,
                    mode='lines+markers',
                    name='Train Sharpe',
                    line={"color": self.dark_colors[0]},
                    showlegend=False
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=window_nums,
                    y=test_sharpes,
                    mode='lines+markers',
                    name='Test Sharpe',
                    line={"color": self.dark_colors[1]},
                    showlegend=False
                ),
                row=1, col=2
            )

            # Scatter plots for correlation
            fig.add_trace(
                go.Scatter(
                    x=train_returns,
                    y=test_returns,
                    mode='markers',
                    name='Returns Correlation',
                    marker={"color": self.dark_colors[2], "size": 8},
                    showlegend=False
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=train_sharpes,
                    y=test_sharpes,
                    mode='markers',
                    name='Sharpe Correlation',
                    marker={"color": self.dark_colors[3], "size": 8},
                    showlegend=False
                ),
                row=2, col=2
            )

            # Add diagonal lines for perfect correlation
            if train_returns and test_returns:
                min_ret, max_ret = min(train_returns, test_returns), max(train_returns, test_returns)
                fig.add_trace(
                    go.Scatter(
                        x=[min_ret, max_ret],
                        y=[min_ret, max_ret],
                        mode='lines',
                        line={"dash": "dash", "color": "gray"},
                        showlegend=False
                    ),
                    row=2, col=1
                )

            if train_sharpes and test_sharpes:
                min_sharpe, max_sharpe = min(train_sharpes, test_sharpes), max(train_sharpes, test_sharpes)
                fig.add_trace(
                    go.Scatter(
                        x=[min_sharpe, max_sharpe],
                        y=[min_sharpe, max_sharpe],
                        mode='lines',
                        line={"dash": "dash", "color": "gray"},
                        showlegend=False
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                height=None,
                width=None,
                title_text="🚶 Walk-Forward Analysis - Performance Stability",
                template='plotly_dark',
                showlegend=True
            )
            fig.show()

            return {"success": True}

        except Exception as e:
            print(f"⚠️ Walk-forward plot failed: {e}")
            return {"success": False, "error": str(e)}

    # Performance summary table functionality removed
