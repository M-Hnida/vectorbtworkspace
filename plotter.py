"""
Visualization Module
Handles all plotting and visualization for trading strategy analysis.
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt
# Set VectorBT global settings for consistent plotting
vbt.settings.set_theme("dark")
# Note: VectorBT plotting settings are handled per-plot call

def create_performance_plots(portfolios: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """Create performance plots for multiple portfolios."""
    visualizer = TradingVisualizer()
    return visualizer.plot_comprehensive_analysis(portfolios, strategy_name)

class TradingVisualizer:
    """Streamlined visualization leveraging VectorBT's built-in plotting."""

    def __init__(self):
        self.dark_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

    def plot_comprehensive_analysis(self, portfolios, strategy_name: str = "Trading Strategy",
                                  mc_results: Optional[Dict[str, Any]] = None,
                                  wf_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Plot everything: portfolios, Monte Carlo, and walk-forward analysis."""

        try:
            # Handle single portfolio or dict of portfolios
            if portfolios is None:
                print("⚠️ No portfolios provided")
                return {"success": False, "error": "No portfolios provided"}

            # Convert single portfolio to dict format
            if not isinstance(portfolios, dict):
                portfolios = {"Portfolio": portfolios}

            # Check for empty portfolios dict
            if not portfolios:
                print("⚠️ Empty portfolios dictionary provided")
                return {"success": False, "error": "Empty portfolios dictionary"}

            # Plot portfolios
            self._plot_portfolios(portfolios, strategy_name)

            # Plot analysis results if available
            if mc_results and 'error' not in mc_results:
                print("🎲 Plotting Monte Carlo analysis...")
                self._plot_monte_carlo(mc_results)

            if wf_results and 'error' not in wf_results:
                print("🚶 Plotting walk-forward analysis...")
                self._plot_walkforward(wf_results)

            print("✅ Comprehensive analysis completed.")
            return {"success": True}

        except Exception as e:
            print(f"⚠️ Comprehensive analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def _plot_portfolios(self, portfolios: Dict[str, Any], strategy_name: str) -> None:
        """Plot individual portfolios and comparison using VectorBT native functionality."""
        print("📊 Creating portfolio visualizations...")

        # Plot each portfolio individually with VectorBT native methods
        for name, portfolio in portfolios.items():
            try:
                print(f"📈 Creating VectorBT native plot for {name}...")

                # Use VectorBT's comprehensive plotting with all subplots
                fig = portfolio.plot(
                    template='plotly_dark',
                    height=None,
                    width=None
                )
                fig.update_layout(
                    title=f"📊 {strategy_name} Strategy - {name} Performance"
                )
                fig.show()

                # VectorBT native rolling Sharpe using built-in methods
                self._plot_vectorbt_rolling_metrics(portfolio, f"{strategy_name} - {name}")

            except Exception as e:
                print(f"⚠️ Failed to plot {name}: {e}")
                continue

        # Create VectorBT native comparison plot if multiple portfolios
        if len(portfolios) > 1:
            self._create_vectorbt_comparison(portfolios, strategy_name)

    def _create_vectorbt_comparison(self, portfolios: Dict[str, Any], strategy_name: str):
        """Create VectorBT native comparison plot for multiple portfolios."""
        try:
            print("📊 Creating VectorBT native portfolio comparison...")

            portfolio_list = list(portfolios.values())


            if len(portfolio_list) > 1:
                # Use VectorBT's native multi-portfolio value plotting
                fig = go.Figure()

                for name, portfolio in portfolios.items():
                    value_series = portfolio.value()
                    fig.add_trace(go.Scatter(
                        x=value_series.index,
                        y=value_series.values,
                        mode='lines',
                        name=name,
                        line={"width": 2}
                    ))

                fig.update_layout(
                    title=f"📊 {strategy_name} Strategy - Portfolio Comparison",
                    template='plotly_dark',
                    height=None,
                    width=None
                )
                fig.show()

                # VectorBT native metrics comparison using stats()
                self._plot_vectorbt_metrics_comparison(portfolios, strategy_name)

        except Exception as e:
            print(f"⚠️ VectorBT comparison plot failed: {e}")



    def _plot_monte_carlo(self, mc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Plot Monte Carlo results with enhanced visualizations."""
        try:
            if 'analysis' not in mc_results:
                return {"success": False, "reason": "no_analysis"}

            base_metrics = mc_results.get('base_metrics', {})
            simulations = mc_results.get('simulations', [])

            if not simulations:
                return {"success": False, "reason": "no_simulations"}

            # Extract data for plotting
            returns_data = [sim['metrics']['return'] for sim in simulations]
            sharpes_data = [sim['metrics']['sharpe'] for sim in simulations]

            # Extract rolling Sharpe data if available
            rolling_sharpe_data = []
            for sim in simulations:
                if 'rolling_sharpe' in sim:
                    rolling_sharpe_data.append(sim['rolling_sharpe'])

            if not returns_data or not sharpes_data:
                return {"success": False, "reason": "no_data"}

            # Create distribution plots using standard Plotly
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Histogram(
                x=returns_data,
                nbinsx=30,
                name='Returns',
                marker_color=self.dark_colors[0],
                opacity=0.7
            ))

            # Add base return reference line
            base_return = base_metrics.get('total_return', 0)
            fig_returns.add_vline(
                x=base_return,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Base: {base_return:.2f}%"
            )
            fig_returns.update_layout(
                title="Return Distribution (%)",
                xaxis_title="Return (%)",
                template='plotly_dark',
                height=None,
                width=None
            )
            fig_returns.show()

            # Sharpe distribution plot
            fig_sharpe = go.Figure()
            fig_sharpe.add_trace(go.Histogram(
                x=sharpes_data,
                nbinsx=30,
                name='Sharpe',
                marker_color=self.dark_colors[1],
                opacity=0.7
            ))

            # Add base Sharpe reference line
            base_sharpe = base_metrics.get('sharpe_ratio', 0)
            fig_sharpe.add_vline(
                x=base_sharpe,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Base: {base_sharpe:.3f}"
            )
            fig_sharpe.update_layout(
                title="Sharpe Ratio Distribution",
                xaxis_title="Sharpe Ratio",
                template='plotly_dark',
                height=None,
                width=None
            )
            fig_sharpe.show()

            # Enhanced rolling Sharpe analysis using VectorBT native plotting
            if rolling_sharpe_data:
                # Convert rolling data to VectorBT format for efficient plotting
                sample_size = min(10, len(rolling_sharpe_data))
                rolling_series_list = []

                for i in range(sample_size):
                    rolling_data = rolling_sharpe_data[i]
                    if len(rolling_data) > 0:
                        rolling_series_list.append(pd.Series(rolling_data, name=f'Sim {i+1}'))

                if rolling_series_list:
                    # Plot rolling Sharpe evolution
                    fig_rolling = go.Figure()
                    for i, series in enumerate(rolling_series_list[:5]):  # Show first 5 for clarity
                        fig_rolling.add_trace(go.Scatter(
                            y=series.values,
                            mode='lines',
                            name=series.name,
                            line={"width": 1, "color": self.dark_colors[i % len(self.dark_colors)]},
                            opacity=0.6
                        ))

                    fig_rolling.update_layout(
                        title="Rolling Sharpe Evolution - Monte Carlo Simulations",
                        yaxis_title="Rolling Sharpe Ratio",
                        template='plotly_dark',
                        height=None,
                        width=None
                    )
                    fig_rolling.show()

                # Stability analysis
                sharpe_stds = [np.std(rolling_data) if len(rolling_data) > 1 else 0
                              for rolling_data in rolling_sharpe_data]

                if sharpe_stds:
                    fig_stability = go.Figure()
                    fig_stability.add_trace(go.Histogram(
                        x=sharpe_stds,
                        nbinsx=20,
                        name='Sharpe Volatility',
                        marker_color=self.dark_colors[3],
                        opacity=0.7
                    ))
                    fig_stability.update_layout(
                        title="Sharpe Ratio Stability Distribution",
                        xaxis_title="Sharpe Volatility",
                        template='plotly_dark',
                        height=None,
                        width=None
                    )
                    fig_stability.show()

            return {"success": True}

        except Exception as e:
            print(f"⚠️ Monte Carlo plot failed: {e}")
            return {"success": False, "error": str(e)}

    def _plot_walkforward(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Plot enhanced walk-forward analysis results with multiple asset support."""
        try:
            if 'windows' not in wf_results:
                return {"success": False, "reason": "no_windows"}

            windows = wf_results['windows']
            if not windows:
                return {"success": False, "reason": "empty_windows"}

            # Check if we have multiple assets
            has_multiple_assets = any('asset_results' in w for w in windows)

            if has_multiple_assets:
                return self._plot_multi_asset_walkforward(wf_results)
            return self._plot_single_asset_walkforward(wf_results)

        except Exception as e:
            print(f"⚠️ Walk-forward plot failed: {e}")
            return {"success": False, "error": str(e)}

    def _plot_single_asset_walkforward(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Plot walk-forward analysis for single asset."""
        windows = wf_results['windows']

        # Extract data from windows
        window_nums = [w['window'] for w in windows]
        train_returns = [w['train_metrics'].get('total_return', 0) for w in windows]
        test_returns = [w['test_metrics'].get('total_return', 0) for w in windows]
        train_sharpes = [w['train_metrics'].get('sharpe_ratio', 0) for w in windows]
        test_sharpes = [w['test_metrics'].get('sharpe_ratio', 0) for w in windows]

        # Extract rolling Sharpe if available
        rolling_sharpe_train = []
        rolling_sharpe_test = []
        for w in windows:
            if 'rolling_sharpe_train' in w:
                rolling_sharpe_train.extend(w['rolling_sharpe_train'])
            if 'rolling_sharpe_test' in w:
                rolling_sharpe_test.extend(w['rolling_sharpe_test'])

        # Create enhanced subplots
        subplot_titles = [
            'Returns by Window (%)', 'Sharpe Ratio by Window',
            'Train vs Test Returns', 'Train vs Test Sharpe'
        ]

        if rolling_sharpe_train or rolling_sharpe_test:
            subplot_titles.extend(['Rolling Sharpe Evolution', 'Performance Degradation'])
            rows, cols = 3, 2
        else:
            rows, cols = 2, 2

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles
        )

        # Plot returns and Sharpe ratios over windows
        fig.add_trace(
            go.Scatter(
                x=window_nums, y=train_returns,
                mode='lines+markers', name='Train Returns',
                line={"color": self.dark_colors[0], "width": 3}
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=window_nums, y=test_returns,
                mode='lines+markers', name='Test Returns',
                line={"color": self.dark_colors[1], "width": 3}
            ), row=1, col=1
        )

        # Sharpe over windows
        fig.add_trace(
            go.Scatter(
                x=window_nums, y=train_sharpes,
                mode='lines+markers', name='Train Sharpe',
                line={"color": self.dark_colors[0], "width": 3},
                showlegend=False
            ), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=window_nums, y=test_sharpes,
                mode='lines+markers', name='Test Sharpe',
                line={"color": self.dark_colors[1], "width": 3},
                showlegend=False
            ), row=1, col=2
        )

        # Correlation scatter plots
        fig.add_trace(
            go.Scatter(
                x=train_returns, y=test_returns,
                mode='markers', name='Returns Correlation',
                marker={"color": self.dark_colors[2], "size": 10},
                showlegend=False
            ), row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=train_sharpes, y=test_sharpes,
                mode='markers', name='Sharpe Correlation',
                marker={"color": self.dark_colors[3], "size": 10},
                showlegend=False
            ), row=2, col=2
        )

        # Add diagonal reference lines
        self._add_diagonal_lines(fig, train_returns, test_returns, 2, 1)
        self._add_diagonal_lines(fig, train_sharpes, test_sharpes, 2, 2)

        # Add rolling Sharpe plots if available
        if (rolling_sharpe_train or rolling_sharpe_test) and rows > 2:
            if rolling_sharpe_train:
                fig.add_trace(
                    go.Scatter(
                        y=rolling_sharpe_train,
                        mode='lines', name='Rolling Sharpe (Train)',
                        line={"color": self.dark_colors[0], "width": 2}
                    ), row=3, col=1
                )
            if rolling_sharpe_test:
                fig.add_trace(
                    go.Scatter(
                        y=rolling_sharpe_test,
                        mode='lines', name='Rolling Sharpe (Test)',
                        line={"color": self.dark_colors[1], "width": 2}
                    ), row=3, col=1
                )

            # Performance degradation analysis
            degradation = [test - train for train, test in zip(train_returns, test_returns)]
            fig.add_trace(
                go.Scatter(
                    x=window_nums, y=degradation,
                    mode='lines+markers', name='Performance Degradation',
                    line={"color": self.dark_colors[4], "width": 2}
                ), row=3, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=2)

        fig.update_layout(
            height=None,
            width=None,
            title_text="🚶 Walk-Forward Analysis - VectorBT Enhanced Performance Stability",
            template='plotly_dark',
            showlegend=True
        )
        fig.show()

        return {"success": True}

    def _plot_multi_asset_walkforward(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Plot walk-forward analysis for multiple assets."""
        windows = wf_results['windows']

        # Extract asset names from first window
        first_window = windows[0]
        asset_names = list(first_window.get('asset_results', {}).keys())

        if not asset_names:
            return self._plot_single_asset_walkforward(wf_results)

        # Create comparison plots for multiple assets
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Train Returns by Asset (%)',
                'Test Returns by Asset (%)',
                'Train vs Test Comparison',
                'Asset Performance Ranking'
            )
        )

        window_nums = [w['window'] for w in windows]

        # Plot each asset's performance
        for i, asset in enumerate(asset_names):
            color = self.dark_colors[i % len(self.dark_colors)]

            # Extract asset-specific data
            train_returns = []
            test_returns = []
            for w in windows:
                asset_data = w.get('asset_results', {}).get(asset, {})
                train_returns.append(asset_data.get('train_return', 0))
                test_returns.append(asset_data.get('test_return', 0))

            # Train returns
            fig.add_trace(
                go.Scatter(
                    x=window_nums, y=train_returns,
                    mode='lines+markers', name=f'{asset} Train',
                    line={"color": color, "width": 2}
                ), row=1, col=1
            )

            # Test returns
            fig.add_trace(
                go.Scatter(
                    x=window_nums, y=test_returns,
                    mode='lines+markers', name=f'{asset} Test',
                    line={"color": color, "width": 2, "dash": "dash"}
                ), row=1, col=2
            )

            # Train vs Test scatter
            fig.add_trace(
                go.Scatter(
                    x=train_returns, y=test_returns,
                    mode='markers', name=f'{asset}',
                    marker={"color": color, "size": 8},
                    showlegend=False
                ), row=2, col=1
            )

        # Asset ranking analysis
        final_window = windows[-1]
        asset_performance = []
        for asset in asset_names:
            asset_data = final_window.get('asset_results', {}).get(asset, {})
            test_return = asset_data.get('test_return', 0)
            asset_performance.append((asset, test_return))

        asset_performance.sort(key=lambda x: x[1], reverse=True)
        assets_sorted, returns_sorted = zip(*asset_performance)

        fig.add_trace(
            go.Bar(
                x=list(assets_sorted), y=list(returns_sorted),
                name='Final Test Returns',
                marker_color=self.dark_colors[:len(assets_sorted)]
            ), row=2, col=2
        )

        fig.update_layout(
            height=None,
            width=None,
            title_text="🚶 Multi-Asset Walk-Forward Analysis - VectorBT Enhanced",
            template='plotly_dark',
            showlegend=True
        )
        fig.show()

        return {"success": True}

    def _add_diagonal_lines(self, fig, x_data, y_data, row, col):
        """Add diagonal reference lines for correlation plots."""
        if x_data and y_data:
            all_values = list(x_data) + list(y_data)
            min_val, max_val = min(all_values), max(all_values)
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    line={"dash": "dash", "color": "gray", "width": 1},
                    showlegend=False
                ), row=row, col=col
            )

    def _plot_vectorbt_rolling_metrics(self, portfolio, title: str) -> None:
        """Plot rolling metrics using VectorBT native functionality."""
        try:
            print(f"📈 Creating VectorBT rolling metrics for {title}...")

            # Use VectorBT's native rolling metrics calculation
            returns = portfolio.returns()
            window = 60  # Single 60-day rolling window

            # Create simple rolling Sharpe plot with one line
            if len(returns) > window:
                rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode='lines',
                    name='Rolling Sharpe (60D)',
                    line={"color": self.dark_colors[0], "width": 2}
                ))

                fig.update_layout(
                    title=f"📊 {title} - Rolling Sharpe Ratio",
                    yaxis_title="Sharpe Ratio",
                    template='plotly_dark',
                    height=None,
                    width=None
                )
                fig.show()

            # VectorBT native drawdown plot
            fig_dd = portfolio.plot_drawdowns(
                template='plotly_dark',
                height=None,
                width=None
            )
            fig_dd.update_layout(title=f"📊 {title} - Drawdown Analysis")
            fig_dd.show()

        except Exception as e:
            print(f"⚠️ VectorBT rolling metrics plot failed: {e}")

    def _plot_vectorbt_metrics_comparison(self, portfolios: Dict[str, Any], strategy_name: str) -> None:
        """Plot metrics comparison using VectorBT native functionality."""
        try:
            print("📊 Creating VectorBT metrics comparison...")

            # Use VectorBT's optimized stats extraction
            stats_list = [p.stats() for p in portfolios.values()]
            stats_df = pd.DataFrame(stats_list, index=portfolios.keys())

            # Select key metrics for comparison
            key_metrics = ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]']
            available_metrics = [m for m in key_metrics if m in stats_df.columns]

            if available_metrics and len(portfolios) > 1:
                # Use pandas plotting with VectorBT styling
                comparison_df = stats_df[available_metrics]

                # Create bar plot using pandas native plotting
                fig = comparison_df.plot.bar(
                    title=f"📊 {strategy_name} - Portfolio Metrics Comparison",
                    template='plotly_dark',
                    height=None,
                    width=None
                )
                fig.update_layout(
                    xaxis_title="Portfolio",
                    yaxis_title="Metric Value",
                    showlegend=True
                )
                fig.show()

        except Exception as e:
            print(f"⚠️ VectorBT metrics comparison failed: {e}")

    # Performance summary table functionality removed
