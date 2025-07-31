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

# Configure VectorBT global settings for consistent plotting
vbt.settings.set_theme("dark")
vbt.settings['plotting']['layout']["template"] = "plotly_dark"
# Don't set height/width to None - this causes addition error when redefined later
# vbt.settings['plotting']['layout']['height'] = None
# vbt.settings['plotting']['layout']['width'] = None


def create_performance_plots(portfolios: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """Create performance plots for multiple portfolios."""
    return plot_comprehensive_analysis(portfolios, strategy_name)


def plot_comprehensive_analysis(portfolios, strategy_name: str = "Trading Strategy",
                                mc_results: Optional[Dict[str, Any]] = None,
                                wf_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Plot everything: portfolios, Monte Carlo, and walk-forward analysis."""
    try:
        if not portfolios:
            print("⚠️ No portfolios provided")
            return {"success": False, "error": "No portfolios provided"}

        if not isinstance(portfolios, dict):
            portfolios = {"Portfolio": portfolios}

        _plot_portfolios(portfolios, strategy_name)

        if mc_results and 'error' not in mc_results:
            print("🎲 Plotting Monte Carlo analysis...")
            _plot_monte_carlo(mc_results)

        if wf_results and 'error' not in wf_results:
            print("🚶 Plotting walk-forward analysis...")
            _plot_walkforward(wf_results)

        print("✅ Comprehensive analysis completed.")
        return {"success": True}

    except Exception as e:
        print(f"⚠️ Comprehensive analysis failed: {e}")
        return {"success": False, "error": str(e)}

def _plot_portfolios(portfolios: Dict[str, Any], strategy_name: str) -> None:
    """Plot individual portfolios and comparison using VectorBT native functionality."""
    print("📊 Creating portfolio visualizations...")

    # Plot each portfolio individually
    for name, portfolio in portfolios.items():
        if not _validate_portfolio(portfolio, name):
            continue
            
        try:
            _plot_single_portfolio(portfolio, name, strategy_name)
        except Exception as e:
            print(f"⚠️ Failed to plot {name}: {e}")
            continue

    # Create comparison plot if multiple portfolios
    if len(portfolios) > 1:
        _create_vectorbt_comparison(portfolios, strategy_name)


def _validate_portfolio(portfolio: Any, name: str) -> bool:
    """Validate portfolio before plotting."""
    if portfolio is None:
        print(f"⚠️ Skipping {name}: portfolio is None")
        return False
        
    try:
        stats = portfolio.stats()
        if stats is None or len(stats) == 0 or not stats.get('Total Trades', 0):
            print(f"⚠️ Skipping {name}: no trades")
            return False

        value_series = portfolio.value()
        if value_series is None or len(value_series) == 0 or value_series.isna().all():
            print(f"⚠️ Skipping {name}: no valid value data")
            return False

        if portfolio.orders is None:
            print(f"⚠️ Portfolio orders is None for {name}")
            return False

        return True
        
    except Exception as e:
        print(f"⚠️ Portfolio validation failed for {name}: {e}")
        return False


def _plot_single_portfolio(portfolio: Any, name: str, strategy_name: str) -> None:
    """Plot a single portfolio."""
    print(f"📈 Creating VectorBT plot for {name}...")
    print(f"✅ Portfolio validation passed for {name}")
    
    try:
        fig = portfolio.plot(template='plotly_dark')
        fig.update_layout(
            title=f"📊 {strategy_name} Strategy - {name} Performance",
            height=600,
            width=1200
        )
        fig.show()
        
    except Exception as plot_error:
        print(f"⚠️ VectorBT plot failed for {name}: {plot_error}")
        print(f"   Portfolio type: {type(portfolio)}")
        print(f"   Portfolio stats available: {hasattr(portfolio, 'stats')}")
        
def _create_vectorbt_comparison(portfolios: Dict[str, Any], strategy_name: str):
    """Create VectorBT native comparison plot for multiple portfolios."""
    try:
        print("📊 Creating portfolio comparison...")
        if len(portfolios) <= 1:
            return

        fig = go.Figure()
        
        for name, portfolio in portfolios.items():
            try:
                if portfolio is None:
                    continue

                value_series = portfolio.value()
                if value_series is None or len(value_series) == 0:
                    continue

                first_value = value_series.iloc[0]
                if pd.isna(first_value) or first_value == 0:
                    continue
                    
                normalized_values = (value_series / first_value) * 100
                fig.add_trace(go.Scatter(
                    x=normalized_values.index, y=normalized_values.values,
                    mode='lines', name=name, line={"width": 3}
                ))
            except Exception as e:
                print(f"⚠️ Failed to add {name} to comparison: {e}")

        fig.update_layout(
            title=f"📊 {strategy_name} Strategy - Portfolio Comparison (Normalized)",
            yaxis_title="Normalized Value (Start = 100)", xaxis_title="Date",
            template='plotly_dark', height=600, width=1200
        )
        fig.show()
    

    except Exception as e:
        print(f"⚠️ VectorBT comparison plot failed: {e}")

def _plot_monte_carlo(mc_results: Dict[str, Any]) -> Dict[str, Any]:
    """Plot Monte Carlo results with parameter sensitivity analysis."""
    try:
        simulations = mc_results.get('simulations', [])
        statistics = mc_results.get('statistics', {})

        if not simulations:
            return {"success": False, "reason": "no_simulations"}

        returns_data = [sim['total_return'] for sim in simulations]
        if not returns_data:
            return {"success": False, "reason": "no_data"}

        # Create the subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Monte Carlo Return Distribution',
                'Parameter Sensitivity Analysis', 
                'Percentile Analysis',
                'Performance vs Random'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Add all subplot components
        _add_mc_histogram(fig, returns_data, statistics)
        _add_parameter_sensitivity(fig, simulations)
        _add_mc_percentiles(fig, returns_data, statistics)
        _add_mc_comparison(fig, statistics)

        # Configure layout and show
        fig.update_layout(
            title="Monte Carlo Analysis - Parameter Sensitivity",
            template='plotly_dark',
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        _update_mc_axes(fig)
        fig.show()

        _print_mc_summary(statistics)
        return {"success": True}

    except Exception as e:
        print(f"⚠️ Monte Carlo plot failed: {e}")
        return {"success": False, "error": str(e)}


def _add_mc_histogram(fig: go.Figure, returns_data: list, statistics: dict) -> None:
    """Add clean histogram """
    if not returns_data:
        return

    # Debug: Print data characteristics
    print(f"   Debug - Returns data: {len(returns_data)} simulations")
    print(f"   Debug - Returns range: [{min(returns_data):.3f}%, {max(returns_data):.3f}%]")
    print(f"   Debug - Returns sample: {returns_data[:5]}")

    fig.add_trace(go.Histogram(
        x=returns_data, nbinsx=50, name='Random Returns',
        opacity=0.7, marker_color='lightblue', showlegend=True
    ), row=1, col=1)

    actual_return = statistics.get('actual_return')
    if actual_return is not None:
        # Get the maximum frequency for proper scaling
        hist_counts, _ = np.histogram(returns_data, bins=50)
        max_count = max(hist_counts) if len(hist_counts) > 0 else 1

        fig.add_shape(
            type="line", x0=actual_return, x1=actual_return, y0=0, y1=max_count,
            line={"dash": "dash", "color": "red", "width": 3},
            xref="x1", yref="y1"
        )
        fig.add_annotation(
            x=actual_return, y=max_count * 0.9, text=f"Strategy: {actual_return:.2f}%",
            showarrow=True, arrowcolor="red", xref="x1", yref="y1"
        )

    # REMOVED: Confusing 90% confidence box overlay


def _add_parameter_sensitivity(fig: go.Figure, simulations: list) -> None:
    """Add parameter sensitivity analysis subplot."""
    if not simulations:
        return

    # Extract parameter values and returns
    param1_values = [sim.get('param1', 0) for sim in simulations]
    param2_values = [sim.get('param2', 0) for sim in simulations]
    returns = [sim['total_return'] for sim in simulations]
    
    # Debug: Print data characteristics
    print(f"   Debug - Param1 values: {param1_values}")
    print(f"   Debug - Param2 values: {param2_values}")
    print(f"   Debug - Returns: {returns}")

    # Check if we have valid parameter data
    if all(p == 0 or pd.isna(p) for p in param1_values) or all(p == 0 or pd.isna(p) for p in param2_values):
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='text',
            text=["No valid parameter data available", "for sensitivity analysis"],
            textposition="middle center",
            showlegend=False
        ), row=1, col=2)
        return

    # Validate that param1 and param2 have non-zero and non-NaN values
    valid_simulations = [
        (param1, param2, ret) 
        for param1, param2, ret in zip(param1_values, param2_values, returns) 
        if param1 != 0 and not pd.isna(param1) and param2 != 0 and not pd.isna(param2)
    ]

    if not valid_simulations:
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='text',
            text=["No valid parameter data available", "for sensitivity analysis"],
            textposition="middle center",
            showlegend=False
        ), row=1, col=2)
        return

    param1_values, param2_values, returns = zip(*valid_simulations)

    # Create scatter plot with color-coded returns
    fig.add_trace(go.Scatter(
        x=param1_values, y=param2_values,
        mode='markers',
        marker=dict(
            size=8,
            color=returns,
            colorscale='Viridis',
            colorbar=dict(title="Return (%)", x=0.95, y=0.5),
            showscale=True
        ),
        text=[f"Return: {r:.2f}%" for r in returns],
        hovertemplate="<b>Param 1:</b> %{x}<br>" +
                      "<b>Param 2:</b> %{y}<br>" +
                      "<b>Return:</b> %{marker.color:.2f}%<br>" +
                      "<extra></extra>",
        showlegend=False,
        name='Parameter Combinations'
    ), row=1, col=2)

    # Update axes labels
    fig.update_xaxes(title_text="Parameter 1", row=1, col=2)
    fig.update_yaxes(title_text="Parameter 2", row=1, col=2)


def _add_mc_percentiles(fig: go.Figure, returns_data: list, statistics: dict) -> None:
    """Add clear percentile analysis - shows how strategy ranks vs random."""
    if not returns_data:
        return

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = [np.percentile(returns_data, p) for p in percentiles]

    # Debug: Print percentile data
    print(f"   Debug - Percentiles: {list(zip(percentiles, percentile_values))}")
    print(f"   Debug - Percentile range: [{min(percentile_values):.3f}%, {max(percentile_values):.3f}%]")

    fig.add_trace(go.Scatter(
        x=percentiles, y=percentile_values, mode='lines+markers',
        name='Random Performance Curve', line={'color': 'cyan', 'width': 3},
        marker={'size': 8}, showlegend=True
    ), row=2, col=1)

    # Add strategy performance point and line
    actual_return = statistics.get('actual_return')
    if actual_return is not None:
        # Calculate the actual percentile rank of the strategy
        percentile_rank = statistics.get('percentile_rank', 50)
        beats_percent = percentile_rank  # percentile_rank IS the % of strategies beaten

        # Add strategy point on the curve
        fig.add_trace(go.Scatter(
            x=[percentile_rank], y=[actual_return], mode='markers',
            name='Your Strategy', marker={'color': 'red', 'size': 12, 'symbol': 'diamond'},
            showlegend=True
        ), row=2, col=1)

        # Add horizontal reference line
        fig.add_shape(
            type="line", x0=0, x1=100, y0=actual_return, y1=actual_return,
            line={"dash": "dash", "color": "rgba(255,0,0,0.5)", "width": 2},
            xref="x3", yref="y3"
        )

        # Clear explanation of what percentile means
        explanation = f'Your Strategy: {actual_return:.2f}%<br>Beats {beats_percent:.1f}% of random strategies'
        fig.add_annotation(
            x=percentile_rank + 5, y=actual_return, text=explanation,
            showarrow=True, arrowcolor="red", xref="x3", yref="y3",
            bgcolor="red", bordercolor="white", borderwidth=1,
            font=dict(color="white", size=10)
        )


def _add_mc_comparison(fig: go.Figure, statistics: dict) -> None:
    """Add performance comparison subplot to Monte Carlo plot."""
    mean_random = statistics.get('mean_return', 0)
    std_random = statistics.get('std_return', 1)
    actual_return = statistics.get('actual_return', 0)

    # Debug print to check values
    print(f"   Debug - Mean Random: {mean_random:.3f}%, Std: {std_random:.3f}%, Strategy: {actual_return:.3f}%")

    # Create comparison data with proper labels
    categories = ['Random\nMean', 'Strategy', 'Random\n+1σ', 'Random\n-1σ']
    values = [mean_random, actual_return, mean_random + std_random, mean_random - std_random]
    colors = ['lightgray', 'red', 'lightgreen', 'orange']

    # Add bars with better formatting
    for i, (category, val, color) in enumerate(zip(categories, values, colors)):
        showlegend = i < 2  # Only show legend for main items
        fig.add_trace(go.Bar(
            x=[category], y=[val], marker_color=color,
            name=category.replace('\n', ' ') if showlegend else None,
            showlegend=showlegend,
            text=f'{val:.2f}%',
            textposition='outside'
        ), row=2, col=2)

    # Add zero reference line
    fig.add_shape(
        type="line", x0=-0.5, x1=3.5, y0=0, y1=0,
        line={"dash": "dot", "color": "white", "width": 1},
        xref="x4", yref="y4"
    )

    # Add performance interpretation
    performance_text = "Outperforming" if actual_return > mean_random else "Underperforming"
    fig.add_annotation(
        x=1.5, y=max(values) * 0.8, text=f"Strategy is {performance_text} vs Random",
        showarrow=False, xref="x4", yref="y4", font=dict(size=12, color="white")
    )


def _update_mc_axes(fig: go.Figure) -> None:
    """Update axis labels for Monte Carlo plot."""
    axis_updates = [
        (1, 1, "Return (%)", "Frequency"),
        (1, 2, "Parameter 1", "Parameter 2"),
        (2, 1, "Percentile", "Return (%)"),
        (2, 2, "Category", "Return (%)")
    ]
    
    for row, col, x_title, y_title in axis_updates:
        fig.update_xaxes(title_text=x_title, row=row, col=col)
        fig.update_yaxes(title_text=y_title, row=row, col=col)
    
    # Set specific axis ranges for better visualization
    fig.update_yaxes(range=[0, 100], row=1, col=2)  # Percentile rank 0-100%
    fig.update_xaxes(tickangle=45, row=2, col=2)  # Rotate category labels


def _print_mc_summary(statistics: dict) -> None:
    """Print Monte Carlo parameter sensitivity summary."""
    print("\n📊 Monte Carlo Parameter Sensitivity Analysis:")
    
    actual_return = statistics.get('actual_return')
    mean_random = statistics.get('mean_return', 0)
    std_random = statistics.get('std_return', 0)
    
    print(f"   Strategy Return: {actual_return:.3f}%" if actual_return else "   No strategy return available")
    print(f"   Random Mean: {mean_random:.3f}% ± {std_random:.3f}%")
    print(f"   Random Range: [{mean_random - std_random:.3f}%, {mean_random + std_random:.3f}%]")
    
    # Performance interpretation
    if actual_return is not None and mean_random is not None:
        outperformance = actual_return - mean_random
        print(f"   Performance vs Random: {outperformance:+.3f}% ({'Better' if outperformance > 0 else 'Worse'})")
        
    print("   Parameter Sensitivity: Visualized in subplot 2")


def _plot_walkforward(wf_results: Dict[str, Any]) -> Dict[str, Any]:
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
            return _plot_multi_asset_walkforward(wf_results)
        return _plot_single_asset_walkforward(wf_results)

    except Exception as e:
        print(f"⚠️ Walk-forward plot failed: {e}")
        return {"success": False, "error": str(e)}

def _plot_single_asset_walkforward(wf_results: Dict[str, Any]) -> Dict[str, Any]:
    """Plot walk-forward analysis for single asset."""
    windows = wf_results['windows']

    # Extract data from windows
    window_nums = [w['window'] for w in windows]
    train_returns = [w['train_stats'].get('Total Return [%]', 0) for w in windows]
    test_returns = [w['test_stats'].get('Total Return [%]', 0) for w in windows]
    train_sharpes = [w['train_stats'].get('Sharpe Ratio', 0) for w in windows]
    test_sharpes = [w['test_stats'].get('Sharpe Ratio', 0) for w in windows]

    # Extract rolling Sharpe if available
    rolling_sharpe_train = [item for w in windows if 'rolling_sharpe_train' in w for item in w['rolling_sharpe_train']]
    rolling_sharpe_test = [item for w in windows if 'rolling_sharpe_test' in w for item in w['rolling_sharpe_test']]

    # Create subplots
    has_rolling = rolling_sharpe_train or rolling_sharpe_test
    rows, cols = (3, 2) if has_rolling else (2, 2)
    subplot_titles = ['Returns by Window (%)', 'Sharpe Ratio by Window', 'Train vs Test Returns', 'Train vs Test Sharpe']
    if has_rolling:
        subplot_titles.extend(['Rolling Sharpe Evolution', 'Performance Degradation'])

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    # Add traces for returns and Sharpe ratios
    traces = [
        (window_nums, train_returns, 'Train Returns', 1, 1),
        (window_nums, test_returns, 'Test Returns', 1, 1),
        (window_nums, train_sharpes, 'Train Sharpe', 1, 2),
        (window_nums, test_sharpes, 'Test Sharpe', 1, 2),
        (train_returns, test_returns, 'Returns Correlation', 2, 1),
        (train_sharpes, test_sharpes, 'Sharpe Correlation', 2, 2)
    ]

    for x, y, name, row, col in traces:
        mode = 'markers' if 'Correlation' in name else 'lines+markers'
        showlegend = 'Sharpe' not in name or 'Train' in name
        fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name, showlegend=showlegend), row=row, col=col)

    # Add diagonal reference lines
    _add_diagonal_lines(fig, train_returns, test_returns, 2, 1)
    _add_diagonal_lines(fig, train_sharpes, test_sharpes, 2, 2)

    # Add rolling Sharpe plots if available
    if has_rolling and rows > 2:
        if rolling_sharpe_train:
            fig.add_trace(go.Scatter(y=rolling_sharpe_train, mode='lines', name='Rolling Sharpe (Train)'), row=3, col=1)
        if rolling_sharpe_test:
            fig.add_trace(go.Scatter(y=rolling_sharpe_test, mode='lines', name='Rolling Sharpe (Test)'), row=3, col=1)

        # Performance degradation analysis
        degradation = [test - train for train, test in zip(train_returns, test_returns)]
        fig.add_trace(go.Scatter(x=window_nums, y=degradation, mode='lines+markers', name='Performance Degradation'), row=3, col=2)
        fig.add_shape(type="line", x0=min(window_nums), x1=max(window_nums), y0=0, y1=0,
                     line={"dash": "dash", "color": "gray"}, xref="x6", yref="y6")

    fig.update_layout(
        title_text="🚶 Walk-Forward Analysis - VectorBT Enhanced Performance Stability",
        template='plotly_dark', showlegend=True
    )
    fig.show()
    return {"success": True}

def _plot_multi_asset_walkforward(wf_results: Dict[str, Any]) -> Dict[str, Any]:
    """Plot walk-forward analysis for multiple assets."""
    windows = wf_results['windows']
    first_window = windows[0]
    asset_names = list(first_window.get('asset_results', {}).keys())

    if not asset_names:
        return _plot_single_asset_walkforward(wf_results)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Train Returns by Asset (%)', 'Test Returns by Asset (%)',
            'Train vs Test Comparison', 'Asset Performance Ranking'
        )
    )

    window_nums = [w['window'] for w in windows]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Plot asset performance data
    _plot_asset_returns(fig, windows, asset_names, window_nums, colors)
    _plot_asset_ranking(fig, windows[-1], asset_names)

    fig.update_layout(
        height=None, width=None,
        title_text="🚶 Multi-Asset Walk-Forward Analysis - VectorBT Enhanced",
        template='plotly_dark', showlegend=True
    )
    fig.show()
    return {"success": True}


def _plot_asset_returns(fig: go.Figure, windows: list, asset_names: list, 
                       window_nums: list, colors: list) -> None:
    """Plot asset returns for multi-asset walkforward analysis."""
    for i, asset in enumerate(asset_names):
        color = colors[i % len(colors)]
        train_returns, test_returns = _extract_asset_data(windows, asset)

        # Add traces for train, test, and comparison
        fig.add_trace(go.Scatter(
            x=window_nums, y=train_returns, mode='lines+markers',
            name=f'{asset} Train', line={"color": color, "width": 2}
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=window_nums, y=test_returns, mode='lines+markers',
            name=f'{asset} Test', line={"color": color, "width": 2, "dash": "dash"}
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=train_returns, y=test_returns, mode='markers',
            name=f'{asset}', marker={"color": color, "size": 8}, showlegend=False
        ), row=2, col=1)


def _extract_asset_data(windows: list, asset: str) -> tuple:
    """Extract train and test returns for a specific asset."""
    train_returns = []
    test_returns = []
    
    for w in windows:
        asset_data = w.get('asset_results', {}).get(asset, {})
        train_returns.append(asset_data.get('train_return', 0))
        test_returns.append(asset_data.get('test_return', 0))
        
    return train_returns, test_returns


def _plot_asset_ranking(fig: go.Figure, final_window: dict, asset_names: list) -> None:
    """Plot asset performance ranking."""
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
            name='Final Test Returns'
        ), row=2, col=2
    )


def _add_diagonal_lines(fig, x_data, y_data, row, col):
    """Add diagonal reference lines to scatter plots."""
    if not x_data or not y_data:
        return

    all_values = list(x_data) + list(y_data)
    min_val, max_val = min(all_values), max(all_values)

    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode='lines',
        line={'dash': 'dash', 'color': 'gray'}, name='Perfect Correlation', showlegend=False
    ), row=row, col=col)


def create_comparison_plot(results: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """Create default vs optimized comparison plot."""
    try:
        # Extract stats from both backtests
        def extract_stats(backtest_key):
            if backtest_key not in results:
                return None
            for _, timeframes in results[backtest_key].items():
                for _, result in timeframes.items():
                    if 'portfolio' in result:
                        return result['portfolio'].stats()
            return None

        default_stats = extract_stats('default_backtest')
        optimized_stats = extract_stats('full_backtest')

        if not default_stats or not optimized_stats:
            return {"success": False, "reason": "missing_stats"}

        # Create comparison chart
        metrics_names = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Total Trades']
        default_values = [
            float(default_stats.get('Total Return [%]', 0)),
            float(default_stats.get('Sharpe Ratio', 0)),
            float(default_stats.get('Max Drawdown [%]', 0)),
            float(default_stats.get('Win Rate [%]', 0)),
            int(default_stats.get('Total Trades', 0))
        ]
        optimized_values = [
            float(optimized_stats.get('Total Return [%]', 0)),
            float(optimized_stats.get('Sharpe Ratio', 0)),
            float(optimized_stats.get('Max Drawdown [%]', 0)),
            float(optimized_stats.get('Win Rate [%]', 0)),
            int(optimized_stats.get('Total Trades', 0))
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Default Parameters', x=metrics_names, y=default_values, 
                           marker_color='lightblue', opacity=0.7))
        fig.add_trace(go.Bar(name='Optimized Parameters', x=metrics_names, y=optimized_values, 
                           marker_color='red', opacity=0.7))

        fig.update_layout(
            title=f'{strategy_name} Strategy: Default vs Optimized Parameters',
            xaxis_title='Metrics', yaxis_title='Values', barmode='group',
            template='plotly_dark', height=600
        )
        fig.show()

        # Print improvement summary
        print("\n📈 Optimization Impact Summary:")
        print(f"   Return: {default_values[0]:.2f}% → {optimized_values[0]:.2f}% ({optimized_values[0] - default_values[0]:+.2f}%)")
        print(f"   Sharpe: {default_values[1]:.3f} → {optimized_values[1]:.3f} ({optimized_values[1] - default_values[1]:+.3f})")
        print(f"   Max DD: {default_values[2]:.2f}% → {optimized_values[2]:.2f}% ({optimized_values[2] - default_values[2]:+.2f}%)")
        print(f"   Win Rate: {default_values[3]:.1f}% → {optimized_values[3]:.1f}% ({optimized_values[3] - default_values[3]:+.1f}%)")

        return {"success": True}

    except Exception as e:
        print(f"⚠️ Comparison plot failed: {e}")
        return {"success": False, "error": str(e)}

