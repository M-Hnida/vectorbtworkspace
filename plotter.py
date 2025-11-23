"""
Plotting utilities for trading strategy analysis.
Refactored for modularity, fail-fast debugging, and UI integration.
"""

import warnings
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt

# --- Global Settings (User Preference) ---
warnings.filterwarnings("ignore")
vbt.settings.set_theme("dark")
vbt.settings['plotting']['layout']["template"] = "plotly_dark"


def create_visualizations(results: Dict[str, Any], strategy_name: str = "Trading Strategy") -> Dict[str, go.Figure]:
    """
    Main entry point. Generates all visualization figures from the results dictionary.
    
    Returns:
        Dict[str, go.Figure]: A dictionary of Plotly figures ready for rendering or UI embedding.
    """
    figures: Dict[str, go.Figure] = {}

    # 1. Extract and Plot Portfolios (Individual & Comparison)
    portfolios = _extract_portfolios_from_results(results)
    if portfolios:
        # Plot individual portfolios
        for name, pf in portfolios.items():
            figures[f"portfolio_{name}"] = _plot_single_portfolio(pf, name, strategy_name)
        
        # Plot comparison if multiple portfolios exist
        if len(portfolios) > 1:
            figures["portfolio_comparison"] = _create_comparison_plot(portfolios, strategy_name)

    # 2. Monte Carlo Analysis
    if mc_results := results.get('monte_carlo'):
        # Standard MC Dashboard
        figures["monte_carlo_dashboard"] = _plot_monte_carlo_dashboard(mc_results)
        
        # Path Analysis (previously dead code, now integrated)
        if 'simulated_returns' in mc_results:
             figures["monte_carlo_paths"] = plot_path_mc_results(mc_results)

    # 3. Walk-Forward Analysis
    if wf_results := results.get('walkforward'):
        # Detect if multi-asset or single-asset based on result structure
        windows = wf_results.get('windows', [])
        if windows:
            has_multiple_assets = any('asset_results' in w for w in windows)
            if has_multiple_assets:
                figures["walk_forward_analysis"] = _plot_multi_asset_walkforward(wf_results)
            else:
                figures["walk_forward_analysis"] = _plot_single_asset_walkforward(wf_results)

    # 4. Strategy Optimization Comparison (Default vs Optimized)
    if 'default_backtest' in results and 'full_backtest' in results:
        figures["optimization_impact"] = _create_optimization_comparison(results, strategy_name)

    return figures


def render_figures(figures: Dict[str, go.Figure]) -> None:
    """
    Helper function to display all figures in a Notebook environment.
    Use this if you are not integrating into a Dashboard UI.
    """
    if not figures:
        print("No figures generated.")
        return

    print(f"Rendering {len(figures)} visualization(s)...")
    for name, fig in figures.items():
        print(f"--- {name} ---")
        fig.show()


# --- Internal Plotting Logic ---

def _plot_single_portfolio(portfolio: Any, name: str, strategy_name: str) -> go.Figure:
    """Generates a plot for a single portfolio."""
    fig = portfolio.plot(template='plotly_dark')
    fig.update_layout(
        title=f"{strategy_name} - {name}",
        height=None,
        width=None
    )
    return fig


def _create_comparison_plot(portfolios: Dict[str, Any], strategy_name: str) -> go.Figure:
    """Creates a normalized comparison plot for multiple portfolios."""
    fig = go.Figure()
    
    # Sort: Put '_default' strategies first for baseline comparison
    ordered = sorted(portfolios.items(), key=lambda kv: (not kv[0].endswith('_default'), kv[0]))
    
    for name, portfolio in ordered:
        value_series = portfolio.value()
        # Normalize to 100 for valid comparison across different asset prices
        normalized_values = (value_series / value_series.iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=normalized_values.index, 
            y=normalized_values.values,
            mode='lines', 
            name=name, 
            line={"width": 3}
        ))

    fig.update_layout(
        title=f"{strategy_name} - Portfolio Comparison (Normalized)",
        yaxis_title="Normalized Value (Start = 100)", 
        xaxis_title="Date",
        template='plotly_dark'
    )
    return fig


def _plot_monte_carlo_dashboard(mc_results: Dict[str, Any]) -> go.Figure:
    """Generates the main Monte Carlo dashboard (4 subplots)."""
    simulations = mc_results.get('simulations', [])
    statistics = mc_results.get('statistics', {})
    
    if not simulations:
        # Return empty figure with annotation if no data (fail-soft for empty data, fail-fast for bugs)
        fig = go.Figure()
        fig.add_annotation(text="No Monte Carlo simulations found", showarrow=False)
        return fig

    returns_data = [sim['total_return'] for sim in simulations if 'total_return' in sim]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Return Distribution',
            'Parameter Sensitivity', 
            'Simulation Paths',
            'Performance Comparison'
        ]
    )

    _add_histogram(fig, returns_data, statistics)
    _add_parameter_sensitivity(fig, simulations)
    _add_simulation_paths(fig, simulations, mc_results)
    _add_mc_comparison(fig, statistics)

    fig.update_layout(
        title="Monte Carlo Analysis Dashboard",
        template='plotly_dark',
        height=900 
    )
    return fig


def plot_path_mc_results(mc_results: Dict[str, Any]) -> go.Figure:
    """
    Plots the specific Path Randomization results (Equity curves, Sharpe dist, etc.).
    Previously dead code, now integrated.
    """
    stats = mc_results['statistics']
    mc_returns = mc_results['simulated_returns']
    mc_sharpes = mc_results['simulated_sharpes']
    equity_paths = mc_results.get('equity_paths')
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Return Distribution',
            'Sharpe Ratio Distribution',
            'Equity Paths (Sample)',
            'Max Drawdown Distribution'
        ),
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # 1. Return distribution
    fig.add_trace(
        go.Histogram(x=mc_returns, name='MC Returns', nbinsx=50, 
                    marker_color='lightblue', showlegend=False),
        row=1, col=1
    )
    fig.add_vline(x=stats['original_return'], line_dash="dash", line_color="red",
                 annotation_text=f"Orig: {stats['original_return']:.2f}%", row=1, col=1)
    
    # 2. Sharpe distribution
    fig.add_trace(
        go.Histogram(x=mc_sharpes, name='MC Sharpe', nbinsx=50,
                    marker_color='lightgreen', showlegend=False),
        row=1, col=2
    )
    fig.add_vline(x=stats['original_sharpe'], line_dash="dash", line_color="red",
                 annotation_text=f"Orig: {stats['original_sharpe']:.3f}", row=1, col=2)
    
    # 3. Sample equity paths (Limit to 100 to prevent browser lag)
    if equity_paths is not None and len(equity_paths) > 0:
        for equity in equity_paths[:100]:
            fig.add_trace(
                go.Scatter(y=equity, mode='lines', line=dict(color='lightgray', width=0.5),
                          showlegend=False, opacity=0.3),
                row=2, col=1
            )
    
    # 4. Max Drawdown distribution
    fig.add_trace(
        go.Histogram(x=mc_results.get('simulated_max_dds', []), name='MC Max DD', nbinsx=50,
                    marker_color='salmon', showlegend=False),
        row=2, col=2
    )
    if 'original_max_dd' in stats:
        fig.add_vline(x=stats['original_max_dd'], line_dash="dash", line_color="red",
                     annotation_text=f"Orig: {stats['original_max_dd']:.2f}%", row=2, col=2)
    
    fig.update_layout(
        title_text=f"Path Randomization Monte Carlo ({mc_results.get('n_simulations', '?')} runs)",
        template='plotly_dark',
        height=800
    )
    return fig


def _create_optimization_comparison(results: Dict[str, Any], strategy_name: str) -> go.Figure:
    """Creates a bar chart comparing Default vs Optimized metrics."""
    # Logic corrected to handle potential multiple tickers, here we take the average or first valid set
    default_stats = _extract_stats_aggregate(results, 'default_backtest')
    optimized_stats = _extract_stats_aggregate(results, 'full_backtest')
    
    if not default_stats or not optimized_stats:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for comparison", showarrow=False)
        return fig

    metrics_map = {
        'Total Return [%]': 'Total Return (%)',
        'Sharpe Ratio': 'Sharpe Ratio',
        'Max Drawdown [%]': 'Max Drawdown (%)',
        'Win Rate [%]': 'Win Rate (%)',
        'Total Trades': 'Total Trades'
    }
    
    metrics_names = list(metrics_map.values())
    
    # Helper to safely extract and format values
    def get_vals(stats):
        vals = []
        for key in metrics_map.keys():
            val = stats.get(key, 0)
            vals.append(float(val) if val is not None else 0.0)
        return vals

    default_values = get_vals(default_stats)
    optimized_values = get_vals(optimized_stats)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Default Parameters', x=metrics_names, y=default_values, marker_color='lightblue', opacity=0.7))
    fig.add_trace(go.Bar(name='Optimized Parameters', x=metrics_names, y=optimized_values, marker_color='red', opacity=0.7))
    
    fig.update_layout(
        title=f'{strategy_name}: Optimization Impact',
        xaxis_title='Metrics', 
        yaxis_title='Values', 
        barmode='group', 
        template='plotly_dark'
    )
    return fig


def _plot_single_asset_walkforward(wf_results: Dict[str, Any]) -> go.Figure:
    """Walk-forward analysis for a single asset."""
    windows = wf_results['windows']
    
    window_nums = [w['window'] for w in windows]
    train_sharpes = [w.get('train_sharpe', 0) for w in windows]
    test_sharpes = [w.get('test_sharpe', 0) for w in windows]

    fig = make_subplots(rows=2, cols=2, subplot_titles=['Sharpe by Window', 'Train vs Test Sharpe', 'Performance Degradation', 'Window Summary'])

    # Traces
    fig.add_trace(go.Scatter(x=window_nums, y=train_sharpes, mode='lines+markers', name='Train Sharpe'), row=1, col=1)
    fig.add_trace(go.Scatter(x=window_nums, y=test_sharpes, mode='lines+markers', name='Test Sharpe'), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_sharpes, y=test_sharpes, mode='markers', name='Sharpe Correlation'), row=1, col=2)

    # Diagonal reference line
    if train_sharpes and test_sharpes:
        min_val = min(min(train_sharpes), min(test_sharpes))
        max_val = max(max(train_sharpes), max(test_sharpes))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                               line={'dash': 'dash', 'color': 'gray'}, name='Perfect Correlation', showlegend=False), row=1, col=2)

    # Degradation
    degradation = [test - train for train, test in zip(train_sharpes, test_sharpes)]
    fig.add_trace(go.Scatter(x=window_nums, y=degradation, mode='lines+markers', name='Performance Degradation'), row=2, col=1)
    fig.add_shape(type="line", x0=min(window_nums), x1=max(window_nums), y0=0, y1=0,
                 line={"dash": "dash", "color": "gray"}, xref="x4", yref="y4", row=2, col=1)

    fig.update_layout(title_text="Walk-Forward Analysis (Single Asset)", template='plotly_dark')
    return fig


def _plot_multi_asset_walkforward(wf_results: Dict[str, Any]) -> go.Figure:
    """Walk-forward analysis for multiple assets."""
    windows = wf_results['windows']
    first_window = windows[0]
    asset_names = list(first_window.get('asset_results', {}).keys())

    fig = make_subplots(rows=2, cols=2, subplot_titles=('Train Returns by Asset (%)', 'Test Returns by Asset (%)', 'Train vs Test Comparison', 'Asset Performance Ranking'))
    window_nums = [w['window'] for w in windows]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i, asset in enumerate(asset_names):
        color = colors[i % len(colors)]
        train_returns = [w.get('asset_results', {}).get(asset, {}).get('train_return', 0) for w in windows]
        test_returns = [w.get('asset_results', {}).get(asset, {}).get('test_return', 0) for w in windows]

        fig.add_trace(go.Scatter(x=window_nums, y=train_returns, mode='lines+markers', name=f'{asset} Train', line={"color": color, "width": 2}), row=1, col=1)
        fig.add_trace(go.Scatter(x=window_nums, y=test_returns, mode='lines+markers', name=f'{asset} Test', line={"color": color, "width": 2, "dash": "dash"}), row=1, col=2)
        fig.add_trace(go.Scatter(x=train_returns, y=test_returns, mode='markers', name=f'{asset}', marker={"color": color, "size": 8}, showlegend=False), row=2, col=1)

    # Asset ranking (Final window)
    final_window = windows[-1]
    asset_performance = [(asset, final_window.get('asset_results', {}).get(asset, {}).get('test_return', 0)) for asset in asset_names]
    asset_performance.sort(key=lambda x: x[1], reverse=True)
    assets_sorted, returns_sorted = zip(*asset_performance)
    
    fig.add_trace(go.Bar(x=list(assets_sorted), y=list(returns_sorted), name='Final Test Returns'), row=2, col=2)
    fig.update_layout(title_text="Multi-Asset Walk-Forward Analysis", template='plotly_dark', showlegend=True)
    return fig


# --- Helpers & Data Extraction ---

def _extract_portfolios_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts portfolios into a flat dictionary with unique names."""
    portfolios = {}
    keys_to_check = [('default_backtest', '_default'), ('full_backtest', '_optimized')]
    
    for key, suffix in keys_to_check:
        if key in results:
            data_dict = results[key]
            # Handle nested structure: Symbol -> Timeframe -> Portfolio
            for symbol, timeframes in data_dict.items():
                for timeframe, portfolio in timeframes.items():
                    # Duck typing check for VectorBT portfolio object
                    if hasattr(portfolio, 'stats') and hasattr(portfolio, 'plot'):
                        portfolios[f"{symbol}_{timeframe}{suffix}"] = portfolio
    return portfolios


def _extract_stats_aggregate(results: Dict[str, Any], key: str) -> Optional[Dict[str, float]]:
    """
    Extracts stats. If multiple portfolios exist for the key (e.g. multiple assets),
    this logic defaults to the FIRST valid one for now to maintain comparison logic.
    TODO: Implement weighted average aggregation if needed for portfolio of portfolios.
    """
    if key in results:
        for symbol, timeframes in results[key].items():
            for tf, portfolio in timeframes.items():
                if hasattr(portfolio, 'stats'):
                    # Fail-fast Note: If .stats() fails, we let it crash here.
                    return portfolio.stats()
    return None


# --- Sub-component builders for Monte Carlo (Modularized) ---

def _add_histogram(fig: go.Figure, returns_data: list, statistics: dict) -> None:
    fig.add_trace(go.Histogram(
        x=returns_data, nbinsx=30, name='MC Returns', marker_color='lightblue'
    ), row=1, col=1)

    if actual_return := statistics.get('actual_return'):
        fig.add_vline(x=actual_return, line_dash="dash", line_color="red",
                      annotation_text=f"Strategy: {actual_return:.2f}%", row=1, col=1)


def _add_parameter_sensitivity(fig: go.Figure, simulations: list) -> None:
    """Adds parameter sensitivity bar chart and heatmap."""
    # Logic extracted to keep main function clean
    param_data = _extract_parameter_data(simulations)
    if not param_data:
        return

    param_names = [n for n in param_data.keys() if n != 'returns']
    returns = pd.Series(param_data['returns'], dtype=float)
    if not param_names or returns.empty:
        return

    # Correlation Analysis
    scores = []
    for name in param_names:
        vals = pd.Series(param_data[name])
        mask = ~(vals.isna() | returns.isna())
        if mask.sum() >= 3:
            # We allow correlation calculation to fail silently (math errors) 
            # but structure errors will crash
            try:
                corr = float(returns[mask].corr(vals[mask], method='spearman'))
                scores.append((name, abs(corr)))
            except ValueError:
                continue
    
    if not scores:
        return
        
    scores.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*scores)
    
    fig.add_trace(go.Bar(x=[str(n) for n in names], y=list(vals), 
                        marker_color='cornflowerblue', name='Param Importance'), row=1, col=2)

    # Simple Heatmap for top 2 params
    varying = [n for n in param_names if len(set(v for v in param_data[n] if v is not None)) > 1]
    if len(varying) == 2:
        _add_heatmap(fig, varying, param_data, returns)


def _add_heatmap(fig: go.Figure, varying_params: list, param_data: dict, returns: pd.Series):
    x_param, y_param = varying_params
    x_vals, y_vals = pd.Series(param_data[x_param]), pd.Series(param_data[y_param])
    
    mask = ~(x_vals.isna() | y_vals.isna() | returns.isna())
    if mask.sum() < 4:
        return

    df = pd.DataFrame({'x': x_vals[mask].values, 'y': y_vals[mask].values, 'ret': returns[mask].values})
    
    # Simple binning
    df['xb'] = df['x'].astype(str)
    df['yb'] = df['y'].astype(str)
    
    try:
        piv_pivot = df.groupby(['yb', 'xb'])['ret'].mean().reset_index().pivot(index='yb', columns='xb', values='ret')
        fig.add_trace(go.Heatmap(z=piv_pivot.values, x=[str(c) for c in piv_pivot.columns],
                                y=[str(i) for i in piv_pivot.index], colorscale='RdYlGn',
                                colorbar=dict(title="Mean Return", x=1.1), name='Heatmap'), row=1, col=2)
    except Exception:
        # Pivot failures are data-dependent and acceptable to skip in visualization
        pass


def _extract_parameter_data(simulations: list) -> Optional[dict]:
    if not simulations: return None
    first_sim = next((s for s in simulations if isinstance(s, dict) and 'parameters' in s), None)
    if not first_sim: return None
    
    param_names = list(first_sim.get('parameters', {}).keys())
    if not param_names: return None
    
    param_data = {name: [] for name in param_names}
    param_data['returns'] = []
    
    for sim in simulations:
        if 'total_return' not in sim: continue
        params = sim.get('parameters', {})
        for name in param_names:
            param_data[name].append(params.get(name))
        param_data['returns'].append(sim['total_return'])
        
    return param_data


def _add_simulation_paths(fig: go.Figure, simulations: list, mc_results: Dict[str, Any]) -> None:
    path_matrix = mc_results.get('path_matrix')
    if path_matrix is None or path_matrix.size == 0:
        return
    
    T, N = path_matrix.shape
    time_axis = list(range(T))
    
    # Plot subset of paths
    max_paths = min(50, N)
    step = max(1, N // max_paths)
    
    for i in range(0, N, step)[:max_paths]:
        path_values = path_matrix[:, i]
        fig.add_trace(go.Scatter(
            x=time_axis, y=path_values, mode='lines',
            line={'color': 'rgba(100,149,237,0.15)', 'width': 1},
            showlegend=False, hoverinfo='skip'
        ), row=2, col=1)
    
    # Median Path
    median_path = np.nanmedian(path_matrix, axis=1)
    fig.add_trace(go.Scatter(x=time_axis, y=median_path, mode='lines', name='Median Path',
                            line={'color': 'orange', 'width': 3}), row=2, col=1)


def _add_mc_comparison(fig: go.Figure, statistics: dict) -> None:
    mean_random = statistics.get('mean_return', 0.0)
    std_random = statistics.get('std_return', 0.0)
    actual_return = statistics.get('actual_return', 0.0)

    categories = ['Random Mean', 'Strategy']
    values = [mean_random, actual_return]
    colors = ['lightgray', 'red']

    fig.add_trace(go.Bar(x=categories, y=values, marker_color=colors, name='Perf Comparison'), row=2, col=2)