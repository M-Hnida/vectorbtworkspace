"""Plotting utilities for trading strategy analysis."""

import warnings
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt

warnings.filterwarnings("ignore")
vbt.settings.set_theme("dark")
vbt.settings['plotting']['layout']["template"] = "plotly_dark"


def plot_comprehensive_analysis(portfolios: Dict[str, Any], strategy_name: str = "Trading Strategy",
                              mc_results: Optional[Dict[str, Any]] = None,
                              wf_results: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Plot comprehensive analysis with error handling."""
    if not portfolios:
        print("No portfolios to plot")
        return None

    try:
        _plot_portfolios(portfolios, strategy_name)
        
        if mc_results:
            _plot_monte_carlo(mc_results)
            
        if wf_results:
            return _plot_walkforward(wf_results)
            
        return {"success": True}
    except Exception as e:
        print(f"Plotting failed: {e}")
        return {"success": False, "error": str(e)}


def _plot_portfolios(portfolios: Dict[str, Any], strategy_name: str) -> None:
    """Plot individual portfolios and comparison."""
    print("Creating portfolio visualizations...")
    
    ordered = sorted(portfolios.items(), key=lambda kv: (not kv[0].endswith('_default'), kv[0]))
    for name, portfolio in ordered:
        _plot_single_portfolio(portfolio, name, strategy_name)

    if len(portfolios) > 1:
        _create_comparison_plot(portfolios, strategy_name)


def _plot_single_portfolio(portfolio: Any, name: str, strategy_name: str) -> None:
    """Plot a single portfolio."""
    print(f"Creating plot for {name}...")
    
    fig = portfolio.plot(template='plotly_dark')
    fig.update_layout(
        title=f"{strategy_name} - {name}",
        height=None,
        width=None
    )
    fig.show()


def _create_comparison_plot(portfolios: Dict[str, Any], strategy_name: str) -> None:
    """Create comparison plot for multiple portfolios."""
    print("Creating portfolio comparison...")
    
    fig = go.Figure()
    ordered = sorted(portfolios.items(), key=lambda kv: (not kv[0].endswith('_default'), kv[0]))
    
    for name, portfolio in ordered:
        value_series = portfolio.value()
        normalized_values = (value_series / value_series.iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=normalized_values.index, 
            y=normalized_values.values,
            mode='lines', 
            name=name, 
            line={"width": 3}
        ))

    fig.update_layout(
        title=f"{strategy_name} - Portfolio Comparison",
        yaxis_title="Normalized Value (Start = 100)", 
        xaxis_title="Date",
        template='plotly_dark'
    )
    fig.show()


def _plot_monte_carlo(mc_results: Dict[str, Any]) -> None:
    """Plot Monte Carlo results with consolidated logic."""
    simulations = mc_results.get('simulations', [])
    statistics = mc_results.get('statistics', {})
    
    if not simulations:
        print("No Monte Carlo simulations to plot")
        return

    # Simplified returns data extraction
    returns_data = [sim['total_return'] for sim in simulations if 'total_return' in sim]
    if not returns_data:
        print("No valid returns data found")
        return

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
        title="Monte Carlo Analysis",
        template='plotly_dark'
    )
    
    fig.show()
    _print_mc_summary(statistics)


def _add_histogram(fig: go.Figure, returns_data: list, statistics: dict) -> None:
    """Add histogram of Monte Carlo returns."""
    fig.add_trace(go.Histogram(
        x=returns_data,
        nbinsx=30,
        name='MC Returns',
        marker_color='lightblue'
    ), row=1, col=1)

    actual_return = statistics.get('actual_return')
    if actual_return:
        fig.add_vline(
            x=actual_return, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Strategy: {actual_return:.2f}%",
            row=1, col=1
        )


def _calculate_parameter_importance(param_names: list, param_data: dict, returns: pd.Series) -> list:
    """Calculate parameter importance using Spearman correlation."""
    scores = []
    for name in param_names:
        vals = pd.Series(param_data[name])
        mask = ~(vals.isna() | returns.isna())
        if mask.sum() >= 3:
            corr = float(returns[mask].corr(vals[mask], method='spearman'))
            scores.append((name, abs(corr)))
    return scores


def _extract_parameter_data(simulations: list) -> Optional[dict]:
    """Extract parameter values and returns from simulations."""
    if not simulations:
        return None
    
    # Find first simulation with parameters
    first_sim = next((s for s in simulations if isinstance(s, dict) and 'parameters' in s), None)
    if not first_sim:
        return None
    
    param_names = list(first_sim.get('parameters', {}).keys())
    if not param_names:
        return None
    
    # Initialize data structure
    param_data = {name: [] for name in param_names}
    param_data['returns'] = []
    
    # Extract data from all simulations
    for sim in simulations:
        if not isinstance(sim, dict) or 'total_return' not in sim:
            continue
            
        parameters = sim.get('parameters', {})
        if not parameters:
            continue
            
        # Add parameter values (default to 0 if missing)
        for name in param_names:
            param_data[name].append(parameters.get(name, 0))
        param_data['returns'].append(sim['total_return'])
    
    # Return None if no returns were found
    return param_data if param_data['returns'] else None


def _add_parameter_sensitivity(fig: go.Figure, simulations: list) -> None:
    """Add parameter sensitivity panel with simplified logic."""
    if not simulations:
        return

    param_data = _extract_parameter_data(simulations)
    if not param_data:
        return

    param_names = [n for n in param_data.keys() if n != 'returns']
    returns = pd.Series(param_data['returns'], dtype=float)
    if len(param_names) == 0 or returns.empty:
        return

    # Calculate parameter importance using Spearman correlation
    scores = _calculate_parameter_importance(param_names, param_data, returns)
    if not scores:
        return
        
    scores.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*scores)
    
    # Add parameter importance bar chart
    fig.add_trace(go.Bar(
        x=[str(n) for n in names],
        y=list(vals),
        marker_color='cornflowerblue',
        name='Param Importance (|Spearman|)'
    ), row=1, col=2)
    fig.update_yaxes(title_text="|Spearman|", row=1, col=2)

    # Top parameter violin plot
    top_param = scores[0][0]
    top_vals = pd.Series(param_data[top_param])
    mask = ~(top_vals.isna() | returns.isna())
    if mask.sum() >= 5:
        q = pd.qcut(top_vals[mask], q=min(5, mask.sum()), duplicates='drop')
        grouped = pd.DataFrame({'bin': q.astype(str), 'ret': returns[mask].values})
        med = grouped.groupby('bin')['ret'].median().sort_values()
        ordered_bins = list(med.index)
        
        for b in ordered_bins:
            vals = grouped[grouped['bin'] == b]['ret'].values
            if len(vals) == 0:
                continue
            label = f"{top_param} {b}"
            fig.add_trace(go.Violin(
                y=vals,
                x=[label]*len(vals),
                name=label,
                line_color='lightblue',
                meanline_visible=True,
                showlegend=False
            ), row=1, col=2)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)

    # 2D heatmap for exactly two varying params
    varying = [n for n in param_names if len(set([v for v in param_data[n] if v is not None])) > 1]
    if len(varying) == 2:
        x_param, y_param = varying
        x_vals = pd.Series(param_data[x_param])
        y_vals = pd.Series(param_data[y_param])
        mask = ~(x_vals.isna() | y_vals.isna() | returns.isna())
        if mask.sum() >= 4:
            df = pd.DataFrame({
                'x': x_vals[mask].values,
                'y': y_vals[mask].values,
                'ret': returns[mask].values
            })
            
            def bucket(s):
                uniq = np.unique(s)
                if len(uniq) > 12:
                    return pd.qcut(s, q=8, duplicates='drop').astype(str)
                return s.astype(str)
            
            df['xb'] = bucket(pd.Series(df['x']))
            df['yb'] = bucket(pd.Series(df['y']))
            piv = df.groupby(['yb', 'xb'])['ret'].mean().reset_index()
            piv_pivot = piv.pivot(index='yb', columns='xb', values='ret')
            
            fig.add_trace(go.Heatmap(
                z=piv_pivot.values,
                x=[str(c) for c in piv_pivot.columns],
                y=[str(i) for i in piv_pivot.index],
                colorscale='RdYlGn',
                colorbar=dict(title="Mean Return (%)"),
                name='Param Heatmap',
                showscale=True
            ), row=1, col=2)
            fig.update_xaxes(title_text=str(x_param), row=1, col=2)
            fig.update_yaxes(title_text=str(y_param), row=1, col=2)

def _add_simulation_paths(fig: go.Figure, simulations: list, mc_results: Optional[Dict[str, Any]] = None) -> None:
    """Add simulation paths panel using path_matrix from Monte Carlo results."""
    if not mc_results or 'path_matrix' not in mc_results:
        return
    
    path_matrix = mc_results['path_matrix']
    if path_matrix is None or path_matrix.size == 0:
        return
    
    # path_matrix shape is (T, N) where T=time steps, N=simulations
    T, N = path_matrix.shape
    
    # Plot a sample of paths (max 50 for better visualization)
    max_paths = min(50, N)
    step = max(1, N // max_paths)
    path_indices = range(0, N, step)[:max_paths]
    
    time_axis = list(range(T))
    
    # Plot individual paths with low opacity
    for i, path_idx in enumerate(path_indices):
        path_values = path_matrix[:, path_idx]
        
        # Skip paths with no variation or invalid data
        if not np.isfinite(path_values).any() or np.all(path_values == path_values[0]):
            continue
        
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=path_values,
            mode='lines',
            name='MC Paths' if i == 0 else None,
            showlegend=(i == 0),
            line={'color': 'rgba(100,149,237,0.15)', 'width': 1},
            hovertemplate="Time: %{x}<br>Return: %{y:.2f}%<extra></extra>"
        ), row=2, col=1)
    
    # Calculate and plot median path (50th percentile)
    median_path = np.nanmedian(path_matrix, axis=1)
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=median_path,
        mode='lines',
        name='Median Path',
        line={'color': 'orange', 'width': 3},
        showlegend=True,
        hovertemplate="Time: %{x}<br>Median Return: %{y:.2f}%<extra></extra>"
    ), row=2, col=1)
    
    # Add 5th and 95th percentile bands
    percentile_5 = np.nanpercentile(path_matrix, 5, axis=1)
    percentile_95 = np.nanpercentile(path_matrix, 95, axis=1)
    
    # Add shaded area for confidence interval
    fig.add_trace(go.Scatter(
        x=time_axis + time_axis[::-1],
        y=list(percentile_95) + list(percentile_5[::-1]),
        fill='toself',
        fillcolor='rgba(100,149,237,0.1)',
        line={'color': 'rgba(255,255,255,0)'},
        name='90% Confidence',
        showlegend=True,
        hoverinfo='skip'
    ), row=2, col=1)
    
    # Add strategy performance line if available
    statistics = mc_results.get('statistics', {})
    actual_return = statistics.get('actual_return')
    if actual_return is not None:
        # Create a horizontal line at the actual return level
        fig.add_trace(go.Scatter(
            x=[0, T-1],
            y=[actual_return, actual_return],
            mode='lines',
            name='Strategy Return',
            line={'color': 'red', 'width': 3, 'dash': 'dash'},
            showlegend=True,
            hovertemplate=f"Strategy Return: {actual_return:.2f}%<extra></extra>"
        ), row=2, col=1)


def _add_mc_comparison(fig: go.Figure, statistics: dict) -> None:
    """Add performance comparison subplot to Monte Carlo plot."""
    mean_random = statistics.get('mean_return', 0.0)
    std_random = statistics.get('std_return', 0.0)
    actual_return = statistics.get('actual_return', 0.0)

    values = np.array([mean_random, actual_return, mean_random + std_random, mean_random - std_random], dtype=float)
    if not np.isfinite(values).any():
        return

    categories = ['Random\nMean', 'Strategy', 'Random\n+1σ', 'Random\n-1σ']
    colors = ['lightgray', 'red', 'lightgreen', 'orange']

    # Add all bar traces in a loop
    bar_configs = zip(categories, values, colors, [True, True, False, False])
    for category, val, color, showlegend in bar_configs:
        text_position = 'outside' if abs(val) < 1 else 'inside'
        text_color = 'white' if text_position == 'inside' else color

        fig.add_trace(go.Bar(
            x=[category], y=[val], marker_color=color,
            name=category.replace('\n', ' ') if showlegend else None,
            showlegend=showlegend,
            text=f'{val:.2f}%',
            textposition=text_position,
            textfont={'color': text_color, 'size': 10}
        ), row=2, col=2)

    fig.add_shape(
        type="line", x0=-0.5, x1=3.5, y0=0, y1=0,
        line={"dash": "dot", "color": "white", "width": 1},
        xref="x4", yref="y4"
    )

    vmax = float(np.nanmax(values))
    vmin = float(np.nanmin(values))
    if vmax == vmin:
        pad = 1.0 or abs(vmax) * 0.15
        ymin, ymax = vmin - pad, vmax + pad
    else:
        span = vmax - vmin
        pad = span * 0.15
        ymin, ymax = vmin - pad, vmax + pad
    fig.update_yaxes(range=[ymin, ymax], row=2, col=2)

    performance_text = "Outperforming" if actual_return > mean_random else "Underperforming"
    fig.add_annotation(
        x=1.5, y=ymax - (ymax - ymin) * 0.08, text=f"Strategy is {performance_text} vs Random",
        showarrow=False, xref="x4", yref="y4", font=dict(size=11, color="white"),
        bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1
    )


def _print_mc_summary(statistics: dict) -> None:
    """Print Monte Carlo parameter sensitivity summary."""
    print("\nMonte Carlo Parameter Sensitivity Analysis:")
    
    actual_return = statistics.get('actual_return')
    mean_random = statistics.get('mean_return', 0)
    std_random = statistics.get('std_return', 0)
    
    # Print strategy return info
    if actual_return:
        print(f"   Strategy Return: {actual_return:.3f}%")
    else:
        print("   No strategy return available")
        
    print(f"   Random Mean: {mean_random:.3f}% ± {std_random:.3f}%")
    print(f"   Random Range: [{mean_random - std_random:.3f}%, {mean_random + std_random:.3f}%]")
    
    # Print performance comparison
    if actual_return is not None and mean_random is not None:
        outperformance = actual_return - mean_random
        performance_desc = 'Better' if outperformance > 0 else 'Worse'
        print(f"   Performance vs Random: {outperformance:+.3f}% ({performance_desc})")
        
    print("   Parameter Sensitivity: Visualized in subplot 2")


def _plot_walkforward(wf_results: Dict[str, Any]) -> Dict[str, Any]:
    """Plot walk-forward analysis results with consolidated logic."""
    try:
        if 'windows' not in wf_results:
            return {"success": False, "reason": "no_windows"}

        windows = wf_results['windows']
        if not windows:
            return {"success": False, "reason": "empty_windows"}

        has_multiple_assets = any('asset_results' in w for w in windows)
        return _plot_multi_asset_walkforward(wf_results) if has_multiple_assets else _plot_single_asset_walkforward(wf_results)

    except Exception as e:
        print(f"Walk-forward plot failed: {e}")
        return {"success": False, "error": str(e)}


def _plot_single_asset_walkforward(wf_results: Dict[str, Any]) -> Dict[str, Any]:
    """Plot walk-forward analysis for single asset."""
    windows = wf_results['windows']
    
    window_nums = [w['window'] for w in windows]
    train_returns = [w['train_stats'].get('Total Return [%]', 0) for w in windows]
    test_returns = [w['test_stats'].get('Total Return [%]', 0) for w in windows]
    train_sharpes = [w['train_stats'].get('Sharpe Ratio', 0) for w in windows]
    test_sharpes = [w['test_stats'].get('Sharpe Ratio', 0) for w in windows]

    rolling_sharpe_train = [item for w in windows if 'rolling_sharpe_train' in w for item in w['rolling_sharpe_train']]
    rolling_sharpe_test = [item for w in windows if 'rolling_sharpe_test' in w for item in w['rolling_sharpe_test']]

    has_rolling = rolling_sharpe_train or rolling_sharpe_test
    rows, cols = (3, 2) if has_rolling else (2, 2)
    subplot_titles = ['Returns by Window (%)', 'Sharpe Ratio by Window', 'Train vs Test Returns', 'Train vs Test Sharpe']
    if has_rolling:
        subplot_titles.extend(['Rolling Sharpe Evolution', 'Performance Degradation'])

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    # Add line/point traces
    trace_configs = [
        (window_nums, train_returns, 'Train Returns', 1, 1, 'lines+markers'),
        (window_nums, test_returns, 'Test Returns', 1, 1, 'lines+markers'),
        (window_nums, train_sharpes, 'Train Sharpe', 1, 2, 'lines+markers'),
        (window_nums, test_sharpes, 'Test Sharpe', 1, 2, 'lines+markers'),
        (train_returns, test_returns, 'Returns Correlation', 2, 1, 'markers'),
        (train_sharpes, test_sharpes, 'Sharpe Correlation', 2, 2, 'markers')
    ]

    for x, y, name, row, col, mode in trace_configs:
        showlegend = 'Sharpe' not in name or 'Train' in name
        fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name, showlegend=showlegend), row=row, col=col)

    _add_diagonal_lines(fig, train_returns, test_returns, 2, 1)
    _add_diagonal_lines(fig, train_sharpes, test_sharpes, 2, 2)

    # Add rolling sharpe traces if available
    if has_rolling and rows > 2:
        if rolling_sharpe_train:
            fig.add_trace(go.Scatter(y=rolling_sharpe_train, mode='lines', name='Rolling Sharpe (Train)'), row=3, col=1)
        if rolling_sharpe_test:
            fig.add_trace(go.Scatter(y=rolling_sharpe_test, mode='lines', name='Rolling Sharpe (Test)'), row=3, col=1)

        degradation = [test - train for train, test in zip(train_returns, test_returns)]
        fig.add_trace(go.Scatter(x=window_nums, y=degradation, mode='lines+markers', name='Performance Degradation'), row=3, col=2)
        fig.add_shape(type="line", x0=min(window_nums), x1=max(window_nums), y0=0, y1=0,
                     line={"dash": "dash", "color": "gray"}, xref="x6", yref="y6")

    fig.update_layout(
        title_text="Walk-Forward Analysis - VectorBT Enhanced Performance Stability",
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

    _plot_asset_returns(fig, windows, asset_names, window_nums, colors)
    _plot_asset_ranking(fig, windows[-1], asset_names)

    fig.update_layout(
        title_text="Multi-Asset Walk-Forward Analysis - VectorBT Enhanced",
        template='plotly_dark', showlegend=True
    )
    fig.show()
    return {"success": True}


def _plot_asset_returns(fig: go.Figure, windows: list, asset_names: list,
                       window_nums: list, colors: list) -> None:
    """Plot asset returns for multi-asset walkforward analysis."""
    for i, asset in enumerate(asset_names):
        color = colors[i % len(colors)]
        train_returns = [w.get('asset_results', {}).get(asset, {}).get('train_return', 0) for w in windows]
        test_returns = [w.get('asset_results', {}).get(asset, {}).get('test_return', 0) for w in windows]

        # Add traces for train and test returns
        fig.add_trace(go.Scatter(
            x=window_nums, y=train_returns, mode='lines+markers',
            name=f'{asset} Train', line={"color": color, "width": 2}
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=window_nums, y=test_returns, mode='lines+markers',
            name=f'{asset} Test', line={"color": color, "width": 2, "dash": "dash"}
        ), row=1, col=2)

        # Add scatter plot for train vs test comparison
        fig.add_trace(go.Scatter(
            x=train_returns, y=test_returns, mode='markers',
            name=f'{asset}', marker={"color": color, "size": 8}, showlegend=False
        ), row=2, col=1)


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


def _add_diagonal_lines(fig, x_data, y_data, row, col) -> None:
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
        default_stats = _extract_stats_from_results(results, 'default_backtest')
        optimized_stats = _extract_stats_from_results(results, 'full_backtest')
        
        if default_stats is None or optimized_stats is None:
            return {"success": False, "reason": "missing_stats"}

        metrics_names = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Total Trades']
        
        # Extract values using a helper function
        def extract_values(stats, keys):
            return [
                float(stats.get(keys[0], 0)),  # Total Return
                float(stats.get(keys[1], 0)),  # Sharpe Ratio
                float(stats.get(keys[2], 0)),  # Max Drawdown
                float(stats.get(keys[3], 0)),  # Win Rate
                int(stats.get(keys[4], 0))     # Total Trades
            ]
        
        default_values = extract_values(default_stats, ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]', 'Total Trades'])
        optimized_values = extract_values(optimized_stats, ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]', 'Total Trades'])

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Default Parameters',
            x=metrics_names,
            y=default_values,
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            name='Optimized Parameters',
            x=metrics_names,
            y=optimized_values,
            marker_color='red',
            opacity=0.7
        ))

        fig.update_layout(
            title=f'{strategy_name} Strategy: Default vs Optimized Parameters',
            xaxis_title='Metrics',
            yaxis_title='Values',
            barmode='group',
            template='plotly_dark'
        )
        fig.show()

        print("\nOptimization Impact Summary:")
        print(f"   Return: {default_values[0]:.2f}% → {optimized_values[0]:.2f}% ({optimized_values[0] - default_values[0]:+.2f}%)")
        print(f"   Sharpe: {default_values[1]:.3f} → {optimized_values[1]:.3f} ({optimized_values[1] - default_values[1]:+.3f})")
        print(f"   Max DD: {default_values[2]:.2f}% → {optimized_values[2]:.2f}% ({optimized_values[2] - default_values[2]:+.2f}%)")
        print(f"   Win Rate: {default_values[3]:.1f}% → {optimized_values[3]:.1f}% ({optimized_values[3] - default_values[3]:+.1f}%)")
        
        return {"success": True}
        
    except Exception as e:
        print(f"Comparison plot failed: {e}")
        return {"success": False, "error": str(e)}


def _extract_stats_from_results(results: Dict[str, Any], key: str) -> Optional[dict]:
    """Extract stats from results structure."""
    if key in results:
        for symbol, timeframes in results[key].items():
            for tf, portfolio in timeframes.items():
                if hasattr(portfolio, 'stats'):
                    return portfolio.stats()
    return None

def _extract_portfolios_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract portfolios from results structure."""
    portfolios = {}
    
    if 'default_backtest' in results:
        for symbol, timeframes in results['default_backtest'].items():
            for timeframe, portfolio in timeframes.items():
                if hasattr(portfolio, 'stats'):
                    portfolios[f"{symbol}_{timeframe}_default"] = portfolio

    if 'full_backtest' in results:
        for symbol, timeframes in results['full_backtest'].items():
            for timeframe, portfolio in timeframes.items():
                if hasattr(portfolio, 'stats'):
                    portfolios[f"{symbol}_{timeframe}_optimized"] = portfolio
    
    return portfolios

def create_visualizations(results: Dict[str, Any], strategy_name: str) -> Dict[str, Any]: #keep this ; used in main.py
    """Create enhanced visualizations with consolidated logic."""
    try:
        portfolios = _extract_portfolios_from_results(results)
        
        if not portfolios:
            return {}
        
        plot_results = plot_comprehensive_analysis(
            portfolios, strategy_name,
            results.get('monte_carlo', {}),
            results.get('walkforward', {})
        )
        
        if 'default_backtest' in results and 'full_backtest' in results:
            print("Creating Default vs Optimized comparison...")
            comparison_results = create_comparison_plot(results, strategy_name)
            if plot_results is None:
                plot_results = {}
            plot_results['comparison'] = comparison_results
        
        return plot_results or {}
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        return {"success": False, "error": str(e)}