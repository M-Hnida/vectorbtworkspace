"""
Visualization Module
Handles all plotting and visualization for trading strategy analysis.
"""

import warnings
warnings.filterwarnings("ignore")

# Toggle verbose diagnostics across this module
DEBUG = False

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt

# Configure VectorBT global settings for consistent plotting
vbt.settings.set_theme("dark")
vbt.settings['plotting']['layout']["template"] = "plotly_dark"


def plot_comprehensive_analysis(portfolios, strategy_name: str = "Trading Strategy",
                                mc_results: Optional[Dict[str, Any]] = None,
                                wf_results: Optional[Dict[str, Any]] = None):
    """Plot everything: portfolios, Monte Carlo, and walk-forward analysis."""
    if not portfolios:
        print("No portfolios to plot")
        return

    _plot_portfolios(portfolios, strategy_name)
    
    if mc_results:
        _plot_monte_carlo(mc_results)
        
    if wf_results:
        _plot_walkforward(wf_results)

def _plot_portfolios(portfolios: Dict[str, Any], strategy_name: str):
    """Plot individual portfolios and comparison using VectorBT native functionality."""
    print("Creating portfolio visualizations...")

    # Plot each portfolio individually
    # Ensure default portfolios are shown first
    ordered = sorted(portfolios.items(), key=lambda kv: (not kv[0].endswith('_default'), kv[0]))
    for name, portfolio in ordered:
        _plot_single_portfolio(portfolio, name, strategy_name)

    # Create comparison plot if multiple portfolios
    if len(portfolios) > 1:
        _create_comparison_plot(portfolios, strategy_name)

def _plot_single_portfolio(portfolio: Any, name: str, strategy_name: str):
    """Plot a single portfolio."""
    print(f"Creating plot for {name}...")
    
    fig = portfolio.plot(template='plotly_dark')
    fig.update_layout(
        title=f"{strategy_name} - {name}",
        height=None,
        width=None
    )
    fig.show()

def _create_comparison_plot(portfolios: Dict[str, Any], strategy_name: str):
    """Create comparison plot for multiple portfolios."""
    print("Creating portfolio comparison...")
    
    fig = go.Figure()
    
    # Ensure default series are added first for legend consistency
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

def _plot_monte_carlo(mc_results: Dict[str, Any]):
    """Plot Monte Carlo results with parameter sensitivity analysis and path visualization."""
    simulations = mc_results.get('simulations', [])
    statistics = mc_results.get('statistics', {})
    
    if not simulations:
        print("No Monte Carlo simulations to plot"); return

    if DEBUG:
        print(f"Plotting {len(simulations)} Monte Carlo simulations...")

    # Extract returns data
    returns_data = [sim['total_return'] for sim in simulations if 'total_return' in sim]
    
    if not returns_data:
        print("No valid returns data found"); return

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Return Distribution',
            'Parameter Sensitivity', 
            'Simulation Paths',
            'Performance Comparison'
        ]
    )

    # Add histogram
    _add_histogram(fig, returns_data, statistics)
    
    # Add parameter sensitivity
    _add_parameter_sensitivity(fig, simulations)
    
    # Add simulation paths
    _add_simulation_paths(fig, simulations, mc_results)
    
    # Add performance comparison
    _add_mc_comparison(fig, statistics)

    fig.update_layout(
        title="Monte Carlo Analysis",
        template='plotly_dark'
    )
    
    fig.show()
    _print_mc_summary(statistics)

def _add_simulation_paths(fig: go.Figure, simulations: list, mc_results: Optional[Dict[str, Any]] = None) -> None:
    """Add simulation paths panel using path_matrix when available, else fallback to per-sim equity curves."""
    try:
        path_matrix = None
        if isinstance(mc_results, dict):
            # Preferred: use precomputed path_matrix for efficient plotting
            path_matrix = mc_results.get('path_matrix', None)
            # Also set reference for percentile helper if used later
            try:
                setattr(_add_mc_percentiles, "_path_matrix_ref", path_matrix)
            except Exception:
                pass

        # Try path_matrix first
        used = False
        if path_matrix is not None:
            used = _plot_from_path_matrix(fig, path_matrix)

        # Fallback to individual simulation equity curves when available
        if not used and simulations:
            _plot_from_individual_simulations(fig, simulations)
    except Exception as e:
        if DEBUG:
            print(f"Debug: _add_simulation_paths failed: {e}")

def _add_histogram(fig: go.Figure, returns_data: list, statistics: dict):
    """Add histogram of Monte Carlo returns."""
    fig.add_trace(go.Histogram(
        x=returns_data,
        nbinsx=30,
        name='MC Returns',
        marker_color='lightblue'
    ), row=1, col=1)

    # Add strategy performance line
    actual_return = statistics.get('actual_return')
    if actual_return:
        fig.add_vline(
            x=actual_return, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Strategy: {actual_return:.2f}%",
            row=1, col=1
        )

def _extract_parameter_data(simulations: list) -> Optional[dict]:
    """Extract parameter values and returns from simulations dynamically."""
    if not simulations:
        return None
    
    # Get all parameter names from first simulation
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
            
        # Add parameter values
        for name in param_names:
            param_data[name].append(parameters.get(name, 0))
        param_data['returns'].append(sim['total_return'])
    
    return param_data if param_data['returns'] else None


def _add_parameter_sensitivity(fig: go.Figure, simulations: list) -> None:
    """Add parameter sensitivity panel with clearer, more interpretable summaries.
    
    Debug logging added to validate:
      - Extracted parameter names and sample values
      - Returns scale (percent vs fraction) and sign
      - Binning labels used on x-axis to ensure they are strings, not numbers
    """
    if not simulations:
        print("Warning: No simulations provided for parameter sensitivity analysis")
        return

    # Extract
    param_data = _extract_parameter_data(simulations)
    if not param_data:
        print("Warning: No valid parameter data found for sensitivity analysis")
        return

    param_names = [n for n in param_data.keys() if n != 'returns']
    returns = pd.Series(param_data['returns'], dtype=float)
    if len(param_names) == 0 or returns.empty:
        print("Warning: No parameters found in simulation data")
        return

    # Diagnostic: show a compact snapshot
    try:
        sample_n = min(5, len(returns))
        print("[Diag] Sensitivity: params =", param_names)
        for name in param_names[:5]:
            vals = pd.Series(param_data[name])
            print(f"[Diag]   {name}: sample={list(vals.head(sample_n))}, unique_count={vals.nunique(dropna=True)}")
        print(f"[Diag]   returns: sample={list(returns.head(sample_n))}, min={returns.min():.4f}, max={returns.max():.4f}")
        # Heuristic scale check
        if returns.abs().median() <= 1.0:
            print("[Diag]   returns look like FRACTIONS; plotting assumes PERCENT. Upstream may need scaling x100.")
        else:
            print("[Diag]   returns look like PERCENT values.")
    except Exception as e:
        print(f"[Diag] Logging error in parameter sensitivity: {e}")

    # 1) Feature importance style: absolute Spearman correlation per parameter (bar chart)
    import numpy as _np
    import pandas as _pd
    try:
        scores = []
        for name in param_names:
            vals = _pd.Series(param_data[name])
            mask = ~(vals.isna() | returns.isna())
            if mask.sum() >= 3:
                corr = float(returns[mask].corr(vals[mask], method='spearman'))
                scores.append((name, abs(corr)))
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            names, vals = zip(*scores)
            fig.add_trace(go.Bar(
                x=[str(n) for n in list(names)],  # ensure categorical axis
                y=list(vals),
                marker_color='cornflowerblue',
                name='Param Importance (|Spearman|)'
            ), row=1, col=2)
            fig.update_yaxes(title_text="|Spearman|", row=1, col=2)
    except Exception as e:
        print(f"[Diag] Error computing importance scores: {e}")

    # 2) For top-1 parameter, show aggregated performance bands (violin) for interpretability
    if scores:
        top_param = scores[0][0]
        top_vals = _pd.Series(param_data[top_param])
        mask = ~(top_vals.isna() | returns.isna())
        if mask.sum() >= 5:
            # Bin numeric parameters into quantiles for readability
            q = _pd.qcut(top_vals[mask], q=min(5, mask.sum()), duplicates='drop')
            grouped = _pd.DataFrame({'bin': q.astype(str), 'ret': returns[mask].values})
            # Sort bins by median for ordered storytelling
            med = grouped.groupby('bin')['ret'].median().sort_values()
            ordered_bins = list(med.index)
            print(f"[Diag]   top_param={top_param}, bins={ordered_bins}")
            for b in ordered_bins:
                vals = grouped[grouped['bin'] == b]['ret'].values
                if len(vals) == 0:
                    continue
                # Ensure x is string categorical so Plotly uses labels not numeric codes
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

    # 3) Keep existing 2D scatter only when exactly two varying params and low cardinality (clear heatmap alternative)
    varying = [n for n in param_names if len(set([v for v in param_data[n] if v is not None])) > 1]
    if len(varying) == 2:
        x_param, y_param = varying
        x_vals = _pd.Series(param_data[x_param])
        y_vals = _pd.Series(param_data[y_param])
        mask = ~(x_vals.isna() | y_vals.isna() | returns.isna())
        if mask.sum() >= 4:
            # Aggregate to a pivot heatmap for clarity
            try:
                df = _pd.DataFrame({
                    'x': x_vals[mask].values,
                    'y': y_vals[mask].values,
                    'ret': returns[mask].values
                })
                # Reduce cardinality by bucketing if needed
                def bucket(s):
                    uniq = _np.unique(s)
                    if len(uniq) > 12:
                        # numeric bucketing
                        return _pd.qcut(s, q=8, duplicates='drop').astype(str)
                    return s.astype(str)
                df['xb'] = bucket(_pd.Series(df['x']))
                df['yb'] = bucket(_pd.Series(df['y']))
                piv = df.groupby(['yb', 'xb'])['ret'].mean().reset_index()
                piv_pivot = piv.pivot(index='yb', columns='xb', values='ret')
                # Build heatmap
                fig.add_trace(go.Heatmap(
                    z=piv_pivot.values,
                    x=[str(c) for c in list(piv_pivot.columns)],
                    y=[str(i) for i in list(piv_pivot.index)],
                    colorscale='RdYlGn',
                    colorbar=dict(title="Mean Return (%)"),
                    name='Param Heatmap',
                    showscale=True
                ), row=1, col=2)
                fig.update_xaxes(title_text=str(x_param), row=1, col=2)
                fig.update_yaxes(title_text=str(y_param), row=1, col=2)
            except Exception as e:
                print(f"[Diag] Error building 2D heatmap: {e}")

def _plot_from_path_matrix(fig: go.Figure, path_matrix) -> bool:
    """Plot simulation paths from path_matrix."""
    try:
        if DEBUG:
            print("Debug: Attempting to use path_matrix")
        arr = np.asarray(path_matrix, dtype=float)
        
        if arr.ndim == 2 and arr.size > 0:
            T, N = arr.shape
            if DEBUG:
                print(f"Debug: path_matrix shape: {T} time steps, {N} simulations")
            
            max_paths = min(50, N)
            path_indices = np.random.choice(N, max_paths, replace=False) if N > max_paths else np.arange(N)
            
            paths_added = 0
            time_axis = np.arange(T)
            
            for i, path_idx in enumerate(path_indices):
                try:
                    path_values = arr[:, path_idx]
                    
                    if path_values.size < 2 or not np.isfinite(path_values).any():
                        continue
                    
                    # Check for variation
                    valid_values = path_values[np.isfinite(path_values)]
                    if np.all(valid_values == valid_values[0]):
                        if DEBUG:
                            print(f"Debug: Path {path_idx} has no variation, skipping")
                        continue

                    opacity = max(0.1, min(0.3, 20.0 / max_paths))
                    color_rgba = f'rgba(100,149,237,{opacity})'
                    first = (i == 0)
                    fig.add_trace(go.Scatter(
                        x=time_axis, y=path_values, mode='lines',
                        line={'color': color_rgba, 'width': 1},
                        name='MC Simulation' if first else None, showlegend=first,
                        hovertemplate="Day: %{x}<br>Return: %{y:.2f}%<extra></extra>"
                    ), row=2, col=1)
                    paths_added += 1
                    
                except Exception as e:
                    if DEBUG:
                        print(f"Debug: Failed to plot path {path_idx}: {e}")
                    continue
            
            if paths_added > 0:
                if DEBUG:
                    print(f"Debug: Successfully plotted {paths_added} paths from path_matrix")
                return True
            else:
                if DEBUG:
                    print("Debug: No valid paths found in path_matrix")
                
    except Exception as e:
        if DEBUG:
            print(f"Debug: path_matrix plotting failed: {e}")

    return False


def _plot_from_individual_simulations(fig: go.Figure, simulations: list) -> bool:
    """Plot simulation paths from individual simulation equity curves."""
    if DEBUG:
        print("Debug: Attempting to plot individual simulation equity curves")
    max_paths = min(50, len(simulations))
    sample_indices = np.linspace(0, len(simulations)-1, max_paths, dtype=int)
    
    paths_added = 0
    for i, idx in enumerate(sample_indices):
        try:
            sim = simulations[idx]
            if not isinstance(sim, dict):
                continue
                
            equity_curve = sim.get('equity_curve', sim.get('portfolio_value', sim.get('value', None)))
            if equity_curve is None or len(equity_curve) == 0:
                continue
            
            eq_array = np.asarray(equity_curve, dtype=float)
            if eq_array.size < 2:
                continue
            
            valid_mask = np.isfinite(eq_array) & (eq_array != 0)
            if not valid_mask.any():
                continue
            
            first_valid_idx = np.where(valid_mask)[0][0]
            start_value = eq_array[first_valid_idx]
            normalized_curve = ((eq_array / start_value) - 1.0) * 100.0
            
            if not np.isfinite(normalized_curve).any():
                continue
            
            time_axis = np.arange(len(normalized_curve))
            opacity = max(0.1, min(0.3, 20.0 / max_paths))
            
            fig.add_trace(go.Scatter(
                x=time_axis, y=normalized_curve, mode='lines',
                line={'color': f'rgba(100,149,237,{opacity})', 'width': 1},
                name='MC Simulation' if i == 0 else None, showlegend=(i == 0),
                hovertemplate="Day: %{x}<br>Return: %{y:.2f}%<extra></extra>"
            ), row=2, col=1)
            paths_added += 1
            
        except Exception as e:
            if DEBUG:
                print(f"Debug: Failed to plot simulation {idx}: {e}")
            continue
    
    if paths_added > 0:
        if DEBUG:
            print(f"Debug: Successfully plotted {paths_added} simulation paths")
        return True

    return False


def _add_strategy_overlay(fig: go.Figure, statistics: dict) -> None:
    """Add strategy equity curve overlay if available."""
    strategy_equity = statistics.get('strategy_equity_curve', [])
    if strategy_equity and len(strategy_equity) > 1:
        try:
            se = np.asarray(strategy_equity, dtype=float)
            if se.size > 1 and np.isfinite(se[0]) and se[0] != 0:
                strat_norm = (se / se[0] - 1.0) * 100.0
                time_axis = np.arange(len(strat_norm))
                
                fig.add_trace(go.Scatter(
                    x=time_axis, y=strat_norm, mode='lines',
                    line={'color': 'red', 'width': 3}, name='Your Strategy', showlegend=True,
                    hovertemplate="Day: %{x}<br>Strategy Return: %{y:.2f}%<extra></extra>"
                ), row=2, col=1)
        except Exception as e:
            print(f"Debug: Failed to plot strategy equity curve: {e}")

def _add_mc_percentiles(fig: go.Figure,statistics: dict) -> None:
    """Add Monte Carlo simulation paths visualization."""
    
    simulations = statistics.get('simulations', [])
    path_matrix = getattr(_add_mc_percentiles, "_path_matrix_ref", None)
    
    if DEBUG:
        print(f"Debug: Found {len(simulations)} simulations, path_matrix: {path_matrix is not None}")
        if path_matrix is not None:
            print(f"Debug: path_matrix shape: {path_matrix.shape}, dtype: {path_matrix.dtype}")
            print(f"Debug: path_matrix sample values: {path_matrix[:5, :3] if path_matrix.size > 0 else 'empty'}")

        if simulations:
            sample_sim = simulations[0] if simulations else {}
            print(f"Debug: Sample simulation keys: {list(sample_sim.keys())}")
            print(f"Debug: Sample simulation equity_curve: {sample_sim.get('equity_curve', 'None')}")
    
    # Plot simulation paths
    if path_matrix is not None:
        _plot_from_path_matrix(fig, path_matrix)
    elif simulations:
        _plot_from_individual_simulations(fig, simulations)
    
    # Add strategy equity curve overlay if available
    _add_strategy_overlay(fig, statistics)
    


def _add_mc_comparison(fig: go.Figure, statistics: dict) -> None:
    """Add performance comparison subplot to Monte Carlo plot with dynamic scaling."""
    mean_random = statistics.get('mean_return', 0.0)
    std_random = statistics.get('std_return', 0.0)
    actual_return = statistics.get('actual_return', 0.0)

    # Guard for non-finite inputs
    values = np.array([mean_random, actual_return, mean_random + std_random, mean_random - std_random], dtype=float)
    if not np.isfinite(values).any():
        print("Error: Non-finite comparison values, skipping comparison panel")
        return

    categories = ['Random\nMean', 'Strategy', 'Random\n+1σ', 'Random\n-1σ']
    colors = ['lightgray', 'red', 'lightgreen', 'orange']

    # Add bars (compact)
    cats = categories
    vals = values.tolist()
    cols = colors
    for i, (category, val, color) in enumerate(zip(cats, vals, cols)):
        showlegend = i < 2
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

    # Zero reference
    fig.add_shape(
        type="line", x0=-0.5, x1=3.5, y0=0, y1=0,
        line={"dash": "dot", "color": "white", "width": 1},
        xref="x4", yref="y4"
    )

    # Dynamic y-axis scaling with 15% headroom
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

    # Interpretation label placed relative to scaled axis
    performance_text = "Outperforming" if actual_return > mean_random else "Underperforming"
    fig.add_annotation(
        x=1.5, y=ymax - (ymax - ymin) * 0.08, text=f"Strategy is {performance_text} vs Random",
        showarrow=False, xref="x4", yref="y4", font=dict(size=11, color="white"),
        bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1
    )


def _update_mc_axes(fig: go.Figure) -> None:
    """Update axis labels for Monte Carlo plot."""
    axis_updates = [
        (1, 1, "Return (%)", "Frequency"),
        (1, 2, "Parameter 1", "Parameter 2"),
        (2, 1, "Time Steps", "Return (%)"),
        (2, 2, "Category", "Return (%)")
    ]
    
    for row, col, x_title, y_title in axis_updates:
        fig.update_xaxes(title_text=x_title, row=row, col=col)
        fig.update_yaxes(title_text=y_title, row=row, col=col)
    
    # Set specific axis ranges for better visualization
    fig.update_yaxes(range=[0, 100], row=1, col=2)  # Parameter sensitivity
    fig.update_xaxes(tickangle=45, row=2, col=2)  # Rotate category labels
    
    # Auto-scale bar chart to prevent overflow - remove fixed range
    fig.update_yaxes(autorange=True, row=2, col=2)  # Auto-scale for proper fit


def _print_mc_summary(statistics: dict) -> None:
    """Print Monte Carlo parameter sensitivity summary."""
    print("\nMonte Carlo Parameter Sensitivity Analysis:")
    
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
        print(f"Walk-forward plot failed: {e}")
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

    # Plot asset performance data
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
        # Inline extraction (avoid bouncing to a tiny helper)
        train_returns = [w.get('asset_results', {}).get(asset, {}).get('train_return', 0) for w in windows]
        test_returns  = [w.get('asset_results', {}).get(asset, {}).get('test_return', 0)  for w in windows]

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


# Note: _extract_asset_data inlined into _plot_asset_returns to reduce code hopping and declutter.


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
        default_stats = None
        optimized_stats = None
        
        # Extract stats from both backtests - portfolios are stored directly now
        if 'default_backtest' in results:
            for symbol, timeframes in results['default_backtest'].items():
                for tf, portfolio in timeframes.items():
                    if hasattr(portfolio, 'stats'):
                        default_stats = portfolio.stats()
                        break
                if default_stats is not None:
                    break
        
        if 'full_backtest' in results:
            for symbol, timeframes in results['full_backtest'].items():
                for tf, portfolio in timeframes.items():
                    if hasattr(portfolio, 'stats'):
                        optimized_stats = portfolio.stats()
                        break
                if optimized_stats is not None:
                    break
        
        if default_stats is None or optimized_stats is None:
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
        
        # Add bars for comparison
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

        # Print improvement summary
        print("\nOptimization Impact Summary:")
        print(f"   Return: {default_values[0]:.2f}% → {optimized_values[0]:.2f}% ({optimized_values[0] - default_values[0]:+.2f}%)")
        print(f"   Sharpe: {default_values[1]:.3f} → {optimized_values[1]:.3f} ({optimized_values[1] - default_values[1]:+.3f})")
        print(f"   Max DD: {default_values[2]:.2f}% → {optimized_values[2]:.2f}% ({optimized_values[2] - default_values[2]:+.2f}%)")
        print(f"   Win Rate: {default_values[3]:.1f}% → {optimized_values[3]:.1f}% ({optimized_values[3] - default_values[3]:+.1f}%)")
        
        return {"success": True}
        
    except Exception as e:
        print(f"Comparison plot failed: {e}")
        return {"success": False, "error": str(e)}


def create_visualizations(results: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """Create enhanced visualizations with default vs optimized comparison."""
    try:
        portfolios = _extract_portfolios_from_results(results)
        
        if portfolios:
            plot_results = plot_comprehensive_analysis(
                portfolios, strategy_name,
                results.get('monte_carlo', {}),
                results.get('walkforward', {})
            )
            
            # Add comparison plot if we have both default and optimized results
            if 'default_backtest' in results and 'full_backtest' in results:
                print("Creating Default vs Optimized comparison...")
                comparison_results = create_comparison_plot(results, strategy_name)
                if plot_results is None:
                    plot_results = {}
                plot_results['comparison'] = comparison_results
            
            return plot_results or {}
        
        return {}
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        return {"success": False, "error": str(e)}


def _extract_portfolios_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract portfolios from results structure."""
    portfolios = {}
    
    # Get default results first to control order
    if 'default_backtest' in results:
        for symbol, timeframes in results['default_backtest'].items():
            for timeframe, portfolio in timeframes.items():
                if hasattr(portfolio, 'stats'):
                    portfolios[f"{symbol}_{timeframe}_default"] = portfolio

    # Get optimized results after defaults
    if 'full_backtest' in results:
        for symbol, timeframes in results['full_backtest'].items():
            for timeframe, portfolio in timeframes.items():
                if hasattr(portfolio, 'stats'):
                    portfolios[f"{symbol}_{timeframe}_optimized"] = portfolio
    
    return portfolios
