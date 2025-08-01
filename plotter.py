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
                                wf_results: Optional[Dict[str, Any]] = None,
                                show: str = "auto") -> Dict[str, Any]:
    """Plot everything: portfolios, Monte Carlo, and walk-forward analysis.
    
    Args:
        portfolios: dict of name -> vbt.Portfolio
        mc_results: results from optimizer.run_monte_carlo_analysis
        wf_results: walk-forward results
        show: "auto" | "monte_carlo" | "optimized". If "auto", defaults to Monte Carlo if available.
    """
    try:
        if not portfolios:
            print("Warning: No portfolios provided")
            return {"success": False, "error": "No portfolios provided"}

        if not isinstance(portfolios, dict):
            portfolios = {"Portfolio": portfolios}

        # Decide default view
        has_mc = bool(mc_results) and ('error' not in mc_results)
        if show == "auto":
            show = "monte_carlo" if has_mc else "optimized"

        # Always render portfolios first for consistency
        _plot_portfolios(portfolios, strategy_name)

        # MC visualization first if requested/available
        if has_mc:
            print("Plotting Monte Carlo analysis...")
            _plot_monte_carlo(mc_results, preferred_view=show)

        if wf_results and 'error' not in wf_results:
            print("Plotting walk-forward analysis...")
            _plot_walkforward(wf_results)

        print("Comprehensive analysis completed.")
        return {"success": True}

    except Exception as e:
        print(f"Comprehensive analysis failed: {e}")
        return {"success": False, "error": str(e)}

def _plot_portfolios(portfolios: Dict[str, Any], strategy_name: str) -> None:
    """Plot individual portfolios and comparison using VectorBT native functionality."""
    print("Creating portfolio visualizations...")

    # Plot each portfolio individually
    for name, portfolio in portfolios.items():
        if not _validate_portfolio(portfolio, name):
            continue
            
        try:
            _plot_single_portfolio(portfolio, name, strategy_name)
        except Exception as e:
            print(f"Failed to plot {name}: {e}")
            continue

    # Create comparison plot if multiple portfolios
    if len(portfolios) > 1:
        _create_vectorbt_comparison(portfolios, strategy_name)


def _validate_portfolio(portfolio: Any, name: str) -> bool:
    """Validate portfolio before plotting."""
    if portfolio is None:
        print(f"Skipping {name}: portfolio is None")
        return False
        
    try:
        stats = portfolio.stats()
        if stats is None or len(stats) == 0 or not stats.get('Total Trades', 0):
            print(f"Skipping {name}: no trades")
            return False

        value_series = portfolio.value()
        if value_series is None or len(value_series) == 0 or value_series.isna().all():
            print(f"Skipping {name}: no valid value data")
            return False

        # Make orders validation tolerant - feature-detect via hasattr
        if hasattr(portfolio, 'orders') and portfolio.orders is None:
            print(f"Warning: Portfolio orders is None for {name}, but proceeding with valid stats and value")
            # Don't return False - allow plotting if stats and value are valid

        return True
        
    except Exception as e:
        print(f"Portfolio validation failed for {name}: {e}")
        return False


def _plot_single_portfolio(portfolio: Any, name: str, strategy_name: str) -> None:
    """Plot a single portfolio."""
    print(f"Creating VectorBT plot for {name}...")
    print(f"Portfolio validation passed for {name}")
    
    try:
        fig = portfolio.plot(template='plotly_dark')
        fig.update_layout(
            title=f"{strategy_name} Strategy - {name} Performance",
            height=600,
            width=1200
        )
        fig.show()
        
    except Exception as plot_error:
        print(f"VectorBT plot failed for {name}: {plot_error}")
        print(f"   Portfolio type: {type(portfolio)}")
        print(f"   Portfolio stats available: {hasattr(portfolio, 'stats')}")
        
def _create_vectorbt_comparison(portfolios: Dict[str, Any], strategy_name: str):
    """Create VectorBT native comparison plot for multiple portfolios."""
    try:
        print("Creating portfolio comparison...")
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
                print(f"Failed to add {name} to comparison: {e}")

        fig.update_layout(
            title=f"{strategy_name} Strategy - Portfolio Comparison (Normalized)",
            yaxis_title="Normalized Value (Start = 100)", xaxis_title="Date",
            template='plotly_dark', height=600, width=1200
        )
        fig.show()
    

    except Exception as e:
        print(f"VectorBT comparison plot failed: {e}")

def _plot_monte_carlo(mc_results: Dict[str, Any], preferred_view: str = "monte_carlo") -> Dict[str, Any]:
    """Plot Monte Carlo results with parameter sensitivity analysis and path-matrix support."""
    try:
        simulations = mc_results.get('simulations', [])
        statistics = mc_results.get('statistics', {})
        path_matrix = mc_results.get('path_matrix', None)

        # Robustly extract numeric returns_data for histogram
        returns_data = []
        for sim in simulations:
            r = sim.get('total_return')
            if r is None:
                continue
            try:
                r = float(r)
            except Exception:
                continue
            if np.isfinite(r):
                returns_data.append(r)

        # Ensure numpy array for path_matrix
        if path_matrix is not None:
            try:
                path_matrix = np.asarray(path_matrix, dtype=float)
            except Exception:
                path_matrix = None

        if (not simulations or len(returns_data) == 0) and (path_matrix is None or path_matrix.size == 0):
            print("Error: No simulations or path matrix available for Monte Carlo plot")
            return {"success": False, "reason": "no_simulations"}

        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Monte Carlo Return Distribution',
                'Parameter Sensitivity Analysis',
                'Monte Carlo Simulation Paths',
                'Performance vs Random'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Add histogram and parameter sensitivity
        _add_mc_histogram(fig, returns_data, statistics)
        _add_parameter_sensitivity(fig, simulations)

        # Inject path_matrix for downstream function (avoids copying)
        setattr(_add_mc_percentiles, "_path_matrix_ref", path_matrix)

        # Add path view using path_matrix when available; fallback to sampled equity_curve traces
        _add_mc_percentiles(fig, returns_data, statistics)

        # Comparison panel (auto-scale y dynamically from data)
        _add_mc_comparison(fig, statistics)

        # Layout
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
        print(f"Monte Carlo plot failed: {e}")
        return {"success": False, "error": str(e)}


def _add_mc_histogram(fig: go.Figure, returns_data: list, statistics: dict) -> None:
    """Add robust histogram for MC returns with NaN/Inf guards."""
    # Accept both list and np array
    try:
        arr = np.asarray(returns_data, dtype=float)
    except Exception:
        return
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        print("Error: No finite returns for histogram")
        return

    # Use FD rule, fallback to sqrt; ensure at least 5 bins
    try:
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / np.cbrt(arr.size) if iqr > 0 else 0
        if bin_width > 0:
            n_bins = int(np.clip(np.ceil((arr.max() - arr.min()) / bin_width), 5, 100))
        else:
            n_bins = int(np.clip(np.sqrt(arr.size), 5, 50))
    except Exception:
        n_bins = int(np.clip(np.sqrt(arr.size), 5, 50))

    fig.add_trace(go.Histogram(
        x=arr.tolist(),
        nbinsx=int(n_bins),
        name='Random Returns',
        opacity=0.8,
        marker_color='lightblue',
        showlegend=True
    ), row=1, col=1)

    # Strategy marker
    actual_return = statistics.get('actual_return')
    if actual_return is not None:
        try:
            ar = float(actual_return)
        except Exception:
            ar = None
        if ar is not None and np.isfinite(ar):
            hist_counts, _ = np.histogram(arr, bins=int(n_bins))
            max_count = float(hist_counts.max()) if hist_counts.size > 0 else 1.0
            fig.add_shape(
                type="line", x0=ar, x1=ar, y0=0, y1=max_count,
                line={"dash": "dash", "color": "red", "width": 3},
                xref="x1", yref="y1"
            )
            fig.add_annotation(
                x=ar, y=max_count * 0.9, text=f"Strategy: {ar:.2f}%",
                showarrow=True, arrowcolor="red", xref="x1", yref="y1"
            )


def _add_parameter_sensitivity(fig: go.Figure, simulations: list) -> None:
    """Add parameter sensitivity analysis subplot."""
    if not simulations:
        print("Warning: No simulations provided for parameter sensitivity analysis")
        return

    # Extract parameter values and returns with better handling
    param1_values = []
    param2_values = []
    returns = []
    
    # Try different parameter key names that might exist
    possible_param_keys = [
        ['param1', 'param2'], ['parameter1', 'parameter2'],
        ['p1', 'p2'], ['params', None], ['parameters', None]
    ]
    
    for sim in simulations:
        # Try to find parameter values in simulation data
        p1, p2 = None, None
        
        for key_pair in possible_param_keys:
            if key_pair[0] in sim:
                if isinstance(sim[key_pair[0]], (list, tuple)) and len(sim[key_pair[0]]) >= 2:
                    p1, p2 = sim[key_pair[0]][0], sim[key_pair[0]][1]
                    break
                elif key_pair[1] and key_pair[1] in sim:
                    p1, p2 = sim[key_pair[0]], sim[key_pair[1]]
                    break
                elif isinstance(sim[key_pair[0]], (int, float)):
                    p1 = sim[key_pair[0]]
                    # Try to find second parameter
                    if 'param2' in sim:
                        p2 = sim['param2']
                    elif len(simulations) > 1:
                        # Use simulation index as second parameter if no param2
                        p2 = simulations.index(sim)
        
        # If still no parameters found, use defaults
        if p1 is None:
            p1 = sim.get('param1', sim.get('parameter1', sim.get('p1', 0)))
        if p2 is None:
            p2 = sim.get('param2', sim.get('parameter2', sim.get('p2', 0)))
            
        param1_values.append(p1)
        param2_values.append(p2)
        returns.append(sim['total_return'])
    
    # Debug: Print data characteristics
    print(f"   Debug - Param1 values: {len(param1_values)} values, range: [{min(param1_values) if param1_values else 'N/A'}, {max(param1_values) if param1_values else 'N/A'}]")
    print(f"   Debug - Param2 values: {len(param2_values)} values, range: [{min(param2_values) if param2_values else 'N/A'}, {max(param2_values) if param2_values else 'N/A'}]")
    print(f"   Debug - Returns: {len(returns)} values, range: [{min(returns):.3f}%, {max(returns):.3f}%]" if returns else "   Debug - Returns: No data")

    # Check if we have meaningful parameter variation
    param1_unique = len(set(param1_values)) > 1
    param2_unique = len(set(param2_values)) > 1
    
    if not param1_unique and not param2_unique:
        print("Warning: No parameter variation detected - all simulations used same parameters, skipping parameter sensitivity subplot")
        return

    # Filter out invalid values (NaN, None, etc.)
    valid_simulations = []
    for p1, p2, ret in zip(param1_values, param2_values, returns):
        if (p1 is not None and not pd.isna(p1) and
            p2 is not None and not pd.isna(p2) and
            ret is not None and not pd.isna(ret)):
            valid_simulations.append((p1, p2, ret))

    if not valid_simulations:
        print("Warning: No valid parameter combinations found in simulation data, skipping parameter sensitivity subplot")
        return

    param1_values, param2_values, returns = zip(*valid_simulations)

    # Create scatter plot with improved visualization
    fig.add_trace(go.Scatter(
        x=param1_values, y=param2_values,
        mode='markers',
        marker=dict(
            size=10,
            color=returns,
            colorscale='RdYlGn',  # Red-Yellow-Green scale (red=bad, green=good)
            colorbar=dict(
                title="Return (%)", 
                x=0.52,  # Position colorbar to the right of parameter sensitivity plot
                y=0.85,  # Position at top
                len=0.3,  # Make it shorter
                thickness=15,
                xpad=10  # Add padding from the plot
            ),
            showscale=True,
            line=dict(width=1, color='white')  # Add white border to markers
        ),
        text=[f"P1: {p1}<br>P2: {p2}<br>Return: {r:.2f}%" for p1, p2, r in zip(param1_values, param2_values, returns)],
        hovertemplate="<b>Parameter 1:</b> %{x}<br>" +
                      "<b>Parameter 2:</b> %{y}<br>" +
                      "<b>Return:</b> %{marker.color:.2f}%<br>" +
                      "<extra></extra>",
        showlegend=False,
        name='Parameter Combinations'
    ), row=1, col=2)

    # Add best performance marker
    best_idx = returns.index(max(returns))
    fig.add_trace(go.Scatter(
        x=[param1_values[best_idx]], y=[param2_values[best_idx]],
        mode='markers',
        marker=dict(size=15, color='gold', symbol='star', line=dict(width=2, color='black')),
        name='Best Performance',
        showlegend=True,
        hovertemplate=f"<b>Best Combination</b><br>P1: {param1_values[best_idx]}<br>P2: {param2_values[best_idx]}<br>Return: {returns[best_idx]:.2f}%<extra></extra>"
    ), row=1, col=2)

    # Update axes labels with better formatting and positioning
    fig.update_xaxes(title_text="Parameter 1", row=1, col=2, showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(
        title_text="Parameter 2", 
        row=1, col=2, 
        showgrid=True, 
        gridcolor='rgba(255,255,255,0.1)',
        title_standoff=25  # Add more space between y-axis title and plot to avoid colorbar overlap
    )


def _add_mc_percentiles(fig: go.Figure, returns_data: list, statistics: dict) -> None:
    """Add Monte Carlo simulation paths visualization using path_matrix if present."""
    # Try to access path_matrix injected by _plot_monte_carlo
    path_matrix = getattr(_add_mc_percentiles, "_path_matrix_ref", None)

    if path_matrix is not None:
        try:
            arr = np.asarray(path_matrix)
            if arr.ndim == 2 and arr.size > 0:
                T, N = arr.shape
                max_paths = min(100, N)
                cols = np.linspace(0, N - 1, max_paths, dtype=int) if N > max_paths else np.arange(N, dtype=int)
                max_T = 1000
                t_idx = np.arange(0, T, int(np.ceil(T / max_T)), dtype=int) if T > max_T else np.arange(T, dtype=int)

                for j, c in enumerate(cols):
                    series = arr[t_idx, c]
                    if series.size == 0 or not np.isfinite(series).any():
                        continue
                    fig.add_trace(go.Scatter(
                        y=series.astype(float),
                        mode='lines',
                        line={'color': 'rgba(100,149,237,0.25)', 'width': 1},
                        name='MC Simulation' if j == 0 else None,
                        showlegend=(j == 0),
                        hovertemplate="MC Path<br>Return: %{y:.2f}%<extra></extra>"
                    ), row=2, col=1)

                # Overlay strategy equity if provided
                strategy_equity = statistics.get('strategy_equity_curve', [])
                if strategy_equity:
                    se = np.asarray(strategy_equity, dtype=float)
                    if se.size > 1 and np.isfinite(se[0]) and se[0] != 0:
                        strat_norm = (se / se[0] - 1.0) * 100.0
                        if strat_norm.size > t_idx.size:
                            strat_norm = strat_norm[:t_idx.size]
                        fig.add_trace(go.Scatter(
                            y=strat_norm.astype(float),
                            mode='lines',
                            line={'color': 'red', 'width': 3},
                            name='Your Strategy',
                            showlegend=True
                        ), row=2, col=1)
                return
        except Exception as e:
            print(f"MC path_matrix plotting failed: {e}")

    # Legacy fallback: sample equity_curve from simulations if available
    simulations = statistics.get('simulations', [])
    if isinstance(simulations, list) and len(simulations) > 0:
        max_paths = min(50, len(simulations))
        sample_indices = np.linspace(0, len(simulations)-1, max_paths, dtype=int)
        for i, idx in enumerate(sample_indices):
            sim = simulations[idx]
            eq = sim.get('equity_curve')
            if eq is None or len(eq) == 0:
                continue
            s0 = eq[0]
            if s0 is None or s0 == 0 or not np.isfinite(s0):
                continue
            normalized_curve = ((np.asarray(eq, dtype=float) / float(s0)) - 1.0) * 100.0
            normalized_curve = normalized_curve[np.isfinite(normalized_curve)]
            if normalized_curve.size == 0:
                continue
            fig.add_trace(go.Scatter(
                y=normalized_curve.tolist(),
                mode='lines',
                line={'color': 'rgba(100,149,237,0.25)', 'width': 1},
                name='MC Simulation' if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate="MC Path<br>Return: %{y:.2f}%<extra></extra>"
            ), row=2, col=1)
        # Strategy overlay
        strategy_equity = statistics.get('strategy_equity_curve', [])
        if strategy_equity:
            se = np.asarray(strategy_equity, dtype=float)
            if se.size > 1 and np.isfinite(se[0]) and se[0] != 0:
                strat_norm = (se / se[0] - 1.0) * 100.0
                fig.add_trace(go.Scatter(
                    y=strat_norm.astype(float),
                    mode='lines',
                    line={'color': 'red', 'width': 3},
                    name='Your Strategy',
                    showlegend=True
                ), row=2, col=1)
        return

    # Last resort: percentile curve only if no paths available
    if returns_data:
        arr = np.asarray(returns_data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            percentile_values = [float(np.percentile(arr, p)) for p in percentiles]
            fig.add_trace(go.Scatter(
                x=percentiles, y=percentile_values, mode='lines+markers',
                name='Random Performance Curve', line={'color': 'cyan', 'width': 3},
                marker={'size': 8}, showlegend=True
            ), row=2, col=1)
            actual_return = statistics.get('actual_return')
            if actual_return is not None and np.isfinite(actual_return):
                percentile_rank = statistics.get('percentile_rank', 50)
                fig.add_trace(go.Scatter(
                    x=[percentile_rank], y=[actual_return], mode='markers',
                    name='Your Strategy', marker={'color': 'red', 'size': 12, 'symbol': 'diamond'},
                    showlegend=True
                ), row=2, col=1)


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

    # Add bars
    for i, (category, val, color) in enumerate(zip(categories, values.tolist(), colors)):
        showlegend = i < 2
        # Use 'outside' when absolute value is small to keep labels readable
        text_position = 'outside' if abs(val) < 1 else 'inside'
        text_color = 'white' if text_position == 'inside' else color

        fig.add_trace(go.Bar(
            x=[category], y=[val], marker_color=color,
            name=category.replace('\n', ' ') if showlegend else None,
            showlegend=showlegend,
            text=f'{val:.2f}%',
            textposition=text_position,
            textfont=dict(color=text_color, size=10)
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
        height=None, width=None,
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
        import plotly.graph_objects as go
        
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
            template='plotly_dark',
            height=600
        )
        fig.show()

        # Print improvement summary
        print(f"\nOptimization Impact Summary:")
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
        # Collect portfolios for plotting
        portfolios = {}
        default_portfolios = {}
        
        # Get optimized results - portfolios are stored directly now
        if 'full_backtest' in results:
            for symbol, timeframes in results['full_backtest'].items():
                for timeframe, portfolio in timeframes.items():
                    if hasattr(portfolio, 'stats'):
                        portfolios[f"{symbol}_{timeframe}_optimized"] = portfolio
        
        # Get default results (if available) - portfolios are stored directly now
        if 'default_backtest' in results:
            for symbol, timeframes in results['default_backtest'].items():
                for timeframe, portfolio in timeframes.items():
                    if hasattr(portfolio, 'stats'):
                        default_portfolios[f"{symbol}_{timeframe}_default"] = portfolio
                        portfolios[f"{symbol}_{timeframe}_default"] = portfolio
        
        if portfolios:
            # Enhanced plotting with comparison data
            plot_results = plot_comprehensive_analysis(
                portfolios,  # Use positional argument
                strategy_name,
                results.get('monte_carlo', {}),
                results.get('walkforward', {})
            )
            
            # Create default vs optimized comparison plot if both exist
            if default_portfolios and len(portfolios) > len(default_portfolios):
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