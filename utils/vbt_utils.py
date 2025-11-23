import pandas as pd
import vectorbt as vbt
import numpy as np
import plotly.graph_objects as go

# Global Configuration
pd.set_option('future.no_silent_downcasting', True)

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def get_annualization_factor(timeframe="4h"):
    """Returns the annualization factor based on the timeframe string."""
    tf = timeframe.lower()
    if tf == "1h": return np.sqrt(365 * 24) #noqa
    if tf == "4h": return np.sqrt(365 * 6) #noqa
    if tf == "1d": return np.sqrt(365) #noqa
    if tf == "15m": return np.sqrt(365 * 96) #noqa    # Default fallback
    return np.sqrt(365)

# ==============================================================================
# 2. SIZING BUILDERS (The "Strategy Factory")
# ==============================================================================

def build_volatility_sizing_matrix(close, trend_signal, target_vols, window_bars_range, timeframe="4h"):
    """
    Builder for Strategy 1: Volatility Targeting.
    Generates sizing based on Target Vol / Realized Vol.
    """
    returns = close.pct_change()
    ann_factor = get_annualization_factor(timeframe)
    
    sizers = []
    combo_labels = []

    print(f"⚡ Building Volatility Matrix: {len(target_vols)} Targets x {len(window_bars_range)} Windows...")
    
    for w_bars in window_bars_range:
        # Calculate Rolling Volatility (Annualized)
        vol_rolling = returns.rolling(window=w_bars).std() * ann_factor
        
        for target_vol in target_vols:
            # Formula: Target / Actual (Shifted to avoid lookahead)
            weight = (target_vol / vol_rolling).vbt.fshift(1)
            weight = weight.fillna(0.0).replace([np.inf, -np.inf], 0.0).clip(upper=1.0)
            
            # Apply Trend Filter (Go to Cash if False)
            if trend_signal is not None:
                weight = weight.mask(~trend_signal, 0.0)
            
            sizers.append(weight)
            combo_labels.append((target_vol, w_bars))

    # Concatenate into one giant DataFrame
    sizing_df = pd.concat(sizers, axis=1)
    sizing_df.columns = pd.MultiIndex.from_tuples(combo_labels, names=['Target_Vol', 'Window_Bars'])
    
    # Tile (Broadcast) signals to match matrix shape
    if trend_signal is not None:
        # Ensure boolean type
        entries_base = (trend_signal & ~trend_signal.shift(1).fillna(False)).astype(bool)
        exits_base = (~trend_signal & trend_signal.shift(1).fillna(False)).astype(bool)
        
        entries_tiled = entries_base.vbt.tile(len(combo_labels), keys=sizing_df.columns)
        exits_tiled = exits_base.vbt.tile(len(combo_labels), keys=sizing_df.columns)
        
        return sizing_df, entries_tiled, exits_tiled
    
    return sizing_df, None, None

def apply_atr_sizing_grid(entries, exits, data_close, data_high, data_low, 
                          atr_window=14, atr_stop_mult=3.0, risk_range=[0.01, 0.02]):
    """
    Builder for Strategy 2: ATR/Kelly Sizing on top of ANY strategy.
    Takes existing signals (single or multiple columns) and explodes them
    by applying different Risk % settings.
    """
    # 1. Calculate ATR (One calculation for all)
    atr = vbt.ATR.run(data_high, data_low, data_close, window=atr_window).atr
    
    # 2. Calculate Stop Distance (Risk per Share)
    # Shifted 1 bar to ensure we calculate based on PREVIOUS bar's ATR
    risk_per_share = (atr * atr_stop_mult).vbt.fshift(1)
    
    sizers = []
    labels = []
    
    # Normalize inputs to DataFrames
    if isinstance(entries, pd.Series):
        entries = entries.to_frame()
        exits = exits.to_frame()
    
    print(f"⚡ Building ATR Sizing Matrix: {entries.shape[1]} Strategies x {len(risk_range)} Risk Levels...")

    # 3. Cartesian Product: Strategy Params x Risk Params
    for col in entries.columns:
        for risk_pct in risk_range:
            
            # Sizing Math: 
            # We want to risk X% of Equity. 
            # Stop Loss distance % = (ATR * Mult) / Price
            # Target Exposure = Risk % / Stop Loss Distance %
            
            current_price = data_close.vbt.fshift(1) # Price at decision time
            stop_loss_pct = risk_per_share / current_price
            
            # Calculate Weight (Target Percent of Equity)
            target_exposure = risk_pct / stop_loss_pct
            
            # Safety: Clean NaNs/Infs and Cap at 100% (Clip)
            target_exposure = target_exposure.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=1.0)
            
            # Apply Signals Mask (Only hold size when Entry is Active)
            # Note: For pure VBT backtesting, we typically pass the size array 
            # and the Entry triggers. VBT handles the holding logic.
            # But to use 'TargetPercent', we need valid weights.
            # Here we just prepare the weight array.
            
            sizers.append(target_exposure)
            labels.append((col, risk_pct))

    # 4. Final Assembly
    size_df = pd.concat(sizers, axis=1)
    size_df.columns = pd.MultiIndex.from_tuples(labels, names=['Strat_Param', 'Risk_Pct'])
    
    # 5. Signal Tiling (Manual Map)
    # We duplicate the original entry/exit signals to match the new matrix columns
    entries_tiled = pd.DataFrame(index=entries.index, columns=size_df.columns)
    exits_tiled = pd.DataFrame(index=exits.index, columns=size_df.columns)
    
    for col, risk in size_df.columns:
        # Map the original strategy column to the new Risk-Adjusted column
        entries_tiled[(col, risk)] = entries[col]
        exits_tiled[(col, risk)] = exits[col]
        
    return size_df, entries_tiled, exits_tiled

# ==============================================================================
# 3. ANALYSIS & VISUALIZATION (Generic)
# ==============================================================================

def analyze_optimization(portfolio, metric="Calmar"):
    """
    Generic analyzer that works for ANY optimization strategy.
    Automatically detects parameters from the MultiIndex.
    
    Args:
        portfolio: The VBT portfolio object
        metric: Metric to sort by ('Calmar', 'Sharpe', 'Total_Return', 'Sortino')
    """
    print(f"📊 Analyzing Results (Sorting by {metric})...")
    
    # 1. Calculate basic stats
    stats = pd.DataFrame({
        'Total_Return': portfolio.total_return(),
        'Max_DD': portfolio.max_drawdown(),
        'Sharpe': portfolio.sharpe_ratio(),
        'Calmar': portfolio.calmar_ratio(),
        'Sortino': portfolio.sortino_ratio(),
        'Win_Rate': portfolio.trades.win_rate()
    })
    
    # 2. Handle MultiIndex (Flatten params into columns)
    if isinstance(stats.index, pd.MultiIndex):
        # Reset index to move params from Index to Columns
        stats = stats.reset_index()
    
    # 3. Find the Winner
    best_idx = stats[metric].idxmax()
    best_run = stats.loc[best_idx]
    
    print("\n" + "="*50)
    print(f"🏆 BEST RUN ({metric}) 🏆")
    print("="*50)
    
    # Dynamically print parameters (excluding the metrics columns)
    metric_cols = ['Total_Return', 'Max_DD', 'Sharpe', 'Calmar', 'Sortino', 'Win_Rate']
    print("PARAMETERS:")
    for col in stats.columns:
        if col not in metric_cols:
            val = best_run[col]
            # Format formatting if float
            if isinstance(val, float):
                print(f" -> {col}: {val:.4f}")
            else:
                print(f" -> {col}: {val}")
                
    print("-" * 30)
    print("METRICS:")
    print(f" -> Calmar Ratio:   {best_run['Calmar']:.4f}")
    print(f" -> Sharpe Ratio:   {best_run['Sharpe']:.4f}")
    print(f" -> Total Return:   {best_run['Total_Return']:.2%}")
    print(f" -> Max Drawdown:   {best_run['Max_DD']:.2%}")
    print("="*50)
    
    return best_run, stats

def plot_heatmap(results_df, x_col, y_col, metric='Calmar'):
    """
    Generic Heatmap plotter.
    
    Args:
        results_df: The dataframe returned by analyze_optimization
        x_col: Name of the column for X axis (e.g. 'Window_Bars' or 'Risk_Pct')
        y_col: Name of the column for Y axis (e.g. 'Target_Vol' or 'Strat_Param')
        metric: Value to visualize (e.g. 'Calmar')
    """
    # Check if columns exist
    if x_col not in results_df.columns or y_col not in results_df.columns:
        print(f"❌ Error: Columns '{x_col}' or '{y_col}' not found in results.")
        print(f"Available columns: {results_df.columns.tolist()}")
        return

    pivot = results_df.pivot(index=y_col, columns=x_col, values=metric)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        colorbar=dict(title=metric)
    ))
    
    fig.update_layout(
        title=f'Optimization Heatmap: {metric} ({y_col} vs {x_col})',
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_dark',
        height=600
    )
    fig.show()