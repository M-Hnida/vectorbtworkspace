from typing import Dict,Callable
import pandas as pd
import scipy.optimize
import vectorbt as vbt
# =============================================================================
# WALK-FORWARD OPTIMIZATION TEMPLATE
# =============================================================================
def walk_forward_optimize(data: pd.DataFrame, signal_function: Callable,
                         param_grid: Dict, train_ratio: float = 0.75,
                         n_splits: int = 10, metric: str = 'sharpe_ratio') -> Dict[]:
    """
    Vectorbt-optimized walk-forward optimization.
    
    Args:
        data: OHLC DataFrame
        signal_function: Signal generating function
        param_grid: Parameter grid for optimization
        train_ratio: Training set ratio (0.75 = 75% train, 25% test)
        n_splits: Number of WFO splits
        metric: Optimization metric ('sharpe_ratio', 'total_return', etc.)
    
    Returns:
        Dict with WFO results and out-of-sample performance
    """
    
    def get_optimized_split(total_length, frac, n):
        """Calculate optimal split sizes using linear programming."""
        d = total_length / (frac + n * (1 - frac))
        di = frac * d
        do = (1 - frac) * d
        
        # Linear optimization for split sizes
        c = [-(1/frac - 1), 1]
        Aeq = [[1, n]]
        Aub = [[-1, 1], [(1/frac - 1), -1]]
        beq = [total_length]
        bub = [0, 0]
        x0_bounds = (di * 0.5, di * 1.5)
        x1_bounds = (do * 0.5, do * 1.5)
        
        res = scipy.optimize.linprog(
            c, A_eq=Aeq, b_eq=beq, A_ub=Aub, b_ub=bub, 
            bounds=(x0_bounds, x1_bounds),
            integrality=[1, 1], method='highs'
        )
        
        return int(res.x[0]), int(res.x[1])
    
    def wfo_split_func(splits, bounds, index, length_IS, length_OOS):
        """Generate walk-forward splits."""
        if len(splits) == 0:
            new_split = (slice(0, length_IS), slice(length_IS, length_OOS + length_IS))
        else:
            prev_end = bounds[-1][1].stop
            new_split = (
                slice(prev_end - length_IS, prev_end),
                slice(prev_end, prev_end + length_OOS)
            )
        
        if new_split[1].stop > len(index):
            return None
        return new_split
    
    # Calculate optimal split sizes
    length_IS, length_OOS = get_optimized_split(len(data), train_ratio, n_splits)
    
    # Create vectorbt splitter
    splitter = vbt.Splitter.from_split_func(
        data.index,
        wfo_split_func,
        split_args=(vbt.Rep("splits"), vbt.Rep("bounds"), vbt.Rep("index")),
        split_kwargs={'length_IS': length_IS, 'length_OOS': length_OOS},
        set_labels=["IS", "OOS"]
    )
    
    # Create vectorized strategy using IndicatorFactory
    def strategy_wrapper(close, **params):
        """Wrapper for signal function to work with vectorbt."""
        # Convert close to DataFrame if needed
        if isinstance(close, np.ndarray):
            close_df = pd.DataFrame({'close': close}, index=data.index[:len(close)])
        else:
            close_df = pd.DataFrame({'close': close})
            
        # Add OHLC columns if available
        if len(data.columns) >= 4:
            close_df['open'] = data['open'].iloc[:len(close_df)]
            close_df['high'] = data['high'].iloc[:len(close_df)]
            close_df['low'] = data['low'].iloc[:len(close_df)]
        
        signals = signal_function(close_df, **params)
        
        # Return entries and exits
        entries = signals.entries.reindex(close_df.index, fill_value=False)
        exits = signals.exits.reindex(close_df.index, fill_value=False)
        
        return entries, exits
    
    # Create vectorbt IndicatorFactory
    strategy_factory = vbt.IndicatorFactory(
        class_name='WFOStrategy',
        short_name='wfo_strat',
        input_names=['close'],
        param_names=list(param_grid.keys()),
        output_names=['entries', 'exits']
    ).from_apply_func(strategy_wrapper, **{k: v[0] for k, v in param_grid.items()})
    
    # Run strategy with parameter grid
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    strategy_results = strategy_factory.run(
        data['close'],
        **{name: values for name, values in param_grid.items()},
        param_product=True
    )
    
    # Performance evaluation function
    def evaluate_performance(data_slice, strategy_slice, metric_name='sharpe_ratio'):
        """Evaluate strategy performance on data slice."""
        pf = vbt.Portfolio.from_signals(
            data_slice,
            entries=strategy_slice.entries,
            exits=strategy_slice.exits,
            init_cash=DEFAULT_CONFIG['init_cash'],
            fees=DEFAULT_CONFIG['fees']
        )
        return getattr(pf, metric_name)()
    
    # Training performance across all splits
    train_performance = splitter.apply(
        evaluate_performance,
        data['close'],
        strategy_results,
        metric,
        execute_kwargs=dict(show_progress=True, clear_cache=50),
        set_='IS',
        merge_func='concat'
    )
    
    # Get best parameters for each split
    try:
        best_params_per_split = train_performance.groupby(['split']).idxmax()
        
        # Extract optimized signals for out-of-sample testing
        optimized_entries = []
        optimized_exits = []
        oos_performance = []
        
        for split_idx in best_params_per_split.index:
            try:
                best_param_combo = best_params_per_split.loc[split_idx]
                
                # Get out-of-sample signals with best parameters
                oos_entries = splitter['OOS'].take(strategy_results.entries)[split_idx][best_param_combo]
                oos_exits = splitter['OOS'].take(strategy_results.exits)[split_idx][best_param_combo]
                
                # Remove parameter level indices if they exist
                if hasattr(oos_entries, 'droplevel') and len(oos_entries.columns.names) > 1:
                    param_levels = [name for name in oos_entries.columns.names if 'param' in str(name).lower()]
                    if param_levels:
                        oos_entries = oos_entries.droplevel(param_levels, axis=1)
                        oos_exits = oos_exits.droplevel(param_levels, axis=1)
                
                optimized_entries.append(oos_entries)
                optimized_exits.append(oos_exits)
                
                # Calculate OOS performance
                oos_data = splitter['OOS'].take(data['close'])[split_idx]
                oos_pf = vbt.Portfolio.from_signals(
                    oos_data,
                    entries=oos_entries,
                    exits=oos_exits,
                    init_cash=DEFAULT_CONFIG['init_cash'],
                    fees=DEFAULT_CONFIG['fees']
                )
                oos_performance.append(getattr(oos_pf, metric)())
                
            except Exception as e:
                print(f"Error processing split {split_idx}: {e}")
                continue
        
        if optimized_entries and optimized_exits:
            # Combine all out-of-sample results
            final_entries = pd.concat(optimized_entries).sort_index()
            final_exits = pd.concat(optimized_exits).sort_index()
            
            # Create final portfolio
            final_portfolio = vbt.Portfolio.from_signals(
                data['close'],
                entries=final_entries,
                exits=final_exits,
                init_cash=DEFAULT_CONFIG['init_cash'],
                fees=DEFAULT_CONFIG['fees']
            )
            
            # Calculate parameter stability
            param_stability = {}
            for param_name in param_names:
                param_values = [best_params_per_split.loc[i][1] if isinstance(best_params_per_split.loc[i], tuple) 
                              else best_params_per_split.loc[i] for i in best_params_per_split.index]
                param_stability[param_name] = {
                    'mean': np.mean(param_values),
                    'std': np.std(param_values),
                    'stability_ratio': 1 - (np.std(param_values) / np.mean(param_values)) if np.mean(param_values) != 0 else 0
                }
            
            return {
                'final_portfolio': final_portfolio,
                'oos_performance': oos_performance,
                'parameter_stability': param_stability,
                'best_params_per_split': best_params_per_split,
                'train_performance': train_performance,
                'splitter': splitter,
                'n_splits': len(optimized_entries),
                'avg_oos_performance': np.mean(oos_performance),
                'oos_performance_stability': np.std(oos_performance)
            }
            
    except Exception as e:
        print(f"Walk-forward optimization failed: {e}")
        return {
            'error': str(e),
            'splitter': splitter,
            'train_performance': train_performance if 'train_performance' in locals() else None
        }

import data_manager
from strategies.momentum import generate_signals
data = data_manager.load_ohlc_csv('data/BTCUSD_1h_2011-2025.csv')

# WFO with your optimizer
wfo_results = walk_forward_optimize(
    data=data,
    signal_function=generate_signals(data),
    param_grid={'rsi_period': [10, 14, 20], 'ma_period': [20, 50, 100]},
    train_ratio=0.75,
    n_splits=10,
    metric='sharpe_ratio'
)

print(f"Average OOS Performance: {wfo_results['avg_oos_performance']:.3f}")
print(f"Parameter Stability: {wfo_results['parameter_stability']}")