#!/usr/bin/env python3
"""
Walk-Forward Analysis - Simple and Practical Implementation
"""

import pandas as pd
from strategies import get_strategy_signal_function
from typing import Dict, Any
from backtest import run_backtest

 
def run_walkforward_analysis(strategy, data: pd.DataFrame) -> Dict[str, Any]:
    """Run walk-forward analysis with proper train/test metrics."""
    try:
        
        print(f"📊 Walk-forward analysis on {len(data)} bars")
        
        # Simple walk-forward: split data into windows
        window_size = max(50, len(data) // 4)  # 25% windows, min length safeguard
        step_size = max(25, window_size // 2)  # 50% overlap, min step safeguard
        
        signal_func = get_strategy_signal_function(strategy.name)
        windows = []
        
        for i in range(0, len(data) - window_size, step_size):
            if len(windows) >= 5:  # Limit to 5 windows for testing
                break
                
            train_end = i + window_size
            # Use a test window roughly equal to train size to increase chance of trades
            test_end = min(train_end + window_size, len(data))
            
            if test_end - train_end < 20:  # Skip if test window too small
                continue
                
            train_data = data.iloc[i:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Generate signals and run backtests for both periods
            try:
                primary_tf = strategy.get_required_timeframes()[0]
                # Train period
                train_signals = signal_func(
                    {primary_tf: train_data},
                    strategy.parameters
                )
                train_entries = getattr(train_signals, 'entries', None)
                train_exits = getattr(train_signals, 'exits', None)
                train_trades_hint = int(train_entries.sum()) if hasattr(train_entries, 'sum') else -1

                train_portfolio = run_backtest(train_data, train_signals)
                train_stats = train_portfolio.stats()
                
                # Test period
                test_signals = signal_func(
                    {primary_tf: test_data},
                    strategy.parameters
                )
                test_entries = getattr(test_signals, 'entries', None)
                test_exits = getattr(test_signals, 'exits', None)
                test_trades_hint = int(test_entries.sum()) if hasattr(test_entries, 'sum') else -1

                test_portfolio = run_backtest(test_data, test_signals)
                test_stats = test_portfolio.stats()

                # Fallback: if Total Return missing, compute from value series
                def total_return_pct(pf):
                    try:
                        v = pf.value()
                        if v is not None and len(v) > 1 and float(v.iloc[0]) != 0:
                            return float((v.iloc[-1] / v.iloc[0] - 1.0) * 100.0)
                    except Exception:
                        pass
                    return float(test_stats.get('Total Return [%]', 0.0))

                # Attach diagnostics
                train_ret = float(train_stats.get('Total Return [%]', 0.0))
                test_ret = float(test_stats.get('Total Return [%]', 0.0))
                if test_ret == 0.0:
                    test_ret = total_return_pct(test_portfolio)
                    test_stats['Total Return [%]'] = test_ret

                windows.append({
                    'window': len(windows) + 1,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'train_stats': train_stats,
                    'test_stats': test_stats,
                    'diagnostics': {
                        'train_trades_hint': train_trades_hint,
                        'test_trades_hint': test_trades_hint
                    }
                })
                
                print(f"✅ Window {len(windows)}: Train {train_stats['Total Return [%]']:.2f}% (trades~{train_trades_hint}), Test {test_stats['Total Return [%]']:.2f}% (trades~{test_trades_hint})")
                
            except Exception as e:
                print(f"⚠️ Window {len(windows) + 1} failed: {e}")
                continue
        
        if not windows:
            return {'error': 'No successful walk-forward windows'}
        
        # Calculate summary statistics
        test_returns = [w['test_stats']['Total Return [%]'] for w in windows]
        avg_oos_performance = sum(test_returns) / len(test_returns)
        
        return {
            'windows': windows,
            'avg_oos_performance': avg_oos_performance,
            'parameter_stability': 'stable',  # Simplified for now
            'summary': f"Completed {len(windows)} walk-forward windows, avg OOS return: {avg_oos_performance:.2f}%"
        }
        
    except Exception as e:
        print(f"⚠️ Walk-forward analysis failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    print("Walk-forward analysis module loaded successfully")