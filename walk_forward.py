#!/usr/bin/env python3
"""
Walk-Forward Analysis - Simple and Practical Implementation
"""

import pandas as pd

from typing import Dict, Any
import vectorbt as vbt

 
def run_walkforward_analysis(strategy, data: pd.DataFrame) -> Dict[str, Any]:
    """Run walk-forward analysis with proper train/test metrics."""
    try:
        
        print(f"📊 Walk-forward analysis on {len(data)} bars")
        
        # Simple walk-forward: split data into windows
        window_size = max(50, len(data) // 4)  # 25% windows, min length safeguard
        step_size = max(25, window_size // 2)  # 50% overlap, min step safeguard
        
        from strategy_registry import create_portfolio
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
            
            # Run backtests for both periods
            try:
                # Train period
                train_portfolio = create_portfolio(strategy.name, train_data, strategy.parameters)
                train_stats = train_portfolio.stats()
                
                # Get trade count from portfolio
                try:
                    train_trades_hint = len(train_portfolio.trades.records_readable)
                except:
                    train_trades_hint = 0
                
                # Test period
                test_portfolio = create_portfolio(strategy.name, test_data, strategy.parameters)
                test_stats = test_portfolio.stats()
                
                # Get trade count from portfolio
                try:
                    test_trades_hint = len(test_portfolio.trades.records_readable)
                except:
                    test_trades_hint = 0

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
        
        # Calculate cumulative return across all test periods
        # This simulates what would happen if we traded the strategy sequentially
        cumulative_return = 1.0
        for ret in test_returns:
            cumulative_return *= (1 + ret / 100)
        total_cumulative_return = (cumulative_return - 1) * 100
        
        return {
            'windows': windows,
            'avg_oos_performance': avg_oos_performance,
            'total_cumulative_return': total_cumulative_return,
            'parameter_stability': 'stable',  # Simplified for now
            'summary': f"Completed {len(windows)} walk-forward windows, avg OOS return: {avg_oos_performance:.2f}%"
        }
        
    except Exception as e:
        print(f"⚠️ Walk-forward analysis failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    print("Walk-forward analysis module loaded successfully")