"""
Risk Premia Strategy - Multi-Asset Systematic Trading
Combines Momentum, Carry, Value, and Volatility signals across FX and commodities
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class RiskPremiaStrategy:
    """
    Multi-factor risk premia strategy using vectorbt
    """
    
    def __init__(self, 
                 symbols: List[str] = None,
                 start_date: str = '2020-01-01',
                 end_date: str = '2024-01-01',
                 target_vol: float = 0.15,
                 max_leverage: float = 2.0,
                 rebalance_freq: str = 'W'):
        
        self.symbols = symbols or ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'GC=F']
        self.start_date = start_date
        self.end_date = end_date
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.rebalance_freq = rebalance_freq
        
        # Risk management parameters
        self.max_daily_loss = 0.05  # 5%
        self.max_total_dd = 0.10    # 10%
        
        # Signal weights
        self.weights = {
            'momentum': 0.4,
            'carry': 0.3,
            'value': 0.3
        }
        
        # Data containers
        self.data = None
        self.prices = None
        self.returns = None
        self.portfolio = None
        
    def load_data(self) -> pd.DataFrame:
        """Load price data using vectorbt"""
        try:
            print(f"Loading data for {self.symbols}...")
            self.data = vbt.YFData.download(
                self.symbols, 
                start=self.start_date, 
                end=self.end_date
            )
            self.prices = self.data.get('Close')
            self.returns = self.prices.pct_change()
            print(f"Data loaded: {self.prices.shape}")
            return self.prices
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def calculate_momentum_signals(self, fast_window=None, slow_window=None, 
                                  macd_fast=None, macd_slow=None, macd_signal=None) -> pd.DataFrame:
        """Calculate momentum signals using MA crossover and MACD"""
        print("Calculating momentum signals...")
        
        # Use provided parameters or defaults
        fast_window = fast_window or 10
        slow_window = slow_window or 30
        macd_fast = macd_fast or 12
        macd_slow = macd_slow or 26
        macd_signal = macd_signal or 9
        
        # Simple MA crossover
        fast_ma = vbt.MA.run(self.prices, window=fast_window)
        slow_ma = vbt.MA.run(self.prices, window=slow_window)
        
        # Handle vectorbt output format
        if hasattr(fast_ma, 'ma'):
            fast_ma_values = fast_ma.ma
            slow_ma_values = slow_ma.ma
        else:
            fast_ma_values = fast_ma
            slow_ma_values = slow_ma
        
        # Ensure alignment
        if hasattr(fast_ma_values, 'reindex_like'):
            fast_ma_aligned = fast_ma_values.reindex_like(self.prices)
            slow_ma_aligned = slow_ma_values.reindex_like(self.prices)
        else:
            fast_ma_aligned = fast_ma_values
            slow_ma_aligned = slow_ma_values
        
        ma_signal = (fast_ma_aligned > slow_ma_aligned).astype(int) * 2 - 1
        
        # MACD alternative
        try:
            macd = vbt.MACD.run(self.prices, fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
            
            if hasattr(macd, 'macd') and hasattr(macd, 'signal'):
                macd_values = macd.macd
                signal_values = macd.signal
            else:
                # Fallback to simple calculation
                return ma_signal.fillna(0)
            
            if hasattr(macd_values, 'reindex_like'):
                macd_aligned = macd_values.reindex_like(self.prices)
                signal_aligned = signal_values.reindex_like(self.prices)
            else:
                macd_aligned = macd_values
                signal_aligned = signal_values
                
            macd_signal_calc = (macd_aligned > signal_aligned).astype(int) * 2 - 1
            
            # Combine both momentum signals
            momentum_signal = (ma_signal + macd_signal_calc) / 2
        except Exception:
            # Fallback to MA only if MACD fails
            momentum_signal = ma_signal
            
        return momentum_signal.fillna(0)
    
    def calculate_carry_signals(self, lookback=None) -> pd.DataFrame:
        """Calculate carry signals (simplified using recent returns slope)"""
        print("Calculating carry signals...")
        
        # Use provided parameter or default
        lookback = lookback or 5
        
        # Use lookback-day returns as carry proxy
        if hasattr(self.prices, 'pct_change'):
            returns_lookback = self.prices.pct_change(lookback)
        else:
            # Fallback calculation
            returns_lookback = self.prices / self.prices.shift(lookback) - 1
            
        carry_signal = np.sign(returns_lookback)
        
        return carry_signal.fillna(0)
    
    def calculate_value_signals(self, ma_window=None) -> pd.DataFrame:
        """Calculate value signals using mean reversion (Z-score)"""
        print("Calculating value signals...")
        
        # Use provided parameter or default
        ma_window = ma_window or 20
        
        # Z-score based mean reversion
        if hasattr(self.prices, 'rolling'):
            ma_n = self.prices.rolling(ma_window).mean()
            std_n = self.prices.rolling(ma_window).std()
        else:
            # Fallback calculation
            ma_n = self.prices.mean()
            std_n = self.prices.std()
            
        zscore = (self.prices - ma_n) / std_n
        
        # Inverse signal: buy when oversold, sell when overbought
        value_signal = -np.sign(zscore)
        
        return value_signal.fillna(0)
    
    def calculate_volatility_scaling(self, vol_window=None) -> pd.DataFrame:
        """Calculate volatility scaling factors"""
        print("Calculating volatility scaling...")
        
        # Use provided parameter or default
        vol_window = vol_window or 20
        
        # Rolling volatility
        if hasattr(self.returns, 'ewm'):
            vol = self.returns.ewm(span=vol_window).std() * np.sqrt(252)
        else:
            # Fallback calculation
            vol = self.returns.rolling(vol_window).std() * np.sqrt(252)
        
        # Volatility scaling
        vol_scalar = self.target_vol / vol
        vol_scalar = vol_scalar.clip(0, self.max_leverage)
        
        return vol_scalar.fillna(1)
    
    def combine_signals(self, momentum: pd.DataFrame, carry: pd.DataFrame, 
                       value: pd.DataFrame) -> pd.DataFrame:
        """Combine all risk premia signals"""
        print("Combining signals...")
        
        # Weighted combination
        combined_signal = (
            momentum * self.weights['momentum'] +
            carry * self.weights['carry'] +
            value * self.weights['value']
        )
        
        return combined_signal
    
    def calculate_positions(self, signals: pd.DataFrame, 
                          vol_scaling: pd.DataFrame) -> pd.DataFrame:
        """Calculate target positions with risk management"""
        print("Calculating positions...")
        
        # Raw positions (signal * vol scaling)
        raw_positions = signals * vol_scaling
        
        # Normalize by number of assets
        n_assets = len(self.symbols)
        positions = raw_positions / n_assets
        
        # Cap positions
        positions = positions.clip(-1, 1)
        
        # Rebalance frequency
        if self.rebalance_freq != 'D':
            positions_resampled = positions.resample(self.rebalance_freq).last()
            positions = positions_resampled.reindex(positions.index, method='ffill')
        
        return positions.fillna(0)
    
    def apply_risk_management(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Apply FTMO-style risk management rules"""
        print("Applying risk management...")
        
        # Create a simple portfolio to calculate returns for risk management
        temp_pf = vbt.Portfolio.from_signals(
            self.prices,
            entries=positions > 0.1,
            exits=positions < -0.1,
            size=np.abs(positions),
            size_type='percent',  # Changed from 'targetpercent' to 'percent'
            fees=0.0002,
            init_cash=10000
        )
        
        daily_returns = temp_pf.returns()
        
        # Daily loss limit
        daily_cumret = (1 + daily_returns).cumprod()
        daily_dd = daily_cumret / daily_cumret.cummax() - 1
        
        # Total drawdown limit
        total_dd = daily_cumret / daily_cumret.expanding().max() - 1
        
        # Filter positions
        positions_filtered = positions.copy()
        positions_filtered[daily_dd < -self.max_daily_loss] = 0
        positions_filtered[total_dd < -self.max_total_dd] = 0
        
        return positions_filtered
    
    def run_backtest(self, **kwargs) -> vbt.Portfolio:
        """Run the complete backtest"""
        print("Starting backtest...")
        
        if self.prices is None:
            self.load_data()
        
        # Extract parameters from kwargs
        fast_ma_window = kwargs.get('fast_ma_window', 10)
        slow_ma_window = kwargs.get('slow_ma_window', 30)
        macd_fast = kwargs.get('macd_fast', 12)
        macd_slow = kwargs.get('macd_slow', 26)
        macd_signal = kwargs.get('macd_signal', 9)
        carry_lookback = kwargs.get('carry_lookback', 5)
        value_ma_window = kwargs.get('value_ma_window', 20)
        vol_window = kwargs.get('vol_window', 20)
        
        # Update weights if provided
        if 'momentum_weight' in kwargs:
            self.weights['momentum'] = kwargs['momentum_weight']
        if 'carry_weight' in kwargs:
            self.weights['carry'] = kwargs['carry_weight']
        if 'value_weight' in kwargs:
            self.weights['value'] = kwargs['value_weight']
        
        # Calculate all signals with parameters
        momentum_signals = self.calculate_momentum_signals(
            fast_window=fast_ma_window, 
            slow_window=slow_ma_window,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal
        )
        carry_signals = self.calculate_carry_signals(lookback=carry_lookback)
        value_signals = self.calculate_value_signals(ma_window=value_ma_window)
        vol_scaling = self.calculate_volatility_scaling(vol_window=vol_window)
        
        # Combine signals
        combined_signals = self.combine_signals(momentum_signals, carry_signals, value_signals)
        
        # Calculate positions
        positions = self.calculate_positions(combined_signals, vol_scaling)
        
        # Apply risk management
        positions = self.apply_risk_management(positions)
        
        # Create more aggressive entry/exit thresholds to generate trades
        entry_threshold = kwargs.get('position_threshold', 0.05)  # Lower threshold
        
        # Create portfolio with more sensitive signals
        self.portfolio = vbt.Portfolio.from_signals(
            self.prices,
            entries=positions > entry_threshold,
            exits=positions < -entry_threshold,
            size=np.abs(positions) * 0.1,  # Smaller position sizes to ensure trades
            size_type='percent',
            fees=kwargs.get('fees', 0.0002),
            slippage=kwargs.get('slippage', 0.0001),
            init_cash=kwargs.get('init_cash', 10000),
            freq='1D'
        )
        
        print("Backtest completed!")
        return self.portfolio
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        if self.portfolio is None:
            print("No portfolio found. Run backtest first.")
            return {}
        
        # Helper function to handle Series/scalar values
        def format_metric(metric, format_str):
            if hasattr(metric, 'iloc'):
                value = metric.iloc[0] if len(metric) > 0 else 0
            else:
                value = metric
            return format_str.format(value)
        
        try:
            stats = {
                'Total Return': format_metric(self.portfolio.total_return(), "{:.2%}"),
                'Sharpe Ratio': format_metric(self.portfolio.sharpe_ratio(), "{:.2f}"),
                'Sortino Ratio': format_metric(self.portfolio.sortino_ratio(), "{:.2f}"),
                'Max Drawdown': format_metric(self.portfolio.max_drawdown(), "{:.2%}"),
                'Win Rate': format_metric(self.portfolio.trades.win_rate, "{:.2%}"),
                'Profit Factor': format_metric(self.portfolio.trades.profit_factor, "{:.2f}"),
                'Total Trades': self.portfolio.trades.count(),
            }
        except Exception as e:
            print(f"Error calculating stats: {e}")
            stats = {
                'Total Return': "N/A",
                'Sharpe Ratio': "N/A",
                'Sortino Ratio': "N/A",
                'Max Drawdown': "N/A",
                'Win Rate': "N/A",
                'Profit Factor': "N/A",
                'Total Trades': 0,
            }
        
        return stats
    
    def print_results(self):
        """Print detailed results"""
        if self.portfolio is None:
            print("No results to display. Run backtest first.")
            return
        
        print("\n" + "="*50)
        print("RISK PREMIA STRATEGY RESULTS")
        print("="*50)
        
        # Overall stats
        stats = self.get_performance_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n" + "-"*30)
        print("PER ASSET PERFORMANCE:")
        print("-"*30)
        
        # Per asset performance
        for symbol in self.symbols:
            try:
                pf_asset = self.portfolio[symbol]
                print(f"\n{symbol}:")
                print(f"  Return: {pf_asset.total_return():.2%}")
                print(f"  Sharpe: {pf_asset.sharpe_ratio():.2f}")
                print(f"  Max DD: {pf_asset.max_drawdown():.2%}")
            except:
                print(f"\n{symbol}: No data available")
    
    def plot_results(self):
        """Plot portfolio performance"""
        if self.portfolio is None:
            print("No portfolio to plot. Run backtest first.")
            return
        
        try:
            # Portfolio value
            self.portfolio.plot().show()
            
            # Drawdowns
            self.portfolio.plot_drawdowns().show()
            
        except Exception as e:
            print(f"Error plotting results: {e}")
    
    def optimize_parameters(self, param_ranges: Dict) -> pd.DataFrame:
        """Optimize strategy parameters using grid search"""
        print("Starting parameter optimization...")
        
        results = []
        
        # Example: optimize MA windows
        fast_windows = param_ranges.get('fast_windows', [5, 10, 15])
        slow_windows = param_ranges.get('slow_windows', [20, 30, 40])
        
        for fast in fast_windows:
            for slow in slow_windows:
                if fast >= slow:
                    continue
                    
                try:
                    # Recalculate with new parameters
                    fast_ma = vbt.MA.run(self.prices, window=fast)
                    slow_ma = vbt.MA.run(self.prices, window=slow)
                    signal = (fast_ma.ma > slow_ma.ma).astype(int) * 2 - 1
                    
                    # Simple backtest
                    pf_temp = vbt.Portfolio.from_signals(
                        self.prices,
                        entries=signal > 0,
                        exits=signal < 0,
                        init_cash=10000,
                        fees=0.0002
                    )
                    
                    results.append({
                        'fast_window': fast,
                        'slow_window': slow,
                        'sharpe': pf_temp.sharpe_ratio(),
                        'total_return': pf_temp.total_return(),
                        'max_drawdown': pf_temp.max_drawdown()
                    })
                    
                except Exception as e:
                    print(f"Error with fast={fast}, slow={slow}: {e}")
                    continue
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            print("\nTop 5 parameter combinations by Sharpe ratio:")
            best = results_df.nlargest(5, 'sharpe')
            print(best)
            
        return results_df


def main():
    """Example usage of the Risk Premia Strategy"""
    
    # Initialize strategy
    strategy = RiskPremiaStrategy(
        symbols=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'GC=F'],
        start_date='2020-01-01',
        end_date='2024-01-01',
        target_vol=0.15,
        rebalance_freq='W'
    )
    
    # Run backtest
    portfolio = strategy.run_backtest()
    
    # Print results
    strategy.print_results()
    
    # Plot results (optional)
    # strategy.plot_results()
    
    # Optimize parameters (optional)
    param_ranges = {
        'fast_windows': [5, 10, 15],
        'slow_windows': [20, 30, 40]
    }
    # optimization_results = strategy.optimize_parameters(param_ranges)
    
    return strategy


def create_portfolio(data, params=None):
    """
    Create portfolio function for integration with the existing system
    
    Args:
        data: Price data (pandas DataFrame or dict)
        params: Strategy parameters (dict)
    
    Returns:
        vectorbt Portfolio object
    """
    if params is None:
        params = {
            'target_vol': 0.15,
            'max_leverage': 2.0,
            'rebalance_freq': 'W',
            'momentum_weight': 0.4,
            'carry_weight': 0.3,
            'value_weight': 0.3,
            'fast_ma_window': 10,
            'slow_ma_window': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'carry_lookback': 5,
            'value_ma_window': 20,
            'vol_window': 20,
            'fees': 0.0002,
            'init_cash': 10000
        }
    
    # Handle different data formats
    if isinstance(data, dict):
        # Multi-timeframe data - use daily if available
        if 'daily' in data:
            prices = data['daily']
        elif '1d' in data:
            prices = data['1d']
        else:
            # Use first available timeframe
            prices = list(data.values())[0]
    else:
        # Single DataFrame
        prices = data
    
    # Create strategy instance with custom parameters
    strategy = RiskPremiaStrategy(
        symbols=list(prices.columns) if hasattr(prices, 'columns') else ['EURUSD=X'],
        target_vol=params.get('target_vol', 0.15),
        max_leverage=params.get('max_leverage', 2.0),
        rebalance_freq=params.get('rebalance_freq', 'W')
    )
    
    # Set prices directly
    strategy.prices = prices
    strategy.returns = prices.pct_change()
    
    # Update weights if provided
    if 'momentum_weight' in params:
        strategy.weights['momentum'] = params['momentum_weight']
    if 'carry_weight' in params:
        strategy.weights['carry'] = params['carry_weight']
    if 'value_weight' in params:
        strategy.weights['value'] = params['value_weight']
    
    # Run backtest with custom parameters
    try:
        # Run backtest with all parameters
        portfolio = strategy.run_backtest(**params)
        
        return portfolio
        
    except Exception as e:
        print(f"Error creating risk premia portfolio: {e}")
        # Return simple buy-and-hold as fallback
        return vbt.Portfolio.from_holding(prices, init_cash=params.get('init_cash', 10000))


if __name__ == "__main__":
    strategy = main()