"""Streamlined visualization using VectorBT's native plotting capabilities."""
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt

# Set VectorBT dark theme
vbt.settings.set_theme("dark")


class TradingVisualizer:
    """Streamlined visualization leveraging VectorBT's built-in plotting."""
    
    def __init__(self):
        # Use VectorBT's dark theme colors
        self.dark_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    
    def create_dashboard(self, portfolios: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Create streamlined dashboard using VectorBT's native plotting."""
        print("📊 Creating streamlined visualization dashboard...")
        
        try:
            main_portfolio = portfolios['full']
            
            # Main VectorBT plot (includes equity, returns, drawdown, trades automatically)
            print("📈 Creating main portfolio plot...")
            fig = main_portfolio.plot()
            fig.update_layout(
                title="📊 Portfolio Performance Dashboard - Complete Analysis", 
                height=800,
                showlegend=True
            )
            fig.show()
            
            # Additional analysis only if available and successful
            if results.get('monte_carlo', {}).get('success'):
                print("🎲 Creating Monte Carlo analysis...")
                self._plot_monte_carlo(results['monte_carlo'])
                
            if results.get('walkforward', {}).get('success'):
                print("🚶 Creating walk-forward analysis...")
                self._plot_walkforward(results['walkforward'])
            
            print("✅ Dashboard creation completed successfully")
            return {"success": True}
            
        except Exception as e:
            print(f"⚠️ Dashboard creation failed: {e}")
            return {"error": str(e)}
    

    
    def _plot_monte_carlo(self, mc_results: Dict[str, Any]) -> Dict[str, Any]:
        """Plot Monte Carlo results with proper error handling."""
        try:
            # Validate data first
            permuted_sharpes = mc_results.get('permuted_sharpes', [])
            permuted_pfs = mc_results.get('permuted_profit_factors', [])
            original_sharpe = mc_results.get('original_sharpe', 0)
            original_pf = mc_results.get('original_profit_factor', 0)
            
            if not permuted_sharpes and not permuted_pfs:
                print("ℹ️ No Monte Carlo data to plot")
                return {"success": False, "reason": "no_data"}
            
            # Create subplot only if we have data
            n_cols = sum([bool(permuted_sharpes), bool(permuted_pfs)])
            if n_cols == 0:
                return {"success": False, "reason": "no_valid_data"}
            
            subplot_titles = []
            if permuted_sharpes:
                subplot_titles.append('Sharpe Ratio Distribution')
            if permuted_pfs:
                subplot_titles.append('Profit Factor Distribution')
            
            fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles)
            
            col_idx = 1
            
            # Sharpe distribution
            if permuted_sharpes and len(permuted_sharpes) > 0:
                fig.add_trace(
                    go.Histogram(x=permuted_sharpes, nbinsx=20, name='Permuted Sharpe',
                               marker_color=self.dark_colors[0], opacity=0.7),
                    row=1, col=col_idx
                )
                # Add vertical line for original value
                fig.add_vline(x=original_sharpe, line_dash="dash", line_color="red", 
                             annotation_text=f"Original: {original_sharpe:.3f}")
                col_idx += 1
            
            # Profit factor distribution
            if permuted_pfs and len(permuted_pfs) > 0:
                fig.add_trace(
                    go.Histogram(x=permuted_pfs, nbinsx=20, name='Permuted PF',
                               marker_color=self.dark_colors[1], opacity=0.7),
                    row=1, col=col_idx
                )
                # Add vertical line for original value
                fig.add_vline(x=original_pf, line_dash="dash", line_color="red",
                             annotation_text=f"Original: {original_pf:.3f}")
            
            fig.update_layout(
                height=400, 
                title_text="🎲 Monte Carlo Permutation Test Results", 
                template='plotly_dark',
                showlegend=True
            )
            fig.show()
            
            return {"success": True}
            
        except Exception as e:
            print(f"⚠️ Monte Carlo plot failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _plot_walkforward(self, wf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Plot walk-forward analysis results."""
        try:
            results_df = wf_results.get('results')
            if results_df is None or results_df.empty:
                return {"success": False, "reason": "no_data"}
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Sharpe Ratio', 'Returns %', 'Max Drawdown %')
            )
            
            # Sharpe ratio over windows
            fig.add_trace(
                go.Scatter(x=results_df['window'], y=results_df['sharpe'],
                         mode='lines+markers', name='Sharpe Ratio',
                         line={"color": self.dark_colors[0]}),
                row=1, col=1
            )
            
            # Returns over windows
            fig.add_trace(
                go.Scatter(x=results_df['window'], y=results_df['return'],
                         mode='lines+markers', name='Returns %',
                         line={"color": self.dark_colors[1]}),
                row=1, col=2
            )
            
            # Max drawdown over windows
            fig.add_trace(
                go.Scatter(x=results_df['window'], y=results_df['max_dd'],
                         mode='lines+markers', name='Max DD %',
                         line={"color": self.dark_colors[2]}),
                row=1, col=3
            )
            
            fig.update_layout(
                height=400, 
                title_text="🚶 Walk-Forward Analysis - Performance Stability", 
                template='plotly_dark',
                showlegend=True
            )
            fig.show()
            
            return {"success": True}
            
        except Exception as e:
            print(f"⚠️ Walk-forward plot failed: {e}")
            return {"success": False, "error": str(e)}
    
    def plot_portfolio(self, portfolio) -> Dict[str, Any]:
        """Main portfolio plotting using VectorBT's comprehensive plot method."""
        try:
            # Use VectorBT's comprehensive plot method (includes equity, returns, drawdown, trades)
            fig = portfolio.plot()
            fig.update_layout(template='plotly_dark')
            fig.show()
            return {"success": True}
            
        except Exception as e:
            print(f"⚠️ Portfolio plot failed: {e}")
            # Fallback to equity curve only
            try:
                equity = portfolio.value()
                fig = equity.vbt.plot()
                fig.update_layout(title="Portfolio Equity Curve", template='plotly_dark')
                fig.show()
                return {"success": True, "fallback": True}
            except Exception as e2:
                print(f"⚠️ Fallback plot also failed: {e2}")
                return {"success": False, "error": str(e2)}
    def get_stats_summary(self, portfolio) -> Dict[str, Any]:
        """Get portfolio statistics summary."""
        try:
            stats = portfolio.stats()
            return {
                "total_return": stats.get('Total Return [%]', 0),
                "sharpe_ratio": stats.get('Sharpe Ratio', 0),
                "max_drawdown": stats.get('Max Drawdown [%]', 0),
                "win_rate": stats.get('Win Rate [%]', 0),
                "total_trades": stats.get('# Trades', 0)
            }
        except Exception as e:
            print(f"⚠️ Stats summary failed: {e}")
            return {}


