"""Main trading system orchestrator."""
from typing import Dict, List, Union, Any, Optional
import pandas as pd
import numpy as np
from .base import BaseStrategy, BaseDataLoader
from .data_loader import CSVDataLoader
from .portfolio import PortfolioManager
from .config import ConfigManager, StrategyConfig, SystemConfig
from .optimizer import ParameterOptimizer
from .walkforward import WalkForwardAnalyzer
from .monte_carlo import MonteCarloAnalyzer
from .visualization import TradingVisualizer
from .analysis import OverfittingAnalyzer, StatisticalValidator, MultiAssetAnalyzer
from . import metrics

class TradingSystem:
    """Main trading system orchestrator."""

    def __init__(self, config_manager: ConfigManager, data_loader: BaseDataLoader):
        self.config_manager = config_manager
        self.system_config = config_manager.get_system_config()
        self.data_loader = data_loader  # Use injected data loader
        self.portfolio_manager: Optional[PortfolioManager] = None

    def run_strategy(self, strategy: BaseStrategy, symbols: Union[str, List[str]],
                    run_optimization: bool = True, run_walkforward: bool = True,
                    run_monte_carlo: bool = True) -> Dict[str, Any]:
        """Run complete strategy analysis pipeline."""

        # No longer initialize data loader here - use injected instance
        if not self.data_loader:
            raise ValueError("Data loader not initialized")

        # Normalize symbols to list
        if isinstance(symbols, str):
            symbols = [symbols]


        # Get strategy config
        strategy_config = self.config_manager.strategy_configs[strategy.name]

        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(symbols, strategy_config.portfolio)

        # Load data
        print(f"📊 Loading data for symbols: {symbols}")
        data = self._load_data(symbols)

        if data.empty:
            print("❌ No data loaded")
            return {"success": False, "error": "No data loaded"}

        print(f"📏 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")

        # Run optimization if enabled
        best_params = strategy.parameters.copy()
        optimization_results = None

        if run_optimization and strategy_config.optimization_grid:
            try:
                optimizer = ParameterOptimizer(strategy, strategy_config)
                optimization_results = optimizer.optimize(data)
                best_params = optimization_results['param_combination']
            except Exception as e:
                print(f"⚠️ Optimization failed: {e}")
                run_optimization = False

        # Update strategy with best parameters
        strategy.parameters.update(best_params)

        # Run backtests on different data splits
        portfolios = self._run_backtests(data, strategy)

        # Run walk-forward analysis if enabled
        walkforward_results = None
        if run_walkforward:
            try:
                wf_analyzer = WalkForwardAnalyzer(strategy, strategy_config)
                walkforward_results = wf_analyzer.run_analysis(data, best_params)
            except Exception as e:
                print(f"⚠️ Walk-forward analysis failed: {e}")

        # Run Monte Carlo analysis if enabled
        monte_carlo_results = None
        if run_monte_carlo:
            try:
                print("🎲 Starting Monte Carlo Analysis...")
                mc_analyzer = MonteCarloAnalyzer(strategy, strategy_config)
                monte_carlo_results = mc_analyzer.run_analysis(
                    data, best_params, portfolios['full'], runs=20  # Reduced runs for speed
                )
                if monte_carlo_results.get('success'):
                    print("✅ Monte Carlo analysis completed successfully")
                else:
                    print("⚠️ Monte Carlo analysis completed with issues")
            except Exception as e:
                print(f"⚠️ Monte Carlo analysis failed: {e}")
                monte_carlo_results = {"success": False, "error": str(e)}

        # Run additional analysis
        additional_analysis = self._run_additional_analysis(portfolios, data, symbols)

        # Compile final results
        results = {
            "success": True,
            "strategy": strategy.name,
            "symbols": symbols,
            "data_period": f"{data.index[0]} to {data.index[-1]}",
            "total_bars": len(data),
            "best_parameters": best_params,
            "portfolios": portfolios,
            "optimization": optimization_results,
            "walkforward": walkforward_results,
            "monte_carlo": monte_carlo_results,
            "additional_analysis": additional_analysis
        }

        # Run comprehensive visualization
        self._run_visualizations(results, data, symbols)

        # Print comprehensive summary
        self._print_comprehensive_summary(results)

        return results

    def _load_data(self, symbols: List[str]) -> pd.DataFrame:
        """Load data for symbols."""
        if len(symbols) == 1:
            # Single symbol
            symbol = symbols[0]
            # For now, default to 1h interval - this could be configurable
            data = self.data_loader.load(symbol, '1h')
            return data
        else:
            # Multi-symbol - load all symbols and combine
            print(f"📊 Loading data for {len(symbols)} symbols...")
            all_data = {}

            for symbol in symbols:
                symbol_data = self.data_loader.load(symbol, '1h')
                if not symbol_data.empty:
                    all_data[symbol] = symbol_data
                    print(f"   ✅ {symbol}: {len(symbol_data)} bars loaded")
                else:
                    print(f"   ❌ {symbol}: No data loaded")

            if not all_data:
                print("❌ No data loaded for any symbol")
                return pd.DataFrame()

            # For now, return the first successfully loaded symbol's data
            # In a full multi-asset implementation, this would be a combined DataFrame
            first_symbol = list(all_data.keys())[0]
            print(f"📊 Using {first_symbol} as primary symbol for analysis")
            return all_data[first_symbol]

    def _run_backtests(self, data: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """Run backtests on train/test/full datasets."""
        print("🔄 Running backtests on different data splits...")

        # Split data
        split_ratio = 0.7
        split_idx = int(len(data) * split_ratio)

        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()

        portfolios = {}

        # Train backtest
        print("📊 Train Analysis:")
        train_entries, train_exits = strategy.generate_signals(train_data)
        # Pass only close price, not entire DataFrame
        train_portfolio = self.portfolio_manager.create_portfolio(train_data['close'], train_entries, train_exits)
        portfolios['train'] = train_portfolio
        self._print_portfolio_metrics(train_portfolio, "Train")

        # Test backtest
        print("\n🔄 Running backtest for test dataset...")
        print("📊 Test Analysis:")
        test_entries, test_exits = strategy.generate_signals(test_data)
        # Pass only close price, not entire DataFrame
        test_portfolio = self.portfolio_manager.create_portfolio(test_data['close'], test_entries, test_exits)
        portfolios['test'] = test_portfolio
        self._print_portfolio_metrics(test_portfolio, "Test")

        # Full backtest
        print("\n🔄 Running backtest for full dataset...")
        print("📊 Full Analysis:")
        full_entries, full_exits = strategy.generate_signals(data)
        # Pass only close price, not entire DataFrame
        full_portfolio = self.portfolio_manager.create_portfolio(data['close'], full_entries, full_exits)
        portfolios['full'] = full_portfolio
        self._print_portfolio_metrics(full_portfolio, "Full")

        # Performance comparison
        self._print_performance_comparison(portfolios)

        return portfolios

    def _print_portfolio_metrics(self, portfolio, name: str):
        """Print metrics for a single portfolio."""
        try:
            portfolio_metrics = metrics.calc_metrics(portfolio, name)

            # Map the actual metric keys from calc_metrics
            key_metrics = [
                ('Sharpe Ratio', 'sharpe', ''),
                ('Calmar Ratio', 'calmar', ''),
                ('Total Return', 'return', '%'),
                ('Max Drawdown', 'max_dd', '%'),
                ('Total Trades', 'trades', ''),
                ('Win Rate', 'win_rate', '%'),
                ('Profit Factor', 'profit_factor', ''),
                ('Average Win', 'avg_win', '$'),
                ('Average Loss', 'avg_loss', '$'),
                ('Volatility', 'volatility', '%'),
                ('VaR (95%)', 'var_95', '%'),
                ('CVaR (95%)', 'cvar_95', '%')
            ]

            for display_name, key, unit in key_metrics:
                if key in portfolio_metrics:
                    value = portfolio_metrics[key]
                    if isinstance(value, (int, float)):
                        if unit == '%':
                            print(f"   {display_name}: {value:.2f}%")
                        elif unit == '$':
                            print(f"   {display_name}: ${value:.2f}")
                        else:
                            print(f"   {display_name}: {value:.3f}")
                    else:
                        print(f"   {display_name}: {value}")

        except Exception as e:
            print(f"   ⚠️ Error calculating metrics: {e}")

    def _print_performance_comparison(self, portfolios: Dict[str, Any]):
        """Print performance comparison table."""
        print("\n📊 Performance Analysis:")

        try:
            comparison_data = []

            for name, portfolio in portfolios.items():
                portfolio_metrics = metrics.calc_metrics(portfolio, name)
                comparison_data.append({
                    'Dataset': name.title(),
                    'Total Return [%]': f"{portfolio_metrics.get('return', 0):.3f}",
                    'Sharpe Ratio': f"{portfolio_metrics.get('sharpe', 0):.3f}",
                    'Max Drawdown [%]': f"{portfolio_metrics.get('max_dd', 0):.3f}",
                    'Total Trades': f"{portfolio_metrics.get('trades', 0):.0f}",
                    'Win Rate [%]': f"{portfolio_metrics.get('win_rate', 0):.3f}"
                })

            # Create comparison table
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False, justify='right'))

        except Exception as e:
            print(f"   ⚠️ Error creating comparison table: {e}")

    def _print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print comprehensive results summary."""
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE STRATEGY ANALYSIS SUMMARY")
        print("="*60)

        print(f"✅ Strategy: {results['strategy']}")
        print(f"✅ Symbols: {results['symbols']}")
        print(f"✅ Data Period: {results['data_period']}")
        print(f"✅ Total Bars: {results['total_bars']}")

        # Optimization summary
        if results.get('optimization'):
            print("✅ Parameter Optimization: COMPLETED")
        else:
            print("⚠️ Parameter Optimization: SKIPPED")

        # Walk-forward summary
        if results.get('walkforward') and results['walkforward'].get('success'):
            wf = results['walkforward']
            print(f"✅ Walk-Forward Analysis: {wf['stability_assessment']}")
        else:
            print("⚠️ Walk-Forward Analysis: FAILED/SKIPPED")

        # Monte Carlo summary
        if results.get('monte_carlo') and results['monte_carlo'].get('success'):
            mc = results['monte_carlo']
            print(f"✅ Monte Carlo Validation: {mc['interpretation']}")
        else:
            print("⚠️ Monte Carlo Validation: FAILED/SKIPPED")

        # Statistical validation summary
        if results.get('monte_carlo') or results.get('walkforward'):
            validator = StatisticalValidator()
            validation_summary = validator.create_validation_summary(results)
            results['statistical_validation'] = validation_summary

        # Visualization summary
        if results.get('visualizations') and not results['visualizations'].get('error'):
            print("✅ Comprehensive Visualizations: COMPLETED")
        else:
            print("⚠️ Visualizations: FAILED/INCOMPLETE")

        print("="*60)

    def _run_additional_analysis(self, portfolios: Dict[str, Any], data: pd.DataFrame,
                               symbols: List[str]) -> Dict[str, Any]:
        """Run additional analysis components."""
        analysis_results = {}

        # Overfitting analysis
        if 'train' in portfolios and 'test' in portfolios:
            overfitting_analyzer = OverfittingAnalyzer()
            analysis_results['overfitting'] = overfitting_analyzer.analyze_overfitting(
                portfolios['train'], portfolios['test']
            )

        # Multi-asset analysis (if applicable)
        if len(symbols) > 1:
            multi_asset_analyzer = MultiAssetAnalyzer()
            analysis_results['correlation'] = multi_asset_analyzer.analyze_asset_correlation(data, symbols)
            analysis_results['asset_performance'] = multi_asset_analyzer.plot_asset_performance(portfolios, symbols)
        else:
            print("📊 Multi-asset correlation analysis...")
            print("Not enough assets for correlation.")

        return analysis_results

    def _run_visualizations(self, results: Dict[str, Any], data: pd.DataFrame, symbols: List[str]):
        """Run comprehensive visualization pipeline."""
        print("\n📊 Creating visualizations...")

        try:
            visualizer = TradingVisualizer()

            # Create main dashboard
            dashboard_results = visualizer.create_dashboard(
                results['portfolios'], results
            )

            # Add dashboard results to main results
            results['visualizations'] = dashboard_results

            print("✅ Visualization phase completed")

        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            results['visualizations'] = {"error": str(e)}
