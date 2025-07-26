"""Advanced analysis components for trading strategies."""
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Any
import pandas as pd
from . import metrics


class OverfittingAnalyzer:
    """Analyzes potential overfitting in strategy results."""
    
    def analyze_overfitting(self, train_portfolio, test_portfolio) -> Dict[str, Any]:
        """Analyze overfitting risk between train and test results."""
        print("\n⚠️  Overfitting Analysis (Train vs. Test):")
        
        try:
            # Calculate metrics for both periods
            train_metrics = metrics.calc_metrics(train_portfolio)
            test_metrics = metrics.calc_metrics(test_portfolio)
            
            # Compare key metrics
            sharpe_diff = train_metrics['sharpe'] - test_metrics['sharpe']
            return_diff = train_metrics['return'] - test_metrics['return']
            
            print(f"   Sharpe Ratio: {train_metrics['sharpe']:.3f} vs {test_metrics['sharpe']:.3f} (Diff: {sharpe_diff:.3f})")
            print(f"   Total Return: {train_metrics['return']:.2f}% vs {test_metrics['return']:.2f}% (Diff: {return_diff:.2f}%)")
            
            # Determine overfitting risk level
            overfitting_risk = self._assess_overfitting_risk(sharpe_diff, return_diff)
            print(f"   Overfitting Risk Level: {overfitting_risk}")
            
            return {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "sharpe_difference": sharpe_diff,
                "return_difference": return_diff,
                "overfitting_risk": overfitting_risk
            }
            
        except Exception as e:
            print(f"   ⚠️ Error in overfitting analysis: {e}")
            return {"error": str(e)}
    
    def _assess_overfitting_risk(self, sharpe_diff: float, return_diff: float) -> str:
        """Assess overfitting risk level based on performance differences."""
        # Simple heuristic for overfitting assessment
        if abs(sharpe_diff) > 1.0 or abs(return_diff) > 50:
            return "HIGH"
        elif abs(sharpe_diff) > 0.5 or abs(return_diff) > 25:
            return "MODERATE"
        else:
            return "LOW"


class StatisticalValidator:
    """Provides statistical validation and significance testing."""
    
    def create_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive statistical validation summary."""
        print("\n🎯 STATISTICAL VALIDATION SUMMARY:")
        
        validation_results = {}
        
        # Walk-forward analysis validation
        if results.get('walkforward') and results['walkforward'].get('success'):
            wf_result = results['walkforward']
            stability_status = "✅ STABLE" if wf_result['stability_assessment'] == 'STABLE' else "❌ UNSTABLE"
            print(f"   Walk-Forward Analysis: {stability_status}")
            validation_results['walkforward_stable'] = wf_result['stability_assessment'] == 'STABLE'
        else:
            print("   Walk-Forward Analysis: ❌ NOT COMPLETED")
            validation_results['walkforward_stable'] = False
        
        # Monte Carlo validation
        if results.get('monte_carlo') and results['monte_carlo'].get('success'):
            mc_result = results['monte_carlo']
            significance_status = mc_result['interpretation']
            print(f"   Monte Carlo Permutation Test: {significance_status}")
            validation_results['monte_carlo_significant'] = mc_result['is_significant']
        else:
            print("   Monte Carlo Permutation Test: ❌ NOT COMPLETED")
            validation_results['monte_carlo_significant'] = False
        
        # Overall validation assessment
        overall_valid = validation_results.get('walkforward_stable', False) and validation_results.get('monte_carlo_significant', False)
        overall_status = "✅ VALIDATED" if overall_valid else "⚠️ REQUIRES FURTHER VALIDATION"
        print(f"   Overall Statistical Validation: {overall_status}")
        
        validation_results['overall_validated'] = overall_valid
        validation_results['summary_status'] = overall_status
        
        return validation_results


class MultiAssetAnalyzer:
    """Handles multi-asset analysis and correlation."""
    
    def analyze_asset_correlation(self, data: pd.DataFrame, symbols: List[str]) -> Dict[str, Any]:
        """Analyze correlation between multiple assets."""
        if len(symbols) < 2:
            print("Not enough assets for correlation analysis.")
            return {"success": False, "reason": "insufficient_assets"}
        
        try:
            print("📊 Multi-asset correlation analysis...")
            
            # Calculate returns for each asset
            returns_data = {}
            for symbol in symbols:
                # Extract the 'close' price for each symbol from the MultiIndex DataFrame
                if (symbol, 'close') in data.columns:
                    returns_data[symbol] = data[(symbol, 'close')].pct_change().dropna()
                elif symbol in data.columns: # Fallback for single-level index (e.g., single asset data)
                    returns_data[symbol] = data[symbol].pct_change().dropna()
            
            if len(returns_data) < 2:
                print("Not enough valid asset data for correlation.")
                return {"success": False, "reason": "insufficient_data"}
            
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            print("Asset Correlation Matrix:")
            print(correlation_matrix.round(3))
            
            return {
                "success": True,
                "correlation_matrix": correlation_matrix,
                "returns_data": returns_df
            }
            
        except Exception as e:
            print(f"⚠️ Correlation analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def plot_asset_performance(self, portfolios: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
        """Plot performance comparison across multiple assets."""
        if len(symbols) <= 1:
            print("ℹ️ Single asset portfolio - skipping multi-asset comparison")
            return {"success": False, "reason": "single_asset"}
        
        try:
            print("📊 Creating Performance by Asset Plot...")
            
            # This would be implemented when we have true multi-asset support
            # For now, just acknowledge the request
            print("📊 Multi-asset performance plotting not yet implemented")
            
            return {"success": True, "note": "Multi-asset plotting placeholder"}
            
        except Exception as e:
            print(f"⚠️ Asset performance plot failed: {e}")
            return {"success": False, "error": str(e)}