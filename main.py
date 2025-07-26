"""Main entry point for the trading system."""
"""Main entry point for the trading system."""
import sys
import os
import warnings
from typing import Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config import ConfigManager
from core.trading_system import TradingSystem
from core.strategies.orb_strategy import ORBStrategy
from core.data_loader import CSVDataLoader  # New import


def run_orb_strategy() -> Dict[str, Any]:
    """Run the ORB strategy with full analysis pipeline."""
    try:
        # Load configuration
        config_manager = ConfigManager('config')
        strategy_config = config_manager.load_config('orb')

        # Initialize data loader
        data_loader = CSVDataLoader(csv_directory='project/data')

        # Create trading system with data loader
        trading_system = TradingSystem(config_manager, data_loader=data_loader)

        # Create strategy
        strategy = ORBStrategy(strategy_config.__dict__)

        # Get symbols from config
        symbols = strategy_config.data_requirements.get('symbols', ['EURUSD'])

        # Run complete strategy analysis with reduced Monte Carlo runs for speed
        results = trading_system.run_strategy(
            strategy=strategy,
            symbols=symbols,
            run_optimization=True,
            run_walkforward=True,
            run_monte_carlo=True
        )

        return results

    except Exception as e:
        print(f"❌ Error running ORB strategy: {e}")
        return {"success": False, "error": str(e)}

# ... rest of main.py remains unchanged ...


def main():
    """Main entry point."""
    print("🚀 Starting Trading System")
    print("="*50)

    # Run ORB strategy
    results = run_orb_strategy()

    if results.get("success"):
        print("\n✅ Strategy execution completed successfully!")
    else:
        print(f"\n❌ Strategy execution failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
