"""Main entry point for the trading system."""
import sys
import os
import warnings
import importlib
from typing import Dict, Any, Type

# Suppress warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config import ConfigManager
from core.trading_system import TradingSystem
from core.base import BaseStrategy
from core.data_loader import CSVDataLoader


def get_strategy_class(strategy_name: str) -> Type[BaseStrategy]:
    """Dynamically import and return the strategy class."""
    try:
        # Convert strategy name to class name (e.g., 'orb' -> 'ORBStrategy')
        class_name = ''.join(word.capitalize() for word in strategy_name.split('_')) + 'Strategy'
        module_name = f'core.strategies.{strategy_name}_strategy'
        
        print(f"\n� Looking for strategy class '{class_name}' in module '{module_name}'")
        
        # Import the strategy module directly
        module = importlib.import_module(module_name)
        
        if hasattr(module, class_name):
            strategy_class = getattr(module, class_name)
            if issubclass(strategy_class, BaseStrategy):
                print(f"✅ Found valid strategy class {class_name}")
                return strategy_class
            else:
                print(f"❌ {class_name} found but does not inherit from BaseStrategy")
        else:
            print(f"❌ {class_name} not found in module")
            
        raise ImportError(f"Could not find valid {class_name} class")
        
    
    except Exception as e:
        print(f"⚠️ Error loading strategy: {e}")
        raise ImportError(f"Failed to load strategy {strategy_name}: {str(e)}") from e


def run_strategy(strategy_name: str) -> Dict[str, Any]:
    """Run any strategy with full analysis pipeline."""
    try:
        # Load configuration
        config_manager = ConfigManager('config')
        strategy_config = config_manager.load_config(strategy_name)

        # Initialize data loader
        data_loader = CSVDataLoader(csv_directory='data')

        # Create trading system with data loader
        trading_system = TradingSystem(config_manager, data_loader=data_loader)

        # Dynamically load and create strategy
        strategy_class = get_strategy_class(strategy_name)
        strategy = strategy_class(strategy_config.__dict__)

        # Get symbols from config
        symbols = strategy_config.data_requirements.get('symbols', ['EURUSD'])

        # Run complete strategy analysis
        results = trading_system.run_strategy(
            strategy=strategy,
            symbols=symbols,
            run_optimization=True,
            run_walkforward=True,
            run_monte_carlo=True
        )

        return {"success": True, "results": results}

    except Exception as e:
        print(f"❌ Error running {strategy_name} strategy: {e}")
        return {"success": False, "error": str(e)}

# ... rest of main.py remains unchanged ...


def main():
    """Main entry point."""
    print("🚀 Starting Trading System")
    print("="*50)

    # Get available strategies
    config_dir = 'config'
    available_strategies = [
        os.path.splitext(f)[0] 
        for f in os.listdir(config_dir) 
        if f.endswith('.yaml')
    ]

    print("\n📊 Available Strategies:")
    for i, strategy in enumerate(available_strategies, 1):
        print(f"{i}. {strategy}")

    # Get strategy choice
    try:
        choice = int(input("\nSelect strategy number: ")) - 1
        strategy_name = available_strategies[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection")
        return

    print(f"\n🔄 Running {strategy_name} strategy...")
    results = run_strategy(strategy_name)

    if results["success"]:
        print("\n✅ Strategy execution completed successfully!")
    else:
        print(f"\n❌ Strategy execution failed: {results['error']}")


if __name__ == "__main__":
    main()
