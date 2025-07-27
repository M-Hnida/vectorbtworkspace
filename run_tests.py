import unittest
import sys
import os

def discover_and_run_tests():
    """Discover and run all unit tests."""
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add project root to Python path
    sys.path.append(current_dir)
    
    # Discover tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir=os.path.join(current_dir, 'tests'),
        pattern='test_*.py',
        top_level_dir=current_dir
    )
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = discover_and_run_tests()
    sys.exit(exit_code)