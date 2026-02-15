"""
Strategy package for VectorFlow.
Strategies are auto-discovered by portfolio_builder.py at runtime.
"""

# Available strategies are dynamically discovered from this directory.
# Each strategy module must export a `create_portfolio(data, params)` function.

# To add a new strategy:
# 1. Create a new file in vectorflow/strategies/ (e.g., my_strategy.py)
# 2. Implement create_portfolio(data, params) function
# 3. Add default parameters to config/my_strategy.yaml
# 4. The strategy will be auto-discovered on next run