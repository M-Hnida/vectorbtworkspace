# VectorBT Optimization Guide

## Overview

This guide explains how to use VectorBT's vectorization capabilities to dramatically speed up parameter optimization. The key is using the `@vbt.parametrize` decorator which can provide 10-100x performance improvements.

## Key Improvements Made

### 1. Added Vectorized Optimization Functions

- `optimize_strategy_parameters_vectorized()` - Main vectorized optimization function
- `create_vectorized_strategy_function()` - Creates parametrized strategy functions
- `optimize_with_parametrize_decorator()` - Direct use of @vbt.parametrize (recommended)

### 2. Fixed Import Issues

- Added inline `calc_metrics()` function using `portfolio.stats()`
- Added validation functions inline
- Fixed all import errors and typos

### 3. Performance Comparison

| Method | Speed | Use Case |
|--------|-------|----------|
| Sequential | 1x (baseline) | Legacy compatibility |
| Parallel | 2-4x | CPU-bound improvements |
| **Vectorized** | **10-100x** | **Recommended approach** |

## Usage Examples

### Method 1: Direct @vbt.parametrize (Fastest)

```python
import vectorbt as vbt
from optimizer import optimize_with_parametrize_decorator

# Define optimization grid
optimization_grid = {
    'bbands_period': [10, 15, 20, 25, 30],
    'bbands_std': [1.5, 2.0, 2.5]
}

# Run optimization
result = optimize_with_parametrize_decorator(
    data, 
    optimization_grid, 
    {'verbose': True}
)

print(f"Best parameters: {result['best_params']}")
print(f"Best Sharpe: {result['best_metrics']['sharpe_ratio']:.3f}")
```

### Method 2: Strategy Class Integration

```python
from optimizer import optimize_strategy_parameters
from base import StrategyConfig

# Create strategy config
config = StrategyConfig(
    name="bollinger_strategy",
    parameters={},
    optimization_grid={
        'bbands_period': [10, 15, 20, 25, 30],
        'bbands_std': [1.5, 2.0, 2.5]
    }
)

# Run optimization (automatically tries vectorized first)
result = optimize_strategy_parameters(data, YourStrategyClass, config)
```

### Method 3: Pure VectorBT Approach

```python
@vbt.parametrize(
    bbands_period=[10, 15, 20, 25, 30],
    bbands_std=[1.5, 2.0, 2.5],
    product=True  # Generate all combinations
)
def bollinger_strategy(close, bbands_period, bbands_std):
    # Calculate indicators
    bb = vbt.talib('BBANDS').run(close, timeperiod=bbands_period, 
                                nbdevup=bbands_std, nbdevdn=bbands_std)
    
    # Generate signals
    entries = close < bb.real1  # Below lower band
    exits = close > bb.real0    # Above upper band
    
    # Create portfolio
    return vbt.Portfolio.from_signals(close=close, entries=entries, exits=exits)

# Execute all combinations at once
portfolios = bollinger_strategy(data['close'])

# Get best result
best_idx = portfolios.sharpe_ratio().idxmax()
print(f"Best combination: {best_idx}")
```

## Key Benefits of Vectorization

### 1. Speed
- **10-100x faster** than sequential optimization
- Processes all parameter combinations simultaneously
- Leverages NumPy/Numba optimizations

### 2. Memory Efficiency
- VectorBT handles memory management automatically
- Efficient storage of multiple portfolio results
- Built-in result aggregation

### 3. Built-in Analytics
- Automatic calculation of all standard metrics
- Easy comparison across parameter combinations
- Integrated visualization capabilities

### 4. Scalability
- Handles thousands of parameter combinations
- GPU acceleration possible (with proper setup)
- Minimal code changes required

## Best Practices

### 1. Parameter Grid Design
```python
# ✅ Good: Reasonable number of combinations
optimization_grid = {
    'period': [10, 20, 30],      # 3 values
    'threshold': [1.5, 2.0, 2.5] # 3 values
}  # Total: 9 combinations

# ❌ Avoid: Too many combinations
optimization_grid = {
    'period': list(range(5, 100)),     # 95 values
    'threshold': np.linspace(0.1, 5, 50) # 50 values
}  # Total: 4,750 combinations (may be too slow)
```

### 2. Data Preparation
```python
# Ensure clean data
data = data.dropna()
data.index = pd.to_datetime(data.index)

# Use appropriate frequency
freq = pd.infer_freq(data.index) or '1H'
```

### 3. Memory Management
```python
# For large datasets, consider chunking
if len(data) > 50000:
    # Use smaller parameter grids or data subsets
    data = data.iloc[-20000:]  # Last 20k bars
```

## Testing Your Implementation

Run the test script to verify performance:

```bash
python test_vectorized_optimizer.py
```

Expected output:
- Execution time: < 5 seconds for 28 combinations
- Speed: > 5 combinations/second
- Clear performance improvement over sequential methods

## Migration from Sequential to Vectorized

### Step 1: Update Function Calls
```python
# Old
result = optimize_strategy_parameters(data, strategy_class, config)

# New (automatic fallback)
result = optimize_strategy_parameters(data, strategy_class, config)  # Same call!
```

### Step 2: Use Direct Vectorization (Optional)
```python
# For maximum performance
result = optimize_with_parametrize_decorator(data, optimization_grid)
```

### Step 3: Update Strategy Classes (Optional)
Add vectorization support to your strategy classes by implementing the parametrized signal generation pattern.

## Troubleshooting

### Common Issues

1. **"Vectorized optimization failed"**
   - Falls back to sequential automatically
   - Check parameter grid format
   - Ensure strategy supports vectorization

2. **Memory errors**
   - Reduce parameter grid size
   - Use smaller data samples
   - Increase system RAM

3. **Slow performance**
   - Verify VectorBT installation
   - Check for data quality issues
   - Consider GPU acceleration

### Performance Tips

1. Use `product=True` in @vbt.parametrize for full combinations
2. Minimize data preprocessing inside parametrized functions
3. Use VectorBT's built-in indicators when possible
4. Profile your code to identify bottlenecks

## Conclusion

The vectorized optimization approach provides significant performance improvements with minimal code changes. The automatic fallback ensures compatibility while the direct @vbt.parametrize approach offers maximum speed for new implementations.

Key takeaway: **Use vectorization for parameter optimization to achieve 10-100x speed improvements** while maintaining the same API and functionality.