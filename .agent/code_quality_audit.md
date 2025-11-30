# VectorFlow Code Quality Audit & Refactoring Plan

**Date:** 2025-11-30
**Objective:** Comprehensive code quality review and refactoring based on PEP 8 and best practices

---

## 🔍 CRITICAL ISSUES FOUND

### 1. **Missing `strategy_registry` Module**
- **Severity:** HIGH ❌
- **Files Affected:**
  - `vectorflow/utils/config_validator.py` (line 17)
  - `examples/debug_path_mc.py` (line 4)
- **Issue:** Code imports `from strategy_registry import ...` but this module doesn't exist
- **Location:** The module was likely renamed to `vectorflow/core/portfolio_builder.py`
- **Impact:** Code will fail at runtime
- **Fix:** Update imports to use `vectorflow.core.portfolio_builder`

### 2. **Dead/Unused Code**
- **File:** `examples/debug_path_mc.py`
  - Imports non-existent `monte_carlo_path` module (line 3)
  - Imports non-existent `strategy_registry` module (line 4)
  - This entire example file appears to be outdated

---

## 📋 CODE QUALITY ISSUES BY CATEGORY

### A. Import Organization & Consolidation

#### ❌ `config_validator.py`
```python
# Line 17: Dynamic import inside function
from strategy_registry import _STRATEGIES
```
**Issues:**
- Dynamic import (bad practice, hard to track dependencies)
- Imports non-existent module
- Should use absolute imports from vectorflow package

**Fix:**
```python
from vectorflow.core.portfolio_builder import _STRATEGIES
```

#### ❌ `path_randomization.py`
```python
# Lines 244: Duplicate imports in __main__ block
import vectorbt as vbt
import pandas as pd
import numpy as np
```
**Issues:**
- These are already imported at the top (lines 9-11)
- Unnecessary duplication in test block

**Fix:** Remove duplicate imports from `__main__` block

---

### B. Magic Numbers & Strings

#### ❌ `config_validator.py`
```python
# Line 114: Magic string
print(f"{'=' * 60}")

# Line 31: Magic regex pattern
pattern = r'params\.get\(["\']([^"\']+)["\']'
```
**Fix:** Extract to constants:
```python
SEPARATOR_WIDTH = 60
PARAM_GET_PATTERN = r'params\.get\(["\']([^"\']+)["\']'
```

#### ❌ `path_randomization.py`
```python
# Lines 173, 231: Magic numbers for progress reporting
if (i + 1) % 200 == 0:

# Lines 160, 218: Magic number for Sharpe annualization
sharpe = (np.mean(sampled) / np.std(sampled)) * np.sqrt(n_trades)
sharpe = (np.mean(shuffled) / np.std(shuffled)) * np.sqrt(252)
```
**Fix:** Extract to constants:
```python
PROGRESS_LOG_INTERVAL = 200
TRADING_DAYS_PER_YEAR = 252
```

---

### C. Naming Conventions

#### ✅ Generally Good
Most variable and function names follow PEP 8:
- `load_strategy_config` ✓
- `run_path_randomization_mc` ✓
- `get_available_strategies` ✓

#### ❌ Issues Found:
1. **config_validator.py** - `strategy_name` parameter shadowing in line 244
   ```python
   def suggest_parameter_mapping(strategy_name: str, config: Dict) -> Optional[Dict[str, str]]:
       # ...
       for config_name, strategy_name_param in suggestions.items():
           print(f"   '{config_name}' → '{strategy_name_param}'")
   ```
   - Variable `strategy_name_param` is confusing (not actually strategy name)
   - Better: `suggested_param_name`

---

### D. Error Handling Issues

#### ❌ `config_validator.py`
```python
# Line 36: Bare except with silent failure
except Exception:
    return None
```
**Issues:**
- Too broad exception catching
- Silent failures make debugging hard
- No logging of what went wrong

**Fix:**
```python
except (ImportError, AttributeError, ValueError) as e:
    logger.debug(f"Could not extract parameters for {strategy_name}: {e}")
    return None
```

#### ❌ `path_randomization.py`
```python
# Lines 136, 195: Generic error without context
if len(trades_df) == 0:
    raise ValueError("Portfolio has no trades to bootstrap")
```
**Improvement:** Add more context about the portfolio state

---

### E. Code Duplication

#### ❌ `path_randomization.py`
**Duplicate code** in `_bootstrap_trades` and `_shuffle_returns`:
- Lines 156-171 (equity calculation, Sharpe, max DD)
- Lines 213-228 (identical logic)

**Fix:** Extract to helper function:
```python
def _calculate_simulation_metrics(returns_array, n_periods):
    """Calculate equity curve, Sharpe ratio, and max drawdown."""
    equity = (1 + returns_array).cumprod()
    total_return = (equity[-1] - 1) * 100
    
    if np.std(returns_array) > 0:
        sharpe = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(n_periods)
    else:
        sharpe = 0.0
    
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    max_dd = np.abs(np.min(drawdown))
    
    return total_return, sharpe, max_dd, equity
```

---

### F. Type Hints & Validation

#### ✅ Good Coverage
- Most functions have proper type hints
- Return types specified

#### ❌ Inconsistencies:
1. `config_validator.py` line 168: Uses old-style tuple return
   ```python
   def suggest_parameter_mapping(...) -> Optional[Dict[str, str]]:
   ```
   Should specify `Dict[str, str]` keys/values more clearly

2. Missing validation for dictionary parameters before access

---

### G. Documentation & Comments

#### ✅ Good:
- Comprehensive docstrings in most functions
- Module-level documentation present

#### ❌ Needs Improvement:
1. **config_validator.py**: Commented-out code (lines 53-56)
   ```python
   # Note: Parameter extraction often fails, which is fine - configs are optional
   # result['warnings'].append(
   #     f"Could not extract expected parameters from strategy '{strategy_name}'"
   # )
   ```
   **Fix:** Remove or convert to proper documentation

---

## 🔧 SPECIFIC FILE REFACTORING TASKS

### `config_validator.py`

**Priority: HIGH**

1. **Fix broken import** (line 17)
   - Replace `from strategy_registry import _STRATEGIES`
   - With `from vectorflow.core.portfolio_builder import _STRATEGIES`

2. **Extract constants** (top of file):
   ```python
   SEPARATOR_WIDTH = 60
   PARAM_GET_PATTERN = r'params\.get\(["\']([^"\']+)["\']'
   NON_STRATEGY_PARAMS = {
       "primary_timeframe", "primary_symbol", "data_source",
       "csv_path", "initial_cash", "fee"
   }
   ```

3. **Improve error handling** (line 36):
   - Add specific exception types
   - Add logging for debugging

4. **Remove commented code** (lines 53-56)

5. **Rename variable** (line 244):
   - `strategy_name_param` → `suggested_param_name`

### `path_randomization.py`

**Priority: MEDIUM**

1. **Extract constants** (top of file):
   ```python
   PROGRESS_LOG_INTERVAL = 200
   TRADING_DAYS_PER_YEAR = 252
   PERCENTAGE_MULTIPLIER = 100
   ```

2. **Remove duplicate imports** (lines 244-246)

3. **Extract calculation logic** to helper function:
   - Create `_calculate_simulation_metrics()`
   - Eliminates duplication between bootstrap and shuffle methods

4. **Improve error messages**:
   - Add portfolio state information to exceptions

### `examples/debug_path_mc.py`

**Priority: HIGH**

1. **Fix broken imports**:
   - Line 3: `from monte_carlo_path import` → module doesn't exist
   - Line 4: `from strategy_registry import` → use `vectorflow.core.portfolio_builder`

2. **Consider deprecating or updating**:
   - This file appears outdated
   - Either fix completely or move to archived examples

---

## ✅ VALIDATION CHECKLIST

### Files to Test After Refactoring:
- [ ] `vectorflow/utils/config_validator.py`
- [ ] `vectorflow/validation/path_randomization.py`
- [ ] `examples/debug_path_mc.py`

### Integration Tests:
- [ ] Import `config_validator` and verify `_STRATEGIES` loads
- [ ] Run path randomization MC on sample portfolio
- [ ] Verify all strategy configs validate correctly

### Import Map to Verify:
```
vectorflow/
├── core/
│   ├── portfolio_builder.py → _STRATEGIES, create_portfolio, get_available_strategies
│   ├── config_manager.py → load_strategy_config, save_config
│   ├── data_loader.py → load_ohlc_csv
│   └── constants.py → All application constants
├── utils/
│   ├── config_validator.py → validate_strategy_config, quick_validate
│   └── portfolio_metrics.py
├── validation/
│   ├── path_randomization.py → run_path_randomization_mc
│   └── walk_forward.py
└── strategies/
    └── [strategy files] → create_portfolio functions
```

---

## 📊 PRIORITY RANKING

### 🔴 Critical (Must Fix Immediately)
1. Fix broken imports in `config_validator.py`
2. Fix or remove `examples/debug_path_mc.py`

### 🟡 High (Should Fix Soon)
1. Extract magic numbers to constants
2. Remove code duplication in path_randomization
3. Improve error handling with specific exceptions

### 🟢 Medium (Nice to Have)
1. Consolidate imports
2. Remove commented code
3. Enhance documentation

---

## 🎯 RECOMMENDED EXECUTION ORDER

1. **Phase 1 - Critical Fixes** (Do First)
   - Fix `config_validator.py` imports
   - Fix or deprecate `examples/debug_path_mc.py`
   - Test that system runs without errors

2. **Phase 2 - Code Quality** (Do Second)
   - Extract constants in both files
   - Remove duplicate imports
   - Clean up error handling

3. **Phase 3 - Refactoring** (Do Third)
   - Extract duplicate calculation logic
   - Improve naming consistency
   - Enhance documentation

---

## 🔍 SUMMARY STATISTICS

- **Critical Issues:** 2 (broken imports)
- **Code Duplication:** 2 instances (60+ duplicate lines)
- **Magic Numbers:** 5 instances
- **Error Handling Issues:** 3 instances
- **Dead Code:** 1 file (examples/debug_path_mc.py)
- **Import Issues:** 4 instances

**Overall Code Quality Score:** 7/10
- ✅ Good structure and organization
- ✅ Proper type hints
- ✅ Good documentation
- ❌ Broken imports (critical)
- ❌ Code duplication
- ❌ Magic numbers not extracted

