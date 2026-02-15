# VectorFlow TODOs

## HIGH PRIORITY
- [x] Trade direction markers (long/short) in plots
- [x] Path Monte Carlo in CLI (already wired up!)
- [x] Buy & Hold benchmark comparison

## MEDIUM PRIORITY
- [ ] Multi-timeframe data loading optimization
- [ ] Statistical validation module

## LOW PRIORITY
- [ ] Dashboard UI
- [ ] Enhanced documentation

---

## 🔴 HIGH PRIORITY (Next Steps)

### 1. **Add Trade Direction Lines to Portfolio Plots** ⏳ IN PROGRESS
**Status**: Attempted but file corruption occurred  
**Goal**: Add visual lines connecting entry/exit points, colored by P&L (green=win, red=loss)  
**Location**: `vectorflow/visualization/plotters.py` - `_plot_single_portfolio()` function  
**Why**: Makes it much clearer when trades occur and their outcomes (like backtesting.py style)  
**Estimated effort**: 30min  

### 2. **Fix Multi-Timeframe Data Loading for Single-TF Strategies**
**Status**: Identified issue  
**Problem**: Strategies that only need primary_timeframe still load ALL timeframes  
**Impact**: Significant performance hit (e.g., loading 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w unnecessarily)  
**Solution**: Check if strategy `needs_multi_timeframe()`, if false, only load primary  
**Location**: `cli.py` - `load_data()` method  
**Estimated effort**: 1h  

### 3. **Validate Path Monte Carlo is Using Correct Method**
**Issue**: User noticed "Parameter Monte Carlo" ran when selecting "Path Randomization Monte Carlo"
**Check needed**:
  - Verify `run_path_randomization_mc()` is called (not `run_monte_carlo_analysis()`)
  - Confirm plot shows equity paths, not parameter distributions  
  - Test with stddev_grid strategy
**Estimated effort**: 30min testing

---

## 🟡 MEDIUM PRIORITY

### 4. **Implement Buy-and-Hold Benchmark Comparison**
**Why**: Users need automatic comparison to know if strategy beats simple B&H  
**Implementation**: 3 lines in `print_summary()`:
```python
bh_portfolio = vbt.Portfolio.from_holding(close=close, init_cash=10000)
bh_return = bh_portfolio.total_return()
print(f"� Buy & Hold: {bh_return:.2f}% | {'✅' if return > bh_return else '⚠️'} Strategy: {return - bh_return:+.2f}%")
```
**Estimated effort**: 15min

### 5. **Statistical Validation Module** (`validation/statistical_tests.py`)
**Goal**: Prevent overfitting and data snooping  
**Features**:
1. Sharpe significance test (Jobson & Korkie 1981)
2. Minimum trades requirement (warn if < 30)
3. Autocorrelation test (Ljung-Box) - detect lookahead bias
4. Win rate sanity checks (flag if >95% or <20%)
5. Multiple testing adjustment (Harvey 2017 haircut)

**Integration**: Auto-run after backtest, print warnings  
**Estimated effort**: 2-3h  
**Impact**: Critical - prevents self-deception

### 6. **Interactive Validation Checklist** (`validation/checklist.py`)
**Categories** (17 checks total):
- Data Quality (5): lookahead, survivorship bias, timezone, gaps, volume
- Strategy Logic (4): past-only signals, execution delays, rebalancing frequency
- Optimization (4): walk-forward, holdout set, consistency, parameter count
- Robustness (4): path MC passed, Sharpe significant, min trades, beats B&H

**Modes**: Interactive CLI, Auto-validate, PDF report  
**Command**: `python main.py --strategy X --validate`  
**Estimated effort**: 3-4h

---

## 🟢 LOW PRIORITY (Nice to Have)

### 7. **Setup.py & PyPI Distribution**
- Create proper `setup.py` for `pip install vectorflow`
- Update `pyproject.toml` with build config
- Publish to PyPI (optional)
**Estimated effort**: 2h

### 8. **Enhanced Documentation**
- Expand README with quickstart guide
- Add strategy development tutorial
- Create examples/ directory with working demos
**Estimated effort**: 3-4h

### 9. **Dashboard UI** (`visualization/dashboard.py`)
- Streamlit or Dash interface
- Live portfolio monitoring
- Interactive parameter tuning
**Estimated effort**: 8-10h

---

## � CODE QUALITY IMPROVEMENTS

### Linting (Ongoing)
- Clean up remaining Ruff warnings
- Fix any type hint inconsistencies
- Remove dead code in plotters.py (if any remains)

### Testing
- Add unit tests for strategies
- Test path Monte Carlo methods (shuffle, bootstrap, block)
- Integration tests for CLI workflows

---

## 🎯 IMMEDIATE NEXT ACTION

**Priority 1**: Fix trade direction lines in plots (complete the attempted work)  
**Why**: Essential UX improvement that was interrupted  
**Steps**:
1. Restore `_plot_single_portfolio()` properly
2. Add trade lines colored by P&L
3. Test with stddev_grid strategy
4. Verify browser performance with 1000+ trades limit

**Priority 2**: Verify Path MC is working correctly (not showing param MC)  
**Priority 3**: Add Buy & Hold benchmark (quick win)

---

## 📝 NOTES

### Monte Carlo Naming Clarity
✅ Successfully separated:
- **Parameter MC** (`optimization/param_monte_carlo.py`) - Tests parameter sensitivity
- **Path MC** (`validation/path_randomization.py`) - Tests sequence dependency

### Data Loading Strategy
✅ Now checks:
1. `config.csv_path` (root level)
2. `config.parameters.csv_path` (fallback)
3. Auto-discover `data/*.csv` (last resort)

### Multi-Timeframe Support
- Template allows both single DataFrame and Dict[timeframe, DataFrame]
- Auto-detection via `strategy_needs_multi_timeframe()`
- Performance optimization needed for single-TF strategies