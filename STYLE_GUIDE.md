# Code Style Guide

## 🎯 Philosophie
Écrire du code qui est à la fois **concis et clair**. Utiliser des mots complets quand c'est possible, mais éviter les noms inutilement longs. Prioriser la lisibilité et la cohérence.

## 📋 Conventions de Nommage

### Classes de Stratégie

1. **Acronymes** : Utiliser des majuscules pour les acronymes
   ```python
   class ORBStrategy(BaseStrategy):  # Opening Range Breakout
   class LTIStrategy(BaseStrategy):  # Logical Trading Indicator
   ```

2. **Mots Réguliers** : Utiliser le CamelCase standard
   ```python
   class MomentumStrategy(BaseStrategy)
   class TrendFollowingStrategy(BaseStrategy)
   ```

### **Functions**
- **✅ DO**: Use verb-noun patterns that clearly describe the action
- **✅ DO**: Keep names between 2-4 words maximum
- **❌ AVOID**: Overly long descriptive names
- **❌ AVOID**: Single letter or unclear abbreviations

```python
# ✅ Good Examples
def calc_metrics()
def plot_results()
def run_backtest()
def select_column()
def get_scalar()

# ❌ Bad Examples  
def run_monte_carlo_permutation_test()  # Too long
def safe_select_portfolio_column()      # Too long
def calc()                              # Too short
def p()                                 # Unclear
```

### **Variables**
- **✅ DO**: Use descriptive nouns, avoid unnecessary prefixes
- **✅ DO**: Use consistent abbreviations across the codebase
- **❌ AVOID**: Hungarian notation or type prefixes

```python
# ✅ Good Examples
portfolio = run_backtest()
config = load_config()
data = load_data()
returns = portfolio.returns()
sharpe_ratio = returns.mean() / returns.std()

# ❌ Bad Examples
portfolio_config_dict = {}     # Redundant suffixes
p_tuple = ()                   # Unclear abbreviation  
df_sym = data.get()           # Unclear prefix
cum_returns = portfolio.cumulative_returns()  # Inconsistent with returns
```

### **Parameters**
- **✅ DO**: Use full words for important parameters
- **✅ DO**: Use standard abbreviations for common concepts

```python
# ✅ Good Examples
def run_backtest(data, signals, config, timeframe):
def add_indicators(df, config):
def risk_size_atr(data, portfolio, indicators, timeframe):

# ❌ Bad Examples
def run_backtest(data, signals, portfolio_config, timeframe_to_use):
def add_indicators(df, indicators_config):
```

## 🔧 **Standard Abbreviations**

Use these consistently throughout the codebase:

| Concept | Use | Don't Use |
|---------|-----|-----------|
| Configuration | `config` | `cfg`, `configuration`, `settings` |
| Portfolio | `portfolio` | `pf`, `port` |
| Timeframe | `timeframe` | `tf`, `time_frame` |
| Returns | `returns` | `ret`, `cum_returns` |
| Parameters | `params` | `parameters`, `param_dict` |
| Indicators | `indicators` | `ind`, `indicators_config` |
| Column | `col` | `column`, `col_idx` |
| Data | `data` | `df`, `dataset` |

## 📐 **Consistency Rules**

### **1. Helper Functions**
- Prefix with single underscore: `_helper_name()`
- Keep names short and focused

```python
# ✅ Good
def _get_returns()
def _process_data()  
def _add_traces()

# ❌ Bad
def _get_cumulative_returns_safely()
def _create_train_test_split()
```

### **2. Similar Operations**
Use consistent patterns for similar operations:

```python
# Data processing
def load_data()
def process_data()
def save_data()

# Plotting functions  
def plot_results()
def plot_returns()
def plot_metrics()

# Analysis functions
def calc_metrics()
def calc_sharpe()
def calc_returns()
```

### **3. Boolean Variables**
Use clear positive naming:

```python
# ✅ Good
is_valid = True
has_data = len(df) > 0
use_filter = config.get('enable_filter', True)

# ❌ Bad  
valid = True
no_data = len(df) == 0
disable_filter = False
```

## 🏗️ **Module Structure**

### **File Names**
- Use lowercase with underscores
- Be descriptive but concise

```
✅ Good: backtest.py, indicators.py, plotting.py
❌ Bad: monte_carlo_permutation_test.py, strategy_signals.py
```

### **Class Names**
- Use PascalCase
- Avoid redundant suffixes

```python
# ✅ Good
class MarketData:
class Portfolio:

# ❌ Bad  
class MarketDataClass:
class PortfolioManager:
```

## 📊 **Constants**

Use UPPER_CASE for module-level constants:

```python
# ✅ Good
TRADING_DAYS_PER_YEAR = 252
PERCENTAGE_MULTIPLIER = 100
VAR_95_PERCENTILE = 5

# ❌ Bad
trading_days_per_year = 252
TRADING_DAYS_IN_A_YEAR_FOR_CALCULATION = 252
```

## 🔄 **Refactoring Guidelines**

When refactoring long names:

1. **Identify the core action/concept**
2. **Remove redundant words** 
3. **Use standard abbreviations**
4. **Ensure the name is still clear**

```python
# Before
def run_monte_carlo_permutation_test() -> run_monte_carlo_test()
def plot_train_test_split_returns() -> plot_split_returns()  
def safe_select_portfolio_column() -> select_column()

# The refactored names are shorter but still clear
```

## ✅ **Quick Checklist**

Before committing code, ask:

- [ ] Are my function names 2-4 words maximum?
- [ ] Do I use consistent abbreviations?
- [ ] Are variable names descriptive without being verbose?
- [ ] Do similar operations follow the same naming pattern?
- [ ] Would a new developer understand these names?

## 🚫 **Anti-Patterns to Avoid**

```python
# ❌ Overly long names
def calculate_portfolio_performance_metrics_with_risk_adjustment()

# ❌ Unclear abbreviations  
def calc_pf_perf_w_ra()

# ❌ Inconsistent naming
portfolio_config = {}
indicators_cfg = {}
timeframe_settings = {}

# ❌ Redundant prefixes/suffixes
config_dict = {}
df_dataframe = pd.DataFrame()
portfolio_obj = Portfolio()
```

---

**Remember**: Code is read far more often than it's written. Optimize for clarity and consistency.