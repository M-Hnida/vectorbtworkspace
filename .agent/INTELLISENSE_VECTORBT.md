# 🔧 IntelliSense avec VectorBT - Guide Complet

## ❓ **Le Problème: Pourquoi IntelliSense ne détecte pas les attributs Portfolio?**

Quand vous tapez `portfolio.` vous n'avez pas d'autocomplétion pour `value()`, `sharpe_ratio()`, etc.

### **Raisons Techniques:**

VectorBT utilise des **métaclasses dynamiques** qui créent des attributs à la volée au runtime, ce qui rend l'autocomplétion impossible pour les IDEs standards.

```python
# Résultat de notre test:
VectorBT version: 0.28.0
Portfolio class type: <class 'vectorbt.portfolio.base.MetaPortfolio'>
                                                    ^^^^^ Métaclasse!
```

La classe `Portfolio` est générée par `MetaPortfolio` qui ajoute des méthodes dynamiquement:
- ✅ Le code fonctionne parfaitement au runtime
- ❌ Mais l'IDE ne peut pas prédire ce qui sera disponible

---

## 🎯 **Solutions pour Activer IntelliSense**

### **Solution 1: Type Stubs (Recommandée) ✅**

VectorBT fournit des **type stubs** (.pyi files) pour l'autocomplétion.

#### **Installation:**
```bash
# Les stubs sont normalement inclus avec vectorbt
# Si ce n'est pas le cas, vérifier:
pip install —-upgrade vectorbt

# Ou installer manuellement les stubs:
pip install types-vectorbt  # Si disponible
```

#### **Vérification:**
```python
# Dans votre IDE, ceci devrait maintenant fonctionner:
import vectorbt as vbt

portfolio = vbt.Portfolio.from_signals(...)
portfolio.  # <-- IntelliSense devrait proposer: value(), sharpe_ratio(), etc.
```

---

### **Solution 2: Type Hints Manuels (Workaround)**

Si les stubs ne fonctionnent pas, ajoutez des type hints manuels:

#### **A) Fichier de stubs local**

Créez `vectorbt_stubs.pyi` dans votre projet:

```python
# vectorbt_stubs.pyi
from typing import Optional, Union, Any
import pandas as pd
import numpy as np

class Portfolio:
    """VectorBT Portfolio with manual type hints for IntelliSense"""
    
    # Portfolio creation
    @staticmethod
    def from_signals(
        close: pd.Series,
        entries: Union[pd.Series, pd.DataFrame],
        exits: Union[pd.Series, pd.DataFrame],
        short_entries: Optional[Union[pd.Series, pd.DataFrame]] = None,
        short_exits: Optional[Union[pd.Series, pd.DataFrame]] = None,
        size: Union[float, pd.Series, pd.DataFrame] = 1.0,
        size_type: str = "amount",
        init_cash: float = 10000,
        fees: float = 0.0,
        fixed_fees: float = 0.0,
        slippage: float = 0.0,
        freq: Optional[str] = None,
        direction: str = "longonly",
        conflict_mode: str = "opposite",
        accumulate: bool = False,
        **kwargs: Any
    ) -> 'Portfolio': ...
    
    @staticmethod
    def from_orders(
        close: pd.Series,
        size: Union[pd.Series, pd.DataFrame],
        size_type: str = "amount",
        **kwargs: Any
    ) -> 'Portfolio': ...
    
    @staticmethod
    def from_holding(
        close: pd.Series,
        init_cash: float = 10000,
        **kwargs: Any
    ) -> 'Portfolio': ...
    
    # Properties
    @property
    def init_cash(self) -> float: ...
    
    @property
    def trades(self) -> Any: ...  # TradesAccessor
    
    @property
    def orders(self) -> Any: ...  # OrdersAccessor
    
    @property
    def positions(self) -> Any: ...  # PositionsAccessor
    
    @property
    def drawdowns(self) -> Any: ...  # DrawdownsAccessor
    
    # Statistics methods
    def stats(self, **kwargs: Any) -> pd.Series: ...
    
    def total_return(self) -> Union[float, pd.Series]: ...
    
    def sharpe_ratio(self, **kwargs: Any) -> Union[float, pd.Series]: ...
    
    def sortino_ratio(self, **kwargs: Any) -> Union[float, pd.Series]: ...
    
    def calmar_ratio(self, **kwargs: Any) -> Union[float, pd.Series]: ...
    
    def max_drawdown(self, **kwargs: Any) -> Union[float, pd.Series]: ...
    
    def win_rate(self) -> Union[float, pd.Series]: ...
    
    # Time series data
    def value(self) -> Union[pd.Series, pd.DataFrame]: ...
    
    def returns(self, **kwargs: Any) -> Union[pd.Series, pd.DataFrame]: ...
    
    def cumulative_returns(self) -> Union[pd.Series, pd.DataFrame]: ...
    
    def cash(self) -> Union[pd.Series, pd.DataFrame]: ...
    
    def shares(self) -> Union[pd.Series, pd.DataFrame]: ...
    
    # Risk metrics
    def alpha(self, benchmark_rets: Optional[pd.Series] = None) -> Union[float, pd.Series]: ...
    
    def beta(self, benchmark_rets: Optional[pd.Series] = None) -> Union[float, pd.Series]: ...
    
    def downside_risk(self) -> Union[float, pd.Series]: ...
    
    def up_capture(self) -> Union[float, pd.Series]: ...
    
    def down_capture(self) -> Union[float, pd.Series]: ...
    
    # Plotting
    def plot(self, **kwargs: Any) -> Any: ...  # Returns Plotly Figure
    
    def plot_performance(self, **kwargs: Any) -> Any: ...
```

#### **Utilisation:**

```python
# Dans votre code
from vectorbt_stubs import Portfolio  # Pour IntelliSense uniquement

# Ou avec typing.TYPE_CHECKING:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbt_stubs import Portfolio
else:
    import vectorbt as vbt
    Portfolio = vbt.Portfolio

# Maintenant les type hints fonctionnent:
def analyze_strategy(data, params) -> Portfolio:  # ✅ IntelliSense détecte
    portfolio = create_portfolio(data, params)
    
    # ✅ Autocomplétion fonctionne ici
    sharpe = portfolio.sharpe_ratio()
    return_val = portfolio.total_return()
    
    return portfolio
```

---

### **Solution 3: Configuration IDE (VS Code)**

#### **A) Paramètres Pylance**

Ajoutez à votre `settings.json`:

```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.useLibraryCodeForTypes": true,
    "python.analysis.autoImportCompletions": true,
    "python.analysis.completeFunctionParens": true,
    "python.analysis.stubPath": "./typings"
}
```

#### **B) Installer Pylance Extension**

```bash
# Dans VS Code, installer:
# - Python (Microsoft)
# - Pylance (Microsoft)
```

#### **C) Créer typings/vectorbt.pyi**

Même contenu que `vectorbt_stubs.pyi` ci-dessus, mais dans le dossier `typings/`.

---

### **Solution 4: Docstrings + Comments (Minimal)**

Si rien d'autre ne fonctionne, utilisez des commentaires:

```python
import vectorbt as vbt

portfolio = vbt.Portfolio.from_signals(...)

# Available methods (see VectorBT docs):
# - portfolio.value() -> pd.Series : Equity curve
# - portfolio.returns() -> pd.Series : Returns
# - portfolio.sharpe_ratio() -> float : Sharpe ratio
# - portfolio.total_return() -> float : Total return %
# - portfolio.stats() -> pd.Series : All stats
# - portfolio.trades.records_readable -> pd.DataFrame : All trades

sharpe = portfolio.sharpe_ratio()  # Type: float
returns = portfolio.returns()  # Type: pd.Series
```

---

## 📚 **Référence Complète des Méthodes Portfolio**

### **Création de Portfolio**

```python
# Method 1: from_signals (le plus utilisé)
portfolio = vbt.Portfolio.from_signals(
    close=close_prices,          # pd.Series
    entries=entry_signals,       # pd.Series (bool)
    exits=exit_signals,          # pd.Series (bool)
    size=1.0,                    # float ou pd.Series
    size_type="percent",         # "amount", "percent", "targetpercent"
    init_cash=10000,             # float
    fees=0.001,                  # float (0.001 = 0.1%)
    freq="1H",                   # str (important!)
    direction="longonly",        # "longonly", "shortonly", "both"
    accumulate=False,            # bool
)

# Method 2: from_orders (plus flexible)
portfolio = vbt.Portfolio.from_orders(
    close=close_prices,
    size=order_sizes,            # pd.Series ou pd.DataFrame
    size_type="targetpercent",   # Pour rebalancing
    **kwargs
)

# Method 3: from_holding (benchmark)
portfolio = vbt.Portfolio.from_holding(
    close=close_prices,
    init_cash=10000,
    fees=0.001
)
```

### **Statistiques Principales**

```python
# Toutes les stats en un coup
stats = portfolio.stats()  # -> pd.Series avec ~30 métriques

# Métriques individuelles
total_return = portfolio.total_return()      # % (e.g., 45.2)
sharpe = portfolio.sharpe_ratio()            # ratio (e.g., 1.5)
sortino = portfolio.sortino_ratio()          # ratio
calmar = portfolio.calmar_ratio()            # ratio
max_dd = portfolio.max_drawdown()            # % (e.g., -20.5)
```

### **Séries Temporelles**

```python
# Equity curve
equity = portfolio.value()           # pd.Series : valeur du portfolio dans le temps

# Returns
returns = portfolio.returns()        # pd.Series : returns quotidiens
cum_returns = portfolio.cumulative_returns()  # pd.Series : returns cumulatifs

# Cash & Shares
cash = portfolio.cash()              # pd.Series : cash disponible
shares = portfolio.shares()          # pd.Series : shares détenues
```

### **Trades**

```python
# Objet Trades
trades = portfolio.trades

# DataFrame de tous les trades
trades_df = trades.records_readable  # pd.DataFrame

# Métriques des trades
win_rate = trades.win_rate()         # % de trades gagnants
profit_factor = trades.profit_factor()  # profit / loss ratio
expectancy = trades.expectancy()     # gain moyen par trade
avg_win = trades.winning_streak.avg()  # moyenne des winning streaks
```

### **Risk Metrics**

```python
# Benchmark comparison
alpha = portfolio.alpha()            # Alpha vs benchmark
beta = portfolio.beta()              # Beta vs benchmark

# Risk metrics
downside_risk = portfolio.downside_risk()
up_capture = portfolio.up_capture()
down_capture = portfolio.down_capture()
```

### **Drawdowns**

```python
# Objet Drawdowns
drawdowns = portfolio.drawdowns

# Métriques
max_dd = drawdowns.max_drawdown()    # Max drawdown
avg_dd = drawdowns.avg_drawdown()    # Average drawdown
max_duration = drawdowns.max_duration()  # Longest DD period
```

### **Plotting**

```python
# Plot complet du portfolio
fig = portfolio.plot()  # -> plotly Figure
fig.show()

# Customisation
fig = portfolio.plot(
    subplot_settings={
        "orders": {"visible": False}
    }
)

# Performance plot
fig = portfolio.plot_performance()
```

---

## 🚀 **Configuration Recommandée pour VectorFlow**

### **Créer `typings/vectorbt/__init__.pyi`**

```python
# typings/vectorbt/__init__.pyi
"""Type stubs pour VectorBT - autocomplétion IDE"""

from typing import Union, Optional, Any
import pandas as pd

class Portfolio:
    # Les méthodes les plus utilisées
    @staticmethod
    def from_signals(...) -> 'Portfolio': ...
    
    def stats(self) -> pd.Series: ...
    def sharpe_ratio(self) -> Union[float, pd.Series]: ...
    def total_return(self) -> Union[float, pd.Series]: ...
    def max_drawdown(self) -> Union[float, pd.Series]: ...
    def value(self) -> Union[pd.Series, pd.DataFrame]: ...
    def returns(self) -> Union[pd.Series, pd.DataFrame]: ...
    def beta(self) -> Union[float, pd.Series]: ...
    
    @property
    def trades(self) -> Any: ...
```

### **Ajouter à `pyproject.toml` (si vous en avez un)**

```toml
[tool.pyright]
typeCheckingMode = "basic"
stubPath = "./typings"
useLibraryCodeForTypes = true
```

---

## 📝 **Best Practices pour travailler avec VectorBT**

### **1. Utilisez Type Hints pour vos fonctions**

```python
from typing import Dict, Optional
import pandas as pd

def create_portfolio(
    data: pd.DataFrame, 
    params: Optional[Dict] = None
) -> "vbt.Portfolio":  # <- String annotation pour éviter erreurs import
    """Create portfolio - returns VectorBT Portfolio object"""
    import vectorbt as vbt
    
    # ... logique ...
    
    portfolio = vbt.Portfolio.from_signals(...)
    return portfolio

# Maintenant IntelliSense sait que le retour est un Portfolio
def analyze_strategy(data, params):
    portfolio = create_portfolio(data, params)
    
    # ✅ IntelliSense propose sharpe_ratio() ici
    sharpe = portfolio.sharpe_ratio()
    return sharpe
```

### **2. Docstrings explicites**

```python
def extract_portfolio_metrics(portfolio: "vbt.Portfolio") -> Dict[str, float]:
    """
    Extract key metrics from VectorBT portfolio.
    
    Args:
        portfolio: VectorBT Portfolio object with methods:
                   - sharpe_ratio() -> float
                   - total_return() -> float
                   - max_drawdown() -> float
                   
    Returns:
        Dictionary with extracted metrics
    """
    return {
        'sharpe': portfolio.sharpe_ratio(),
        'return': portfolio.total_return(),
        'max_dd': portfolio.max_drawdown()
    }
```

### **3. Constantes pour noms de stats**

```python
# constants.py
STAT_SHARPE_RATIO = "Sharpe Ratio"
STAT_TOTAL_RETURN = "Total Return [%]"
STAT_MAX_DRAWDOWN = "Max Drawdown [%]"

# Usage
stats = portfolio.stats()
sharpe = stats.get(STAT_SHARPE_RATIO, 0.0)  # ✅ Autocomplete sur STAT_*
```

---

## 🐛 **Troubleshooting**

### **IntelliSense ne fonctionne toujours pas?**

1. **Recharger la fenêtre VS Code**
   ```
   Ctrl+Shift+P -> "Reload Window"
   ```

2. **Vérifier la version Python**
   ```bash
   python --version  # Doit être >= 3.8
   ```

3. **Réinstaller VectorBT**
   ```bash
   pip uninstall vectorbt
   pip install vectorbt
   ```

4. **Vérifier le Python interpreter**
   ```
   Ctrl+Shift+P -> "Python: Select Interpreter"
   Choisir le bon environnement virtuel
   ```

5. **Activer verbose logging**
   ```json
   // settings.json
   {
       "python.analysis.logLevel": "Trace"
   }
   ```

---

## ✅ **Conclusion**

**Le problème:** VectorBT utilise des métaclasses dynamiques qui empêchent l'autocomplétion native.

**La solution:** Combinez:
1. Type stubs (`.pyi` files)
2. Type hints manuels (`-> "vbt.Portfolio"`)
3. Configuration IDE (Pylance)
4. Docstrings détaillées

**Bonne nouvelle:** Une fois configuré, vous aurez une excellente autocomplétion pour tous les projets VectorBT ! 🚀

