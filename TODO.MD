✅ STRUCTURE PROPOSÉE (Convention Google)
```
vectorflow/
├── vectorflow/                    # Package principal
│   ├── __init__.py
│   ├── core/                      # Logique métier core
│   │   ├── __init__.py
│   │   ├── portfolio_builder.py  # Anciennement: strategy_registry.py
│   │   ├── data_loader.py         # Anciennement: data_manager.py
│   │   └── constants.py
│   │
│   ├── strategies/                # Stratégies de trading
│   │   ├── __init__.py
│   │   ├── donchian.py
│   │   ├── grid.py
│   │   ├── risk_premia.py
│   │   ├── statistical_arbitrage.py  # Anciennement: sol_btc_stat_arb.py
│   │   ├── supertrend_grid.py
│   │   └── _template.py           # Underscore = private/template
│   │
│   ├── optimization/              # Tout ce qui concerne l'optimisation
│   │   ├── __init__.py
│   │   ├── grid_search.py         # Extraction de optimizer.py
│   │   ├── param_monte_carlo.py   # ⭐ RENOMMÉ (Monte Carlo pour params)
│   │   └── walk_forward.py
│   │
│   ├── validation/                # ⭐ NOUVEAU - Analyses de robustesse
│   │   ├── __init__.py
│   │   ├── path_randomization.py  # ⭐ RENOMMÉ (ex: monte_carlo_path.py)
│   │   └── bootstrap.py           # Futur: autres validations
│   │
│   ├── visualization/             # ⭐ RENOMMÉ
│   │   ├── __init__.py
│   │   ├── plotters.py            # Anciennement: plotter.py
│   │   ├── dashboard.py           # Futur: Streamlit/Dash UI
│   │   └── formatters.py          # Helpers pour formatting
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── portfolio_metrics.py   # Anciennement: vbt_utils.py
│   │   └── config_validator.py
│   │
│   └── cli.py                     # Anciennement: main.py
│
├── config/                        # Configuration files
│   ├── strategies/
│   │   ├── donchian.yaml
│   │   └── ...
│   └── settings.yaml              # Global settings
│
├── tests/                         # ⭐ NOUVEAU dossier
│   ├── __init__.py
│   ├── test_strategies.py         # Anciennement: test_all_strategies.py
│   ├── test_path_randomization.py # Anciennement: test_path_mc_robust.py
│   ├── test_optimization.py
│   └── fixtures/                  # Test data
│
├── examples/                      # ⭐ NOUVEAU - Scripts démo
│   ├── quickstart.py
│   ├── custom_strategy_example.py
│   └── batch_backtest.py
│
├── scripts/                       # Scripts utilitaires (pas exemples)
│   ├── download_data.py
│   └── benchmark_strategies.py
│
├── data/                          # Data files (ignored in git)
├── .agent/                        # Workflow automation
├── requirements.txt
├── setup.py                       # ⭐ NOUVEAU - Pour pip install
├── pyproject.toml                 # ⭐ NOUVEAU - Build config
└── README.md
```
🔄 RENOMMAGES CLÉS (Priorité Haute)
1. Clarifier les Monte Carlo
python
# AVANT
optimizer.py::run_monte_carlo_analysis()         # Confus!
monte_carlo_path.py::run_path_randomization_mc() # Confus!

# APRÈS
optimization/param_monte_carlo.py::run_parameter_sensitivity()
validation/path_randomization.py::run_path_randomization()
2. Fichiers principaux
Ancien	Nouveau	Raison
main.py	cli.py ou vectorflow/cli.py	Convention: main.py = entry point standalone
strategy_registry.py	core/portfolio_builder.py	Plus descriptif
data_manager.py	core/data_loader.py	Verbe → action claire
plotter.py	visualization/plotters.py	Pluriel + dossier
walk_forward.py	optimization/walk_forward.py	Même catégorie que optimisation
vbt_utils.py	utils/portfolio_metrics.py	Descriptif, pas d'abréviation
monte_carlo_path.py	validation/path_randomization.py	⭐ Clair et précis
optimizer.py	Split en optimization/grid_search.py + optimization/param_monte_carlo.py	Séparation responsabilités
3. Noms de fonctions (Cohérence)
python
# Optimizer / Parameter Sensitivity
run_parameter_sensitivity()      # Au lieu de run_monte_carlo_analysis()
sample_random_parameters()       # OK

# Path Randomization
run_path_randomization()         # Au lieu de run_path_randomization_mc()
shuffle_returns()                # OK - privée
bootstrap_trades()               # OK - privée

# Plotting
plot_parameter_sensitivity()     # Au lieu de plot_monte_carlo()
plot_path_randomization()        # Au lieu de plot_path_mc_results()
📝 PLAN DE MIGRATION (Étapes)
Phase 1: Renommages Simples (Pas de breaking changes)
bash
# 1. Créer structure
mkdir -p vectorflow/{core,optimization,validation,visualization}
mkdir tests examples

# 2. Déplacer fichiers (avec alias temporaires)
mv monte_carlo_path.py validation/path_randomization.py
mv walk_forward.py optimization/
mv plotter.py visualization/plotters.py

# 3. Créer __init__.py avec ré-exports pour compatibilité
# vectorflow/__init__.py
from .validation.path_randomization import run_path_randomization as run_path_randomization_mc  # Backward compat
Phase 2: Refactoring optimizer.py
python
# optimization/param_monte_carlo.py
def run_parameter_sensitivity(...):  # Nouveau nom clair
    """Monte Carlo parameter sensitivity analysis."""
    pass

# optimization/grid_search.py  
def run_grid_optimization(...):
    """Grid search optimization."""
    pass
Phase 3: Tests
bash
mv test_*.py tests/
mv debug_*.py examples/ ou scripts/
💡 RECOMMANDATIONS ADDITIONNELLES
Naming Conventions à suivre:
✅ Modules: lowercase_with_underscores.py
✅ Classes: CapWords (ex: ParameterSensitivityAnalyzer)
✅ Fonctions: lowercase_with_underscores()
✅ Constantes: UPPER_WITH_UNDERSCORES
✅ Private: _leading_underscore
Docstrings (Google Style):
python
def run_path_randomization(portfolio, n_simulations=1000, method="shuffle"):
    """Tests strategy robustness by randomizing event sequence.
    
    Args:
        portfolio: VectorBT Portfolio object to analyze
        n_simulations: Number of randomized paths to generate
        method: Randomization method ('shuffle', 'bootstrap', 'block')
    
    Returns:
        dict: Statistics including p-values, percentiles, and distributions
        
    Raises:
        ValueError: If portfolio has no trades or zero variance
    """

---

## 🔬 **VALIDATION STATISTIQUE AUTOMATIQUE** (Éviter overfitting / data snooping)

### Problème
- Actuellement: on regarde Sharpe/Return sans savoir si c'est statistiquement significatif
- Risque: Stratégie profitable par hasard (ex: 5% de 100 tests passeront par pur luck)

### Solution: Module `validation/statistical_tests.py`

**Features à implémenter:**

1. **Sharpe significance test** (Jobson & Korkie 1981)
   - Test si Sharpe > 0 avec robustesse non-normalité
   - Prend en compte skew et kurtosis
   - Output: p-value + conclusion claire

2. **Minimum trades check**
   - Warning si < 30 trades (pas assez pour stats)
   - Calcul automatique intervalle confiance

3. **Autocorrelation test** (Ljung-Box)
   - Détecte lookahead bias potentiel
   - Returns ne devraient PAS être autocorrélés

4. **Win rate sanity check**
   - Flag si >95% (suspect, check lookahead)
   - Flag si <20% (probablement cassé)

5. **Multiple testing adjustment** (Harvey 2017)
   - Si N variantes testées → Bonferroni correction
   - Haircut Sharpe: `adjusted = observed - sqrt(N) * 0.5`

**Intégration dans main.py:**
- Appel automatique après chaque backtest
- Print warnings clairs si fails
- Option `--skip-validation` pour désactiver

**Effort estimé:** 2-3h
**Impact:** Critique - évite self-deception

---

## ✅ **CHECKLIST DE VALIDATION INTERACTIVE**

### Problème
- Pas de guide systématique pour éviter erreurs communes
- Utilisateurs oublient checks basiques (lookahead, survivorship bias, etc.)

### Solution: Module `validation/checklist.py`

**Catégories:**

1. **Data Quality** (5 checks)
   - No future data leakage
   - No survivorship bias
   - Timezone consistency
   - No unexplained gaps
   - Volume data realistic

2. **Strategy Logic** (4 checks)
   - Signals use only past data (t-1, not t)
   - No intraday lookahead (open vs close)
   - Execution delays modeled
   - Rebalancing frequency realistic

3. **Optimization** (4 checks)
   - Walk-forward completed
   - Holdout set tested (20% unseen)
   - Performance consistent across periods
   - Parameter count reasonable (data/params > 20)

4. **Robustness** (4 checks)
   - Path randomization MC passed (p < 0.05)
   - Sharpe statistically significant
   - Minimum 30 trades
   - Beats buy-and-hold benchmark

**Modes:**
- Interactive: CLI avec input y/n
- Auto: Appelle validation stats où possible
- Report: Generate PDF checklist

**Intégration:**
- Appel optionnel: `python main.py --strategy X --validate`
- Output: Score X/17 passed + detailed report

**Effort estimé:** 3-4h  
**Impact:** Fort - pédagogique + prévient erreurs

---

## 📊 **BUY-AND-HOLD BENCHMARK (Simple print)**

### Problème
- Pas de comparaison automatique vs benchmark simple
- Utilisateurs peuvent célébrer stratégie qui ne bat même pas B&H

### Solution: 3 lignes dans `print_aggregated_portfolio_stats()`

```python
# Après ligne 227 dans main.py
# Add buy-and-hold comparison
bh_portfolio = vbt.Portfolio.from_holding(close=primary_data['close'], init_cash=10000)
bh_return = bh_portfolio.total_return()
print(f"📌 Buy & Hold Benchmark: {bh_return:.2f}%")
print(f"   {'✅ Outperforms' if avg_return > bh_return else '⚠️ Underperforms'} B&H by {avg_return - bh_return:+.2f}%")