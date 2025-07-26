# Trading Strategy Framework

Un framework modulaire de backtesting et d'analyse de stratégies de trading utilisant VectorBT.

## 🔄 Flux d'Exécution

Le système suit un flux d'exécution clair et modulaire, orchestré par `main.py` :

### 1. Point d'Entrée (`main.py`)

```python
python main.py
```

Le script principal :
1. Liste les stratégies disponibles (fichiers YAML dans `config/`)
2. Demande à l'utilisateur de choisir une stratégie
3. Initialise le pipeline d'analyse complet

### 2. Configuration (`ConfigManager`)

- Charge les paramètres depuis `config/{strategy}.yaml`
- Structure :
  ```yaml
  strategy_params:
    orb_period: 30
    breakout_threshold: 0.0001
  
  data_requirements:
    symbols: ["EURUSD"]
    timeframe: "1h"
  ```

### 3. Stratégie (`ORBStrategy`, `MomentumStrategy`, etc.)

- Chargée dynamiquement depuis `core/strategies/`
- Hérite de `BaseStrategy`
- Génère les signaux (entrées/sorties) via :
  ```python
  signals = strategy.generate_signals(data, direction='both')
  entries, exits = strategy.combine_signals(signals)
  ```g Strategy Framework
This shi spaghetti code ngl

### Strategy Implementation
- **`core/strategies/`** - Strategy signal generation modules
  - `momentum_strategy.py` - Volatility momentum strategy
  - `lti_strategy.py` - Logical Trading Indicator strategy  
  - `orb_strategy.py` - Opening Range Breakout strategy

### 4. Gestion du Portfolio (`PortfolioManager`)

- Crée et gère les positions
- Supporte :
  - Trading directionnel (long/short/both)
  - Positions multi-actifs
  - Stop-loss dynamiques (ATR)
  ```python
  portfolio = portfolio_manager.create_portfolio(
      data=data,
      entries=entries,
      exits=exits,
      direction='both'
  )
  ```

### 5. Analyse Complète (`TradingSystem`)

Le système exécute une analyse en plusieurs étapes :
1. **Optimisation** : Recherche des meilleurs paramètres
2. **Walk-Forward** : Test de robustesse temporelle
3. **Monte Carlo** : Validation statistique
4. **Visualisation** : Graphiques de performance


## 📊 Structure du Projet

```
project/
│
├── main.py           # Point d'entrée principal
├── core/
│   ├── base.py          # Classes de base
│   ├── portfolio.py     # Gestion des positions
│   ├── trading_system.py # Logique principale
│   │
│   └── strategies/      # Implémentations de stratégies
│       ├── orb_strategy.py
│       └── momentum_strategy.py
│
├── config/          # Fichiers de configuration YAML
│   ├── orb.yaml
│   └── momentum.yaml
│
└── data/           # Données de marché
    └── EURUSD_1H_2009-2025.csv
```

### Format des Données
Les fichiers CSV doivent être placés dans le dossier `data/` avec le format :
- `SYMBOL_TIMEFRAME_DATERANGE.csv` (ex: `EURUSD_1H_2009-2025.csv`)
- Supporte CSV et TSV
- En-têtes auto-détectés

## � Ajout d'une Nouvelle Stratégie

1. Créer une classe de stratégie dans `core/strategies/` :
   ```python
   class MyStrategy(BaseStrategy):
       def generate_signals(self, data: pd.DataFrame) -> dict:
           # Logique de la stratégie
           signals = {}
           signals['long_entries'] = ...
           signals['long_exits'] = ...
           return signals
   ```

2. Ajouter un fichier de configuration dans `config/` :
   ```yaml
   strategy_params:
     param1: value1
     param2: value2
   
   data_requirements:
     symbols: ["EURUSD"]
     timeframe: "1h"
   ```

3. La stratégie sera automatiquement détectée dans `main.py`

## 📈 Métriques de Performance

Chaque stratégie est évaluée selon :
- **Rentabilité** : Total Return, Sharpe Ratio
- **Risque** : Maximum Drawdown, VaR, CVaR
- **Qualité** : Win Rate, Profit Factor
- **Robustesse** : Walk-Forward Efficiency

## 🔧 Configuration Requise

- Python 3.8+
- Dépendances principales :
  ```
  vectorbt
  pandas
  numpy
  pyyaml
  pandas_ta
  ```

## 📋 Style de Code

Le code suit une structure modulaire avec :
- Nommage explicite des variables et fonctions
- Documentation complète (docstrings)
- Gestion des erreurs robuste
- Tests unitaires pour les composants critiques

Pour plus de détails, voir `STYLE_GUIDE.md`.