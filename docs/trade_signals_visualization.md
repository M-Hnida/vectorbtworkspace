# Visualisation des Lignes de Connexion des Trades

## 📊 Vue d'ensemble

La fonction `add_trade_signals()` permet d'ajouter des **lignes connectrices** entre les points d'entrée et de sortie des trades sur les graphiques de portfolio VectorBT.

Cette fonctionnalité est utile pour :
- ✅ Visualiser clairement la durée de chaque position
- ✅ Identifier les patterns d'entrée/sortie
- ✅ Analyser la séquence temporelle des trades
- ✅ Combiner avec d'autres indicateurs techniques

## 🚀 Installation

Cette fonctionnalité fait partie du module `vectorflow.visualization.indicators`.

```python
from vectorflow.visualization.indicators import add_trade_signals
```

## 📖 Utilisation

### Exemple basique

```python
import vectorbt as vbt
from vectorflow.visualization.indicators import add_trade_signals

# Créer votre portfolio
portfolio = vbt.Portfolio.from_signals(
    close=close_prices,
    entries=entry_signals,
    exits=exit_signals,
    init_cash=10000
)

# Plot de base
fig = portfolio.plot()

# Ajouter les lignes de connexion des trades
fig = add_trade_signals(
    portfolio=portfolio,
    fig=fig,
    plot_close=False,
    plot_positions="lines"
)

fig.show()
```

### Avec slicing temporel

```python
# Zoomer sur une période spécifique
start_date = "2023-06-01"
end_date = "2023-12-31"

fig = portfolio[start_date:end_date].plot()
fig = add_trade_signals(
    portfolio=portfolio,
    fig=fig,
    start_date=start_date,
    end_date=end_date,
    plot_positions="lines"
)
fig.show()
```

### Combiné avec des indicateurs

```python
from vectorflow.visualization.indicators import add_indicator, add_trade_signals

# 1. Plot du portfolio
fig = portfolio.plot()

# 2. Ajouter des moyennes mobiles
fig = add_indicator(fig, sma_50, name="SMA 50")
fig = add_indicator(fig, sma_200, name="SMA 200")

# 3. Ajouter les lignes de trades
fig = add_trade_signals(portfolio, fig, plot_positions="lines")

fig.show()
```

## 🎨 Paramètres

### `add_trade_signals()`

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `portfolio` | `vbt.Portfolio` | **requis** | L'objet portfolio VectorBT |
| `fig` | `go.Figure` | **requis** | Figure Plotly existante |
| `plot_close` | `bool` | `False` | Afficher le prix de clôture |
| `plot_positions` | `str` | `"lines"` | Type d'affichage (`"lines"`, `"markers"`, etc.) |
| `start_date` | `str/pd.Timestamp` | `None` | Date de début pour le slicing |
| `end_date` | `str/pd.Timestamp` | `None` | Date de fin pour le slicing |
| `**kwargs` | `dict` | - | Kwargs Plotly additionnels |

### Options de `plot_positions`

- **`"lines"`** : Lignes connectrices entre entrée/sortie (recommandé)
- **`"markers"`** : Marqueurs aux points d'entrée/sortie
- **`"both"`** : Lignes + marqueurs
- **`False`** : Ne pas afficher les positions

## 💡 Exemples avancés

### 1. Multi-symbole

```python
# Pour chaque symbole
for symbol in ["AAPL", "GOOGL", "MSFT"]:
    pf = portfolio_dict[symbol]
    fig = pf.plot()
    fig = add_trade_signals(pf, fig, plot_positions="lines")
    fig.update_layout(title=f"Trades - {symbol}")
    fig.show()
```

### 2. Période spécifique avec indicateurs

```python
# Analyse d'une période critique
crisis_start = "2023-03-01"
crisis_end = "2023-03-31"

fig = portfolio[crisis_start:crisis_end].plot()

# Ajouter RSI
fig = add_indicator(fig, rsi, subplot=True, name="RSI")

# Ajouter les trades
fig = add_trade_signals(
    portfolio, 
    fig,
    start_date=crisis_start,
    end_date=crisis_end,
    plot_positions="lines"
)

fig.update_layout(title="Analyse de crise - Mars 2023")
fig.show()
```

### 3. Comparaison avant/après optimisation

```python
# Portfolio par défaut
fig_default = default_portfolio.plot()
fig_default = add_trade_signals(default_portfolio, fig_default)
fig_default.update_layout(title="Avant optimisation")

# Portfolio optimisé
fig_optimized = optimized_portfolio.plot()
fig_optimized = add_trade_signals(optimized_portfolio, fig_optimized)
fig_optimized.update_layout(title="Après optimisation")

# Afficher côte à côte
fig_default.show()
fig_optimized.show()
```

## 🔧 Intégration avec VectorFlow

Cette fonction s'intègre naturellement avec le workflow VectorFlow :

```python
from vectorflow.core import create_portfolio
from vectorflow.visualization.indicators import add_trade_signals

# 1. Créer le portfolio via VectorFlow
portfolio = create_portfolio(
    strategy_name="ma_crossover",
    symbols=["AAPL"],
    timeframes=["1h"]
)

# 2. Visualiser avec les lignes de trades
fig = portfolio.plot()
fig = add_trade_signals(portfolio, fig, plot_positions="lines")
fig.show()
```

## ⚙️ Détails techniques

### Fonctionnement interne

La fonction `add_trade_signals()` :
1. Slice le portfolio si des dates sont fournies
2. Appelle `portfolio.plot_trade_signals()` de VectorBT
3. Passe les kwargs Plotly directement à la méthode VectorBT
4. Retourne la figure modifiée

### Compatibilité

- ✅ VectorBT >= 0.24.0
- ✅ Plotly >= 5.0.0
- ✅ Compatible avec tous les types de portfolios VectorBT

### Performance

- Les lignes de trades sont ajoutées comme traces Plotly supplémentaires
- Pour un grand nombre de trades (>1000), la visualisation peut être lente
- Recommandation : Utiliser le slicing temporel pour les analyses détaillées

## 🐛 Troubleshooting

### Problème : Les lignes ne s'affichent pas

```python
# Vérifier que le portfolio a des trades
print(f"Nombre de trades: {len(portfolio.trades.records)}")

# Si 0 trade, vérifier les signaux
print(portfolio.stats())
```

### Problème : Erreur lors du slicing

```python
# S'assurer que les dates sont valides
print(f"Dates du portfolio: {portfolio.wrapper.index[0]} à {portfolio.wrapper.index[-1]}")

# Utiliser le bon format
fig = add_trade_signals(
    portfolio, fig,
    start_date=pd.Timestamp("2023-01-01"),  # ✅ Bon
    # start_date="01/01/2023",  # ❌ Mauvais format
)
```

## 📚 Voir aussi

- [`add_indicator()`](./indicators.md) - Ajouter des indicateurs techniques
- [`remove_date_gaps()`](./indicators.md) - Supprimer les gaps de dates
- [VectorBT Documentation](https://vectorbt.dev/) - Documentation officielle

## 🤝 Contribution

Pour signaler un bug ou suggérer une amélioration :
1. Ouvrir une issue sur GitHub
2. Décrire le cas d'usage
3. Fournir un exemple de code minimal

---

**Note** : Cette fonctionnalité utilise la méthode native `plot_trade_signals()` de VectorBT, qui accepte tous les kwargs Plotly standard. Consultez la [documentation Plotly](https://plotly.com/python/) pour les options d'affichage avancées.
