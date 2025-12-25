#!/usr/bin/env python3
"""
Exemple simple d'utilisation de la fonction add_trade_signals()
pour visualiser les connexions entre les trades sur les plots VectorBT.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au sys.path pour importer vectorflow
sys.path.insert(0, str(Path(__file__).parent.parent))

import vectorbt as vbt
import pandas as pd
import numpy as np
from vectorflow.visualization.indicators import add_trade_signals

print("=" * 60)
print("Exemple d'utilisation de add_trade_signals()")
print("=" * 60)

# Créer des données de test
np.random.seed(42)
close = pd.Series(
    np.random.randn(252).cumsum() + 100,
    index=pd.date_range("2023-01-01", periods=252, freq="D"),
)

# Créer des signaux d'entrée/sortie simples (MA crossover)
fast_ma = close.rolling(10).mean()
slow_ma = close.rolling(50).mean()
entries = fast_ma > slow_ma
exits = fast_ma < slow_ma

# Créer le portfolio
portfolio = vbt.Portfolio.from_signals(
    close=close, entries=entries, exits=exits, init_cash=10000, fees=0.001
)

print(f"\n✅ Portfolio créé avec {len(portfolio.trades.records)} trades")

# Créer le plot de base
fig = portfolio.plot(template="plotly_dark")

# Ajouter les lignes de connexion des trades
print("📊 Ajout des lignes de connexion des trades...")
fig = add_trade_signals(portfolio=portfolio, fig=fig)

fig.update_layout(title="Portfolio avec lignes de connexion des trades").show()
