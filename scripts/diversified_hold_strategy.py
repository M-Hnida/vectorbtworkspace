"""
Diversified Hold Strategy - Portefeuille Décorrélé Multi-Classes d'Actifs

Stratégie de Hold avec allocation égale sur instruments décorrélés:
- Forex (majeurs et croisés)
- Indices mondiaux (US, Europe, Asie)
- Cryptomonnaies
- Métaux précieux
- Commodités énergétiques
- Matières premières agricoles

Objectif: Maximiser la diversification et réduire la corrélation du portfolio.
"""

import yfinance as yf
import vectorbt as vbt
import pandas as pd
import numpy as np


def run_strategy():
    """
    Exécute une stratégie Buy & Hold sur un portefeuille décorrélé.
    
    Sélection basée sur:
    1. Diversification géographique (US, Europe, Asie)
    2. Classes d'actifs différentes
    3. Drivers de marché distincts
    4. Corrélations historiquement faibles
    """
    
    # Portfolio décorrélé: Sélection réduite pour minimiser la corrélation
    instruments = [
        # === FOREX ===
        {
            "Category": "Forex",
            "Name": "EUR/USD",
            "Ticker": "EURUSD=X",
            "Role": "Euro vs Dollar",
        },
        {
            "Category": "Forex",
            "Name": "USD/JPY", 
            "Ticker": "USDJPY=X",
            "Role": "Safe Haven",
        },
        
        # === INDICES ===
        {
            "Category": "Indices",
            "Name": "S&P 500",
            "Ticker": "^GSPC",
            "Role": "US Large Cap",
        },
        
        # === CRYPTO ===
        {
            "Category": "Crypto",
            "Name": "Bitcoin",
            "Ticker": "BTC-USD",
            "Role": "Digital Gold",
        },
        
        # === COMMODITIES ===
        {
            "Category": "Precious Metals",
            "Name": "Gold",
            "Ticker": "GC=F",
            "Role": "Inflation Hedge",
        },
        {
            "Category": "Energy",
            "Name": "Crude Oil",
            "Ticker": "CL=F",
            "Role": "Energy",
        },
        {
            "Category": "Agricultural",
            "Name": "Coffee",
            "Ticker": "KC=F",
            "Role": "Softs",
        },
    ]
    
    # Extract tickers
    tickers = [inst["Ticker"] for inst in instruments]
    
    # Download data
    try:
        data = yf.download(tickers, period="2y", interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return None
    
    if data.empty:
        return None
    
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        close_price = data["Close"]
    else:
        if "Close" in data:
            close_price = data[["Close"]]
            close_price.columns = [tickers[0]]
        else:
            return None
    
    # Handle missing data
    close_price = close_price.ffill().dropna(how='all').ffill().bfill()
    
    # Generate entry signals: Buy once at first valid price
    entries = pd.DataFrame(False, index=close_price.index, columns=close_price.columns)
    for col in close_price.columns:
        first_valid = close_price[col].first_valid_index()
        if first_valid is not None:
            entries.loc[first_valid, col] = True
    
    # No exits (hold forever)
    exits = pd.DataFrame(False, index=close_price.index, columns=close_price.columns)
    
    weights = np.ones(len(tickers)) / len(tickers)  # allocation égale

    portfolio = vbt.Portfolio.from_signals(
        close=close_price,
        entries=entries,
        exits=exits,
        init_cash=100_000,
        size=weights,
        size_type='targetpercent',   # ← clé de voûte
        cash_sharing=True,
        fees=0.001,
        freq='1D',
        group_by=False  # ou True si tu veux seulement les stats globales
    )
    
    # === RESULTS ===
    print("\n" + "="*80)
    print("📊 RÉSULTATS DU PORTEFEUILLE (AGRÉGÉ)")
    print("="*80)
    
    # Print only the aggregated stats as requested
    stats_agg = portfolio.stats(group_by=True)
    print(stats_agg)
    
    # === VISUALIZATION ===
    try:
        # Plot aggregated portfolio value
        fig = portfolio.value().vbt.plot(title="Valeur Totale du Portefeuille")
        fig.show()
    except Exception as e:
        print(f"⚠️  Erreur lors de la génération des graphiques: {e}")
    
    return portfolio


if __name__ == "__main__":
    run_strategy()
