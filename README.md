# VectorFlow

Workspace structuré pour backtesting de stratégies de trading sur VectorBT.

## Architecture

```
vectorflow/
├── core/                    
│   ├── config_manager.py         
│   ├── data_loader.py            
│   └── portfolio_builder.py      
├── strategies/              
│   ├── vol_breakout.py           
│   ├── ema_rsi_trend.py          
│   └── strategy_template.py      
├── optimization/            
│   └── grid_search.py            
├── validation/             
│   ├── walk_forward.py           
│   └── path_randomization.py     
└── visualization/          
    └── indicators.py             
```

## Acquisition de Données

### 1. Script de téléchargement intégré

```bash
# Crypto via CCXT (Binance)
python scripts/download_data.py crypto --symbol BTC/USDT --timeframe 1h --days 30

# Forex via Dukascopy
python scripts/download_data.py forex --symbol EURUSD --timeframe h1 --days 365

# Multiple paires
python scripts/download_data.py crypto --symbol BTC/USDT,ETH/USDT --timeframe 5m --days 7
```

### 2. yfinance (stocks/ETF)

```python
import yfinance as yf
import vectorbt as vbt

# Télécharger données
data = yf.download("SPY", start="2020-01-01", end="2024-01-01")

# Sauvegarder pour utilisation avec VectorFlow
data.to_csv("data/SPY.csv")

# Ou utiliser directement avec VectorBT
portfolio = vbt.Portfolio.from_signals(
    close=data["Close"],
    entries=entries,
    exits=exits
)
```

### 3. VectorBT Remote

```python
import vectorbt as vbt

# Télécharger via l'API intégrée de VectorBT
data = vbt.YFData.download(
    "BTC-USD",
    start="2020-01-01",
    end="2024-01-01",
    interval="1h"
)

close = data.get("Close")
```

### 4. Chargement CSV

```python
from vectorflow import load_ohlc_csv

# Charge n'importe quel format CSV
df = load_ohlc_csv("data/BTCUSD.csv")

# Avec filtres temporels
df = load_ohlc_csv(
    "data/BTCUSD.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

## Installation

```bash
pip install -e .
```

## CLI

```bash
python -m vectorflow.cli                    # Mode interactif
python -m vectorflow.cli vol_breakout       # Exécution directe
python -m vectorflow.cli vol_breakout --full # Analyse complète
```

## API

```python
from vectorflow import create_portfolio, load_ohlc_csv
from vectorflow.visualization.indicators import add_trade_signals

data = load_ohlc_csv("data/ohlc.csv")
portfolio = create_portfolio("vol_breakout", data, params)

fig = portfolio.plot()
fig = add_trade_signals(portfolio, fig)
fig.show()
```

## Configuration

Fichier YAML dans `config/`:

```yaml
parameters:
  entry_threshold: 0.5
  stop_atr_mult: 2.0
  initial_cash: 10000
  fee: 0.0004

csv_path: "data/ohlc.csv"

optimization_grid:
  entry_threshold: [0.3, 0.7, 0.1]
  stop_atr_mult: [1.5, 3.0, 0.5]
```

## Dépendances

- Python 3.8+
- vectorbt
- numpy, pandas
- plotly
- pyyaml
