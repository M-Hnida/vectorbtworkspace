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
