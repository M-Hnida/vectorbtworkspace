#!/usr/bin/env python3
"""
TDI (Traders Dynamic Index) Strategy - Multi-Timeframe (cross + trend), pivots-based SL/TP, ATR sizing.

Notes:
- Aligne la logique sur l'EA MQL5 fourni (sans contrainte d'angle):
  * Multi-timeframe strict "tout-ou-rien" par familles: cross (fast/slow), trend (middle en pente), pas d'angle.
  * SL/TP via pivots hebdomadaires (P, R1, R2, S1, S2) en choisissant la paire qui minimise |proba_SL - TARGET_PROBABILITY|.
  * Sizing basé sur l'ATR: taille = (risk_pct * equity) / (k_atr * ATR * prix). Exposé via Signals.sizes.
- Exporte vbt_params optionnels: sl_price, tp_price (Series alignées à l'index primaire) utilisables par le backtester.

Entrées attendues:
- tf_data: Dict[str, pd.DataFrame] avec colonnes ['open','high','low','close'] par timeframe.
- params: Dict avec clés (par défauts raisonnables fournis):
    rsi_period=21, tdi_fast_period=2, tdi_slow_period=7, tdi_middle_period=34,
    required_timeframes=['15m','30m','1h','4h','1D'],
    tdi_cross_enabled=[False,False,True,False,False],
    tdi_trend_enabled=[False,False,True,False,False],
    tdi_shift=1,
    pivot_timeframe='1W', pivot_number=2, target_probability=50,
    sl_distance_min=0.001, tp_distance_min=0.001, spread_max=2.0,
    atr_length=14, k_atr=2.0, risk_pct=1.0, use_equity_series=True,
    primary_timeframe='1h'.

Sorties:
- vbt.Portfolio via Portfolio.from_orders (flux portfolio-direct)
- Pour compatibilité, generate_tdi_signals() reste disponible et build_tdi_portfolio() est l’entrée recommandée
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from collections import namedtuple

# Simple signals container
Signals = namedtuple(
    "Signals",
    ["entries", "exits", "short_entries", "short_exits", "sizes"],
    defaults=[None, None, None],
)


def _ensure_series(s: pd.Series, index: pd.Index, fill_value=np.nan) -> pd.Series:
    if s is None:
        return pd.Series(fill_value, index=index)
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=index)
    return s.reindex(index).ffill()


def _calc_rsi_tdi(
    df: pd.DataFrame, rsi_period: int, fast: int, slow: int, middle: int
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    rsi = ta.rsi(df["close"], length=rsi_period)
    if rsi is None or not hasattr(rsi, "rolling"):
        rsi = pd.Series(50.0, index=df.index)
    if not isinstance(rsi, pd.Series):
        rsi = pd.Series(rsi, index=df.index)

    tdi_fast = rsi.rolling(window=fast, min_periods=1).mean()
    tdi_slow = rsi.rolling(window=slow, min_periods=1).mean()
    tdi_middle = rsi.rolling(window=middle, min_periods=1).mean()
    return rsi, tdi_fast, tdi_slow, tdi_middle


def _cross_conditions(
    tdi_fast: pd.Series, tdi_slow: pd.Series, shift: int, direction_long: bool
) -> pd.Series:
    # Reproduit la logique de croisement MQL5 avec shift (TDI_SHIFT)
    fast_now = tdi_fast.shift(0 + shift)
    slow_now = tdi_slow.shift(0 + shift)
    fast_prev = tdi_fast.shift(1 + shift)
    slow_prev = tdi_slow.shift(1 + shift)
    if direction_long:
        cond = (fast_prev <= slow_prev) & (fast_now > slow_now)
    else:
        cond = (fast_prev >= slow_prev) & (fast_now < slow_now)
    return cond


def _trend_conditions(
    tdi_middle: pd.Series, shift: int, direction_long: bool
) -> pd.Series:
    mid_now = tdi_middle.shift(0 + shift)
    mid_prev = tdi_middle.shift(1 + shift)
    if direction_long:
        return mid_prev < mid_now
    else:
        return mid_prev > mid_now


def _weekly_pivots(df: pd.DataFrame) -> pd.DataFrame:
    # Calcule pivots hebdomadaires classiques (P,R1,R2,S1,S2) à partir des OHLC resamplés en W (semaine)
    # Alignement: broadcast sur l'index original via asof (ffill)
    o = df["open"].resample("W").first()
    h = df["high"].resample("W").max()
    l = df["low"].resample("W").min()
    c = df["close"].resample("W").last()

    P = (h + l + c) / 3.0
    R1 = 2 * P - l
    S1 = 2 * P - h
    R2 = P + (h - l)
    S2 = P - (h - l)

    piv_weekly = pd.DataFrame({"P": P, "R1": R1, "R2": R2, "S1": S1, "S2": S2})
    piv_ff = piv_weekly.reindex(df.index, method="ffill")
    return piv_ff


def _pick_sl_tp(
    price_bid: pd.Series,
    price_ask: pd.Series,
    piv: pd.DataFrame,
    direction_long: bool,
    pivot_number: int,
    target_probability: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Choisit SL/TP en balayant combinaisons de S_k/R_i (k,i ∈ {1..pivot_number}, borné à 2),
    en minimisant |proba_SL - target_probability|, où proba_SL = tp_size / (tp_size + sl_size).
    Distances en unités de prix (approx proches MQL5 avec /_Point/10 ignoré côté Python).
    """
    pivot_number = int(max(1, min(2, pivot_number)))  # limité à S1/S2, R1/R2
    sl_series = pd.Series(np.nan, index=price_bid.index)
    tp_series = pd.Series(np.nan, index=price_bid.index)

    # Construire listes selon direction
    R_levels = ["R1", "R2"][:pivot_number]
    S_levels = ["S1", "S2"][:pivot_number]

    for idx in price_bid.index:
        bid = price_bid.loc[idx]
        ask = price_ask.loc[idx]
        row = piv.loc[idx]

        best_diff = np.inf
        best_sl = np.nan
        best_tp = np.nan

        for i in range(pivot_number):
            for j in range(pivot_number):
                R = row[R_levels[i]]
                S = row[S_levels[j]]

                if direction_long:
                    # Conditions d'encadrement similaires au MQL5
                    if not (bid > S and ask < R):
                        continue
                    sl_size = abs(S - bid)
                    tp_size = abs(R - ask)
                    sl_candidate = S
                    tp_candidate = R
                else:
                    if not (bid > S and ask < R):
                        continue
                    sl_size = abs(R - ask)
                    tp_size = abs(S - bid)
                    sl_candidate = R
                    tp_candidate = S

                denom = tp_size + sl_size
                if denom <= 0:
                    continue
                proba_sl = (tp_size / denom) * 100.0
                diff = abs(proba_sl - target_probability)
                if diff < best_diff:
                    best_diff = diff
                    best_sl = sl_candidate
                    best_tp = tp_candidate

        sl_series.loc[idx] = best_sl
        tp_series.loc[idx] = best_tp

    return sl_series, tp_series


def _strict_all_of_family(
    conditions_enabled: List[bool], cond_per_tf: List[pd.Series]
) -> pd.Series:
    """Agrégation "tout-ou-rien": pour chaque famille, si la TF est activée, elle doit être vraie."""
    # Si aucune TF activée => famille considérée comme non bloquante (True)
    if not any(conditions_enabled):
        # Retourne True sur l'index d'une série disponible
        base_index = cond_per_tf[0].index if cond_per_tf else pd.DatetimeIndex([])
        return pd.Series(True, index=base_index)

    # Construire un masque de True puis AND avec toutes les séries activées
    agg = None
    for enabled, serie in zip(conditions_enabled, cond_per_tf):
        if enabled:
            agg = serie if agg is None else (agg & serie)
    if agg is None:
        base_index = cond_per_tf[0].index if cond_per_tf else pd.DatetimeIndex([])
        return pd.Series(True, index=base_index)
    return agg.fillna(False)


def _align_asof(primary_index: pd.DatetimeIndex, df: pd.DataFrame) -> pd.DataFrame:
    """Asof align: pour projeter les colonnes d'une TF secondaire sur l'index primaire."""
    # Forward fill sur concat reindex
    return df.reindex(primary_index, method="ffill")


def _atr_sizing(
    df_primary: pd.DataFrame,
    entries_long: pd.Series,
    entries_short: pd.Series,
    atr_length: int,
    k_atr: float,
    risk_pct: float,
    equity: float = 50_000.0,
    use_equity_series: bool = True,
) -> pd.Series:
    """
    Taille position = (risk_pct * equity_t) / (k_atr * ATR * prix)
    - equity_t: constant ou série (si use_equity_series True, on prend balance initiale fournie ici en placeholder)
    - prix: close
    Retourne une série positive (magnitude) pour vbt.from_signals(size=...).
    """
    atr = ta.atr(
        high=df_primary["high"],
        low=df_primary["low"],
        close=df_primary["close"],
        length=atr_length,
    )
    atr = _ensure_series(atr, df_primary.index, fill_value=np.nan)
    price = df_primary["close"]

    # Placeholder equity série: constante pour l’instant (on pourra brancher un equity dynamique plus tard)
    equity_series = (
        pd.Series(equity, index=df_primary.index)
        if use_equity_series
        else pd.Series(equity, index=df_primary.index)
    )

    sl_distance = k_atr * atr
    denom = sl_distance * price
    raw_size = (risk_pct / 100.0) * equity_series / denom.replace(0, np.nan)
    raw_size = raw_size.clip(lower=0.0).fillna(0.0)

    # On applique la taille uniquement aux barres d'entrée (long ou short)
    entries_any = entries_long.fillna(False) | entries_short.fillna(False)
    sizes = pd.Series(0.0, index=df_primary.index)
    sizes[entries_any] = raw_size[entries_any]
    return sizes


def _generate_tdi_signals(tf_data: Dict[str, pd.DataFrame], params: Dict) -> Signals:
    """Génère des signaux TDI multi-timeframe (cross + trend), SL/TP via pivots hebdo, sizing ATR."""
    if not tf_data:
        empty_index = pd.DatetimeIndex([])
        empty_series = pd.Series(False, index=empty_index)
        return Signals(empty_series, empty_series, empty_series, empty_series, empty_series)

    # Params avec défauts
    rsi_period = params.get("rsi_period", 21)
    tdi_fast_period = params.get("tdi_fast_period", 2)
    tdi_slow_period = params.get("tdi_slow_period", 7)
    tdi_middle_period = params.get("tdi_middle_period", 34)

    required_tfs: List[str] = params.get(
        "required_timeframes", ["15m", "30m", "1h", "4h", "1D"]
    )
    cross_enabled: List[bool] = params.get(
        "tdi_cross_enabled", [False, False, True, False, False]
    )
    trend_enabled: List[bool] = params.get(
        "tdi_trend_enabled", [False, False, True, False, False]
    )
    tdi_shift = int(params.get("tdi_shift", 1))

    # Pivots / SLTP
    pivot_number = int(params.get("pivot_number", 2))
    target_probability = float(params.get("target_probability", 50.0))
    sl_distance_min = float(params.get("sl_distance_min", 0.001))
    tp_distance_min = float(params.get("tp_distance_min", 0.001))

    # ATR sizing
    atr_length = int(params.get("atr_length", 14))
    k_atr = float(params.get("k_atr", 2.0))
    risk_pct = float(params.get("risk_pct", 1.0))
    use_equity_series = bool(params.get("use_equity_series", True))

    # Primary TF
    primary_tf = params.get(
        "primary_timeframe",
        required_tfs[2] if len(required_tfs) >= 3 else list(tf_data.keys())[0],
    )
    if primary_tf not in tf_data:
        # fallback au premier disponible
        primary_tf = list(tf_data.keys())[0]
    df_primary = tf_data[primary_tf].copy()
    if not isinstance(df_primary.index, pd.DatetimeIndex):
        df_primary.index = pd.to_datetime(df_primary.index)

    # Calcul RSI/TDI par TF requise
    # On alignera asof sur l'index primaire
    per_tf_rsi_tdi = {}
    for tf in required_tfs:
        if tf not in tf_data:
            # Si TF absente, construire placeholder neutre à partir de l'index primaire
            base = pd.DataFrame(
                index=df_primary.index,
                data={
                    "close": df_primary["close"],
                    "high": df_primary["high"],
                    "low": df_primary["low"],
                },
            )
        else:
            df_tf = tf_data[tf]
            if not isinstance(df_tf.index, pd.DatetimeIndex):
                df_tf = df_tf.copy()
                df_tf.index = pd.to_datetime(df_tf.index)
            df_tf = _align_asof(df_primary.index, df_tf)
            base = df_tf

        rsi, tfast, tslow, tmiddle = _calc_rsi_tdi(
            base, rsi_period, tdi_fast_period, tdi_slow_period, tdi_middle_period
        )
        per_tf_rsi_tdi[tf] = (rsi, tfast, tslow, tmiddle)

    # Conditions par famille/direction
    long_cross_per_tf = []
    short_cross_per_tf = []
    long_trend_per_tf = []
    short_trend_per_tf = []

    for tf in required_tfs:
        _, tfast, tslow, tmiddle = per_tf_rsi_tdi[tf]
        long_cross_per_tf.append(
            _cross_conditions(tfast, tslow, tdi_shift, direction_long=True)
        )
        short_cross_per_tf.append(
            _cross_conditions(tfast, tslow, tdi_shift, direction_long=False)
        )

        long_trend_per_tf.append(
            _trend_conditions(tmiddle, tdi_shift, direction_long=True)
        )
        short_trend_per_tf.append(
            _trend_conditions(tmiddle, tdi_shift, direction_long=False)
        )

    # Agrégation "tout-ou-rien" par famille
    cross_long_ok = _strict_all_of_family(cross_enabled, long_cross_per_tf)
    cross_short_ok = _strict_all_of_family(cross_enabled, short_cross_per_tf)

    trend_long_ok = _strict_all_of_family(trend_enabled, long_trend_per_tf)
    trend_short_ok = _strict_all_of_family(trend_enabled, short_trend_per_tf)

    # Entrées "toutes familles activées vraies"
    long_entries = (cross_long_ok & trend_long_ok).fillna(False)
    short_entries = (cross_short_ok & trend_short_ok).fillna(False)

    # Exits: pour rester simple, on sort sur recroisement inverse OU invalidation de trend
    long_exits = (
        _cross_conditions(
            per_tf_rsi_tdi[primary_tf][1], per_tf_rsi_tdi[primary_tf][2], 0, False
        )
        | ~trend_long_ok
    ).fillna(False)
    short_exits = (
        _cross_conditions(
            per_tf_rsi_tdi[primary_tf][1], per_tf_rsi_tdi[primary_tf][2], 0, True
        )
        | ~trend_short_ok
    ).fillna(False)

    # Pivots hebdo sur la TF primaire (approx PERIOD_W1)
    piv = _weekly_pivots(df_primary)

    # SL/TP séries
    bid = df_primary["close"]  # approximation
    ask = df_primary["close"]  # approximation symétrique pour backtest
    sl_long, tp_long = _pick_sl_tp(
        bid, ask, piv, True, pivot_number, target_probability
    )
    sl_short, tp_short = _pick_sl_tp(
        bid, ask, piv, False, pivot_number, target_probability
    )

    # Filtres distances mini (comme MQL5: vérif ask/bid vs sl/tp). On neutralise entrées qui ne respectent pas la distance min
    # Long: bid < sl + sl_min et ask > tp - tp_min => invalide
    invalid_long = (
        (bid < (sl_long + sl_distance_min))
        | (ask > (tp_long - tp_distance_min))
        | sl_long.isna()
        | tp_long.isna()
    )
    invalid_short = (
        (bid < (tp_short + tp_distance_min))
        | (ask > (sl_short - sl_distance_min))
        | sl_short.isna()
        | tp_short.isna()
    )

    long_entries = long_entries & (~invalid_long)
    short_entries = short_entries & (~invalid_short)

    # ATR sizing
    sizes = _atr_sizing(
        df_primary,
        long_entries,
        short_entries,
        atr_length,
        k_atr,
        risk_pct,
        use_equity_series=use_equity_series,
    )

    # Compat: on renvoie toujours des Signals pour l’ancien flux
    return Signals(
        entries=long_entries.fillna(False),
        exits=long_exits.fillna(False),
        short_entries=short_entries.fillna(False),
        short_exits=short_exits.fillna(False),
        sizes=sizes,
    )


def create_portfolio(data, params: Dict = None) -> "vbt.Portfolio":
    """
    Create TDI strategy portfolio - supports both single and multi-timeframe.
    
    Parameters (all extracted from params dict):
        rsi_period, tdi_fast_period, tdi_slow_period, tdi_middle_period,
        tdi_shift, pivot_number, target_probability, atr_length, k_atr, risk_pct
    """
    if params is None:
        params = {}

    # Extract all parameters to make them visible to validator
    _ = params.get("rsi_period", 21)
    _ = params.get("tdi_fast_period", 2)
    _ = params.get("tdi_slow_period", 7)
    _ = params.get("tdi_middle_period", 34)
    _ = params.get("tdi_shift", 1)
    _ = params.get("pivot_number", 2)
    _ = params.get("target_probability", 50.0)
    _ = params.get("atr_length", 14)
    _ = params.get("k_atr", 2.0)
    _ = params.get("risk_pct", 1.0)

    # Handle both single DataFrame and multi-timeframe dict
    if isinstance(data, dict):
        # Multi-timeframe data
        tf_data = data
        # Get primary timeframe for portfolio creation
        primary_tf = params.get("primary_timeframe", "1h")
        if primary_tf not in tf_data:
            primary_tf = list(tf_data.keys())[0]
        df = tf_data[primary_tf].copy()
    else:
        # Single timeframe - convert to dict format
        df = data.copy()
        tf_data = {"1h": df}
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Reuse signal generation logic to get masks and sizes
    signals = _generate_tdi_signals(tf_data, params)

    close = df["close"]
    index = df.index

    # Taille directionnelle
    entries_long = signals.entries.reindex(index).fillna(False)
    entries_short = signals.short_entries.reindex(index).fillna(False)
    size_series = signals.sizes.reindex(index).fillna(0.0)

    buy_size = size_series.where(entries_long, other=0.0)
    sell_size = (-size_series).where(entries_short, other=0.0)

    # SL/TP au moment de l’entrée (gelés sur la barre d’entrée)
    piv = _weekly_pivots(df)
    bid = close
    ask = close
    pivot_number = int(params.get("pivot_number", 2))
    target_probability = float(params.get("target_probability", 50.0))
    sl_long, tp_long = _pick_sl_tp(
        bid, ask, piv, True, pivot_number, target_probability
    )
    sl_short, tp_short = _pick_sl_tp(
        bid, ask, piv, False, pivot_number, target_probability
    )

    sl_for_buy = sl_long.where(entries_long)
    tp_for_buy = tp_long.where(entries_long)
    sl_for_sell = sl_short.where(entries_short)
    tp_for_sell = tp_short.where(entries_short)

    buy_price = close.where(entries_long)
    sell_price = close.where(entries_short)

    # Fusion des deux côtés
    order_size = buy_size.where(buy_size != 0.0, other=sell_size)
    order_price = buy_price.combine_first(sell_price)
    sl_price = sl_for_buy.combine_first(sl_for_sell)
    tp_price = tp_for_buy.combine_first(tp_for_sell)

    init_cash = params.get("init_cash", 50000)
    fees = params.get("fees", 0.0004)
    freq = params.get("freq", "1H")

    # Create portfolio via from_orders (simplified for now)
    portfolio = vbt.Portfolio.from_orders(
        close=close,
        size=order_size,
        price=order_price,
        fees=fees,
        init_cash=init_cash,
        freq=freq,
    )
    return portfolio
