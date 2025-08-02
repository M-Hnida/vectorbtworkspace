"""
Volatility-Adaptive Dual-Trend CMO Strategy (VectorBT wrapper)
- Volatility windows adapt using ATR% vs its rolling mean with clipping
- Dual-trend gating using TEMA fast/slow (short trend) and long_fast/long_slow (long trend)
- CMO z-score trigger with configurable threshold
- Supports long and short entries
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import pandas_ta as ta
from base import Signals

# --- Helpers -------------------------------------------------------------------

def _tema(series: pd.Series, length: int) -> pd.Series:
    return ta.tema(series, length=length)

def _cmo(series: pd.Series, length: int) -> pd.Series:
    # pandas_ta cmo returns Column named 'CMO_{length}'
    return ta.cmo(series, length=length)

def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    z = (series - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)

def _safe_clip(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return series.clip(lower=lo, upper=hi)

def _assert_synced_index(df: pd.DataFrame, cols: List[str]) -> None:
    """Ensure required columns exist, share identical index, and no duplicates."""
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if df.index.has_duplicates:
        raise ValueError("Input dataframe index has duplicates; please deduplicate before running strategy.")
    # Additionally ensure no NaNs are introduced inconsistently
    base_na = df[cols[0]].isna()
    for c in cols[1:]:
        if not df[c].isna().equals(base_na) and (df[c].isna().sum() != df[cols[0]].isna().sum()):
            # Not strictly necessary to be identical, but warn via exception for strict alignment
            raise ValueError(f"Column {c} NaN pattern differs from {cols[0]}; please align/ffill before running.")

# --- Core Logic ----------------------------------------------------------------

def _compute_adaptive_windows(close: pd.Series,
                              high: pd.Series,
                              low: pd.Series,
                              base_window: int,
                              atr_len: int,
                              atr_smooth: int,
                              clip_low: float,
                              clip_high: float) -> pd.Series:
    """
    Compute an integer window multiplier based on ATR% relative to its rolling mean.
    Returns an integer window series used to adapt indicator lengths.
    """
    tr = ta.true_range(high=high, low=low, close=close)
    atr = tr.rolling(atr_len).mean()
    atr_pct = (atr / close).replace(0, np.nan) * 100.0

    atr_mean = atr_pct.rolling(atr_smooth).mean()
    ratio = (atr_pct / atr_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Clip to avoid extreme adaptation
    ratio_clipped = _safe_clip(ratio, clip_low, clip_high)

    # Convert to integer window scale around base_window
    adaptive_window = (base_window * ratio_clipped).round().astype(int)
    adaptive_window = adaptive_window.clip(lower=max(3, int(base_window * clip_low)),
                                           upper=int(base_window * clip_high))
    return adaptive_window

def _scale_tema_lengths(fast:int, slow:int, long_fast:int, long_slow:int, scale: float) -> Dict[str, int]:
    """Proportionally scale TEMA lengths preserving ratios; clamp to >=3 ints."""
    def clamp(x: float) -> int:
        return max(3, int(round(x)))
    return {
        'fast': clamp(fast * scale),
        'slow': clamp(slow * scale),
        'long_fast': clamp(long_fast * scale),
        'long_slow': clamp(long_slow * scale),
    }

def _dual_trend_gates(close: pd.Series,
                      fast: int, slow: int,
                      long_fast: int, long_slow: int) -> Dict[str, pd.Series]:
    tema_fast = _tema(close, fast)
    tema_slow = _tema(close, slow)
    tema_long_fast = _tema(close, long_fast)
    tema_long_slow = _tema(close, long_slow)

    short_trend_up = tema_fast > tema_slow
    short_trend_down = tema_fast < tema_slow
    long_trend_up = tema_long_fast > tema_long_slow
    long_trend_down = tema_long_fast < tema_long_slow

    return {
        'short_trend_up': short_trend_up.fillna(False),
        'short_trend_down': short_trend_down.fillna(False),
        'long_trend_up': long_trend_up.fillna(False),
        'long_trend_down': long_trend_down.fillna(False),
        'tema_fast': tema_fast,
        'tema_slow': tema_slow,
        'tema_long_fast': tema_long_fast,
        'tema_long_slow': tema_long_slow
    }

def _cmo_z_trigger(close: pd.Series,
                   cmo_len: int,
                   z_win: int,
                   z_thresh: float) -> Dict[str, pd.Series]:
    cmo = _cmo(close, cmo_len)
    # pandas_ta returns a Series named like 'CMO_...'; ensure it's Series
    if isinstance(cmo, pd.DataFrame):
        cmo = cmo.iloc[:, 0]
    cmo = cmo.astype(float).fillna(0.0)
    cmo_z = _zscore(cmo, z_win)

    long_trigger = cmo_z > z_thresh
    short_trigger = cmo_z < -z_thresh

    return {
        'cmo': cmo,
        'cmo_z': cmo_z,
        'long_trigger': long_trigger.fillna(False),
        'short_trigger': short_trigger.fillna(False)
    }

def _generate_signals_df(df: pd.DataFrame, params: Dict) -> Signals:
    # Defaults
    base_window = int(params.get('base_window', 50))
    atr_len = int(params.get('atr_len', 14))
    atr_smooth = int(params.get('atr_smooth', 50))
    atr_clip_low = float(params.get('atr_clip_low', 0.6))
    atr_clip_high = float(params.get('atr_clip_high', 1.5))

    fast = int(params.get('tema_fast', 10))
    slow = int(params.get('tema_slow', 30))
    long_fast = int(params.get('tema_long_fast', 50))
    long_slow = int(params.get('tema_long_slow', 150))

    cmo_len = int(params.get('cmo_len', 20))
    cmo_z_win = int(params.get('cmo_z_window', 50))
    cmo_z_thresh = float(params.get('cmo_z_threshold', 0.5))

    require_both_trends = bool(params.get('require_both_trends', True))
    allow_shorts = bool(params.get('allow_shorts', True))

    # New behavior toggles
    use_adaptive_tema = bool(params.get('use_adaptive_tema', False))
    confirm_on_close = bool(params.get('confirm_on_close', True))
    discrete_exits = bool(params.get('discrete_exits', False))

    data = df.copy()
    data.columns = [c.lower() for c in data.columns]
    _assert_synced_index(data, ['open', 'high', 'low', 'close'])

    data = data.dropna().copy()

    # Volatility-adaptive window series
    aw = _compute_adaptive_windows(
        data['close'], data['high'], data['low'],
        base_window, atr_len, atr_smooth, atr_clip_low, atr_clip_high
    )

    # Proportional adaptive TEMA: derive a single effective set of lengths using prior-bar window
    if use_adaptive_tema:
        # Scale relative to base_window using previous bar to avoid lookahead
        eff_window = aw.shift(1).fillna(method='bfill')
        # Compute median scale over history for a stable run-level scale
        # scale = median(eff_window/base_window)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_series = (eff_window / float(base_window)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        scale = float(np.nanmedian(scale_series.values)) if len(scale_series) > 0 else 1.0
        scaled = _scale_tema_lengths(fast, slow, long_fast, long_slow, scale)
        fast_eff, slow_eff = scaled['fast'], scaled['slow']
        long_fast_eff, long_slow_eff = scaled['long_fast'], scaled['long_slow']
    else:
        fast_eff, slow_eff = fast, slow
        long_fast_eff, long_slow_eff = long_fast, long_slow

    # Compute trend gates with effective lengths
    gates = _dual_trend_gates(
        data['close'], fast_eff, slow_eff, long_fast_eff, long_slow_eff
    )

    # Compute CMO z triggers
    cmo_blk = _cmo_z_trigger(
        data['close'], cmo_len, cmo_z_win, cmo_z_thresh
    )

    # Regime filter
    if require_both_trends:
        long_regime = gates['short_trend_up'] & gates['long_trend_up']
        short_regime = gates['short_trend_down'] & gates['long_trend_down']
    else:
        long_regime = gates['long_trend_up']
        short_regime = gates['long_trend_down']

    # Base conditions
    long_entry_cond = (long_regime & cmo_blk['long_trigger']).fillna(False)
    long_exit_cond = ((~long_regime) | (cmo_blk['cmo_z'] < 0)).fillna(False)

    if allow_shorts:
        short_entry_cond = (short_regime & cmo_blk['short_trigger']).fillna(False)
        short_exit_cond = ((~short_regime) | (cmo_blk['cmo_z'] > 0)).fillna(False)
    else:
        short_entry_cond = pd.Series(False, index=data.index)
        short_exit_cond = pd.Series(False, index=data.index)

    # Closed-bar confirmation with edge detection
    if confirm_on_close:
        # entries as rising edge
        long_entries = (long_entry_cond & ~long_entry_cond.shift(1).fillna(False))
        short_entries = (short_entry_cond & ~short_entry_cond.shift(1).fillna(False))
        if discrete_exits:
            # exits as rising edge of exit condition
            long_exits = (long_exit_cond & ~long_exit_cond.shift(1).fillna(False))
            short_exits = (short_exit_cond & ~short_exit_cond.shift(1).fillna(False))
        else:
            long_exits = long_exit_cond
            short_exits = short_exit_cond
    else:
        # original continuous behavior
        long_entries = long_entry_cond
        short_entries = short_entry_cond
        long_exits = long_exit_cond
        short_exits = short_exit_cond

    # Ensure boolean dtype
    long_entries = long_entries.fillna(False).astype(bool)
    long_exits = long_exits.fillna(False).astype(bool)
    short_entries = short_entries.fillna(False).astype(bool)
    short_exits = short_exits.fillna(False).astype(bool)

    return Signals(
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits
    )

# --- Public API kept stable ----------------------------------------------------

def generate_vectorbt_signals(tf_data: Dict[str, pd.DataFrame], params: Dict) -> Signals:
    """
    Generate signals using the volatility-adaptive dual-trend CMO strategy.
    Signature preserved for registry compatibility.
    """
    if not tf_data:
        empty_index = pd.DatetimeIndex([])
        empty_series = pd.Series(False, index=empty_index)
        return Signals(empty_series, empty_series, empty_series, empty_series)

    primary_tf = params.get('primary_timeframe', list(tf_data.keys())[0])
    if primary_tf not in tf_data:
        primary_tf = list(tf_data.keys())[0]

    primary_df = tf_data[primary_tf]
    if not isinstance(primary_df.index, pd.DatetimeIndex):
        primary_df.index = pd.to_datetime(primary_df.index)

    return _generate_signals_df(primary_df, params)

def get_vectorbt_required_timeframes(params: Dict) -> List[str]:
    """
    Return required timeframes; keep default to 1h unless configured.
    """
    return params.get('required_timeframes', ['1h'])

# Optional hook for per-strategy Portfolio params (e.g., custom stop sizing)
def get_vbt_params(primary_data: pd.DataFrame, params: Dict) -> Dict:
    """
    Provide optional per-bar parameters to vbt.Portfolio.from_signals if needed.
    Returns {} unless sizing is explicitly requested.
    Supported params:
      - size_mode: 'none'|'fixed_cash'|'risk_atr'|'units' (default 'none')
      - fixed_cash: float (cash per trade if size_mode='fixed_cash')
      - risk_per_trade: float (e.g., 0.01)
      - atr_len_size: int (default 14)
      - atr_mult_stop: float (default 2.0) -> SL distance in ATR multiples
      - stop_mode: 'atr'|'percent' (default 'atr')
      - stop_percent: float (e.g., 0.02 for 2%)
      - take_profit_mult: float or None (optional TP as ATR multiples if stop_mode='atr')
    """
    size_mode = params.get('size_mode', 'none')
    if size_mode in (None, 'none'):
        return {}

    close = primary_data['close'].astype(float)
    high = primary_data['high'].astype(float)
    low = primary_data['low'].astype(float)

    # Compute ATR for sizing if needed
    atr_len_size = int(params.get('atr_len_size', 14))
    tr = ta.true_range(high=high, low=low, close=close)
    atr = tr.rolling(atr_len_size).mean()

    out: Dict[str, object] = {}

    if size_mode == 'fixed_cash':
        # VectorBT expects size_type='cash' with size as cash amount per entry
        fixed_cash = float(params.get('fixed_cash', 0.0))
        if fixed_cash <= 0:
            return {}
        out['size'] = float(fixed_cash)
        out['size_type'] = 'cash'

    elif size_mode == 'units':
        units = float(params.get('units', 0.0))
        if units <= 0:
            return {}
        out['size'] = float(units)
        out['size_type'] = 'amount'  # number of units/shares

    elif size_mode == 'risk_atr':
        # Risk-based units per entry: units = (risk_cash) / (stop_distance)
        # stop_distance from ATR multiples or percent of price
        risk_per_trade = float(params.get('risk_per_trade', 0.01))
        equity = float(params.get('equity', params.get('initial_cash', 10000.0)))
        risk_cash = max(0.0, risk_per_trade) * equity
        stop_mode = params.get('stop_mode', 'atr')
        atr_mult_stop = float(params.get('atr_mult_stop', 2.0))
        stop_percent = float(params.get('stop_percent', 0.02))

        if stop_mode == 'atr':
            stop_dist = (atr * atr_mult_stop).replace(0, np.nan)
        else:
            stop_dist = (close * stop_percent).replace(0, np.nan)

        # Units computed per bar; clip extremely large
        units = (risk_cash / stop_dist.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        units = units.clip(lower=0.0, upper=1e9)

        out['size'] = units
        out['size_type'] = 'amount'

        # Provide stop levels if desired (SL only here)
        if stop_mode == 'atr':
            sl_distance = (atr * atr_mult_stop).fillna(method='bfill')
        else:
            sl_distance = (close * stop_percent).fillna(method='bfill')

        # VectorBT from_signals supports sl_stop/tp_take as absolute price levels
        out['sl_stop'] = (close - sl_distance).clip(lower=0.0)
        tp_mult = params.get('take_profit_mult', None)
        if tp_mult is not None:
            try:
                tp_mult_f = float(tp_mult)
                out['tp_take'] = (close + atr * float(tp_mult_f)).clip(lower=0.0)
            except Exception:
                pass

    # Return assembled kwargs for vbt.Portfolio.from_signals
    return out

