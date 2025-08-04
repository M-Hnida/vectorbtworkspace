"""
Volatility-Adaptive Dual-Trend CMO Strategy (VectorBT wrapper)
- Volatility windows adapt using ATR% vs its rolling mean with clipping
- Dual-trend gating using TEMA fast/slow (short trend) and long_fast/long_slow (long trend)
- CMO z-score trigger with configurable threshold
- Supports long and short entries
- Optional bar-by-bar adaptive TEMA lengths (no lookahead)
- Optional regime filter using slope(TEMA long) + ATR% percentile
- Optional CMO band trigger (CMO > MA + k*std)
- Optional global market filter (e.g., BTC above TEMA200 on higher TF)
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import pandas_ta as ta
from collections import namedtuple

# Simple signals container
Signals = namedtuple('Signals', ['entries', 'exits', 'short_entries', 'short_exits'], defaults=[None, None])

# --- Helpers -------------------------------------------------------------------

def _tema(series: pd.Series, length: int) -> pd.Series:
    return ta.tema(series, length=length)

def _tema_slope(series: pd.Series, length: int, slope_win: int = 5) -> pd.Series:
    """Slope (first difference) of TEMA over slope_win bars."""
    tema = _tema(series, length)
    return tema.diff(slope_win)

def _tema_dynamic_lengths(
    base_lengths: Dict[str, int],
    scale_series: pd.Series
) -> Dict[str, pd.Series]:
    """
    Produce per-bar dynamic lengths for TEMA based on a scale series (shifted to avoid lookahead).
    Returns dict of Series: {'fast','slow','long_fast','long_slow'} with integer lengths per bar.
    """
    def clamp_series(x: pd.Series) -> pd.Series:
        xi = x.round().astype('float').clip(lower=3.0)
        return xi.round().astype(int)

    fast = clamp_series(base_lengths['fast'] * scale_series)
    slow = clamp_series(base_lengths['slow'] * scale_series)
    long_fast = clamp_series(base_lengths['long_fast'] * scale_series)
    long_slow = clamp_series(base_lengths['long_slow'] * scale_series)
    return {'fast': fast, 'slow': slow, 'long_fast': long_fast, 'long_slow': long_slow}

def _tema_variable(series: pd.Series, lengths: pd.Series) -> pd.Series:
    """
    Compute TEMA with variable length per bar by recomputing on unique windows and aligning.
    This is heavier but avoids lookahead by using shifted scale_series beforehand.
    """
    out = pd.Series(index=series.index, dtype=float)
    for L in sorted(lengths.dropna().unique()):
        mask = lengths == L
        out.loc[mask] = _tema(series, int(L)).loc[mask]
    return out

def _cmo(series: pd.Series, length: int) -> pd.Series:
    """Return CMO as Series, normalizing any single-column DataFrame to Series."""
    cmo = ta.cmo(series, length=length)
    # Normalize once: squeeze avoids repeated isinstance checks
    if isinstance(cmo, pd.DataFrame):
        cmo = cmo.squeeze(axis=1)
    return cmo.astype(float)

def _zscore(series: pd.Series, window: int) -> pd.Series:
    # Use pandas_ta SMA/STDEV which can be faster depending on version
    mean = ta.sma(series, length=window)
    std = ta.stdev(series, length=window, ddof=0)
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

def _compute_tr_atr(close: pd.Series, high: pd.Series, low: pd.Series, atr_len: int) -> Dict[str, pd.Series]:
    """Compute True Range and ATR once and reuse across the module."""
    tr = ta.true_range(high=high, low=low, close=close)
    atr = tr.rolling(atr_len).mean()
    return {'tr': tr, 'atr': atr}

def _compute_adaptive_windows(close: pd.Series,
                              high: pd.Series,
                              low: pd.Series,
                              base_window: int,
                              atr_len: int,
                              atr_smooth: int,
                              clip_low: float,
                              clip_high: float,
                              precomputed: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
    """
    Compute an integer window multiplier based on ATR% relative to its rolling mean.
    Returns an integer window series used to adapt indicator lengths.
    """
    if precomputed and 'atr' in precomputed:
        atr = precomputed['atr']
    else:
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

def _dual_trend_gates_variable(close: pd.Series,
                               fast_series: pd.Series, slow_series: pd.Series,
                               long_fast_series: pd.Series, long_slow_series: pd.Series) -> Dict[str, pd.Series]:
    """Dual trend gates computed with variable TEMA lengths per bar."""
    tema_fast = _tema_variable(close, fast_series)
    tema_slow = _tema_variable(close, slow_series)
    tema_long_fast = _tema_variable(close, long_fast_series)
    tema_long_slow = _tema_variable(close, long_slow_series)

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

def _cmo_z_trigger_from_cmo(cmo: pd.Series,
                            z_win: int,
                            z_thresh: float) -> Dict[str, pd.Series]:
    """Derive z-score triggers from a precomputed CMO (avoid recomputation)."""
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

def _cmo_band_trigger_from_cmo(cmo: pd.Series,
                               ma_win: int,
                               k_std: float) -> Dict[str, pd.Series]:
    """CMO band trigger from a precomputed CMO (avoid recomputation)."""
    if isinstance(cmo, pd.DataFrame):
        cmo = cmo.iloc[:, 0]
    cmo = cmo.astype(float)
    ma = cmo.rolling(ma_win).mean()
    sd = cmo.rolling(ma_win).std(ddof=0)
    upper = (ma + k_std * sd)
    lower = (ma - k_std * sd)
    long_trigger = (cmo > upper)
    short_trigger = (cmo < lower)
    return {
        'cmo': cmo.fillna(0.0),
        'cmo_ma': ma.bfill().fillna(0.0),
        'cmo_sd': sd.bfill().fillna(0.0),
        'upper': upper.bfill().fillna(0.0),
        'lower': lower.bfill().fillna(0.0),
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

    # Align defaults with TemaTrendFollowing:
    # - short-term TEMA on primary TF: 10/80
    # - long-term TEMA on higher TF: 20/70 (applied via global filter below)
    fast = int(params.get('tema_fast', 10))
    slow = int(params.get('tema_slow', 80))
    long_fast = int(params.get('tema_long_fast', 20))
    long_slow = int(params.get('tema_long_slow', 70))

    cmo_len = int(params.get('cmo_len', 20))
    cmo_z_win = int(params.get('cmo_z_window', 50))
    cmo_z_thresh = float(params.get('cmo_z_threshold', 0.5))

    # New CMO band params
    use_cmo_band = bool(params.get('use_cmo_band', False))
    cmo_band_ma = int(params.get('cmo_band_ma', 50))
    cmo_band_k = float(params.get('cmo_band_k', 1.0))

    require_both_trends = bool(params.get('require_both_trends', True))
    allow_shorts = bool(params.get('allow_shorts', True))

    # New behavior toggles
    # Disable proportional adaptation by default to mirror TemaTrendFollowing
    use_adaptive_tema = bool(params.get('use_adaptive_tema', False))
    use_adaptive_tema_bar = bool(params.get('use_adaptive_tema_bar', False))  # bar-by-bar adaptive (kept off)
    confirm_on_close = bool(params.get('confirm_on_close', True))
    discrete_exits = bool(params.get('discrete_exits', False))

    # Regime filter 2: slope + ATR percentile
    # Light mode: disabled by default to avoid over-filtering; can be enabled explicitly
    use_regime_slope_atr = bool(params.get('use_regime_slope_atr', False))
    slope_win = int(params.get('slope_win', 5))
    atr_pct_percentile = float(params.get('atr_pct_percentile', 0.5))  # 0..1, e.g., 0.4 ~ 40th

    # Global market filter aligned with TemaTrendFollowing (use 4h TEMA 20/70)
    use_global_filter = bool(params.get('use_global_filter', True))
    global_tf = params.get('global_tf', '4h')  # default to 4h
    global_col_close = params.get('global_col_close', 'close')
    # if provided, fallback to old single TEMA len; otherwise use pair (20/70)
    global_tema_len = int(params.get('global_tema_len', 200))
    global_tema_fast = int(params.get('global_tema_fast', long_fast))
    global_tema_slow = int(params.get('global_tema_slow', long_slow))
    global_condition = params.get('global_condition', 'above')  # kept for compatibility

    # Single-pass normalization to avoid extra copies
    data = df.copy()
    data.columns = [c.lower() for c in data.columns]
    _assert_synced_index(data, ['open', 'high', 'low', 'close'])
    # Force numeric dtype once
    for col in ['open', 'high', 'low', 'close']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()  # no extra .copy() needed; we're not chaining afterwards

    # Precompute TR/ATR once for reuse
    _pre = _compute_tr_atr(data['close'], data['high'], data['low'], atr_len)
    
    # Volatility-adaptive window series (reuse ATR)
    # Apply light-mode tighter clipping defaults if not explicitly provided
    if 'atr_clip_low' not in params:
        atr_clip_low = 0.8
    if 'atr_clip_high' not in params:
        atr_clip_high = 1.3

    aw = _compute_adaptive_windows(
        data['close'], data['high'], data['low'],
        base_window, atr_len, atr_smooth, atr_clip_low, atr_clip_high,
        precomputed=_pre
    )

    # Proportional adaptive TEMA: derive a single effective set of lengths using prior-bar window
    if use_adaptive_tema and not use_adaptive_tema_bar:
        eff_window = aw.shift(1).bfill()
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_series = (eff_window / float(base_window)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        scale = float(np.nanmedian(scale_series.values)) if len(scale_series) > 0 else 1.0
        scaled = _scale_tema_lengths(fast, slow, long_fast, long_slow, scale)
        fast_eff, slow_eff = scaled['fast'], scaled['slow']
        long_fast_eff, long_slow_eff = scaled['long_fast'], scaled['long_slow']

        gates = _dual_trend_gates(
            data['close'], fast_eff, slow_eff, long_fast_eff, long_slow_eff
        )
    elif use_adaptive_tema_bar:
        # Item 1: bar-by-bar adaptive TEMA lengths using shifted scale to avoid lookahead
        eff_window = aw.shift(1).bfill()
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_series = (eff_window / float(base_window)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        # Clip scale to reasonable bounds
        scale_series = scale_series.clip(lower=atr_clip_low, upper=atr_clip_high)

        base_lengths = {'fast': fast, 'slow': slow, 'long_fast': long_fast, 'long_slow': long_slow}
        dyn_lengths = _tema_dynamic_lengths(base_lengths, scale_series)

        gates = _dual_trend_gates_variable(
            data['close'],
            dyn_lengths['fast'], dyn_lengths['slow'],
            dyn_lengths['long_fast'], dyn_lengths['long_slow']
        )
    else:
        gates = _dual_trend_gates(
            data['close'], fast, slow, long_fast, long_slow
        )

    # Compute CMO once and derive triggers based on mode (avoid recomputation)
    _cmo_raw = _cmo(data['close'], cmo_len).fillna(0.0)
    # ADX filter (TemaTrendFollowing used ADX > 40)
    use_adx_filter = bool(params.get('use_adx_filter', True))
    adx_len = int(params.get('adx_len', 14))
    adx_threshold = float(params.get('adx_threshold', 40.0))
    if use_adx_filter:
        adx_df = ta.adx(high=data['high'], low=data['low'], close=data['close'], length=adx_len)
        if isinstance(adx_df, pd.DataFrame):
            adx_col = None
            for col in adx_df.columns:
                if 'ADX' in col.upper():
                    adx_col = col
                    break
            adx_series = adx_df[adx_col] if adx_col is not None else adx_df.iloc[:, -1]
        else:
            adx_series = adx_df
        adx_ok = pd.to_numeric(adx_series, errors='coerce').bfill().fillna(0.0) > adx_threshold
    else:
        adx_ok = pd.Series(True, index=data.index)

    # Light mode: encourage a single CMO engine; if neither explicitly set, default to z-score
    if use_cmo_band:
        cmo_blk = _cmo_band_trigger_from_cmo(_cmo_raw, cmo_band_ma, cmo_band_k)
        cmo_long_trig = cmo_blk['long_trigger']
        cmo_short_trig = cmo_blk['short_trigger']
        cmo_bearish = (cmo_blk['cmo'] < cmo_blk['lower'])
        cmo_bullish = (cmo_blk['cmo'] > cmo_blk['upper'])
        # Keep a z proxy for exit logic uniformity
        cmo_z_val = _zscore(cmo_blk['cmo'], cmo_z_win)
    else:
        cmo_blk = _cmo_z_trigger_from_cmo(_cmo_raw, cmo_z_win, cmo_z_thresh)
        cmo_long_trig = cmo_blk['long_trigger']
        cmo_short_trig = cmo_blk['short_trigger']
        cmo_z_val = cmo_blk['cmo_z']

    # Regime filter (legacy both-trends vs long-only trend)
    if require_both_trends:
        long_regime = gates['short_trend_up'] & gates['long_trend_up']
        short_regime = gates['short_trend_down'] & gates['long_trend_down']
    else:
        long_regime = gates['long_trend_up']
        short_regime = gates['long_trend_down']

    # Item 2: slope(TEMA long) + ATR% percentile
    if use_regime_slope_atr:
        tema_long = gates['tema_long_fast']  # choose long_fast for slope proxy
        tema_long_slope = tema_long.diff(slope_win)
        # ATR% computation aligned to data (reuse precomputed ATR)
        atr_sz = _pre['atr']
        atr_pct = (atr_sz / data['close']).replace(0, np.nan) * 100.0
        # rolling percentile proxy via rolling rank/quantile; pandas lacks exact rolling percentile,
        # approximate with zscore vs rolling mean/std then compare to normal quantile:
        atr_mean = atr_pct.rolling(atr_smooth).mean()
        atr_std = atr_pct.rolling(atr_smooth).std(ddof=0)
        atr_z = (atr_pct - atr_mean) / atr_std.replace(0, np.nan)
        # threshold from percentile -> z cut (for median 0.5 -> 0 z)
        # Map simple percentiles: 0.4 ~ -0.253, 0.5 ~ 0.0, 0.6 ~ 0.253
        pct = np.clip(atr_pct_percentile, 0.01, 0.99)
        z_thresh = float(pd.Series(pct).apply(lambda p: 0.0 if abs(p-0.5) < 1e-9 else (0.253 if p>0.5 else -0.253)).iloc[0])
        atr_ok = atr_z > z_thresh
        slope_ok = tema_long_slope > 0
        # refine regimes
        long_regime = long_regime & slope_ok & atr_ok
        short_regime = short_regime & (~slope_ok) & atr_ok

    # Global market filter (Tema-style): use 4h TEMA fast/slow (20/70) gating by default
    if use_global_filter and isinstance(params.get('_tf_data_ref'), dict):
        tf_data: Dict[str, pd.DataFrame] = params['_tf_data_ref']
        if global_tf in tf_data:
            gdf = tf_data[global_tf].copy()
            if not isinstance(gdf.index, pd.DatetimeIndex):
                gdf.index = pd.to_datetime(gdf.index)
            gdf.columns = [c.lower() for c in gdf.columns]
            global_col = global_col_close if global_col_close in gdf.columns else 'close'
            gclose = gdf[global_col].astype(float)
            # Prefer paired TEMA(20/70) when provided, else fallback to single TEMA200 model
            if 'global_tema_fast' in params or 'global_tema_slow' in params or (long_fast and long_slow):
                gtema_fast = _tema(gclose, global_tema_fast)
                gtema_slow = _tema(gclose, global_tema_slow)
                g_up = (gtema_fast > gtema_slow)
                g_down = (gtema_fast < gtema_slow)
            else:
                gtema = _tema(gclose, global_tema_len)
                g_up = (gclose > gtema)
                g_down = (gclose < gtema)
            g_up = g_up.reindex(data.index, method='ffill').fillna(False)
            g_down = g_down.reindex(data.index, method='ffill').fillna(False)
            long_regime = long_regime & g_up
            short_regime = short_regime & g_down
        # else: leave regimes unchanged

    # Base conditions (include ADX filter)
    long_entry_cond = (long_regime & cmo_long_trig & adx_ok).fillna(False)
    if use_cmo_band:
        long_exit_cond = ((~long_regime) | (~cmo_bullish)).fillna(False)
    else:
        long_exit_cond = ((~long_regime) | (cmo_z_val < 0)).fillna(False)

    if allow_shorts:
        short_entry_cond = (short_regime & cmo_short_trig & adx_ok).fillna(False)
        if use_cmo_band:
            # band mode: exit when CMO returns inside the band or regime breaks
            short_exit_cond = ((~short_regime) | (~cmo_bearish)).fillna(False)
        else:
            # z-score mode: exit when cmo_z crosses above 0 or regime breaks
            short_exit_cond = ((~short_regime) | (cmo_z_val > 0)).fillna(False)
    else:
        short_entry_cond = pd.Series(False, index=data.index)
        short_exit_cond = pd.Series(False, index=data.index)

    # Closed-bar confirmation with edge detection
    if confirm_on_close:
        # entries as rising edge
        # Use explicit boolean conversion to avoid FutureWarning
        long_prev = long_entry_cond.shift(1).fillna(False)
        short_prev = short_entry_cond.shift(1).fillna(False)
        # Convert to bool explicitly to avoid downcasting warning
        long_prev = long_prev.astype(bool, copy=False)
        short_prev = short_prev.astype(bool, copy=False)
        long_entries = (long_entry_cond & ~long_prev).astype(bool)
        short_entries = (short_entry_cond & ~short_prev).astype(bool)
        if discrete_exits:
            # exits as rising edge of exit condition
            # Use explicit boolean conversion to avoid FutureWarning
            long_exit_prev = long_exit_cond.shift(1).fillna(False)
            short_exit_prev = short_exit_cond.shift(1).fillna(False)
            # Convert to bool explicitly to avoid downcasting warning
            long_exit_prev = long_exit_prev.astype(bool, copy=False)
            short_exit_prev = short_exit_prev.astype(bool, copy=False)
            long_exits = (long_exit_cond & ~long_exit_prev).astype(bool)
            short_exits = (short_exit_cond & ~short_exit_prev).astype(bool)
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

    # Optional debug
    if bool(params.get('verbose_debug', False)):
        try:
            print("[vectorbt] entries L/S:", int(long_entries.sum()), int(short_entries.sum()))
            print("[vectorbt] exits   L/S:", int(long_exits.sum()), int(short_exits.sum()))
        except Exception:
            pass

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

    # Pass a reference of tf_data into params for optional global filter alignment
    params = dict(params)
    params['_tf_data_ref'] = tf_data

    return _generate_signals_df(primary_df, params)

def get_vectorbt_required_timeframes(params: Dict) -> List[str]:
    """
    Return required timeframes; default to primary TF and optional global TF for filter if provided.
    """
    # Light defaults: if global filter is enabled and a global_tf is set, include it
    default = ['1h']
    if bool(params.get('use_global_filter', True)) and params.get('global_tf'):
        return [params.get('primary_timeframe', default[0]), params['global_tf']]
    return params.get('required_timeframes', default)

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

    # Compute ATR for sizing if needed (reuse helper for consistency)
    atr_len_size = int(params.get('atr_len_size', 14))
    _pre_size = _compute_tr_atr(close, high, low, atr_len_size)
    atr = _pre_size['atr']

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

        # Units computed per bar; apply leverage scaling to emulate 3x (default) and clip
        leverage = float(params.get('leverage', 3.0))
        units = (risk_cash / stop_dist.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        units = (units * max(0.0, leverage)).clip(lower=0.0, upper=1e12)

        out['size'] = units
        out['size_type'] = 'amount'

        # Provide stop levels and default TP aligned to TemaTrendFollowing (SL=4*ATR, TP=3*ATR)
        if stop_mode == 'atr':
            sl_distance = (atr * atr_mult_stop).bfill()
        else:
            sl_distance = (close * stop_percent).bfill()

        allow_shorts = bool(params.get('allow_shorts', True))
        long_sl = (close - sl_distance).clip(lower=0.0)
        tp_mult = params.get('take_profit_mult', 3.0)  # default TP 3*ATR
        if tp_mult is not None:
            try:
                long_tp = (close + atr * float(tp_mult)).clip(lower=0.0)
            except Exception:
                long_tp = None
        else:
            long_tp = None

        short_sl = (close + sl_distance).clip(lower=0.0)
        if tp_mult is not None:
            try:
                short_tp = (close - atr * float(tp_mult)).clip(lower=0.0)
            except Exception:
                short_tp = None
        else:
            short_tp = None
        
        # If shorts allowed and we can provide both, pass arrays that vectorbt can use.
        # from_signals uses the same sl_stop/tp_take for any position; to support asymmetry,
        # we pass long-oriented arrays; vectorbt will still close shorts on crossing sl_stop/tp_take.
        # Provide the more conservative (wider) of the two as unified arrays to avoid premature exits.
        if allow_shorts:
            # Choose conservative combo: farther SL (max distance) and closer TP (min favorable)
            unified_sl = pd.concat([long_sl, short_sl], axis=1).max(axis=1)
            if long_tp is not None and short_tp is not None:
                unified_tp = pd.concat([long_tp, short_tp], axis=1).min(axis=1).clip(lower=0.0)
            else:
                unified_tp = long_tp if long_tp is not None else short_tp
            out['sl_stop'] = unified_sl
            if unified_tp is not None:
                out['tp_take'] = unified_tp
        else:
            out['sl_stop'] = long_sl
            if long_tp is not None:
                out['tp_take'] = long_tp

    # Optional debug prints
    if bool(params.get('verbose_debug', False)):
        try:
            mode = size_mode
            tp = params.get('take_profit_mult', 3.0 if size_mode == 'risk_atr' else None)
            print(f"[vectorbt] sizing mode={mode}, atr_mult_stop={params.get('atr_mult_stop', 4.0)}, tp_mult={tp}, leverage={params.get('leverage', 3.0)}")
            if 'sl_stop' in out:
                qs = out['sl_stop'].dropna()
                if len(qs) > 0:
                    print("[vectorbt] sl_stop quantiles:", qs.quantile([0.05,0.5,0.95]).to_dict())
            if 'tp_take' in out and out['tp_take'] is not None:
                qt = out['tp_take'].dropna()
                if len(qt) > 0:
                    print("[vectorbt] tp_take quantiles:", qt.quantile([0.05,0.5,0.95]).to_dict())
        except Exception:
            pass

    # Return assembled kwargs for vbt.Portfolio.from_signals
    return out


def create_vectorbt_portfolio(data: pd.DataFrame, params: Dict = None) -> 'vbt.Portfolio':
    """Create VectorBT portfolio directly from data and parameters."""
    import vectorbt as vbt
    
    if params is None:
        params = {
            'base_window': 50,
            'atr_len': 14,
            'tema_fast': 10,
            'tema_slow': 80,
            'cmo_len': 20,
            'cmo_z_threshold': 0.5,
            'allow_shorts': True
        }
        
    # Generate signals
    tf_data = {'1h': data}  # VectorBT strategy uses single timeframe primarily
    signals = generate_vectorbt_signals(tf_data, params)
    
    # Get VBT parameters
    vbt_params = get_vbt_params(data, params)
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=data['close'],
        entries=signals.entries,
        exits=signals.exits,
        short_entries=signals.short_entries,
        short_exits=signals.short_exits,
        init_cash=10000,
        fees=0.001,
        **vbt_params
    )
    
    return portfolio

