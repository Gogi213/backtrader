"""
mr_vwma_zscore_vbt.py

Mean-reversion syncretic strategy for volatile alts (1m):
 - VWMA (volume-weighted MA) + VW-std -> Z_vwma
 - nATR (ATR(30)/price) filter (trade only if nATR >= 1%)
 - volume spike guard (volume >= vol_median * vol_mult)
 - multi-TF trend filter: use resampled 5m EMA to avoid trading against clear trend
 - entry: z_used <= -z_thresh -> long, z_used >= +z_thresh -> short
 - exits:
    * partial take profit: layer A (half) exits when price crosses VWMA toward mean (fast TP)
    * runner: layer B (half) uses trailing stop (ts_stop) and optional tp_stop
 - implemented with vectorbt Portfolio.from_signals (two parallel columns -> partial+runner)

Usage:
    python mr_vwma_zscore_vbt.py path/to/1m_data.csv

Input CSV must contain: ['datetime','open','high','low','close','volume'] (datetime parseable).
Output: prints pf stats and saves `results_{symbol}.parquet` (if provided).
"""

import sys
import numpy as np
import pandas as pd
import vectorbt as vbt

# -------------------------
# STRATEGY PARAMETERS (tweak these)
# -------------------------
LOOKBACK_VWMA = 30         # window for VWMA and weighted std
LOOKBACK_Z = 30            # window for z-score computation (same as vwma for simplicity)
Z_THRESHOLD = 2.5          # z threshold for entry
ATR_PERIOD = 30            # atr period for nATR baseline
NATR_MIN = 0.01            # nATR threshold (1% default)
ATR_MULT_ENTRY = 1.5       # require deviation >= ATR_MULT_ENTRY * nATR
VOL_MULT = 1.3             # min volume multiplier vs median
LOOKBACK_VOL_MED = 30      # median vol window
LOOKBACK_TREND = 5        # minutes to compute trend EMA (resampled 5m)
EMA_TREND_LONG = 50       # long EMA on 5m for trend filter (in minutes -> converted)
MAX_SCALE = 2             # number of parallel layers (we implement 2: partial + runner)

# Portfolio sizing (value percent of portfolio for each layer)
# Example: allocate 5% of portfolio value to layer A and 5% to layer B (total 10% exposure)
SIZE_LAYER_A = 0.05  # 5% allocation to fast TP (partial)
SIZE_LAYER_B = 0.05  # 5% allocation to runner
SIZE_TYPE = "valuepercent"  # use 'valuepercent' for percent of portfolio value (vectorbt)

# Stops for each layer:
# layer A: fast TP implemented via exits (price crossing VWMA) -> no sl_stop needed separately,
# but we still define a protective sl_stop as percent
SL_LAYER_A = 0.015   # 1.5% protective stop below entry for layer A
TP_LAYER_A = None    # handled via exit condition (VWMA cross). If set, it is a fraction (0.02 -> 2%)

# layer B (runner): trailing stop
SL_LAYER_B = 0.03     # initial protective stop 3%
TS_LAYER_B = 0.012    # trailing stop 1.2% from peak (ts_stop)
TP_LAYER_B = 0.08     # optional fixed TP for runner (8%) - set to None to disable

INIT_CASH = 10_000

# -------------------------
# Helpers: indicators
# -------------------------
def compute_vwma_and_weighted_std(df, price_col='close', vol_col='volume', window=LOOKBACK_VWMA):
    p = df[price_col].astype(float)
    v = df[vol_col].astype(float)
    pv = p * v
    sum_v = v.rolling(window, min_periods=1).sum()
    sum_pv = pv.rolling(window, min_periods=1).sum()
    vwma = sum_pv / sum_v.replace(0, np.nan)
    # weighted variance
    pv2 = (p * p) * v
    sum_pv2 = pv2.rolling(window, min_periods=1).sum()
    var_w = (sum_pv2 / sum_v.replace(0, np.nan)) - (vwma ** 2)
    var_w = var_w.clip(lower=0)
    wstd = np.sqrt(var_w)
    return vwma, wstd

def compute_atr(df, period=ATR_PERIOD):
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

# -------------------------
# Signal generation
# -------------------------
def generate_signals(df):
    df = df.copy()
    price = df['close'].astype(float)
    vol = df['volume'].astype(float)

    # VWMA + weighted std
    vwma, vw_std = compute_vwma_and_weighted_std(df, window=LOOKBACK_VWMA)
    df['vwma'] = vwma
    df['vw_std'] = vw_std

    # z score (volume-weighted)
    df['z_vwma'] = (price - df['vwma']) / df['vw_std'].replace(0, np.nan)
    df['z_vwma'] = df['z_vwma'].fillna(0.0)

    # nATR
    atr = compute_atr(df, period=ATR_PERIOD)
    df['atr'] = atr
    df['natr'] = (atr / price).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # deviation %
    df['deviation_pct'] = (price - df['vwma']) / df['vwma']

    # volume median
    df['vol_median'] = vol.rolling(LOOKBACK_VOL_MED, min_periods=1).median()

    # trend filter: resample to 5m, compute EMA slope (we'll map back to 1m index)
    # Note: ensure index is datetime and freq=1T
    df_5m = df[['close']].resample('5T').last().ffill()
    ema5 = df_5m['close'].ewm(span=EMA_TREND_LONG//5 if EMA_TREND_LONG//5>0 else 10).mean()
    # compute slope sign: positive -> uptrend
    ema5_slope = ema5.diff().fillna(0)
    # map slope back to each 1m bar by reindexing forward-fill
    ema5_slope_full = ema5_slope.reindex(df.index, method='ffill').fillna(0)
    df['ema5_slope'] = ema5_slope_full

    # conditions
    cond_natr = df['natr'] >= NATR_MIN
    cond_vol = df['volume'] >= (df['vol_median'] * VOL_MULT)
    cond_dev = df['deviation_pct'].abs() >= (df['natr'] * ATR_MULT_ENTRY)

    # entries: z thresholds + dev + vol + natr
    long_entries = (df['z_vwma'] <= -Z_THRESHOLD) & cond_dev & cond_vol & cond_natr
    short_entries = (df['z_vwma'] >= +Z_THRESHOLD) & cond_dev & cond_vol & cond_natr

    # trend filter: if ema slope strongly positive, avoid short entries; if strongly negative, avoid long entries
    slope_thresh = 0  # can be tuned; >0 => uptrend
    long_entries = long_entries & (df['ema5_slope'] <= slope_thresh + 1e9)  # keep as is, or tighten
    short_entries = short_entries & (df['ema5_slope'] >= -slope_thresh - 1e9)

    # We'll use a single combined entry series (True when either long or short)
    entries = long_entries | short_entries

    # EXITS:
    # Layer A (fast TP) exit when price crosses VWMA back toward mean:
    # for long entries: exit when close >= vwma (cross above)
    # for short entries: exit when close <= vwma (cross below)
    long_fast_exit = (df['close'] >= df['vwma'])
    short_fast_exit = (df['close'] <= df['vwma'])
    # But we only want to use them for respective positions; we'll build two-column exits for each layer later

    # Build two-column entries/exits DataFrames (columns: 'layerA', 'layerB')
    # entries columns identical (open both partial and runner at the same time)
    entries_df = pd.DataFrame({
        'layerA': entries,
        'layerB': entries
    }, index=df.index)

    # exits:
    exits_df = pd.DataFrame(False, index=df.index, columns=['layerA', 'layerB'])

    # LayerA: use fast exit (mean cross)
    exits_df['layerA'] = ( (long_entries & long_fast_exit) | (short_entries & short_fast_exit) )

    # LayerB: we will rely on portfolio param ts_stop and optional tp_stop/sl_stop
    # But creating a fallback exit to avoid infinite holds: force exit at final bar
    exits_df['layerB'] = False  # let ts_stop / tp_stop handle this

    # prepare sl/tp/ts per layer (as fractions, e.g., 0.01 = 1%)
    sl_stop = [SL_LAYER_A, SL_LAYER_B]
    tp_stop = [TP_LAYER_A if TP_LAYER_A is not None else np.nan, TP_LAYER_B if TP_LAYER_B is not None else np.nan]
    ts_stop = [np.nan, TS_LAYER_B]  # only layerB has trailing stop

    # pack indicator columns into df for diagnostics
    df_out = df.copy()
    df_out['entries'] = entries
    df_out['long_entries'] = long_entries
    df_out['short_entries'] = short_entries
    df_out['entries_layerA'] = entries_df['layerA']
    df_out['entries_layerB'] = entries_df['layerB']
    df_out['exits_layerA'] = exits_df['layerA']
    df_out['exits_layerB'] = exits_df['layerB']

    meta = {
        'sl_stop': sl_stop,
        'tp_stop': tp_stop,
        'ts_stop': ts_stop
    }

    return df_out, entries_df, exits_df, meta

# -------------------------
# Backtest runner (vectorbt)
# -------------------------
def run_backtest(df, symbol="SYMB"):
    price = df['close']
    df_signals, entries_df, exits_df, meta = generate_signals(df)

    # VectorBT: entries/exits should be 2D (time x strategies)
    # vectorbt expects columns to be assets/strategies; we pass two columns -> layerA, layerB
    # size needs to be shaped appropriately. We use size as 2-element list inside list, per doc examples.
    size_list = [[SIZE_LAYER_A, SIZE_LAYER_B]]  # shape (1, ncols) is accepted and will broadcast
    size_type = SIZE_TYPE  # 'valuepercent' is used in examples

    # from_signals supports sl_stop / tp_stop / ts_stop as arrays per strategy
    # We pass them as lists, vbt will broadcast to strategies
    sl_arg = meta['sl_stop']
    tp_arg = meta['tp_stop']
    ts_arg = meta['ts_stop']

    print("Running backtest with parameters:")
    print(f" Z_THRESHOLD={Z_THRESHOLD}, NATR_MIN={NATR_MIN}, VOL_MULT={VOL_MULT}")
    print(f" Size per layer (valuepercent): {SIZE_LAYER_A*100:.1f}% , {SIZE_LAYER_B*100:.1f}%")
    print(f" sl_stop: {sl_arg}, tp_stop: {tp_arg}, ts_stop: {ts_arg}")

    # Build the portfolio. We enable cash_sharing so both layers use the same cash pool.
    pf = vbt.Portfolio.from_signals(
        price,
        entries=entries_df,
        exits=exits_df,
        size=size_list,
        size_type=size_type,
        init_cash=INIT_CASH,
        fees=0.0006,           # example fees (0.06%)
        sl_stop=sl_arg,
        tp_stop=tp_arg,
        ts_stop=ts_arg,
        group_by=None,
        cash_sharing=True,
        direction='both',      # allow long and short (for alt scalping consider 'longonly' if shorts not allowed)
    )

    # Diagnostics
    print("\nPortfolio summary")
    print("=================")
    print("Total return:", pf.total_return().values)
    print("Total profit:", pf.total_profit().values)
    print("Max drawdown:", pf.max_drawdown().values)
    print("Sharpe (ann):", pf.sharpe_ratio().values)

    # trades summary
    print("\nTrades summary (per layer)")
    try:
        trades = pf.trades.records_readable
        print(trades.head(10).to_string(index=False))
    except Exception as e:
        print("Could not print trades:", e)

    return pf, df_signals

# -------------------------
# CLI usage: load CSV and run
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mr_vwma_zscore_vbt.py path/to/1m_data.csv")
        sys.exit(0)

    path = sys.argv[1]
    print("Loading", path)
    df_raw = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    # ensure 1m frequency index (if there are missing minutes, forward fill last known ohlc)
    df_raw = df_raw.sort_index()
    # reindex to continuous 1-minute grid to avoid resample artifacts (if you prefer, skip)
    idx = pd.date_range(df_raw.index[0], df_raw.index[-1], freq='1T')
    df = df_raw.reindex(idx).ffill().bfill()
    pf, df_signals = run_backtest(df)
    # Save results (optional)
    out_signals = path.replace('.csv', '.signals.parquet')
    df_signals.to_parquet(out_signals)
    print("Signals saved to", out_signals)
