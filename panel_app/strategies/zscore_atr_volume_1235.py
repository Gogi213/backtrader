import vectorbt as vbt
import pandas as pd
import numpy as np

def run_zscore_atr_volume_strategy_1235(df: pd.DataFrame, window: int = 30, z_thresh: float = 2.0, atr_len: int = 30, vol_z_thresh: float = 0.5, atr_min: float = 0.01, fee: float = 0.00035, init_cash: float = 10000, leverage: float = 1.0):
    close = df['close']
    volume = df['volume']
    sma = close.rolling(window).mean()
    sigma = close.rolling(window).std()
    z = (close - sma) / sigma
    high = df['high'].rolling(atr_len).max()
    low = df['low'].rolling(atr_len).min()
    atr = (high - low) / sma
    vol_med = volume.rolling(window).median()
    vol_z = (volume - vol_med) / vol_med
    entries_long = (z < -z_thresh) & (atr > atr_min) & (vol_z > vol_z_thresh)
    exits_long = (z > 0) | (atr <= atr_min) | (vol_z <= 0)
    entries_short = (z > z_thresh) & (atr > atr_min) & (vol_z > vol_z_thresh)
    exits_short = (z < 0) | (atr <= atr_min) | (vol_z <= 0)
    entries_long = entries_long & ~(exits_long | exits_long.shift(1, fill_value=False))
    entries_short = entries_short & ~(exits_short | exits_short.shift(1, fill_value=False))
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries_long,
        exits=exits_long,
        short_entries=entries_short,
        short_exits=exits_short,
        size=init_cash * leverage,
        size_type='value',
        fees=fee,
        init_cash=init_cash
    )
    signals = {
        'z': z,
        'atr': atr,
        'vol_z': vol_z,
        'entries_long': entries_long,
        'exits_long': exits_long,
        'entries_short': entries_short,
        'exits_short': exits_short
    }
    return pf, signals
