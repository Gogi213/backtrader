import numpy as np
import pandas as pd
import vectorbt as vbt

def zscore_vbt_signals(close: pd.Series, window=60, entry_z=2.0, exit_z=0.0, max_hold=30):
    ma = close.rolling(window=window, min_periods=1).mean()
    std = close.rolling(window=window, min_periods=1).std(ddof=0)
    std = std.replace(0, np.nan).ffill().fillna(1e-8)
    z = (close - ma) / std
    entries_long = z <= -entry_z
    entries_short = z >= entry_z
    exits_long = z >= exit_z
    exits_short = z <= exit_z
    if max_hold is not None and max_hold > 0:
        def apply_max_hold(entries, exits):
            entries_idx = np.where(entries)[0]
            exits_out = exits.copy()
            for ei in entries_idx:
                subsequent = np.arange(ei + 1, len(entries))
                natural_exits = subsequent[exits[subsequent]]
                if natural_exits.size == 0:
                    forced_idx = min(ei + max_hold, len(entries) - 1)
                    exits_out[forced_idx] = True
                else:
                    first_exit = natural_exits[0]
                    if first_exit - ei > max_hold:
                        forced_idx = min(ei + max_hold, len(entries) - 1)
                        exits_out[forced_idx] = True
            return exits_out
        entries_long_arr = entries_long.values.astype(bool)
        entries_short_arr = entries_short.values.astype(bool)
        exits_long_arr = exits_long.values.astype(bool)
        exits_short_arr = exits_short.values.astype(bool)
        exits_long_arr = apply_max_hold(entries_long_arr, exits_long_arr)
        exits_short_arr = apply_max_hold(entries_short_arr, exits_short_arr)
        exits_long = pd.Series(exits_long_arr, index=close.index)
        exits_short = pd.Series(exits_short_arr, index=close.index)
    return entries_long, exits_long, entries_short, exits_short, z

def run_zscore_vbt_strategy(ohlcv_df, window=60, entry_z=2.0, exit_z=0.0, max_hold=30, init_cash=10000, fee=0.0005):
    close = ohlcv_df['close']
    entries_long, exits_long, entries_short, exits_short, z = zscore_vbt_signals(
        close, window=window, entry_z=entry_z, exit_z=exit_z, max_hold=max_hold)
    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries_long,
        exits=exits_long,
        short_entries=entries_short,
        short_exits=exits_short,
        init_cash=init_cash,
        fees=fee,
        freq='min'
    )
    return pf, z
