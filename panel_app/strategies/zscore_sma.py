import vectorbt as vbt
import pandas as pd
import numpy as np


def run_zscore_sma_strategy(
    df: pd.DataFrame,
    sma_len: int = 50,
    z_window: int = 20,
    z_thresh: float = 2.0,
    sl_pct: float = 1.0,
    tp_pct: float = 3.0,
    fee: float = 0.00035,
    init_cash: float = 10000,
    leverage: float = 1.0,
    use_natr_sl_tp: bool = False,
    natr_len: int = 30,
):
    """
    Z-Score + SMA стратегия.

    Длинная позиция:
    - close > SMA(sma_len)
    - Z-Score(window=z_window) <= -z_thresh

    Короткая позиция:
    - close < SMA(sma_len)
    - Z-Score(window=z_window) >= z_thresh

    Стоп-лосс и тейк-профит указываются в процентах (sl_pct, tp_pct).
    """
    # Cast to float32 to reduce memory and speed up ops
    close = df['close'].astype('float32')
    open_ = df['open'].astype('float32')
    high = df['high'].astype('float32')
    low = df['low'].astype('float32')
    sma = close.rolling(sma_len).mean()
    sigma = close.rolling(z_window).std()
    # Защита от деления на ноль
    sigma = sigma.replace(0, np.nan)
    z = (close - close.rolling(z_window).mean()) / sigma

    # Условия входа
    long_cond = (close > sma) & (z <= -abs(z_thresh))
    short_cond = (close < sma) & (z >= abs(z_thresh))

    # Исключаем одновременные и встречные сигналы: если оба, то игнорируем
    both = long_cond & short_cond
    long_cond = long_cond & ~both
    short_cond = short_cond & ~both

    # Эдж-де-бaунс: вход только на первом баре появления условия
    raw_long = long_cond & ~long_cond.shift(1, fill_value=False)
    raw_short = short_cond & ~short_cond.shift(1, fill_value=False)
    # Исполнение на следующем баре по цене открытия
    entries_long = raw_long.shift(1, fill_value=False)
    entries_short = raw_short.shift(1, fill_value=False)
    # Для ускорения передаём numpy-массивы булевых значений
    entries_long = entries_long.to_numpy(dtype=bool)
    entries_short = entries_short.to_numpy(dtype=bool)

    # Выходы не задаем явно: полагаемся на SL/TP (меньше аллокаций)
    exits_long = None
    exits_short = None

    # Рассчёт SL/TP
    if use_natr_sl_tp:
        # True Range и ATR
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(int(natr_len)).mean()
        # Нормированный ATR
        natr = (atr / close).astype('float32')
        # Множители nATR (напр., tp=3 => 3*nATR)
        sl = (float(sl_pct) * natr) if sl_pct is not None else None
        tp = (float(tp_pct) * natr) if tp_pct is not None else None
    else:
        sl = float(sl_pct) / 100.0 if sl_pct is not None else None
        tp = float(tp_pct) / 100.0 if tp_pct is not None else None

    pf = vbt.Portfolio.from_signals(
        # Используем open как прайс для исполнения ордеров, но подаем весь OHLC для корректной работы SL/TP
        close=open_,
        high=high,
        low=low,
        entries=entries_long,
        exits=exits_long,
        short_entries=entries_short,
        short_exits=exits_short,
        size=init_cash * leverage,
        size_type='value',
        fees=fee,
        init_cash=init_cash,
        sl_stop=sl,
        tp_stop=tp,
        direction='both',
        # Управление позициями: без пирамидинга, не разворачиваемся — игнорируем встречный вход пока есть позиция
        accumulate=False,
        upon_opposite_entry='Ignore',
    )

    signals = {
        'sma': sma,
        'z': z,
        'entries_long': entries_long,
        'entries_short': entries_short,
    }
    return pf, signals
