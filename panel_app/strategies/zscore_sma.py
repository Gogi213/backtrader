import vectorbt as vbt
import pandas as pd
import numpy as np
from ..utils.natr import apply_natr_filter, calculate_natr_sl_tp


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
    use_natr_filter: bool = False,
    natr_filter_min_pct: float = 0.5,
):
    """
    Z-Score + SMA стратегия (исправленная версия).

    Логика входа:
    Длинная позиция (покупка):
    - close > SMA(sma_len) - цена выше долгосрочного тренда  
    - Z-Score <= -z_thresh - цена сильно упала относительно среднего (oversold)

    Короткая позиция (продажа):
    - close < SMA(sma_len) - цена ниже долгосрочного тренда
    - Z-Score >= z_thresh - цена сильно выросла относительно среднего (overbought)

    Исполнение (реалистичная модель):
    - Условия проверяются на close bar N (edge detection)
    - Сигнал формируется на открытии bar N+1  
    - Исполнение по open цене bar N+1 (≈ close bar N)
    - SL/TP работают с реальными high/low ценами

    Параметры SL/TP:
    - Если use_natr_sl_tp=False: sl_pct/tp_pct как проценты (1.0 = 1%)
    - Если use_natr_sl_tp=True: sl_pct/tp_pct как множители nATR
      
    Пример nATR режима:
    - nATR = 2%, sl_pct = 1.5 → SL = 2% * 1.5 = 3%
    - nATR = 2%, tp_pct = 3.0 → TP = 2% * 3.0 = 6%
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

    # Условия входа (исправлена логика)
    long_cond = (close > sma) & (z <= -z_thresh)  # убрали abs()
    short_cond = (close < sma) & (z >= z_thresh)

    # Применяем фильтр по nATR если включен
    if use_natr_filter and natr_filter_min_pct is not None:
        long_cond, short_cond = apply_natr_filter(
            long_cond, short_cond, df, natr_filter_min_pct, natr_len
        )

    # Исключаем одновременные и встречные сигналы: если оба, то игнорируем
    both = long_cond & short_cond
    long_cond = long_cond & ~both
    short_cond = short_cond & ~both

    # Edge detection: вход только на первом баре появления условия
    raw_long = long_cond & ~long_cond.shift(1, fill_value=False)
    raw_short = short_cond & ~short_cond.shift(1, fill_value=False)
    
    # Исполнение на следующем баре по цене открытия (реалистично)
    # Это правильно: сигнал на close[N] -> исполнение на open[N+1]
    entries_long = raw_long.shift(1, fill_value=False) 
    entries_short = raw_short.shift(1, fill_value=False)

    # Выходы не задаем явно: полагаемся на SL/TP (меньше аллокаций)
    exits_long = None
    exits_short = None

    # Рассчёт SL/TP
    if use_natr_sl_tp:
        # Используем nATR множители
        sl, tp = calculate_natr_sl_tp(df, sl_pct, tp_pct, natr_len)
    else:
        # Используем проценты
        sl = float(sl_pct) / 100.0 if sl_pct is not None else None
        tp = float(tp_pct) / 100.0 if tp_pct is not None else None

    pf = vbt.Portfolio.from_signals(
        # ВОЗВРАЩЕНО: логика была правильной изначально
        # Сигнал на close[N] -> исполнение на open[N+1] 
        # Поэтому используем open как цену для расчета PnL
        close=open_,    # цена исполнения = цена для PnL расчета
        high=high,      # максимум для SL/TP
        low=low,        # минимум для SL/TP  
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
