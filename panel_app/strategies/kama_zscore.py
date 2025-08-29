"""
KAMA + адаптивный Z-Score стратегия для коротких таймфреймов (15s).

Логика:
- KAMA как трендовый фильтр
- Z-Score с адаптивным окном (на основе nATR) для mean reversion входов
- Входы против краткосрочного движения в направлении долгосрочного тренда
"""
import vectorbt as vbt
import pandas as pd
import numpy as np
from ..utils.natr import apply_natr_filter, calculate_natr_sl_tp


def calculate_kama(close: pd.Series, period: int = 20, fast_sc: int = 2, slow_sc: int = 30):
    """
    Рассчитывает Kaufman's Adaptive Moving Average (KAMA).
    
    Args:
        close: Цены закрытия
        period: Период для расчета эффективности (ER)
        fast_sc: Быстрая константа сглаживания
        slow_sc: Медленная константа сглаживания
    
    Returns:
        pd.Series: KAMA значения
    """
    # Расчет Efficiency Ratio (ER)
    change = (close - close.shift(period)).abs()
    volatility = (close - close.shift(1)).abs().rolling(period).sum()
    
    # Защита от деления на ноль
    volatility = volatility.replace(0, np.nan)
    er = change / volatility
    
    # Константы сглаживания
    fastest_sc = 2.0 / (fast_sc + 1)
    slowest_sc = 2.0 / (slow_sc + 1)
    
    # Адаптивная константа сглаживания
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    
    # Расчет KAMA
    kama = pd.Series(index=close.index, dtype='float64')
    kama.iloc[period] = close.iloc[period]  # Начальное значение
    
    for i in range(period + 1, len(close)):
        if pd.notna(sc.iloc[i]) and pd.notna(kama.iloc[i-1]):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        else:
            kama.iloc[i] = close.iloc[i]
    
    return kama


def calculate_adaptive_zscore(close: pd.Series, natr_pct: pd.Series, 
                             base_window: int = 10, natr_multiplier: float = 50):
    """
    Рассчитывает Z-Score с адаптивным окном на основе nATR.
    
    Args:
        close: Цены закрытия
        natr_pct: nATR в процентах
        base_window: Базовое окно для Z-Score
        natr_multiplier: Множитель nATR для адаптации окна
    
    Returns:
        pd.Series: Адаптивный Z-Score
    """
    zscore = pd.Series(index=close.index, dtype='float64')
    
    for i in range(base_window, len(close)):
        # Адаптивное окно: больше волатильность = больше окно
        if pd.notna(natr_pct.iloc[i]):
            adaptive_window = int(base_window + natr_pct.iloc[i] * natr_multiplier)
            adaptive_window = max(base_window, min(adaptive_window, i))  # Ограничения
        else:
            adaptive_window = base_window
        
        # Расчет Z-Score для адаптивного окна
        window_close = close.iloc[i-adaptive_window:i]
        if len(window_close) >= 3:  # Минимум точек для расчета
            mean_val = window_close.mean()
            std_val = window_close.std()
            if std_val > 0:
                zscore.iloc[i] = (close.iloc[i] - mean_val) / std_val
    
    return zscore


def run_kama_zscore_strategy(
    df: pd.DataFrame,
    kama_period: int = 20,
    kama_fast_sc: int = 2,
    kama_slow_sc: int = 30,
    z_base_window: int = 10,
    z_natr_multiplier: float = 50,
    z_threshold: float = 2.0,
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
    KAMA + адаптивный Z-Score стратегия.

    Логика входа:
    Длинная позиция (покупка):
    - KAMA направлен вверх (восходящий тренд)
    - Адаптивный Z-Score <= -z_threshold (oversold относительно краткосрочного среднего)

    Короткая позиция (продажа):
    - KAMA направлен вниз (нисходящий тренд)  
    - Адаптивный Z-Score >= z_threshold (overbought относительно краткосрочного среднего)

    Исполнение (реалистичная модель):
    - Условия проверяются на close bar N
    - Сигнал формируется на открытии bar N+1  
    - Исполнение по open цене bar N+1
    - SL/TP работают с реальными high/low ценами

    Параметры SL/TP:
    - Если use_natr_sl_tp=False: sl_pct/tp_pct как проценты (1.0 = 1%)
    - Если use_natr_sl_tp=True: sl_pct/tp_pct как множители nATR
      
    Адаптивность:
    - Окно Z-Score = z_base_window + nATR% * z_natr_multiplier
    - Высокая волатильность → большее окно → менее чувствительные сигналы
    """
    # Cast to float32 для оптимизации
    close = df['close'].astype('float32')
    open_ = df['open'].astype('float32')
    high = df['high'].astype('float32')
    low = df['low'].astype('float32')
    
    # Расчет KAMA
    kama = calculate_kama(close, kama_period, kama_fast_sc, kama_slow_sc)
    
    # Определение направления тренда по KAMA
    kama_trend_up = kama > kama.shift(1)
    kama_trend_down = kama < kama.shift(1)
    
    # Расчет nATR для адаптивного окна
    from ..utils.natr import calculate_natr
    natr_pct = calculate_natr(df, period=natr_len, as_percentage=True)
    
    # Адаптивный Z-Score
    adaptive_zscore = calculate_adaptive_zscore(
        close, natr_pct, z_base_window, z_natr_multiplier
    )
    
    # Условия входа
    long_cond = kama_trend_up & (adaptive_zscore <= -z_threshold)
    short_cond = kama_trend_down & (adaptive_zscore >= z_threshold)
    
    # Применяем фильтр по nATR если включен
    if use_natr_filter and natr_filter_min_pct is not None:
        long_cond, short_cond = apply_natr_filter(
            long_cond, short_cond, df, natr_filter_min_pct, natr_len
        )
    
    # Исключаем одновременные сигналы
    both = long_cond & short_cond
    long_cond = long_cond & ~both
    short_cond = short_cond & ~both
    
    # Edge detection + реалистичное исполнение
    raw_long = long_cond & ~long_cond.shift(1, fill_value=False)
    raw_short = short_cond & ~short_cond.shift(1, fill_value=False)
    
    entries_long = raw_long.shift(1, fill_value=False)
    entries_short = raw_short.shift(1, fill_value=False)
    
    # Выходы через SL/TP
    exits_long = None
    exits_short = None
    
    # Расчёт SL/TP
    if use_natr_sl_tp:
        sl, tp = calculate_natr_sl_tp(df, sl_pct, tp_pct, natr_len)
    else:
        sl = float(sl_pct) / 100.0 if sl_pct is not None else None
        tp = float(tp_pct) / 100.0 if tp_pct is not None else None
    
    # VectorBT Portfolio
    pf = vbt.Portfolio.from_signals(
        close=open_,    # Исполнение по open цене
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
        accumulate=False,
        upon_opposite_entry='Ignore',
    )
    
    signals = {
        'kama': kama,
        'kama_trend_up': kama_trend_up,
        'kama_trend_down': kama_trend_down,
        'adaptive_zscore': adaptive_zscore,
        'natr_pct': natr_pct,
        'entries_long': entries_long,
        'entries_short': entries_short,
    }
    
    return pf, signals
