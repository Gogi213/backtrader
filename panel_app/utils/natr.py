"""
Унифицированные функции для расчета nATR (Normalized ATR).
Используется во всех стратегиях и для отображения метрик.
"""
import pandas as pd
import numpy as np


def calculate_natr(df: pd.DataFrame, period: int = 30, as_percentage: bool = True):
    """
    Рассчитывает нормализованный ATR (nATR).
    
    Args:
        df: DataFrame с колонками 'high', 'low', 'close'
        period: Период для расчета ATR (по умолчанию 30)
        as_percentage: Если True, возвращает в процентах, иначе как долю от цены
        
    Returns:
        pd.Series: nATR значения
    """
    close = df['close'].astype('float64')
    high = df['high'].astype('float64')
    low = df['low'].astype('float64')
    
    # True Range calculation
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR using Wilder's smoothing (RMA/EMA with alpha=1/period)
    atr = true_range.ewm(alpha=1/float(period), adjust=False).mean()
    
    # Normalize by price
    natr = atr / close
    
    if as_percentage:
        natr = natr * 100.0
        
    return natr


def get_natr_at_indices(df: pd.DataFrame, indices: pd.Series, period: int = 30):
    """
    Получает nATR значения на конкретных индексах (например, на входе/выходе из сделок).
    
    Args:
        df: DataFrame с OHLC данными
        indices: Series с индексами для получения nATR
        period: Период для расчета ATR
        
    Returns:
        pd.Series: nATR значения в процентах для каждого индекса
    """
    natr_series = calculate_natr(df, period=period, as_percentage=True)
    
    def _get_natr_value(idx):
        try:
            idx = int(idx)
            if idx < 0 or idx >= len(natr_series):
                return None
            val = natr_series.iloc[idx]
            return float(val) if pd.notna(val) else None
        except Exception:
            return None
    
    return indices.apply(_get_natr_value)


def apply_natr_filter(entries_long, entries_short, df: pd.DataFrame, 
                     min_natr_pct: float, period: int = 30):
    """
    Применяет фильтр по минимальному nATR к сигналам входа.
    
    Args:
        entries_long: Boolean series для лонг входов
        entries_short: Boolean series для шорт входов  
        df: DataFrame с OHLC данными
        min_natr_pct: Минимальный nATR в процентах
        period: Период для расчета ATR
        
    Returns:
        tuple: (filtered_entries_long, filtered_entries_short)
    """
    natr_pct = calculate_natr(df, period=period, as_percentage=True)
    natr_condition = natr_pct >= float(min_natr_pct)
    
    filtered_long = entries_long & natr_condition
    filtered_short = entries_short & natr_condition
    
    return filtered_long, filtered_short


def calculate_natr_sl_tp(df: pd.DataFrame, sl_multiplier: float, tp_multiplier: float, 
                        period: int = 30):
    """
    Рассчитывает Stop Loss и Take Profit на основе nATR множителей.
    
    В nATR режиме пользователь вводит МНОЖИТЕЛИ, а не проценты!
    
    Args:
        df: DataFrame с OHLC данными
        sl_multiplier: Множитель nATR для SL (1.5 = 1.5 * nATR)
        tp_multiplier: Множитель nATR для TP (3.0 = 3.0 * nATR)
        period: Период для расчета ATR
        
    Returns:
        tuple: (sl_levels, tp_levels) как доли от цены для VectorBT
        
    Пример:
        Если nATR = 2% и sl_multiplier = 1.5
        То SL = 2% * 1.5 = 3% от цены входа
    """
    natr = calculate_natr(df, period=period, as_percentage=False)  # как доля
    
    sl_levels = natr * float(sl_multiplier)
    tp_levels = natr * float(tp_multiplier)
    
    return sl_levels, tp_levels
