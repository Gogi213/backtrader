import numpy as np
import pandas as pd
from scipy import stats
import math


def calculate_true_range(high, low, close):
    """
    Вычисление истинного диапазона (True Range) с поддержкой GPU.

    Args:
        high: массив значений максимума
        low: массив значений минимума
        close: массив значений закрытия

    Returns:
        массив значений истинного диапазона
    """
    xp = np
    close_prev = xp.roll(close, 1)
    tr = xp.maximum.reduce([high - low, xp.abs(high - close_prev), xp.abs(low - close_prev)])
    tr[0] = high[0] - low[0]
    return tr


def calculate_atr(high, low, close, period):
    """
    Вычисление среднего истинного диапазона (ATR) с использованием MMA и поддержкой GPU.
    """
    tr = calculate_true_range(high, low, close)
    # Использование pandas ewm - эффективный и проверенный способ
    # Он хорошо работает как с NumPy, так и с CuPy-backed Series в будущем.
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_natr(high, low, close, period):
    """
    Вычисление нормализованного среднего истинного диапазона (NATR)
    
    Args:
        high: массив значений максимума
        low: массив значений минимума
        close: массив значений закрытия
        period: период для вычисления NATR
    
    Returns:
        массив значений NATR в процентах
    """
    atr = calculate_atr(high, low, close, period)
    # Используем np.divide для безопасного деления на ноль
    natr = np.divide(atr, close, out=np.full_like(close, np.nan), where=close!=0) * 100
    return natr


def generate_signals(df, params):
    """
    Генерация сигналов на основе параметров стратегии
    
    Args:
        df: DataFrame с рыночными данными (time, open, high, low, close, volume, long_prints, short_prints)
        params: словарь с параметрами стратегии:
            - vol_period: период для анализа объема (по умолчанию 20)
            - vol_pctl: порог объема в процентах (по умолчанию 1.0)
            - range_period: период для анализа диапазона (по умолчанию 20)
            - rng_pctl: порог диапазона в процентах (по умолчанию 1.0)
            - natr_period: период для NATR (по умолчанию 10)
            - natr_min: минимальный порог NATR в процентах (по умолчанию 0.35)
            - lookback_period: период для фильтра роста (по умолчанию 20)
            - min_growth_pct: минимальный порог роста в процентах (по умолчанию 1.0)
            - stop_loss_pct: стоп-лосс в процентах (по умолчанию 2.0)
            - take_profit_pct: тейк-профит в процентах (по умолчанию 4.0)
    
    Returns:
        список индексов, где сгенерированы сигналы
    """
    # Проверяем, что DataFrame не пуст
    if df.empty:
        return []
    
    # Извлекаем параметры с значениями по умолчанию и округляем float параметры до 2 знаков после запятой
    vol_period = params.get("vol_period", 20)
    vol_pctl = round(params.get("vol_pctl", 1.0), 2) / 100  # Преобразуем в доли и округляем до 2 знаков
    range_period = params.get("range_period", 20)
    rng_pctl = round(params.get("rng_pctl", 1.0), 2) / 100  # Преобразуем в доли и округляем до 2 знаков
    natr_period = params.get("natr_period", 10)
    natr_min = round(params.get("natr_min", 0.35), 2)  # Уже в процентах, округляем до 2 знаков
    lookback_period = params.get("lookback_period", 20)
    min_growth_pct = round(params.get("min_growth_pct", 1.0), 2) / 100 # Преобразуем в доли и округляем до 2 знаков
    
    # Проверяем наличие необходимых столбцов
    required_columns = ['high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Столбец {col} отсутствует в DataFrame")
    
    # Извлекаем массивы для вычислений
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    volume = df['volume'].values

    # Вычисляем дополнительные столбцы
    # Диапазон цены (range)
    price_range = high_prices - low_prices
    
    # Вычисляем NATR
    natr_values = calculate_natr(
        high_prices, low_prices, close_prices,
        natr_period
    )
    
    # Вычисляем процентный рост за lookback_period
    
    # Определяем GPU и CPU функции для вычисления роста
    growth_values = np.zeros(len(close_prices))
    growth_values[lookback_period:] = (
        (close_prices[lookback_period:] - close_prices[:-lookback_period]) /
        close_prices[:-lookback_period]
    )
    
    # Создаем скользящие процентили для объема и диапазона
    # Используем rolling с квантилями для эффективного вычисления процентилей
    vol_rolling = pd.Series(volume).rolling(window=vol_period, min_periods=1)
    range_rolling = pd.Series(price_range).rolling(window=range_period, min_periods=1)
    
    volume_percentiles = vol_rolling.quantile(vol_pctl)  # vol_pctl уже в долях
    range_percentiles = range_rolling.quantile(rng_pctl) # rng_pctl уже в долях
    
    # Создаем условия для всех строк сразу (векторизованно)
    low_vol_condition = volume <= volume_percentiles
    narrow_rng_condition = price_range <= range_percentiles
    high_natr_condition = natr_values > natr_min
    growth_condition = growth_values >= min_growth_pct
    
    # Объединяем первые 4 условия
    signal_conditions = low_vol_condition & narrow_rng_condition & high_natr_condition & growth_condition
    
    # Устанавливаем в False первые несколько индексов, где условия не могут быть вычислены корректно
    min_period = max(vol_period, range_period, natr_period, lookback_period)
    signal_conditions[:min_period] = False
    
    # Добавляем дополнительную проверку: если NATR не определен (NaN), не генерируем сигнал
    natr_is_nan = np.isnan(natr_values)
    signal_conditions = signal_conditions & ~natr_is_nan
    
    # Возвращаем индексы, где выполнены все условия
    # Возвращаем целочисленные позиции индексов, а не сами значения индекса
    return np.where(signal_conditions)[0].tolist()
