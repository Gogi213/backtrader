import numpy as np
import pandas as pd
from numba import njit

@njit
def calculate_true_range_numba(high, low, close):
    """Numba-оптимизированный расчет True Range"""
    n = len(high)
    tr = np.zeros(n, dtype=np.float64)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_c = abs(high[i] - close[i-1])
        l_c = abs(low[i] - close[i-1])
        tr[i] = max(h_l, max(h_c, l_c))

    return tr

@njit
def calculate_atr_numba(high, low, close, period):
    """Numba-оптимизированный расчет ATR с EMA"""
    tr = calculate_true_range_numba(high, low, close)
    n = len(tr)
    atr = np.full(n, np.nan, dtype=np.float64)

    if n >= period:
        # Первое значение - простое среднее
        atr[period-1] = np.mean(tr[:period])

        # EMA
        multiplier = 2.0 / (period + 1)
        for i in range(period, n):
            atr[i] = atr[i-1] + multiplier * (tr[i] - atr[i-1])

    return atr

@njit
def calculate_natr_numba(high, low, close, period):
    """Numba-оптимизированный расчет NATR"""
    atr = calculate_atr_numba(high, low, close, period)
    natr = (atr / close) * 100
    return natr

@njit
def rolling_quantile_numba(arr, window, quantile):
    """Быстрый rolling quantile с Numba (interpolation='lower' как в pandas)"""
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        start = i - window + 1
        window_data = arr[start:i+1].copy()
        window_data.sort()

        # Используем interpolation='lower' как в pandas.quantile()
        # Это означает: берем нижнее значение при расчете позиции
        pos = quantile * (len(window_data) - 1)
        lower_idx = int(np.floor(pos))

        result[i] = window_data[lower_idx]

    return result

@njit
def generate_signals_numba(high, low, close, volume,
                           vol_period, vol_pctl, range_period, rng_pctl,
                           natr_period, natr_min, lookback_period, min_growth_pct):
    """Полностью векторизованная генерация сигналов с Numba"""
    n = len(close)

    # 1. Расчет индикаторов
    price_range = high - low
    natr = calculate_natr_numba(high, low, close, natr_period)

    # 2. Расчет роста цены
    growth_pct = np.zeros(n, dtype=np.float64)
    for i in range(lookback_period, n):
        if close[i - lookback_period] != 0:
            growth_pct[i] = (close[i] - close[i - lookback_period]) / close[i - lookback_period]

    # 3. Rolling quantiles с Numba
    volume_percentiles = rolling_quantile_numba(volume, vol_period, vol_pctl / 100.0)
    range_percentiles = rolling_quantile_numba(price_range, range_period, rng_pctl / 100.0)

    # 4. Проверка условий
    signal_conditions = np.zeros(n, dtype=np.bool_)

    min_period = max(vol_period, range_period, natr_period, lookback_period)
    for i in range(min_period, n):
        if np.isnan(natr[i]) or np.isnan(volume_percentiles[i]) or np.isnan(range_percentiles[i]):
            continue

        low_vol = volume[i] <= volume_percentiles[i]
        narrow_range = price_range[i] <= range_percentiles[i]
        high_natr = natr[i] > natr_min
        growth_ok = growth_pct[i] >= (min_growth_pct / 100.0)

        signal_conditions[i] = low_vol and narrow_range and high_natr and growth_ok

    return signal_conditions

def generate_signals(data, params: dict):
    """
    Универсальная обертка для Numba-оптимизированной генерации сигналов
    Поддерживает как DataFrame, так и NumpyKlinesData (zero-copy)
    """
    # Проверяем тип входных данных
    if isinstance(data, pd.DataFrame):
        # DataFrame path (legacy)
        if data.empty:
            return pd.Series(index=data.index, dtype=bool)

        required_columns = ['high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Столбец {col} отсутствует в DataFrame")

        high = data['high'].values.astype(np.float64)
        low = data['low'].values.astype(np.float64)
        close = data['close'].values.astype(np.float64)
        volume = data['volume'].values.astype(np.float64)
        index = data.index
    else:
        # NumpyKlinesData path (zero-copy!)
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        index = None

    # Извлечение параметров
    vol_period = params.get("vol_period", 20)
    vol_pctl = params.get("vol_pctl", 1.0)
    range_period = params.get("range_period", 20)
    rng_pctl = params.get("rng_pctl", 1.0)
    natr_period = params.get("natr_period", 10)
    natr_min = params.get("natr_min", 0.35)
    lookback_period = params.get("lookback_period", 20)
    min_growth_pct = params.get("min_growth_pct", 1.0)

    # Вызов Numba-функции
    signals = generate_signals_numba(
        high, low, close, volume,
        vol_period, vol_pctl, range_period, rng_pctl,
        natr_period, natr_min, lookback_period, min_growth_pct
    )

    # Возвращаем Series только если был DataFrame
    if index is not None:
        return pd.Series(signals, index=index, dtype=bool)
    else:
        return signals
