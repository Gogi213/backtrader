import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import percentile_filter

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
def calculate_natr_numba(high, low, close, period):
    """Numba-оптимизированный расчет NATR"""
    atr = calculate_atr_numba(high, low, close, period)
    natr = (atr / close) * 100
    return natr

def rolling_quantile_scipy(arr, window, quantile):
    """
    Высокопроизводительный rolling quantile с использованием `scipy.ndimage.percentile_filter`.
    Эта функция написана на C и значительно превосходит наивные реализации.
    """
    # percentile_filter ожидает перцентиль (0-100), а не квантиль (0-1)
    percentile = quantile * 100
    # mode='constant', cval=np.nan обеспечивает обработку границ NaN'ами, как в pandas
    result = percentile_filter(arr, percentile, size=window, mode='constant', cval=np.nan)
    return result

def generate_signals_optimized(high, low, close, volume,
                               vol_period, vol_pctl, range_period, rng_pctl,
                               natr_period, natr_min, lookback_period, min_growth_pct):
    """
    Оптимизированная генерация сигналов:
    - Использует Scipy для rolling quantile
    - Векторизует проверку условий
    - Использует Numba для NATR
    """
    n = len(close)

    # 1. Расчет индикаторов
    price_range = high - low
    natr = calculate_natr_numba(high, low, close, natr_period)  # Numba для NATR

    # 2. Расчет роста цены (векторизованно)
    growth_pct = np.zeros(n, dtype=np.float64)
    growth_pct[lookback_period:] = (close[lookback_period:] - close[:-lookback_period]) / close[:-lookback_period]

    # 3. Rolling quantiles с Scipy (очень быстро)
    volume_percentiles = rolling_quantile_scipy(volume, vol_period, vol_pctl / 100.0)
    range_percentiles = rolling_quantile_scipy(price_range, range_period, rng_pctl / 100.0)

    # 4. Векторизованная проверка условий (БЕЗ цикла!)
    min_period = max(vol_period, range_period, natr_period, lookback_period)

    low_vol = volume <= volume_percentiles
    narrow_range = price_range <= range_percentiles
    high_natr = natr > natr_min
    growth_ok = growth_pct >= (min_growth_pct / 100.0)

    # Комбинируем условия векторно
    signal_conditions = low_vol & narrow_range & high_natr & growth_ok

    # Обнуляем начальный период и NaN
    signal_conditions[:min_period] = False
    signal_conditions[np.isnan(natr) | np.isnan(volume_percentiles) | np.isnan(range_percentiles)] = False

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

    # Вызов оптимизированной функции (pandas + векторизация)
    signals = generate_signals_optimized(
        high, low, close, volume,
        vol_period, vol_pctl, range_period, rng_pctl,
        natr_period, natr_min, lookback_period, min_growth_pct
    )

    # Возвращаем Series только если был DataFrame
    if index is not None:
        return pd.Series(signals, index=index, dtype=bool)
    else:
        return signals
