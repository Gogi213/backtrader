"""
UltraFastKlinesHandler — единый вход для загрузки klines в numpy
Numba-оптимизирован, без pandas, без GUI, без дублирования
Author: HFT System (optimized)
"""
import numpy as np
import os
from typing import Dict, Any, Tuple
from numba import njit, prange
import warnings
import polars as pl

warnings.filterwarnings('ignore', category=UserWarning)


class NumpyKlinesData:
    """
    Ultra-fast numpy-based klines data structure
    Replaces pandas DataFrame with pure numpy arrays
    """

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        self.data = data_dict
        self.columns = list(data_dict.keys())
        self.length = len(data_dict.get('time', []))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self.data[key]
        if isinstance(key, slice):
            new_data = {k: v[key] for k, v in self.data.items()}
            return NumpyKlinesData(new_data)
        raise TypeError(f"Unsupported key type: {type(key)}")

    def __len__(self) -> int:
        return self.length

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def head(self, n: int = 5) -> 'NumpyKlinesData':
        new_data = {k: v[:n] if len(v) > 0 else v for k, v in self.data.items()}
        return NumpyKlinesData(new_data)

    def sort_values(self, by: str) -> 'NumpyKlinesData':
        sort_indices = np.argsort(self.data[by])
        new_data = {k: v[sort_indices] for k, v in self.data.items()}
        return NumpyKlinesData(new_data)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return self.data.copy()


# === Numba-optimized helpers ===

@njit
def _validate_klines_data(times: np.ndarray, opens: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> bool:
    if len(times) == 0:
        return False
    for i in range(len(closes)):
        if closes[i] <= 0:
            return False
    for i in range(len(volumes)):
        if volumes[i] < 0:
            return False
    return True


@njit
def _calculate_derived_fields(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                              closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    price_range = np.empty(n, dtype=np.float64)
    body_size = np.empty(n, dtype=np.float64)

    for i in prange(n):
        price_range[i] = highs[i] - lows[i]
        body_size[i] = abs(closes[i] - opens[i])

    return price_range, body_size


@njit
def _calculate_bollinger_bands(closes: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(closes)
    sma = np.empty(n, dtype=np.float64)
    upper_band = np.empty(n, dtype=np.float64)
    lower_band = np.empty(n, dtype=np.float64)

    for i in range(period - 1):
        sma[i] = np.nan
        upper_band[i] = np.nan
        lower_band[i] = np.nan

    for i in range(period - 1, n):
        sum_close = 0.0
        for j in range(i - period + 1, i + 1):
            sum_close += closes[j]
        sma[i] = sum_close / period

        sum_sq_diff = 0.0
        for j in range(i - period + 1, i + 1):
            diff = closes[j] - sma[i]
            sum_sq_diff += diff * diff
        std = np.sqrt(sum_sq_diff / period)

        upper_band[i] = sma[i] + std_dev * std
        lower_band[i] = sma[i] - std_dev * std

    return sma, upper_band, lower_band


@njit(cache=True)
def _is_sorted(arr: np.ndarray) -> bool:
    """Checks if a 1D numpy array is sorted in non-decreasing order."""
    for i in range(len(arr) - 1):
        if arr[i] > arr[i+1]:
            return False
    return True


# === Основной обработчик ===

class UltraFastKlinesHandler:
    """
    Единый вход для загрузки klines в numpy
    Numba-оптимизирован, без pandas, без GUI
    """

    def __init__(self):
        self.required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    def load_klines(self, csv_path: str) -> NumpyKlinesData:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Klines data file not found: {csv_path}")

        print(f"[UltraFast] Loading klines from {csv_path}...")

        try:
            df = pl.read_csv(csv_path)
            # Standardize column names (e.g., 'Volume' -> 'volume')
            df = df.rename({col: col.lower() for col in df.columns})

            times = df['time'].to_numpy().astype(np.int64)
            opens = df['open'].to_numpy()
            highs = df['high'].to_numpy()
            lows = df['low'].to_numpy()
            closes = df['close'].to_numpy()
            volumes = df['volume'].to_numpy()
        except Exception as e:
            raise IOError(f"Failed to load klines data from {csv_path} using Polars. Error: {e}")

        if not _validate_klines_data(times, opens, highs, lows, closes, volumes):
            raise ValueError("Invalid klines data found")

        data = NumpyKlinesData({
            'time': times,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        # Сортировка по времени (только если необходимо)
        if not _is_sorted(data['time']):
            print("[UltraFast] Warning: Klines data is not sorted by time. Sorting now...")
            data = data.sort_values('time')

        # Дополнительные поля
        price_range, body_size = _calculate_derived_fields(opens, highs, lows, closes)
        data.data['price_range'] = price_range
        data.data['body_size'] = body_size
        data.columns.extend(['price_range', 'body_size'])

        print(f"[UltraFast] Loaded {len(data):,} klines")
        print(f"[DATA] Loaded {len(data)} klines, price range {closes.min():.4f} → {closes.max():.4f}")
        return data

    def get_statistics(self, data: NumpyKlinesData) -> dict:
        closes = data['close']
        return {
            'total': len(data),
            'price_min': float(closes.min()),
            'price_max': float(closes.max()),
            'time_start': int(data['time'][0]),
            'time_end': int(data['time'][-1])
        }


# === Дополнительные функции для совместимости ===

def vectorized_bb_calculation(closes: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Векторизованный расчет полос Боллинджера
    
    Args:
        closes: Массив цен закрытия
        period: Период для расчета SMA
        std_dev: Количество стандартных отклонений
        
    Returns:
        Кортеж (SMA, верхняя полоса, нижняя полоса)
    """
    return _calculate_bollinger_bands(closes, period, std_dev)


def vectorized_signal_generation(closes: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray) -> np.ndarray:
    """
    Векторизованная генерация торговых сигналов на основе полос Боллинджера
    
    Args:
        closes: Массив цен закрытия
        upper_band: Верхняя полоса Боллинджера
        lower_band: Нижняя полоса Боллинджера
        
    Returns:
        Массив сигналов: 1 - покупка, -1 - продажа, 0 - держать
    """
    signals = np.zeros_like(closes, dtype=np.int8)
    
    # Сигнал на покупку: цена касается или пересекает нижнюю полосу
    signals[closes <= lower_band] = 1
    
    # Сигнал на продажу: цена касается или пересекает верхнюю полосу
    signals[closes >= upper_band] = -1
    
    return signals


# === Алиасы для совместимости ===
VectorizedKlinesHandler = UltraFastKlinesHandler