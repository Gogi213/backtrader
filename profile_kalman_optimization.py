"""
Профилирование оптимизации Kalman фильтра
Сравнивает производительность pykalman и Numba-реализации
"""
import cProfile
import pstats
import numpy as np
import time

# Добавляем путь к проекту
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.fast_kalman import fast_kalman_1d

try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False
    print("Warning: pykalman not available, comparison tests will be skipped")


def profile_pykalman(n_samples=100000):
    """Профилирование pykalman реализации"""
    if not PYKALMAN_AVAILABLE:
        print("PyKalman not available, skipping profiling")
        return None
    
    print(f"Profiling PyKalman implementation with {n_samples} samples")
    
    # Создаем тестовые данные
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    
    # Параметры Kalman
    initial_state = prices[0]
    initial_covariance = 1.0
    process_noise = 0.1
    measurement_noise = 5.0
    
    # Создаем профайлер
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Выполняем pykalman реализацию
    start_time = time.time()
    
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=initial_state,
        initial_state_covariance=initial_covariance,
        observation_covariance=measurement_noise,
        transition_covariance=process_noise
    )
    state_means, state_covs = kf.filter(prices)
    fair_values = np.asarray(state_means)[:, 0]
    fair_value_stds = np.sqrt(np.asarray(state_covs)[:, 0, 0])
    
    end_time = time.time()
    profiler.disable()
    
    execution_time = end_time - start_time
    print(f"PyKalman execution time: {execution_time:.4f}s")
    
    # Сохраняем результаты профилирования
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return execution_time, fair_values, fair_value_stds


def profile_numba_kalman(n_samples=100000):
    """Профилирование Numba реализации"""
    print(f"Profiling Numba implementation with {n_samples} samples")
    
    # Создаем тестовые данные
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    
    # Параметры Kalman
    initial_state = prices[0]
    initial_covariance = 1.0
    process_noise = 0.1
    measurement_noise = 5.0
    
    # Создаем профайлер
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Выполняем Numba реализацию
    start_time = time.time()
    
    fair_values, fair_value_stds = fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    
    end_time = time.time()
    profiler.disable()
    
    execution_time = end_time - start_time
    print(f"Numba execution time: {execution_time:.4f}s")
    
    # Сохраняем результаты профилирования
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return execution_time, fair_values, fair_value_stds


def profile_full_strategy(n_samples=100000):
    """Профилирование полной стратегии с оптимизированным Kalman"""
    print(f"Profiling full strategy with optimized Kalman and {n_samples} samples")
    
    # Создаем тестовые данные
    np.random.seed(42)
    price_changes = np.random.randn(n_samples) * 0.01
    prices = np.cumsum(price_changes) + 100
    times = np.arange(len(prices))
    
    # Создаем данные в формате, ожидаемом стратегией
    data = {
        'time': times,
        'close': prices,
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999
    }
    
    # Создаем стратегию
    from src.strategies.turbo_mean_reversion_strategy import HierarchicalMeanReversionStrategy
    strategy = HierarchicalMeanReversionStrategy(symbol='TEST')
    
    # Профилирование полной стратегии
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    results = strategy.vectorized_process_dataset(data)
    end_time = time.time()
    
    profiler.disable()
    
    execution_time = end_time - start_time
    print(f"Full strategy execution time: {execution_time:.4f}s")
    print(f"Total trades: {results.get('total', 0)}")
    
    # Сохраняем результаты профилирования
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    
    return execution_time


def test_equivalence():
    """Тест эквивалентности результатов"""
    if not PYKALMAN_AVAILABLE:
        print("PyKalman not available, skipping equivalence test")
        return
    
    print("Testing equivalence of PyKalman and Numba implementations...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
    
    # Параметры Kalman
    initial_state = prices[0]
    initial_covariance = 1.0
    process_noise = 0.1
    measurement_noise = 5.0
    
    # PyKalman реализация
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=initial_state,
        initial_state_covariance=initial_covariance,
        observation_covariance=measurement_noise,
        transition_covariance=process_noise
    )
    state_means, state_covs = kf.filter(prices)
    fair_values_pykalman = np.asarray(state_means)[:, 0]
    fair_value_stds_pykalman = np.sqrt(np.asarray(state_covs)[:, 0, 0])
    
    # Numba реализация
    fair_values_numba, fair_value_stds_numba = fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    
    # Проверка эквивалентности
    values_diff = np.max(np.abs(fair_values_pykalman - fair_values_numba))
    stds_diff = np.max(np.abs(fair_value_stds_pykalman - fair_value_stds_numba))
    
    print(f"Max difference in fair values: {values_diff:.2e}")
    print(f"Max difference in fair value stds: {stds_diff:.2e}")
    
    if values_diff < 1e-6 and stds_diff < 1e-6:
        print("✅ Results are equivalent (difference < 1e-6)")
    else:
        print("⚠️ Results differ significantly")
    
    return values_diff, stds_diff


def main():
    """Основная функция профилирования"""
    print("=" * 80)
    print("Kalman Optimization Profiling")
    print("=" * 80)
    
    # Параметры тестирования
    n_samples = 100000  # 100K образцов
    
    print(f"Testing with {n_samples} samples")
    print()
    
    # Тест эквивалентности
    test_equivalence()
    print()
    
    # Профилирование pykalman
    if PYKALMAN_AVAILABLE:
        print("1. PyKalman Implementation:")
        pykalman_time, _, _ = profile_pykalman(n_samples)
        print()
    
    # Профилирование Numba
    print("2. Numba Implementation:")
    numba_time, _, _ = profile_numba_kalman(n_samples)
    print()
    
    # Вычисление выигрыша
    if PYKALMAN_AVAILABLE:
        speedup = pykalman_time / numba_time if numba_time > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {pykalman_time - numba_time:.4f}s ({(1 - numba_time/pykalman_time)*100:.1f}%)")
        print()
    
    # Профилирование полной стратегии
    print("3. Full Strategy with Optimized Kalman:")
    full_strategy_time = profile_full_strategy(n_samples)
    print()
    
    # Итоги
    print("=" * 80)
    print("SUMMARY:")
    if PYKALMAN_AVAILABLE:
        print(f"PyKalman time: {pykalman_time:.4f}s")
        print(f"Numba time: {numba_time:.4f}s")
        print(f"Kalman Speedup: {speedup:.2f}x")
    print(f"Full Strategy time: {full_strategy_time:.4f}s")
    print("=" * 80)
    
    # Сохранение детальной статистики в файлы
    print("\nSaving detailed profiling stats to files...")
    
    # Numba реализация
    profiler_numba = cProfile.Profile()
    profiler_numba.enable()
    profile_numba_kalman(n_samples)
    profiler_numba.disable()
    
    stats_numba = pstats.Stats(profiler_numba)
    stats_numba.dump_stats('profile_numba_kalman.prof')
    
    # Полная стратегия
    profiler_strategy = cProfile.Profile()
    profiler_strategy.enable()
    profile_full_strategy(n_samples)
    profiler_strategy.disable()
    
    stats_strategy = pstats.Stats(profiler_strategy)
    stats_strategy.dump_stats('profile_optimized_strategy.prof')
    
    print("✅ Profiling files saved:")
    print("   - profile_numba_kalman.prof")
    print("   - profile_optimized_strategy.prof")
    print("\nTo view detailed stats:")
    print("   python -m pstats profile_numba_kalman.prof")
    print("   python -m pstats profile_optimized_strategy.prof")


if __name__ == "__main__":
    main()