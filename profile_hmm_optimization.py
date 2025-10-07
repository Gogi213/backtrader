"""
Профилирование оптимизации HMM регрессии
Сравнивает производительность до и после оптимизации
"""
import cProfile
import pstats
import numpy as np
from sklearn.mixture import GaussianMixture
import sys
import os
import time

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.turbo_mean_reversion_strategy import (
    HierarchicalMeanReversionStrategy,
    create_rolling_windows
)


def profile_original_hmm(n_samples=100000, window_size=30):
    """Профилирование оригинальной реализации HMM"""
    print(f"Profiling ORIGINAL HMM implementation with {n_samples} samples, window_size={window_size}")
    
    # Создаем тестовые данные
    np.random.seed(42)
    price_changes = np.random.randn(n_samples) * 0.01
    prices = np.cumsum(price_changes) + 100
    times = np.arange(len(prices))
    
    # Создаем стратегию
    strategy = HierarchicalMeanReversionStrategy(symbol='TEST')
    
    # Создаем профайлер
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Выполняем оригинальную реализацию
    start_time = time.time()
    
    # Разделение на train/test
    train_size = max(int(len(prices) * 0.3), window_size + 100)
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]
    
    # HMM модель
    price_changes_train = np.diff(train_prices)
    hmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    hmm_model.fit(price_changes_train.reshape(-1, 1))
    
    # Оригинальная HMM реализация (до оптимизации)
    price_changes_test = np.diff(test_prices)
    n = len(price_changes_test)
    regimes = np.full(len(test_prices), 'TRADE_DISABLED', dtype=object)
    
    if n >= window_size:
        windows = create_rolling_windows(price_changes_test, window_size)
        if windows.size > 0:
            regime_probs = hmm_model.predict_proba(windows.reshape(-1, 1))
            n_windows = windows.shape[0]
            n_comp = regime_probs.shape[1]
            regime_probs = regime_probs.reshape(n_windows, window_size, n_comp)
            regime_probs_last = regime_probs[:, -1, :]
            
            p_trend = regime_probs_last[:, 0]
            p_sideways = regime_probs_last[:, 1]
            
            trend_mask = p_trend > strategy.prob_threshold_trend
            sideways_mask = p_sideways > strategy.prob_threshold_sideways
            indices = np.arange(len(regime_probs_last)) + window_size
            regimes[indices[trend_mask]] = 'TRADE_DISABLED'
            regimes[indices[sideways_mask]] = 'TRADE_ENABLED'
    
    end_time = time.time()
    profiler.disable()
    
    execution_time = end_time - start_time
    print(f"Original implementation execution time: {execution_time:.4f}s")
    
    # Сохраняем результаты профилирования
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return execution_time


def profile_optimized_hmm(n_samples=100000, window_size=30):
    """Профилирование оптимизированной реализации HMM"""
    print(f"Profiling OPTIMIZED HMM implementation with {n_samples} samples, window_size={window_size}")
    
    # Создаем тестовые данные
    np.random.seed(42)
    price_changes = np.random.randn(n_samples) * 0.01
    prices = np.cumsum(price_changes) + 100
    times = np.arange(len(prices))
    
    # Создаем стратегию
    strategy = HierarchicalMeanReversionStrategy(symbol='TEST')
    
    # Создаем профайлер
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Выполняем оптимизированную реализацию
    start_time = time.time()
    
    # Разделение на train/test
    train_size = max(int(len(prices) * 0.3), window_size + 100)
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]
    
    # HMM модель
    price_changes_train = np.diff(train_prices)
    hmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    hmm_model.fit(price_changes_train.reshape(-1, 1))
    
    # Оптимизированная HMM реализация
    price_changes_test = np.diff(test_prices)
    n = len(price_changes_test)
    regimes = np.full(len(test_prices), 'TRADE_DISABLED', dtype=object)
    
    if n >= window_size:
        # OPTIMIZATION: Extract only last diffs instead of predicting on all window elements
        last_diffs = price_changes_test[window_size-1:]
        regime_probs_last = hmm_model.predict_proba(last_diffs.reshape(-1, 1))
        
        p_trend = regime_probs_last[:, 0]
        p_sideways = regime_probs_last[:, 1]
        
        trend_mask = p_trend > strategy.prob_threshold_trend
        sideways_mask = p_sideways > strategy.prob_threshold_sideways
        indices = np.arange(len(regime_probs_last)) + window_size
        regimes[indices[trend_mask]] = 'TRADE_DISABLED'
        regimes[indices[sideways_mask]] = 'TRADE_ENABLED'
    
    end_time = time.time()
    profiler.disable()
    
    execution_time = end_time - start_time
    print(f"Optimized implementation execution time: {execution_time:.4f}s")
    
    # Сохраняем результаты профилирования
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return execution_time


def profile_full_strategy(n_samples=100000):
    """Профилирование полной стратегии до и после оптимизации"""
    print(f"Profiling FULL strategy with {n_samples} samples")
    
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


def main():
    """Основная функция профилирования"""
    print("=" * 80)
    print("HMM Optimization Profiling")
    print("=" * 80)
    
    # Параметры тестирования
    n_samples = 100000  # 100K образцов
    window_size = 30
    
    print(f"Testing with {n_samples} samples, window_size={window_size}")
    print()
    
    # Профилирование оригинальной реализации
    print("1. Original Implementation:")
    original_time = profile_original_hmm(n_samples, window_size)
    print()
    
    # Профилирование оптимизированной реализации
    print("2. Optimized Implementation:")
    optimized_time = profile_optimized_hmm(n_samples, window_size)
    print()
    
    # Вычисление выигрыша
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {original_time - optimized_time:.4f}s ({(1 - optimized_time/original_time)*100:.1f}%)")
    print()
    
    # Профилирование полной стратегии
    print("3. Full Strategy with Optimized HMM:")
    full_strategy_time = profile_full_strategy(n_samples)
    print()
    
    # Итоги
    print("=" * 80)
    print("SUMMARY:")
    print(f"Original HMM time: {original_time:.4f}s")
    print(f"Optimized HMM time: {optimized_time:.4f}s")
    print(f"HMM Speedup: {speedup:.2f}x")
    print(f"Full Strategy time: {full_strategy_time:.4f}s")
    print("=" * 80)
    
    # Сохранение детальной статистики в файлы
    print("\nSaving detailed profiling stats to files...")
    
    # Оригинальная реализация
    profiler_orig = cProfile.Profile()
    profiler_orig.enable()
    profile_original_hmm(n_samples, window_size)
    profiler_orig.disable()
    
    stats_orig = pstats.Stats(profiler_orig)
    stats_orig.dump_stats('profile_original_hmm.prof')
    
    # Оптимизированная реализация
    profiler_opt = cProfile.Profile()
    profiler_opt.enable()
    profile_optimized_hmm(n_samples, window_size)
    profiler_opt.disable()
    
    stats_opt = pstats.Stats(profiler_opt)
    stats_opt.dump_stats('profile_optimized_hmm.prof')
    
    print("✅ Profiling files saved:")
    print("   - profile_original_hmm.prof")
    print("   - profile_optimized_hmm.prof")
    print("\nTo view detailed stats:")
    print("   python -m pstats profile_original_hmm.prof")
    print("   python -m pstats profile_optimized_hmm.prof")


if __name__ == "__main__":
    main()