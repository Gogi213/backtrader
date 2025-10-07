"""
Тесты для оптимизации HMM регрессии
Проверяет эквивалентность результатов до и после оптимизации
"""
import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

# Импортируем необходимые функции
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.turbo_mean_reversion_strategy import create_rolling_windows


def test_hmm_optimization_equivalence():
    """Тест для проверки эквивалентности результатов до/после оптимизации"""
    print("Running HMM optimization equivalence test...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    price_changes = np.random.randn(1000) * 0.01
    hmm_window_size = 30
    
    # Создаем и обучаем HMM модель
    price_changes_train = price_changes[:500]
    hmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    hmm_model.fit(price_changes_train.reshape(-1, 1))
    
    # Оригинальная реализация (до оптимизации)
    price_changes_test = price_changes[500:]
    n = len(price_changes_test)
    
    if n >= hmm_window_size:
        windows = create_rolling_windows(price_changes_test, hmm_window_size)
        regime_probs_orig = hmm_model.predict_proba(windows.reshape(-1, 1))
        n_windows = windows.shape[0]
        n_comp = regime_probs_orig.shape[1]
        regime_probs_orig = regime_probs_orig.reshape(n_windows, hmm_window_size, n_comp)
        regime_probs_last_orig = regime_probs_orig[:, -1, :]
    else:
        regime_probs_last_orig = np.array([])
    
    # Оптимизированная реализация
    if n >= hmm_window_size:
        last_diffs = price_changes_test[hmm_window_size-1:]
        regime_probs_last_opt = hmm_model.predict_proba(last_diffs.reshape(-1, 1))
    else:
        regime_probs_last_opt = np.array([])
    
    # Проверка эквивалентности
    if len(regime_probs_last_orig) > 0 and len(regime_probs_last_opt) > 0:
        assert np.allclose(regime_probs_last_orig, regime_probs_last_opt, rtol=1e-10), \
            "Results differ between original and optimized implementations"
        print("✅ Results are equivalent (difference < 1e-10)")
    else:
        print("⚠️ Test data too small for meaningful comparison")
    
    return True


def test_hmm_optimization_performance():
    """Тест для измерения выигрыша в производительности"""
    print("Running HMM optimization performance test...")
    
    import time
    
    # Создаем большой набор данных
    np.random.seed(42)
    price_changes = np.random.randn(100000) * 0.01  # 100K элементов
    hmm_window_size = 30
    
    # Создаем и обучаем HMM модель
    price_changes_train = price_changes[:30000]
    hmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    hmm_model.fit(price_changes_train.reshape(-1, 1))
    
    # Тест оригинальной реализации
    price_changes_test = price_changes[30000:]
    
    start_time = time.time()
    windows = create_rolling_windows(price_changes_test, hmm_window_size)
    regime_probs_orig = hmm_model.predict_proba(windows.reshape(-1, 1))
    n_windows = windows.shape[0]
    n_comp = regime_probs_orig.shape[1]
    regime_probs_orig = regime_probs_orig.reshape(n_windows, hmm_window_size, n_comp)
    regime_probs_last_orig = regime_probs_orig[:, -1, :]
    orig_time = time.time() - start_time
    
    # Тест оптимизированной реализации
    start_time = time.time()
    last_diffs = price_changes_test[hmm_window_size-1:]
    regime_probs_last_opt = hmm_model.predict_proba(last_diffs.reshape(-1, 1))
    opt_time = time.time() - start_time
    
    # Измерение выигрыша
    speedup = orig_time / opt_time if opt_time > 0 else float('inf')
    print(f"✅ Performance improvement: {speedup:.2f}x")
    print(f"   Original: {orig_time:.4f}s")
    print(f"   Optimized: {opt_time:.4f}s")
    
    # Проверяем, что оптимизация действительно быстрее
    assert speedup > 1.0, "Optimization should be faster than original implementation"
    
    return speedup


def test_hmm_optimization_with_different_window_sizes():
    """Тест оптимизации с разными размерами окон"""
    print("Running HMM optimization test with different window sizes...")
    
    np.random.seed(42)
    price_changes = np.random.randn(5000) * 0.01
    
    for window_size in [10, 20, 30, 50, 100]:
        print(f"Testing with window_size={window_size}")
        
        # Создаем и обучаем HMM модель
        train_size = len(price_changes) // 2
        price_changes_train = price_changes[:train_size]
        hmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        hmm_model.fit(price_changes_train.reshape(-1, 1))
        
        # Тестовые данные
        price_changes_test = price_changes[train_size:]
        n = len(price_changes_test)
        
        if n >= window_size:
            # Оригинальная реализация
            windows = create_rolling_windows(price_changes_test, window_size)
            regime_probs_orig = hmm_model.predict_proba(windows.reshape(-1, 1))
            n_windows = windows.shape[0]
            n_comp = regime_probs_orig.shape[1]
            regime_probs_orig = regime_probs_orig.reshape(n_windows, window_size, n_comp)
            regime_probs_last_orig = regime_probs_orig[:, -1, :]
            
            # Оптимизированная реализация
            last_diffs = price_changes_test[window_size-1:]
            regime_probs_last_opt = hmm_model.predict_proba(last_diffs.reshape(-1, 1))
            
            # Проверка эквивалентности
            assert np.allclose(regime_probs_last_orig, regime_probs_last_opt, rtol=1e-10), \
                f"Results differ for window_size={window_size}"
            
            print(f"  ✅ Results equivalent for window_size={window_size}")
        else:
            print(f"  ⚠️ Test data too small for window_size={window_size}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("HMM Optimization Tests")
    print("=" * 60)
    
    try:
        # Тест эквивалентности
        test_hmm_optimization_equivalence()
        print()
        
        # Тест производительности
        speedup = test_hmm_optimization_performance()
        print()
        
        # Тест с разными размерами окон
        test_hmm_optimization_with_different_window_sizes()
        print()
        
        print("=" * 60)
        print("✅ All tests passed successfully!")
        print(f"Performance improvement: {speedup:.2f}x")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()