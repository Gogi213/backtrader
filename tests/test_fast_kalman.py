"""
Тесты для быстрой Numba-реализации Kalman фильтра
Проверяет эквивалентность результатов с pykalman и производительность
"""
import numpy as np
import pytest
import time

# Импортируем необходимые функции
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.fast_kalman import fast_kalman_1d

try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False
    print("Warning: pykalman not available, equivalence tests will be skipped")


def test_fast_kalman_basic():
    """Базовый тест быстрого Kalman фильтра"""
    print("Running basic fast Kalman test...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    n_points = 1000
    true_state = 100.0
    measurements = np.zeros(n_points)
    
    # Генерируем измерения с шумом
    for i in range(n_points):
        true_state += np.random.randn() * 0.1
        measurements[i] = true_state + np.random.randn() * 1.0
    
    # Параметры Kalman
    initial_state = measurements[0]
    initial_covariance = 1.0
    process_noise = 0.01
    measurement_noise = 1.0
    
    # Применяем быстрый Kalman фильтр
    states, uncertainties = fast_kalman_1d(
        measurements,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    
    # Проверяем результаты
    assert len(states) == n_points, "Incorrect number of states"
    assert len(uncertainties) == n_points, "Incorrect number of uncertainties"
    assert np.all(np.isfinite(states)), "States contain non-finite values"
    assert np.all(np.isfinite(uncertainties)), "Uncertainties contain non-finite values"
    assert np.all(uncertainties >= 0), "Uncertainties should be non-negative"
    
    print("✅ Basic fast Kalman test passed")
    return True


@pytest.mark.skipif(not PYKALMAN_AVAILABLE, reason="pykalman not available")
def test_kalman_pykalman_equivalence():
    """Тест для проверки эквивалентности результатов pykalman и Numba"""
    print("Running Kalman equivalence test...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    n_points = 1000
    prices = np.cumsum(np.random.randn(n_points) * 0.01) + 100
    
    # Параметры Kalman
    initial_state = prices[0]
    initial_covariance = 1.0
    process_noise = 0.1
    measurement_noise = 5.0
    
    # Pykalman реализация
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
    assert np.allclose(fair_values_pykalman, fair_values_numba, rtol=1e-6), \
        "Fair values differ between implementations"
    assert np.allclose(fair_value_stds_pykalman, fair_value_stds_numba, rtol=1e-6), \
        "Fair value stds differ between implementations"
    
    print("✅ Results are equivalent (difference < 1e-6)")
    return True


@pytest.mark.skipif(not PYKALMAN_AVAILABLE, reason="pykalman not available")
def test_kalman_performance():
    """Тест для измерения выигрыша в производительности"""
    print("Running Kalman performance test...")
    
    # Создаем большой набор данных
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100000) * 0.01) + 100  # 100K элементов
    
    # Параметры Kalman
    initial_state = prices[0]
    initial_covariance = 1.0
    process_noise = 0.1
    measurement_noise = 5.0
    
    # Тест pykalman
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
    fair_values_pykalman = np.asarray(state_means)[:, 0]
    pykalman_time = time.time() - start_time
    
    # Тест Numba
    start_time = time.time()
    fair_values_numba, _ = fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    numba_time = time.time() - start_time
    
    # Измерение выигрыша
    speedup = pykalman_time / numba_time
    print(f"✅ Performance improvement: {speedup:.2f}x")
    print(f"   PyKalman: {pykalman_time:.4f}s")
    print(f"   Numba: {numba_time:.4f}s")
    
    # Проверяем, что оптимизация действительно быстрее
    assert speedup > 1.0, "Numba implementation should be faster than pykalman"
    
    return speedup


def test_kalman_with_different_parameters():
    """Тест с разными параметрами Kalman"""
    print("Running Kalman test with different parameters...")
    
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
    
    # Различные наборы параметров
    param_sets = [
        (1.0, 0.1, 5.0),    # стандартные параметры
        (0.5, 0.05, 2.0),   # меньшие значения
        (2.0, 0.2, 10.0),   # большие значения
        (1.0, 0.01, 1.0),   # малый шум процесса
        (1.0, 0.5, 0.1),    # малый шум измерения
    ]
    
    for i, (initial_cov, process_noise, measurement_noise) in enumerate(param_sets):
        print(f"  Testing parameter set {i+1}: cov={initial_cov}, Q={process_noise}, R={measurement_noise}")
        
        # Numba реализация
        states, uncertainties = fast_kalman_1d(
            prices,
            prices[0],
            initial_cov,
            process_noise,
            measurement_noise
        )
        
        # Проверяем результаты
        assert len(states) == len(prices), f"Incorrect number of states for param set {i+1}"
        assert len(uncertainties) == len(prices), f"Incorrect number of uncertainties for param set {i+1}"
        assert np.all(np.isfinite(states)), f"States contain non-finite values for param set {i+1}"
        assert np.all(np.isfinite(uncertainties)), f"Uncertainties contain non-finite values for param set {i+1}"
        assert np.all(uncertainties >= 0), f"Uncertainties should be non-negative for param set {i+1}"
        
        print(f"    ✅ Parameter set {i+1} passed")
    
    print("✅ All parameter sets passed")
    return True


def test_kalman_edge_cases():
    """Тест граничных случаев"""
    print("Running Kalman edge cases test...")
    
    # Тест 1: Пустой массив измерений
    try:
        states, uncertainties = fast_kalman_1d(
            np.array([]),
            100.0,
            1.0,
            0.1,
            1.0
        )
        assert len(states) == 0, "Empty input should produce empty output"
        print("  ✅ Empty input test passed")
    except Exception as e:
        print(f"  ❌ Empty input test failed: {e}")
        raise
    
    # Тест 2: Одно измерение
    try:
        states, uncertainties = fast_kalman_1d(
            np.array([100.0]),
            100.0,
            1.0,
            0.1,
            1.0
        )
        assert len(states) == 1, "Single input should produce single output"
        assert np.isfinite(states[0]), "State should be finite"
        assert np.isfinite(uncertainties[0]), "Uncertainty should be finite"
        print("  ✅ Single input test passed")
    except Exception as e:
        print(f"  ❌ Single input test failed: {e}")
        raise
    
    # Тест 3: Нулевые шумы
    try:
        states, uncertainties = fast_kalman_1d(
            np.array([100.0, 101.0, 102.0]),
            100.0,
            1.0,
            0.0,  # нулевой шум процесса
            0.0   # нулевой шум измерения
        )
        assert len(states) == 3, "Should produce 3 outputs"
        assert np.all(np.isfinite(states)), "States should be finite"
        print("  ✅ Zero noise test passed")
    except Exception as e:
        print(f"  ❌ Zero noise test failed: {e}")
        raise
    
    print("✅ All edge cases passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Fast Kalman Filter Tests")
    print("=" * 60)
    
    try:
        # Базовый тест
        test_fast_kalman_basic()
        print()
        
        # Тест эквивалентности с pykalman
        if PYKALMAN_AVAILABLE:
            test_kalman_pykalman_equivalence()
            print()
        
        # Тест производительности
        if PYKALMAN_AVAILABLE:
            speedup = test_kalman_performance()
            print()
        
        # Тест с разными параметрами
        test_kalman_with_different_parameters()
        print()
        
        # Тест граничных случаев
        test_kalman_edge_cases()
        print()
        
        print("=" * 60)
        print("✅ All tests passed successfully!")
        if PYKALMAN_AVAILABLE:
            print(f"Performance improvement: {speedup:.2f}x")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()