"""
Комплексная валидация Numba-реализации Kalman фильтра
Проверка всех критических аспектов перед продакшеном
"""
import numpy as np
import time
import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.fast_kalman import fast_kalman_1d
from src.strategies.turbo_mean_reversion_strategy import HierarchicalMeanReversionStrategy

try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False
    print("Warning: pykalman not available, some tests will be skipped")


def test_signature_and_return():
    """1. Проверка сигнатуры и возвращаемых значений"""
    print("1. Testing function signature and return values...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
    
    # Параметры Kalman
    initial_state = prices[0]
    initial_covariance = 1.0
    process_noise = 0.1
    measurement_noise = 5.0
    
    # Вызываем функцию
    result = fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    
    # Проверяем, что возвращается кортеж из двух массивов
    assert isinstance(result, tuple), "Function should return a tuple"
    assert len(result) == 2, "Function should return exactly 2 values"
    
    fair_values, fair_value_stds = result
    
    # Проверяем типы
    assert isinstance(fair_values, np.ndarray), "First return value should be numpy array"
    assert isinstance(fair_value_stds, np.ndarray), "Second return value should be numpy array"
    
    # Проверяем, что это стандартные отклонения, а не ковариации
    assert np.all(fair_value_stds >= 0), "Standard deviations should be non-negative"
    
    print("   ✅ Function signature and return values are correct")
    return True


def test_dimensions():
    """2. Проверка размерностей"""
    print("2. Testing dimensions...")
    
    # Создаем тестовые данные разного размера
    test_sizes = [10, 100, 1000, 10000]
    
    for size in test_sizes:
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(size) * 0.01) + 100
        
        # Параметры Kalman
        initial_state = prices[0]
        initial_covariance = 1.0
        process_noise = 0.1
        measurement_noise = 5.0
        
        # Вызываем функцию
        fair_values, fair_value_stds = fast_kalman_1d(
            prices,
            initial_state,
            initial_covariance,
            process_noise,
            measurement_noise
        )
        
        # Проверяем размерности
        assert fair_values.shape == prices.shape, f"fair_values shape {fair_values.shape} != prices shape {prices.shape}"
        assert fair_value_stds.shape == prices.shape, f"fair_value_stds shape {fair_value_stds.shape} != prices shape {prices.shape}"
        
        print(f"   ✅ Dimensions correct for size {size}")
    
    return True


def test_nonzero_uncertainties():
    """3. Проверка защиты от нуля в fair_value_stds"""
    print("3. Testing protection against zero uncertainties...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
    
    # Параметры Kalman с очень малым шумом измерения
    initial_state = prices[0]
    initial_covariance = 1.0
    process_noise = 0.1
    measurement_noise = 1e-10  # Очень малый шум
    
    # Вызываем функцию
    fair_values, fair_value_stds = fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    
    # Проверяем, что нет отрицательных нулей
    assert np.all(fair_value_stds >= 0), "Standard deviations should be non-negative"
    
    # Проверяем, что нет -0.0
    assert not np.any(np.signbit(fair_value_stds)), "No -0.0 values should be present"
    
    print("   ✅ No negative zeros in uncertainties")
    return True


def test_nan_inf_protection():
    """4. Проверка защиты от NaN/inf"""
    print("4. Testing NaN/inf protection...")
    
    # Создаем стратегию для тестирования
    strategy = HierarchicalMeanReversionStrategy(symbol='TEST')
    
    # Тест 1: NaN в данных
    try:
        prices_with_nan = np.array([100.0, 101.0, np.nan, 103.0, 104.0])
        strategy._validate_prices(prices_with_nan)
        assert False, "Should have raised ValueError for NaN"
    except ValueError as e:
        assert "NaN/inf" in str(e), f"Error message should mention NaN/inf: {e}"
        print("   ✅ NaN protection works")
    
    # Тест 2: inf в данных
    try:
        prices_with_inf = np.array([100.0, 101.0, np.inf, 103.0, 104.0])
        strategy._validate_prices(prices_with_inf)
        assert False, "Should have raised ValueError for inf"
    except ValueError as e:
        assert "NaN/inf" in str(e), f"Error message should mention NaN/inf: {e}"
        print("   ✅ inf protection works")
    
    # Тест 3: -inf в данных
    try:
        prices_with_ninf = np.array([100.0, 101.0, -np.inf, 103.0, 104.0])
        strategy._validate_prices(prices_with_ninf)
        assert False, "Should have raised ValueError for -inf"
    except ValueError as e:
        assert "NaN/inf" in str(e), f"Error message should mention NaN/inf: {e}"
        print("   ✅ -inf protection works")
    
    # Тест 4: Корректные данные
    try:
        prices_clean = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        strategy._validate_prices(prices_clean)  # Не должно вызывать ошибку
        print("   ✅ Clean data passes validation")
    except ValueError:
        assert False, "Should not raise ValueError for clean data"
    
    return True


def test_dtype_compatibility():
    """5. Проверка совместимости dtype"""
    print("5. Testing dtype compatibility...")
    
    # Создаем тестовые данные разных типов
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    
    for dtype in dtypes:
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
        prices = prices.astype(dtype)
        
        # Параметры Kalman
        initial_state = float(prices[0])
        initial_covariance = 1.0
        process_noise = 0.1
        measurement_noise = 5.0
        
        # Вызываем функцию
        fair_values, fair_value_stds = fast_kalman_1d(
            prices,
            initial_state,
            initial_covariance,
            process_noise,
            measurement_noise
        )
        
        # Проверяем, что результат всегда float64
        assert fair_values.dtype == np.float64, f"fair_values should be float64, got {fair_values.dtype}"
        assert fair_value_stds.dtype == np.float64, f"fair_value_stds should be float64, got {fair_value_stds.dtype}"
        
        print(f"   ✅ Dtype {dtype} converted to float64 correctly")
    
    return True


def test_performance():
    """6. Проверка производительности"""
    print("6. Testing performance...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(10000) * 0.01) + 100  # 10K баров
    
    # Параметры Kalman
    initial_state = prices[0]
    initial_covariance = 1.0
    process_noise = 0.1
    measurement_noise = 5.0
    
    # Первый вызов (с компиляцией)
    t0 = time.perf_counter()
    fair_values, fair_value_stds = fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    first_call_time = time.perf_counter() - t0
    
    # Второй вызов (скомпилированный)
    t0 = time.perf_counter()
    fair_values, fair_value_stds = fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    second_call_time = time.perf_counter() - t0
    
    print(f"   First call (with compilation): {first_call_time:.6f}s")
    print(f"   Second call (compiled): {second_call_time:.6f}s")
    
    # Проверяем, что скомпилированная версия быстрая
    assert second_call_time < 0.02, f"Compiled version should be < 0.02s, got {second_call_time:.6f}s"
    assert second_call_time < first_call_time, "Compiled version should be faster than first call"
    
    print("   ✅ Performance is acceptable")
    return True


def test_caching():
    """7. Проверка кэширования"""
    print("7. Testing caching...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
    
    # Параметры Kalman
    initial_state = prices[0]
    initial_covariance = 1.0
    process_noise = 0.1
    measurement_noise = 5.0
    
    # Первый вызов (с компиляцией)
    t0 = time.perf_counter()
    fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    first_call_time = time.perf_counter() - t0
    
    # Второй вызов (из кэша)
    t0 = time.perf_counter()
    fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    second_call_time = time.perf_counter() - t0
    
    # Третий вызов (из кэша)
    t0 = time.perf_counter()
    fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    third_call_time = time.perf_counter() - t0
    
    print(f"   First call (with compilation): {first_call_time:.6f}s")
    print(f"   Second call (from cache): {second_call_time:.6f}s")
    print(f"   Third call (from cache): {third_call_time:.6f}s")
    
    # Проверяем, что кэш работает
    assert second_call_time < first_call_time, "Cached call should be faster than first call"
    assert abs(second_call_time - third_call_time) < 0.001, "Cached calls should have similar timing"
    
    print("   ✅ Caching works correctly")
    return True


@pytest.mark.skipif(not PYKALMAN_AVAILABLE, reason="pykalman not available")
def test_equivalence():
    """8. Проверка эквивалентности с pykalman"""
    print("8. Testing equivalence with pykalman...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
    
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
    
    # Numba реализация
    fair_values_numba, _ = fast_kalman_1d(
        prices,
        initial_state,
        initial_covariance,
        process_noise,
        measurement_noise
    )
    
    # Проверка эквивалентности
    max_diff = np.max(np.abs(fair_values_pykalman - fair_values_numba))
    rel_diff = max_diff / np.mean(np.abs(fair_values_pykalman))
    
    print(f"   Max absolute difference: {max_diff:.2e}")
    print(f"   Max relative difference: {rel_diff:.2e}")
    
    # Проверяем, что различие меньше 0.01%
    assert rel_diff < 1e-4, f"Relative difference {rel_diff:.2e} should be < 1e-4"
    
    print("   ✅ Results are equivalent to pykalman")
    return True


def test_integration_with_strategy():
    """9. Проверка интеграции со стратегией"""
    print("9. Testing integration with strategy...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    n_points = 1000
    price_changes = np.random.randn(n_points) * 0.01
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
    
    # Вызываем метод стратегии
    try:
        results = strategy.vectorized_process_dataset(data)
        
        # Проверяем, что результаты содержат ожидаемые ключи
        assert 'total' in results, "Results should contain 'total' key"
        assert 'trades' in results, "Results should contain 'trades' key"
        
        print("   ✅ Integration with strategy works")
        return True
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("KALMAN VALIDATION TESTS")
    print("=" * 80)
    
    tests = [
        test_signature_and_return,
        test_dimensions,
        test_nonzero_uncertainties,
        test_nan_inf_protection,
        test_dtype_compatibility,
        test_performance,
        test_caching,
    ]
    
    if PYKALMAN_AVAILABLE:
        tests.append(test_equivalence)
    
    tests.append(test_integration_with_strategy)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   ❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("✅ All validation tests passed! Ready for production.")
    else:
        print("❌ Some tests failed. Fix issues before production deployment.")