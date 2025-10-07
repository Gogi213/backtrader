"""
Высокопроизводительная 1-D Kalman фильтрация с Numba JIT компиляцией
Оптимизированная замена pykalman для одномерного случая
"""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def fast_kalman_1d(measurements, initial_state, initial_covariance,
                  process_noise, measurement_noise):
    """
    Высокопроизводительная 1-D Kalman фильтрация с Numba JIT компиляцией
    
    Args:
        measurements: массив измерений (numpy.ndarray)
        initial_state: начальное состояние (float)
        initial_covariance: начальная ковариация (float)
        process_noise: шум процесса (Q, float)
        measurement_noise: шум измерения (R, float)
    
    Returns:
        states: отфильтрованные состояния (numpy.ndarray)
        uncertainties: стандартные отклонения (numpy.ndarray)
    """
    # Ensure correct dtype with zero-copy if possible
    y = np.ascontiguousarray(measurements).astype(np.float64)
    n = len(y)
    states = np.zeros(n, dtype=np.float64)
    uncertainties = np.zeros(n, dtype=np.float64)
    
    # Инициализация
    x = initial_state
    P = initial_covariance
    
    for i in range(n):
        # Предсказание (Prediction)
        # Для переходной матрицы [1]: x_pred = x
        x_pred = x
        P_pred = P + process_noise
        
        # Обновление (Update)
        # Kalman gain: K = P_pred / (P_pred + R)
        K = P_pred / (P_pred + measurement_noise)
        
        # Обновление состояния: x = x_pred + K * (measurement - x_pred)
        x = x_pred + K * (y[i] - x_pred)
        
        # Обновление ковариации: P = (1 - K) * P_pred
        P = (1.0 - K) * P_pred
        
        # Сохранение результатов
        states[i] = x
        uncertainties[i] = np.sqrt(P)
    
    # Ensure non-negative uncertainties (avoid -0.0) - vectorized operation
    # Fixed: np.abs with out parameter not supported in numba
    uncertainties = np.abs(uncertainties)
    
    return states, uncertainties


# Batch-версия удалена, так как не используется в текущей реализации.
# При необходимости многопоточной обработки можно использовать joblib.Parallel
# с одиночной функцией fast_kalman_1d.


def test_fast_kalman_1d():
    """
    Тест для проверки корректности быстрой Kalman фильтрации
    """
    print("Testing fast Kalman filter...")
    
    # Создаем тестовые данные
    np.random.seed(42)
    n_points = 1000
    true_state = 100.0
    measurements = np.zeros(n_points)
    
    # Генерируем измерения с шумом
    for i in range(n_points):
        # Истинное состояние меняется случайным образом
        true_state += np.random.randn() * 0.1
        # Измерение с шумом
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
    
    print("✅ Fast Kalman filter test passed")
    return True


if __name__ == "__main__":
    test_fast_kalman_1d()