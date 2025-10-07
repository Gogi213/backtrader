# Спринт: Оптимизация Kalman фильтра через Numba

## Обзор

**Цель спринта:** Заменить pykalman на высокопроизводительную Numba-реализацию 1-D Kalman
**Длительность:** 2-3 дня
**Приоритет:** Критический
**Ожидаемый выигрыш:** 5-10x ускорение (96.4% времени выполнения)

## Проблема

### Текущая НЕЭФФЕКТИВНАЯ реализация

```python
# Текущая реализация с pykalman (строки 287-299 и 537-546)
kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=test_prices[0],
    initial_state_covariance=self.initial_kf_cov,
    observation_covariance=self.measurement_noise_r,
    transition_covariance=self.process_noise_q
)
state_means, state_covs = kf.filter(test_prices)
fair_values = np.asarray(state_means)[:, 0]
fair_value_stds = np.sqrt(np.asarray(state_covs)[:, 0, 0])
```

### Анализ проблемы

1. **Избыточная сложность:** pykalman предназначен для многомерных случаев
2. **Накладные расходы:** Общие матричные операции для 1-D случая
3. **Маскированные массивы:** Дополнительные проверки для numpy masked arrays
4. **Сложность:** O(n) но с высокой константой из-за обобщенности

### Пример масштаба проблемы
- **Текущее время:** 8.122с (96.4% общего времени выполнения)
- **Оптимизация:** 5-10x ускорение → 0.8-1.6с
- **Общий выигрыш:** 5-10x ускорение всей стратегии

## Решение

### ОПТИМИЗИРОВАННАЯ Numba реализация

```python
from numba import njit
import numpy as np

@njit
def fast_kalman_1d(measurements, initial_state, initial_covariance, 
                  process_noise, measurement_noise):
    """
    Высокопроизводительная 1-D Kalman фильтрация с Numba JIT компиляцией
    
    Args:
        measurements: массив измерений
        initial_state: начальное состояние
        initial_covariance: начальная ковариация
        process_noise: шум процесса (Q)
        measurement_noise: шум измерения (R)
    
    Returns:
        states: отфильтрованные состояния
        uncertainties: стандартные отклонения
    """
    n = len(measurements)
    states = np.zeros(n, dtype=np.float64)
    uncertainties = np.zeros(n, dtype=np.float64)
    
    # Инициализация
    x = initial_state
    P = initial_covariance
    
    for i in range(n):
        # Предсказание (Prediction)
        x_pred = x  # Для переходной матрицы [1]
        P_pred = P + process_noise
        
        # Обновление (Update)
        K = P_pred / (P_pred + measurement_noise)  # Kalman gain
        x = x_pred + K * (measurements[i] - x_pred)
        P = (1 - K) * P_pred
        
        # Сохранение результатов
        states[i] = x
        uncertainties[i] = np.sqrt(P)
    
    return states, uncertainties
```

### Анализ решения

1. **Специализация:** Оптимизировано для 1-D случая
2. **JIT компиляция:** Numba компилирует в эффективный машинный код
3. **Простота:** Минимум накладных расходов
4. **Эквивалентность:** Математически идентично pykalman для 1-D случая

## План спринта

### Задача 1: Анализ текущей реализации pykalman (0.5 дня)
- [ ] Изучить текущее использование pykalman в стратегии
- [ ] Идентифицировать все места с Kalman фильтрацией
- [ ] Определить параметры и их типы
- [ ] Создать тестовый набор данных для валидации

### Задача 2: Разработка Numba-реализации (1 день)
- [ ] Реализовать быстрый 1-D Kalman фильтр с Numba
- [ ] Обеспечить эквивалентность результатов с pykalman
- [ ] Оптимизировать для типичных размеров данных
- [ ] Добавить обработку граничных случаев

### Задача 3: Интеграция в стратегию (0.5 дня)
- [ ] Заменить pykalman на Numba-реализацию
- [ ] Обновить оба метода:
  - `vectorized_process_dataset()` (строки 287-299)
  - `turbo_process_dataset()` (строки 537-546)
- [ ] Обеспечить обратную совместимость параметров

### Задача 4: Тестирование и валидация (0.5 дня)
- [ ] Создать тесты для сравнения результатов до/после
- [ ] Проверить эквивалентность на различных наборах данных
- [ ] Провести производительное тестирование
- [ ] Валидировать пограничные случаи

### Задача 5: Документация и отчет (0.5 дня)
- [ ] Обновить код с комментариями об оптимизации
- [ ] Создать отчет о результатах спринта
- [ ] Обновить бэклог

## Технические детали

### Файлы для изменения

1. **`src/strategies/turbo_mean_reversion_strategy.py`**
   - Метод `vectorized_process_dataset()` (строки 287-299)
   - Метод `turbo_process_dataset()` (строки 537-546)

### Код для замены

#### В методе `vectorized_process_dataset()` (строки 287-299)

**ДО:**
```python
# STEP 2: Kalman filter
kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=test_prices[0] if len(test_prices) > 0 else self.initial_kf_mean,
    initial_state_covariance=self.initial_kf_cov,
    observation_covariance=self.measurement_noise_r,
    transition_covariance=self.process_noise_q
)

if len(test_prices) == 0:
    raise ValueError("No test data after train/test split")

state_means, state_covs = kf.filter(test_prices)
fair_values = np.asarray(state_means)[:, 0]
fair_value_stds = np.sqrt(np.asarray(state_covs)[:, 0, 0])
```

**ПОСЛЕ:**
```python
# STEP 2: Fast Kalman filter with Numba JIT compilation
if len(test_prices) == 0:
    raise ValueError("No test data after train/test split")

# OPTIMIZATION: Use fast Numba-compiled 1D Kalman filter
initial_state = test_prices[0] if len(test_prices) > 0 else self.initial_kf_mean
fair_values, fair_value_stds = fast_kalman_1d(
    test_prices,
    initial_state,
    self.initial_kf_cov,
    self.process_noise_q,
    self.measurement_noise_r
)
```

#### В методе `turbo_process_dataset()` (строки 537-546)

**ДО:**
```python
# Kalman filter
kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=test_prices[0] if len(test_prices) > 0 else self.initial_kf_mean,
    initial_state_covariance=self.initial_kf_cov,
    observation_covariance=self.measurement_noise_r,
    transition_covariance=self.process_noise_q
)

state_means, state_covs = kf.filter(test_prices)
fair_values = np.asarray(state_means)[:, 0]
fair_value_stds = np.sqrt(np.asarray(state_covs)[:, 0, 0])
```

**ПОСЛЕ:**
```python
# OPTIMIZATION: Use fast Numba-compiled 1D Kalman filter
initial_state = test_prices[0] if len(test_prices) > 0 else self.initial_kf_mean
fair_values, fair_value_stds = fast_kalman_1d(
    test_prices,
    initial_state,
    self.initial_kf_cov,
    self.process_noise_q,
    self.measurement_noise_r
)
```

### Добавление импорта

В начало файла `src/strategies/turbo_mean_reversion_strategy.py` добавить:
```python
from numba import njit
```

## Тестирование

### Тест эквивалентности результатов

```python
def test_kalman_numba_equivalence():
    """Тест для проверки эквивалентности результатов pykalman и Numba"""
    # Создать тестовые данные
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
```

### Тест производительности

```python
def test_kalman_numba_performance():
    """Тест для измерения выигрыша в производительности"""
    import time
    
    # Создать большой набор данных
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
    
    return speedup
```

## Ожидаемые результаты

### Производительность
- **Выигрыш:** 5-10x на Kalman разделе
- **Общее ускорение:** 5-10x для всей стратегии
- **Экономия памяти:** Сокращение временных массивов

### Качество
- **Эквивалентность результатов:** Математически идентичные результаты
- **Обратная совместимость:** Без изменений в API
- **Стабильность:** Упрощение кода снижает риск ошибок

## Риски и митигация

### Риски
1. **Некорректная реализация:** Результаты могут отличаться от pykalman
2. **Проблемы с Numba:** Совместимость с разными версиями Python
3. **Потеря гибкости:** Numba реализация менее гибкая

### Митигация
1. **Тестирование эквивалентности:** Всестороннее сравнение результатов
2. **Постепенное внедрение:** Возможность отката к pykalman
3. **Комплексное тестирование:** Различные размеры данных и параметры

## Критерии успеха

1. **✅ Эквивалентность результатов:** Отклонение < 0.001%
2. **✅ Выигрыш в производительности:** Минимум 3x на Kalman разделе
3. **✅ Стабильность:** Нет регрессий в существующих тестах
4. **✅ Чистота кода:** Понятные комментарии и документация

## Отчетность

### Ежедневная отчетность
- **Прогресс по задачам:** Статус каждой задачи спринта
- **Проблемы:** Выявленные трудности и блокеры
- **Результаты тестирования:** Промежуточные результаты

### Итоговый отчет
- **Достигнутый выигрыш:** Измеренное ускорение
- **Качество:** Результаты тестирования эквивалентности
- **Проблемы:** Выявленные проблемы и их решения
- **Следующие шаги:** Рекомендации по дальнейшим оптимизациям

## Заключение

Этот спринт нацелен на критическую оптимизацию с максимальным выигрышем. Успешное выполнение даст 5-10x ускорение всей стратегии без потери качества торговых сигналов. После этого спринта общая производительность стратегии должна улучшиться на 5-10x по сравнению с исходной.