# Спринт: Оптимизация HMM регрессии

## Обзор

**Цель спринта:** Оптимизировать избыточные вычисления в HMM разделе стратегии
**Длительность:** 1-2 дня
**Приоритет:** Критический
**Ожидаемый выигрыш:** 20-100x на HMM разделе, потенциально 2x общее ускорение

## Проблема

### Текущая НЕЭФФЕКТИВНАЯ реализация

```python
# Строки 315-325 и 560-567 в turbo_mean_reversion_strategy.py
windows = create_rolling_windows(price_changes_test, self.hmm_window_size)
regime_probs = hmm_model.predict_proba(windows.reshape(-1, 1))  # O(n * window_size)
regime_probs = regime_probs.reshape(n_windows, self.hmm_window_size, n_comp)
regime_probs_last = regime_probs[:, -1, :]  # Используем только последние элементы!
```

### Анализ проблемы

1. **Избыточные вычисления:** `predict_proba()` вызывается на всех элементах всех окон
2. **Сложность:** `O(n * window_size)` вместо `O(n)`
3. **Память:** Создается массив `(n_windows * window_size, 1)`
4. **Используется только:** Последние элементы каждого окна `regime_probs[:, -1, :]`

### Пример масштаба проблемы
- **Для n=100K баров, window=30:** 3M+ предсказаний вместо 100K
- **Для n=1M баров, window=30:** 30M+ предсказаний вместо 1M
- **Коэффициент избыточности:** 30x для window_size=30

## Решение

### ОПТИМИЗИРОВАННАЯ реализация

```python
# Извлекать только последние разности на окно
last_diffs = price_changes_test[self.hmm_window_size-1:]  # Только последние элементы
regime_probs_last = hmm_model.predict_proba(last_diffs.reshape(-1, 1))  # O(n)
p_trend = regime_probs_last[:, 0]
p_sideways = regime_probs_last[:, 1]
p_dead = regime_probs_last[:, 2]
```

### Анализ решения

1. **Устранение избыточности:** Предсказания только для необходимых элементов
2. **Сложность:** `O(n)` вместо `O(n * window_size)`
3. **Память:** Массив `(n_windows, n_components)` вместо `(n_windows * window_size, 1)`
4. **Эквивалентность результатов:** Математически идентично исходной реализации

## План спринта

### Задача 1: Анализ текущей реализации (0.5 дня)
- [ ] Изучить текущую реализацию в `turbo_mean_reversion_strategy.py`
- [ ] Идентифицировать все места с HMM оптимизацией
- [ ] Создать тестовый набор данных для валидации

### Задача 2: Разработка оптимизированной реализации (0.5 дня)
- [ ] Реализовать оптимизированную версию HMM предсказаний
- [ ] Заменить неэффективный код в обоих методах:
  - `vectorized_process_dataset()` (строки 315-325)
  - `turbo_process_dataset()` (строки 560-567)
- [ ] Обеспечить обратную совместимость

### Задача 3: Валидация и тестирование (0.5 дня)
- [ ] Создать тесты для сравнения результатов до/после оптимизации
- [ ] Проверить эквивалентность результатов на различных наборах данных
- [ ] Провести производительное тестирование

### Задача 4: Документация и отчет (0.5 дня)
- [ ] Обновить код с комментариями об оптимизации
- [ ] Создать отчет о результатах спринта
- [ ] Обновить бэклог

## Технические детали

### Файлы для изменения

1. **`src/strategies/turbo_mean_reversion_strategy.py`**
   - Метод `vectorized_process_dataset()` (строки 315-325)
   - Метод `turbo_process_dataset()` (строки 560-567)

### Код для замены

#### В методе `vectorized_process_dataset()` (строки 315-325)

**ДО:**
```python
if n >= self.hmm_window_size and len(price_changes_train) > 0:
    windows = create_rolling_windows(price_changes_test, self.hmm_window_size)
    if windows.size > 0:
        # original code predicted per-element; keep the same reshape/predict_proba behavior
        try:
            regime_probs = hmm_model.predict_proba(windows.reshape(-1, 1))
            # regime_probs shape (n_windows*window_size, n_components)
            # restore to (n_windows, window_size, n_components) if possible
            n_windows = windows.shape[0]
            n_comp = regime_probs.shape[1]
            regime_probs = regime_probs.reshape(n_windows, self.hmm_window_size, n_comp)
            regime_probs_last = regime_probs[:, -1, :]  # last diff in each window

            p_trend = regime_probs_last[:, 0]
            p_sideways = regime_probs_last[:, 1]
            p_dead = regime_probs_last[:, 2]

            trend_mask = p_trend > self.prob_threshold_trend
            sideways_mask = p_sideways > self.prob_threshold_sideways
            # indices align to test_prices positions
            indices = np.arange(len(regime_probs_last)) + self.hmm_window_size
            regimes[indices[trend_mask]] = 'TRADE_DISABLED'
            regimes[indices[sideways_mask]] = 'TRADE_ENABLED'
        except Exception:
            # Fall back: leave regimes as default
            pass
```

**ПОСЛЕ:**
```python
if n >= self.hmm_window_size and len(price_changes_train) > 0:
    # OPTIMIZATION: Extract only last diffs instead of predicting on all window elements
    try:
        # Get only the last difference from each window (O(n) instead of O(n*window_size))
        last_diffs = price_changes_test[self.hmm_window_size-1:]
        regime_probs_last = hmm_model.predict_proba(last_diffs.reshape(-1, 1))
        
        p_trend = regime_probs_last[:, 0]
        p_sideways = regime_probs_last[:, 1]
        p_dead = regime_probs_last[:, 2]

        trend_mask = p_trend > self.prob_threshold_trend
        sideways_mask = p_sideways > self.prob_threshold_sideways
        # indices align to test_prices positions
        indices = np.arange(len(regime_probs_last)) + self.hmm_window_size
        regimes[indices[trend_mask]] = 'TRADE_DISABLED'
        regimes[indices[sideways_mask]] = 'TRADE_ENABLED'
    except Exception:
        # Fall back: leave regimes as default
        pass
```

#### В методе `turbo_process_dataset()` (строки 560-567)

**ДО:**
```python
if n >= self.hmm_window_size and len(price_changes_train) > 0:
    windows = create_rolling_windows(price_changes_test, self.hmm_window_size)
    if windows.size > 0:
        try:
            regime_probs = hmm_model.predict_proba(windows.reshape(-1, 1))
            n_windows = windows.shape[0]
            n_comp = regime_probs.shape[1]
            regime_probs = regime_probs.reshape(n_windows, self.hmm_window_size, n_comp)
            regime_probs_last = regime_probs[:, -1, :]
            p_trend = regime_probs_last[:, 0]
            p_sideways = regime_probs_last[:, 1]
            indices = np.arange(len(regime_probs_last)) + self.hmm_window_size
            trend_mask = p_trend > self.prob_threshold_trend
            sideways_mask = p_sideways > self.prob_threshold_sideways
            regimes[indices[trend_mask]] = 'TRADE_DISABLED'
            regimes[indices[sideways_mask]] = 'TRADE_ENABLED'
        except Exception:
            pass
```

**ПОСЛЕ:**
```python
if n >= self.hmm_window_size and len(price_changes_train) > 0:
    # OPTIMIZATION: Extract only last diffs instead of predicting on all window elements
    try:
        # Get only the last difference from each window (O(n) instead of O(n*window_size))
        last_diffs = price_changes_test[self.hmm_window_size-1:]
        regime_probs_last = hmm_model.predict_proba(last_diffs.reshape(-1, 1))
        
        p_trend = regime_probs_last[:, 0]
        p_sideways = regime_probs_last[:, 1]
        indices = np.arange(len(regime_probs_last)) + self.hmm_window_size
        trend_mask = p_trend > self.prob_threshold_trend
        sideways_mask = p_sideways > self.prob_threshold_sideways
        regimes[indices[trend_mask]] = 'TRADE_DISABLED'
        regimes[indices[sideways_mask]] = 'TRADE_ENABLED'
    except Exception:
        pass
```

## Тестирование

### Тест эквивалентности результатов

```python
def test_hmm_optimization_equivalence():
    """Тест для проверки эквивалентности результатов до/после оптимизации"""
    # Создать тестовые данные
    np.random.seed(42)
    price_changes = np.random.randn(1000)
    hmm_window_size = 30
    
    # Оригинальная реализация
    windows = create_rolling_windows(price_changes, hmm_window_size)
    regime_probs_orig = hmm_model.predict_proba(windows.reshape(-1, 1))
    n_windows = windows.shape[0]
    n_comp = regime_probs_orig.shape[1]
    regime_probs_orig = regime_probs_orig.reshape(n_windows, hmm_window_size, n_comp)
    regime_probs_last_orig = regime_probs_orig[:, -1, :]
    
    # Оптимизированная реализация
    last_diffs = price_changes[hmm_window_size-1:]
    regime_probs_last_opt = hmm_model.predict_proba(last_diffs.reshape(-1, 1))
    
    # Проверка эквивалентности
    assert np.allclose(regime_probs_last_orig, regime_probs_last_opt, rtol=1e-10)
    print("✅ Результаты эквивалентны")
```

### Тест производительности

```python
def test_hmm_optimization_performance():
    """Тест для измерения выигрыша в производительности"""
    import time
    
    # Создать большой набор данных
    np.random.seed(42)
    price_changes = np.random.randn(100000)  # 100K элементов
    hmm_window_size = 30
    
    # Тест оригинальной реализации
    start_time = time.time()
    windows = create_rolling_windows(price_changes, hmm_window_size)
    regime_probs_orig = hmm_model.predict_proba(windows.reshape(-1, 1))
    n_windows = windows.shape[0]
    n_comp = regime_probs_orig.shape[1]
    regime_probs_orig = regime_probs_orig.reshape(n_windows, hmm_window_size, n_comp)
    regime_probs_last_orig = regime_probs_orig[:, -1, :]
    orig_time = time.time() - start_time
    
    # Тест оптимизированной реализации
    start_time = time.time()
    last_diffs = price_changes[hmm_window_size-1:]
    regime_probs_last_opt = hmm_model.predict_proba(last_diffs.reshape(-1, 1))
    opt_time = time.time() - start_time
    
    # Измерение выигрыша
    speedup = orig_time / opt_time
    print(f"✅ Выигрыш в производительности: {speedup:.2f}x")
    print(f"   Оригинал: {orig_time:.4f}s")
    print(f"   Оптимизация: {opt_time:.4f}s")
    
    return speedup
```

## Ожидаемые результаты

### Производительность
- **Выигрыш:** 20-100x на HMM разделе
- **Общее ускорение:** Потенциально 2x для всей стратегии
- **Экономия памяти:** Сокращение в `window_size` раз

### Качество
- **Эквивалентность результатов:** Математически идентичные результаты
- **Обратная совместимость:** Без изменений в API
- **Стабильность:** Снижение риска ошибок из-за упрощения кода

## Риски и митигация

### Риски
1. **Некорректная оптимизация:** Результаты могут отличаться
2. **Проблемы с индексацией:** Сдвиг индексов может вызвать ошибки
3. **Регрессия:** Оптимизация может сломать существующую логику

### Митигация
1. **Тестирование эквивалентности:** Сравнение результатов до/после
2. **Постепенное внедрение:** Сначала в одном методе, затем в другом
3. **Комплексное тестирование:** Различные размеры данных и параметры

## Критерии успеха

1. **✅ Эквивалентность результатов:** Отклонение < 0.001%
2. **✅ Выигрыш в производительности:** Минимум 10x на HMM разделе
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

Этот спринт нацелен на критическую оптимизацию с минимальным риском и максимальным выигрышем. Успешное выполнение даст значительное ускорение стратегии без потери качества торговых сигналов.