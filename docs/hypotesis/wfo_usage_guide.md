# WFO Analyzer — Полное Руководство для Чайников

**Автор:** Claude Code
**Дата:** 2025-10-11
**Для кого:** Трейдер который хочет избежать overfitting и построить робастную стратегию

---

## Содержание

1. [ЧТО ЭТО ТАКОЕ простыми словами](#что-это)
2. [ЗАЧЕМ это нужно](#зачем)
3. [КАК ЭТО РАБОТАЕТ пошагово](#как-работает)
4. [ЧТО ЗА ОКНА и как они работают](#окна)
5. [ЧТО ЗА МЕТРИКИ и как их читать](#метрики)
6. [КАКИЕ ДАННЫЕ использовать](#данные)
7. [КОГДА запускать WFO](#когда)
8. [КАК ЗАПУСТИТЬ пошагово](#запуск)
9. [ЧТО ПОЛУЧИШЬ на выходе](#результат)
10. [ПРИМЕРЫ использования](#примеры)
11. [FAQ - Частые вопросы](#faq)

---

<a name="что-это"></a>
## 1. ЧТО ЭТО ТАКОЕ простыми словами

### Проблема которую решает WFO

**Твоя текущая ситуация:**
```
У тебя есть данные 2020-2024 (5 лет)

Ты запускаешь оптимизацию:
  - Optuna перебирает 1000 параметров
  - Находит "лучшие" параметры: stop_loss=2.5, take_profit=5.0
  - Backtest показывает: Sharpe=3.5, PnL=+500$

Ты думаешь: "Супер! Запущу в live!"

Запускаешь live trading в 2025:
  - Sharpe падает до 0.2
  - PnL = -50$
  - Что пошло не так???
```

**Проблема:** Параметры "заточены" под данные 2020-2024. Они НЕ работают на новых данных 2025!

**Решение:** WFO (Walk-Forward Optimization) — имитирует реальную торговлю на исторических данных.

---

### Что делает WFO простыми словами

**WFO = имитация реального трейдинга**

В реальности ты делаешь так:
1. Анализируешь прошлое (backtest на 2020-2022)
2. Находишь параметры
3. Торгуешь live (2023)
4. Через полгода — переоптимизация на новых данных
5. Repeat

**WFO делает ТО ЖЕ, но на исторических данных:**

```
Твои данные 2020-2024:

WFO Window 1:
  [Train: 2020-2022] → оптимизация → params_1
  [Test: 2023-янв]   → торговля с params_1 → результат_1

WFO Window 2:
  [Train: 2020-2023-июнь] → оптимизация → params_2
  [Test: 2023-июль]       → торговля с params_2 → результат_2

WFO Window 3:
  [Train: 2020-2023-декабрь] → оптимизация → params_3
  [Test: 2024-январь]        → торговля с params_3 → результат_3

...и так далее
```

**Итог:** Ты получаешь РЕАЛИСТИЧНУЮ оценку того, как параметры работают на НОВЫХ данных (которые не участвовали в оптимизации).

---

<a name="зачем"></a>
## 2. ЗАЧЕМ это нужно

### До WFO (твой текущий подход):
```
Данные [=====================================]
         2020                            2024

Оптимизация на ВСЕХ данных → params

Backtest Sharpe: 3.5 ← Это ФЕЙК! Параметры "подогнаны"
Live Sharpe:     0.2 ← Реальность
```

### После WFO:
```
Данные [=====================================]
         2020                            2024

Окно 1: [Train====][Test]
Окно 2:    [Train======][Test]
Окно 3:       [Train========][Test]

OOS Sharpe: 1.5 ← Реалистичная оценка
Live Sharpe: 1.2 ← Близко к OOS! ✅
```

### Зачем конкретно:

1. **Избежать overfitting** — параметры не "подогнаны" под прошлое
2. **Реалистичная оценка** — знаешь как стратегия поведёт себя в live
3. **Робастность** — параметры работают на РАЗНЫХ периодах
4. **Уверенность** — если WFO показал хорошо, в live тоже будет хорошо

---

<a name="как-работает"></a>
## 3. КАК ЭТО РАБОТАЕТ пошагово

### Пошаговый процесс WFO

**Input:** Один датасет (например, BTCUSDT 2020-2024, 1M баров)

**Шаг 1: Резервация Hold-out**
```
Весь датасет: [========================================] 1M баров

Hold-out (10%): [====] ← НЕ ТРОГАЕМ до самого конца!
Данные для WFO: [==================================] 900K баров
```

Hold-out = последние 10% данных. Это "unseen data" для финального теста.

---

**Шаг 2: Создание WFO окон**

WFO analyzer разбивает данные на окна (windows).

**Anchored WFO** (train растёт):
```
900K баров для WFO:

Window 1:
  Train [======]          (40% = 360K bars)
  Test       [==]         (10% = 90K bars)

Window 2:
  Train [==========]      (50% = 450K bars)
  Test           [==]     (10% = 90K bars)

Window 3:
  Train [==============]  (60% = 540K bars)
  Test               [==] (10% = 90K bars)

...
```

**Rolling WFO** (окна скользят):
```
Window 1:
  Train [======]          (фиксированный размер)
  Test       [==]

Window 2:
      Train [======]      (тот же размер, но сдвинут)
      Test       [==]

Window 3:
          Train [======]
          Test       [==]
```

---

**Шаг 3: Оптимизация на каждом окне**

Для КАЖДОГО окна WFO analyzer делает:

```python
# Окно 1
train_data = data[0:360K]      # Train период
test_data = data[360K:450K]    # Test период

# 3.1 Оптимизация на Train (In-Sample)
optimizer = FastStrategyOptimizer(...)
optimizer.optimize(
    data=train_data,
    n_trials=500
)
best_params_1 = optimizer.best_params  # Например: {stop_loss: 2.5, take_profit: 5.0}
IS_sharpe_1 = optimizer.best_value     # Например: 2.8

# 3.2 Тест на Test (Out-of-Sample)
strategy = Strategy(**best_params_1)
results = strategy.backtest(test_data)  # ДАННЫЕ КОТОРЫЕ НЕ ВИДЕЛА ОПТИМИЗАЦИЯ!
OOS_sharpe_1 = results['sharpe_ratio']  # Например: 1.5
OOS_pnl_1 = results['net_pnl']          # Например: +120$

# 3.3 Сравнение
window_1_efficiency = OOS_sharpe_1 / IS_sharpe_1  # 1.5 / 2.8 = 0.536 (53.6%)
```

**Repeat для всех окон (2, 3, 4, ...).**

---

**Шаг 4: Агрегация результатов**

После всех окон собираем метрики:

```python
windows = [
    Window(IS_sharpe=2.8, OOS_sharpe=1.5, OOS_pnl=+120),
    Window(IS_sharpe=3.0, OOS_sharpe=1.8, OOS_pnl=+150),
    Window(IS_sharpe=2.5, OOS_sharpe=1.3, OOS_pnl=+90),
    Window(IS_sharpe=2.9, OOS_sharpe=1.6, OOS_pnl=+110),
]

# WFO Efficiency
mean_IS = (2.8 + 3.0 + 2.5 + 2.9) / 4 = 2.8
mean_OOS = (1.5 + 1.8 + 1.3 + 1.6) / 4 = 1.55
WFO_Efficiency = 1.55 / 2.8 = 0.554 (55.4%) ✅

# Consistency
positive_windows = 4 out of 4
Consistency = 100% ✅

# Stability
std(OOS) = 0.19
Stability = 0.19 / 1.55 = 0.12 ✅ (низкая = хорошо)
```

---

**Шаг 5: Ensemble параметров**

Из всех окон получаем наборы параметров:
```python
Window 1: {stop_loss: 2.5, take_profit: 5.0}
Window 2: {stop_loss: 2.8, take_profit: 4.8}
Window 3: {stop_loss: 2.3, take_profit: 5.2}
Window 4: {stop_loss: 2.6, take_profit: 4.9}

# Финальные параметры = median
final_params = {
    stop_loss: median(2.5, 2.8, 2.3, 2.6) = 2.55,
    take_profit: median(5.0, 4.8, 5.2, 4.9) = 4.95
}
```

---

**Шаг 6: Hold-out тест**

Финальная проверка на "unseen data":

```python
holdout_data = data[900K:1M]  # Последние 10%

strategy = Strategy(**final_params)
holdout_results = strategy.backtest(holdout_data)

holdout_sharpe = 1.4
holdout_pnl = +95$

# Сравнение с In-Sample
holdout_vs_IS = 1.4 / 2.8 = 50%

# Если > 50% → отлично!
# Если > 30% → приемлемо
# Если < 30% → overfit
```

---

<a name="окна"></a>
## 4. ЧТО ЗА ОКНА и как они работают

### Окно (Window) = одна итерация WFO

```
Window состоит из двух частей:

[Train Period] → оптимизация → параметры
[Test Period]  → торговля с найденными параметрами → результат
```

### Типы окон

#### Anchored Windows (растущее train окно)

**Когда использовать:**
- Данных мало в начале периода
- Хочешь учитывать весь исторический опыт
- Рынок эволюционирует медленно

**Визуализация:**
```
Данные: [=============================================]
        2020-01       2022-01       2024-01      2025-01

Window 1:
  Train [=====]
  Test       [=]

Window 2:
  Train [=========]
  Test           [=]

Window 3:
  Train [=============]
  Test               [=]

Window 4:
  Train [=================]
  Test                   [=]
```

**Особенности:**
- Train окно **РАСТЁТ** с каждым окном
- Test окно **ФИКСИРОВАННОГО размера**
- Anchor point = начало данных (2020-01)

**Параметры:**
```python
config = WFOConfig(
    wfo_type='anchored',
    min_train_size=0.4,  # Минимум 40% на train
    test_size=0.1,       # Тест всегда 10%
    step_size=0.1        # Шаг 10%
)

# Создаст окна:
# Window 1: Train 40%, Test 10%
# Window 2: Train 50%, Test 10%
# Window 3: Train 60%, Test 10%
# Window 4: Train 70%, Test 10%
# Window 5: Train 80%, Test 10%
```

---

#### Rolling Windows (скользящее окно)

**Когда использовать:**
- Хочешь адаптироваться к ТЕКУЩИМ условиям рынка
- Старые данные теряют релевантность (crypto, high-freq)
- Быстрая адаптация важнее долгой истории

**Визуализация:**
```
Данные: [=============================================]

Window 1:
  Train [=====]
  Test       [=]

Window 2:
      Train [=====]
      Test       [=]

Window 3:
          Train [=====]
          Test       [=]

Window 4:
              Train [=====]
              Test       [=]
```

**Особенности:**
- Train и Test **ФИКСИРОВАННОГО размера**
- Окна **СКОЛЬЗЯТ** вперёд
- Старые данные "забываются"

**Параметры:**
```python
config = WFOConfig(
    wfo_type='rolling',
    train_size=0.5,   # Train всегда 50%
    test_size=0.1,    # Test всегда 10%
    step_size=0.1     # Шаг 10%
)

# Создаст окна:
# Window 1: Train [0:50%], Test [50:60%]
# Window 2: Train [10:60%], Test [60:70%]
# Window 3: Train [20:70%], Test [70:80%]
# Window 4: Train [30:80%], Test [80:90%]
```

---

### Gap (Embargo Period)

**Что это:** Пауза между Train и Test периодами.

```
Train [======]  GAP [..] Test [==]
```

**Зачем:**
- Избежать information leakage (если есть лаги в данных)
- Реалистичнее имитировать live deployment
- Учесть время на переоптимизацию

**Пример:**
```python
config = WFOConfig(
    gap_size=0.05  # 5% данных
)

# Train: [0:500K bars]
# GAP:   [500K:525K bars] ← НЕ ИСПОЛЬЗУЮТСЯ
# Test:  [525K:625K bars]
```

**Рекомендации:**
- Для crypto: gap = 1-3 дня (0.01-0.03 если данные за год)
- Для stocks: gap = 1-5 дней
- Если нет лагов в данных: gap = 0

---

<a name="метрики"></a>
## 5. ЧТО ЗА МЕТРИКИ и как их читать

### Метрика #1: WFO Efficiency

**Формула:**
```
WFO_Efficiency = mean(OOS_Sharpe_1..N) / mean(IS_Sharpe_1..N)
```

Где:
- `N` = количество WFO окон
- `OOS_Sharpe` = Sharpe на Test периоде (Out-of-Sample)
- `IS_Sharpe` = Sharpe на Train периоде (In-Sample)

**Пример расчёта:**
```python
Windows:
  Window 1: IS=2.8, OOS=1.5
  Window 2: IS=3.0, OOS=1.8
  Window 3: IS=2.5, OOS=1.3
  Window 4: IS=2.9, OOS=1.6

mean_IS = (2.8 + 3.0 + 2.5 + 2.9) / 4 = 2.8
mean_OOS = (1.5 + 1.8 + 1.3 + 1.6) / 4 = 1.55

WFO_Efficiency = 1.55 / 2.8 = 0.554 (55.4%)
```

**Как читать:**

| WFO Efficiency | Оценка | Что делать |
|----------------|--------|------------|
| > 0.7 | 🟢 Отлично | Параметры очень робастны, можно в live |
| 0.5 - 0.7 | 🟢 Хорошо | Приемлемый уровень, можно использовать |
| 0.3 - 0.5 | 🟡 Посредственно | Есть degradation, осторожно |
| < 0.3 | 🔴 Плохо | Сильный overfit, переделать стратегию |

**Что это значит простыми словами:**

- **0.55** = Out-of-sample Sharpe составляет 55% от In-sample Sharpe
- **Degradation** = 45% (100% - 55%)
- **Интерпретация:** Ожидай что в live результаты будут ~50-60% от backtest

**Идеальное значение:**
- 1.0 = нет degradation (нереально)
- 0.7-0.8 = очень хорошо
- 0.5-0.6 = норма для хороших стратегий
- < 0.5 = есть проблемы

---

### Метрика #2: Consistency Score

**Формула:**
```
Consistency = count(positive_OOS_windows) / total_windows
```

**Пример:**
```python
Windows:
  Window 1: OOS_Sharpe = 1.5 (positive ✓)
  Window 2: OOS_Sharpe = 1.8 (positive ✓)
  Window 3: OOS_Sharpe = -0.2 (negative ✗)
  Window 4: OOS_Sharpe = 1.6 (positive ✓)

positive_count = 3
total = 4

Consistency = 3 / 4 = 0.75 (75%)
```

**Как читать:**

| Consistency | Оценка | Что делать |
|-------------|--------|------------|
| = 100% | 🟢 Идеально | Работает на ВСЕХ периодах |
| > 80% | 🟢 Отлично | Высокая универсальность |
| 60-80% | 🟡 Хорошо | Работает на большинстве периодов |
| < 60% | 🔴 Плохо | Не универсальная стратегия |

**Что это значит:**

- **75%** = стратегия прибыльна на 3 из 4 периодов
- **25%** периодов были убыточны
- **Интерпретация:** В live будут периоды просадки, но в целом плюс

**Идеальное значение:**
- 100% = работает всегда (очень редко)
- 80-90% = очень хорошо
- 60-70% = норма

---

### Метрика #3: Stability (Coefficient of Variation)

**Формула:**
```
Stability = std(OOS_Sharpe_1..N) / mean(OOS_Sharpe_1..N)
```

**Пример:**
```python
OOS_Sharpes = [1.5, 1.8, 1.3, 1.6]

mean = 1.55
std = 0.19

Stability = 0.19 / 1.55 = 0.12 (12%)
```

**Как читать:**

| Stability (CV) | Оценка | Что делать |
|----------------|--------|------------|
| < 0.3 | 🟢 Отлично | Стабильные результаты |
| 0.3 - 0.5 | 🟡 Хорошо | Умеренная вариативность |
| 0.5 - 1.0 | 🟡 Посредственно | Высокая вариативность |
| > 1.0 | 🔴 Плохо | Нестабильная стратегия |

**Что это значит:**

- **0.12** = стандартное отклонение 12% от среднего
- **Разброс результатов:** 1.55 ± 0.19 = от 1.36 до 1.74
- **Интерпретация:** Результаты предсказуемы, низкая вариативность

**Идеальное значение:**
- 0 = всегда одинаковые результаты (нереально)
- < 0.2 = очень стабильно
- 0.3-0.5 = норма

---

### Метрика #4: Degradation Ratio

**Формула:**
```
Degradation = 1 - (mean_OOS / mean_IS)
```

**Пример:**
```python
mean_IS = 2.8
mean_OOS = 1.55

Degradation = 1 - (1.55 / 2.8) = 1 - 0.554 = 0.446 (44.6%)
```

**Как читать:**

| Degradation | Оценка | Что делать |
|-------------|--------|------------|
| < 20% | 🟢 Отлично | Минимальная потеря performance |
| 20-40% | 🟢 Хорошо | Приемлемый уровень |
| 40-60% | 🟡 Посредственно | Значительная degradation |
| > 60% | 🔴 Плохо | Сильный overfit |

**Что это значит:**

- **44.6%** = теряешь 44.6% performance от backtest к OOS
- **Ожидай:** Если backtest Sharpe=2.8, в live будет ~1.5
- **Интерпретация:** Норма для алготрейдинга

**Связь с WFO Efficiency:**
```
Degradation ≈ 1 - WFO_Efficiency
```

---

### Сводная таблица интерпретации

**Пример результатов WFO:**
```
WFO Efficiency:  0.554 (55.4%)
Consistency:     75%
Stability:       0.12
Degradation:     44.6%
```

**Вердикт:**
- ✅ WFO Efficiency > 0.5 → Робастная стратегия
- ⚠️ Consistency 75% → Будут периоды просадки
- ✅ Stability < 0.3 → Стабильные результаты
- ✅ Degradation < 50% → Приемлемо

**Итог:** **Стратегия робастна, можно рассмотреть для live.**

---

<a name="данные"></a>
## 6. КАКИЕ ДАННЫЕ использовать

### Вопрос: "Какие данные входят в WFO?"

**Ответ:** СЫРЫЕ данные БЕЗ предварительной оптимизации!

### ❌ НЕПРАВИЛЬНО:

```
Шаг 1: Оптимизируешь стратегию на всех данных
       → находишь "лучшие" параметры

Шаг 2: Запускаешь WFO с этими параметрами
```

**Почему неправильно:** Параметры УЖЕ подогнаны под данные! WFO не покажет реальную картину.

---

### ✅ ПРАВИЛЬНО:

```
Шаг 1: Берёшь СЫРОЙ датасет (BTCUSDT 2020-2024)

Шаг 2: Запускаешь WFO
       → WFO САМ оптимизирует на каждом Train окне
       → WFO САМ тестирует на каждом Test окне

Шаг 3: Получаешь робастные параметры
```

**Почему правильно:** Каждое Train окно оптимизируется НЕЗАВИСИМО. Test данные НЕ ВИДЯТ оптимизацию.

---

### Детальный процесс

**Input для WFO:**
```python
# СЫРОЙ датасет
data = load_klines('data/BTCUSDT_1m.parquet')
# Это просто массив цен/объёмов, БЕЗ параметров

config = WFOConfig(
    strategy_name='ported_from_example',  # Название стратегии
    symbol='BTCUSDT',
    # Параметры стратегии НЕ указываешь!
    # WFO сам их найдёт
)

analyzer = WFOAnalyzer(config, data)
results = analyzer.run_wfo()
```

**Что делает WFO внутри:**
```python
# Для каждого окна:
for window in windows:
    train_data = data[window.train_start:window.train_end]
    test_data = data[window.test_start:window.test_end]

    # ОПТИМИЗАЦИЯ на train
    optimizer = FastStrategyOptimizer(...)
    optimizer.optimize(train_data, n_trials=500)
    best_params = optimizer.best_params  # Находит НОВЫЕ параметры

    # ТЕСТ на test
    strategy = Strategy(**best_params)
    results = strategy.backtest(test_data)  # Test данные НЕ ВИДЕЛИ оптимизацию!
```

---

### Какой датасет выбрать

**Рекомендации:**

1. **Размер:** Минимум 1M баров (лучше 2-5M)
   - Чтобы было достаточно данных для разбиения на окна

2. **Период:** 2-5 лет исторических данных
   - Включает разные рыночные режимы (bull/bear/sideways)

3. **Качество:** Без пропусков, без outliers

4. **Формат:** NumpyKlinesData (твой VectorizedKlinesHandler)

**Пример:**
```python
# Хороший датасет для WFO
data_path = 'data/BTCUSDT_1m_2020-2024.parquet'
# 2M баров, 4 года, разные режимы рынка
```

---

<a name="когда"></a>
## 7. КОГДА запускать WFO

### Timeline твоей работы со стратегией

```
┌─────────────────────────────────────────────────────────┐
│ Этап 1: Разработка стратегии                           │
├─────────────────────────────────────────────────────────┤
│ • Придумываешь логику (entry/exit rules)               │
│ • Пишешь код стратегии                                  │
│ • Определяешь параметры которые будешь оптимизировать  │
│                                                         │
│ НЕ ЗАПУСКАЕШЬ WFO                                      │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│ Этап 2: Быстрая проверка идеи                          │
├─────────────────────────────────────────────────────────┤
│ • Запускаешь простую оптимизацию на 1 датасете         │
│   (FastStrategyOptimizer, 100-200 trials)              │
│ • Смотришь: есть ли edge вообще?                       │
│ • Если Sharpe < 1.0 → идея не работает, дальше не идём │
│                                                         │
│ НЕ ЗАПУСКАЕШЬ WFO (пока рано)                         │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│ Этап 3: WFO VALIDATION ← ЗДЕСЬ ЗАПУСКАЕШЬ WFO         │
├─────────────────────────────────────────────────────────┤
│ • Стратегия показала Sharpe > 1.5 на простом backtest  │
│ • Теперь проверяем РОБАСТНОСТЬ через WFO               │
│ • WFO покажет: работает ли на OOS данных               │
│                                                         │
│ ✅ ЗАПУСКАЕШЬ WFO                                      │
│                                                         │
│ Если WFO Efficiency > 0.5:                             │
│   → Стратегия робастна → идём дальше                   │
│ Если WFO Efficiency < 0.5:                             │
│   → Overfit → упрощаем стратегию → repeat этап 3       │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│ Этап 4: Multi-Dataset Validation                       │
├─────────────────────────────────────────────────────────┤
│ • WFO прошёл на 1 датасете (BTCUSDT)                   │
│ • Теперь проверяем на других символах (ETH, SOL, etc)  │
│ • Запускаешь WFO на топ-10 датасетах                   │
│ • Проверяешь: работает ли универсально                 │
│                                                         │
│ ✅ ЗАПУСКАЕШЬ WFO на каждом датасете                   │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│ Этап 5: Final Ensemble                                 │
├─────────────────────────────────────────────────────────┤
│ • Собираешь параметры со всех WFO runs                 │
│ • Берёшь median/консенсус параметры                    │
│ • Финальный hold-out test                              │
│                                                         │
│ ✅ Используешь результаты WFO                          │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│ Этап 6: Live Trading                                   │
├─────────────────────────────────────────────────────────┤
│ • Запускаешь live с финальными параметрами             │
│ • Мониторишь performance                               │
│ • Если live Sharpe близок к OOS Sharpe → ✅            │
└─────────────────────────────────────────────────────────┘
```

---

### Когда НЕ нужен WFO

❌ **Разработка стратегии** — слишком рано, сначала проверь базовую идею
❌ **Quick backtest** — для быстрых тестов хватит простой оптимизации
❌ **Очень мало данных** — < 500K баров, не хватит для разбиения на окна
❌ **Live monitoring** — WFO для pre-live проверки, не для мониторинга

### Когда НУЖЕН WFO

✅ **Перед live** — ОБЯЗАТЕЛЬНО проверить робастность
✅ **После разработки** — когда стратегия показала хорошие простые backtest
✅ **Multi-dataset validation** — проверка универсальности
✅ **Periodic re-validation** — раз в квартал/год повторять WFO

---

<a name="запуск"></a>
## 8. КАК ЗАПУСТИТЬ пошагово

### Вариант A: Базовый WFO на одном датасете

**Шаг 1: Подготовь данные**

```python
# Убедись что датасет существует
import os
data_path = 'data/BTCUSDT_1m.parquet'

if not os.path.exists(data_path):
    print("❌ Датасет не найден!")
    print("Положи файл в", data_path)
    exit()

print("✅ Датасет найден")
```

---

**Шаг 2: Импорт библиотек**

```python
from src.optimization.wfo_analyzer import WFOAnalyzer, WFOConfig
```

---

**Шаг 3: Создай конфигурацию**

```python
config = WFOConfig(
    # Основное
    strategy_name='ported_from_example',  # Твоя стратегия
    symbol='BTCUSDT',                     # Символ

    # Тип WFO
    wfo_type='anchored',  # 'anchored' или 'rolling'

    # Параметры окон (для anchored)
    min_train_size=0.4,   # Минимум 40% данных на train
    test_size=0.1,        # Тест = 10% данных
    step_size=0.1,        # Шаг 10%
    gap_size=0.0,         # Без embargo (можешь добавить 0.05 = 5%)

    # Hold-out
    holdout_size=0.1,     # Последние 10% не трогаем до конца

    # Оптимизация
    n_trials=500,         # Trials на каждое окно
    objective_metric='sharpe_ratio',
    min_trades=10,
    max_drawdown_threshold=50.0,
    n_jobs=-1             # Использовать все CPU
)
```

**Пояснение параметров:**

- `wfo_type='anchored'` — train растёт, test фиксирован
- `min_train_size=0.4` — первое окно использует 40% данных на train
- `test_size=0.1` — каждое test окно = 10%
- `step_size=0.1` — следующее окно сдвигается на 10%
- `holdout_size=0.1` — последние 10% резервируем для финального теста

**Сколько окон создастся:**
```
Данные для WFO = 100% - holdout = 90%

Окно 1: Train 40%, Test 10% (покрывает 0-50%)
Окно 2: Train 50%, Test 10% (покрывает 0-60%)
Окно 3: Train 60%, Test 10% (покрывает 0-70%)
Окно 4: Train 70%, Test 10% (покрывает 0-80%)
Окно 5: Train 80%, Test 10% (покрывает 0-90%)

Итого: 5 окон
```

---

**Шаг 4: Создай analyzer**

```python
analyzer = WFOAnalyzer(
    config=config,
    data_path=data_path,
    enable_debug=False  # True если хочешь видеть детали
)
```

---

**Шаг 5: Запусти WFO**

```python
print("Запускаем WFO...")
print("Это займёт 10-30 минут в зависимости от данных")
print("-" * 60)

results = analyzer.run_wfo()
```

**Что произойдёт:**
```
[WFO] Loaded 1,000,000 bars from data/BTCUSDT_1m.parquet
[WFO] Strategy: ported_from_example, Symbol: BTCUSDT
[WFO] Type: ANCHORED
[WFO] Reserved hold-out: 100,000 bars (10%)
[WFO] Created 5 windows

--------------------------------------------------------------------------------
Window 1
--------------------------------------------------------------------------------
Train: [0:400000] = 400,000 bars
Test:  [400000:500000] = 100,000 bars

[Train] Optimizing 500 trials...
[Train] Best Sharpe: 2.456
[Train] Best PnL: 245.30
[Train] Trades: 45

[Test]  OOS Sharpe: 1.834
[Test]  OOS PnL: 125.30
[Test]  OOS Trades: 38
[Test]  Window Efficiency: 74.67%

... (окна 2-5)

================================================================================
WFO ANALYSIS SUMMARY
================================================================================

Window Results:
--------------------------------------------------------------------------------
Window   IS Sharpe    OOS Sharpe   OOS PnL      Efficiency
--------------------------------------------------------------------------------
1        2.456        1.834        125.30       74.67%
2        2.687        1.592        98.45        59.25%
3        2.823        2.103        156.78       74.50%
4        2.601        1.456        89.23        55.98%
5        2.745        1.689        110.45       61.53%
--------------------------------------------------------------------------------

Robustness Metrics:
----------------------------------------
WFO Efficiency:    0.658
Consistency Score: 100.00%
Stability (CV):    0.152
Degradation:       34.20%
----------------------------------------

Interpretation:
✅ WFO Efficiency > 0.5 → Strategy is ROBUST
✅ Consistency > 80% → Works on most periods

Final Ensemble Parameters:
--------------------------------------------------------------------------------
  stop_loss_pct: 2.550
  take_profit_pct: 4.950
  entry_logic_mode: Только по принтам
  prints_analysis_period: 3
  prints_threshold_ratio: 2.450
  ...

================================================================================
HOLD-OUT TEST (Final Validation on Unseen Data)
================================================================================
Hold-out size: 100,000 bars

[Hold-out] Sharpe: 1.523
[Hold-out] PnL: 95.45
[Hold-out] Trades: 28
[Hold-out] Win Rate: 64.29%
Hold-out vs IS: 58.53%
✅ Hold-out performance > 50% of IS → GOOD
```

---

**Шаг 6: Сохрани результаты**

```python
analyzer.save_results(results)
# Сохранит в wfo_results/wfo_ported_from_example_BTCUSDT_20251011_143022.json
```

---

**Шаг 7: Интерпретация**

```python
print("\n" + "="*60)
print("ТВОЙ ВЕРДИКТ:")
print("="*60)

if results.wfo_efficiency > 0.5:
    print("✅ Стратегия РОБАСТНА")
    print(f"   WFO Efficiency: {results.wfo_efficiency:.2%}")
    print(f"   Ожидаемый live Sharpe: ~{results.holdout_sharpe:.2f}")
    print("\n📊 Финальные параметры:")
    for key, value in results.final_params.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    print("\n✅ Можно рассмотреть для live trading")
else:
    print("❌ Стратегия НЕ РОБАСТНА")
    print(f"   WFO Efficiency: {results.wfo_efficiency:.2%} (< 0.5)")
    print("\n⚠️  Что делать:")
    print("   1. Упростить стратегию (меньше параметров)")
    print("   2. Изменить логику entry/exit")
    print("   3. Добавить constraints (min_trades, max_DD)")
    print("   4. Попробовать другую objective метрику")
```

---

### Вариант B: WFO на нескольких датасетах

```python
# Список твоих датасетов
datasets = [
    'data/BTCUSDT_1m.parquet',
    'data/ETHUSDT_1m.parquet',
    'data/SOLUSDT_1m.parquet',
    # ... ещё 60
]

all_results = []

for data_path in datasets:
    symbol = data_path.split('/')[-1].replace('_1m.parquet', '')

    print(f"\n{'='*60}")
    print(f"WFO для {symbol}")
    print(f"{'='*60}")

    config = WFOConfig(
        strategy_name='ported_from_example',
        symbol=symbol,
        wfo_type='anchored',
        min_train_size=0.4,
        test_size=0.1,
        step_size=0.1,
        holdout_size=0.1,
        n_trials=300,  # Меньше trials для скорости
        n_jobs=-1
    )

    analyzer = WFOAnalyzer(config, data_path)
    results = analyzer.run_wfo()
    analyzer.save_results(results)

    all_results.append({
        'symbol': symbol,
        'wfo_efficiency': results.wfo_efficiency,
        'consistency': results.consistency_score,
        'holdout_sharpe': results.holdout_sharpe,
        'final_params': results.final_params
    })

# Агрегация по всем датасетам
import numpy as np

avg_efficiency = np.mean([r['wfo_efficiency'] for r in all_results])
avg_consistency = np.mean([r['consistency'] for r in all_results])

print(f"\n{'='*60}")
print(f"SUMMARY: ALL DATASETS")
print(f"{'='*60}")
print(f"Average WFO Efficiency: {avg_efficiency:.2%}")
print(f"Average Consistency: {avg_consistency:.2%}")
print(f"Datasets with Efficiency > 0.5: {sum(1 for r in all_results if r['wfo_efficiency'] > 0.5)} / {len(all_results)}")
```

---

<a name="результат"></a>
## 9. ЧТО ПОЛУЧИШЬ на выходе

### Output #1: Консольный отчёт

См. выше (Шаг 5) — детальный вывод в консоль с таблицами.

---

### Output #2: JSON файл с результатами

**Путь:** `wfo_results/wfo_ported_from_example_BTCUSDT_20251011_143022.json`

**Содержимое:**
```json
{
  "config": {
    "strategy_name": "ported_from_example",
    "symbol": "BTCUSDT",
    "wfo_type": "anchored",
    "n_trials": 500,
    "objective_metric": "sharpe_ratio"
  },
  "windows": [
    {
      "window_id": 1,
      "train_bars": 400000,
      "test_bars": 100000,
      "is_sharpe": 2.456,
      "oos_sharpe": 1.834,
      "oos_pnl": 125.30,
      "oos_trades": 38,
      "oos_max_dd": 12.5,
      "params": {
        "stop_loss_pct": 2.5,
        "take_profit_pct": 5.0,
        "entry_logic_mode": "Только по принтам",
        "prints_analysis_period": 3,
        "prints_threshold_ratio": 2.5
      }
    },
    {
      "window_id": 2,
      ...
    }
  ],
  "metrics": {
    "wfo_efficiency": 0.658,
    "consistency": 1.0,
    "stability": 0.152,
    "degradation": 0.342
  },
  "final_params": {
    "stop_loss_pct": 2.55,
    "take_profit_pct": 4.95,
    "entry_logic_mode": "Только по принтам",
    "prints_analysis_period": 3,
    "prints_threshold_ratio": 2.45
  },
  "holdout": {
    "sharpe": 1.523,
    "pnl": 95.45,
    "trades": 28
  },
  "completed_at": "2025-10-11T14:30:22.123456"
}
```

**Что с этим делать:**
- Сохранить в архив
- Сравнивать разные runs
- Анализировать параметры
- Tracking метрик во времени

---

### Output #3: Финальные параметры для live

**Самое важное:** `results.final_params`

```python
final_params = {
    'stop_loss_pct': 2.55,
    'take_profit_pct': 4.95,
    'entry_logic_mode': 'Только по принтам',
    'prints_analysis_period': 3,
    'prints_threshold_ratio': 2.45,
    'hldir_window': 5,
    'hldir_offset': 1
}
```

**Это параметры которые:**
- Работают на OOS данных
- Прошли WFO validation
- Median из всех окон
- Готовы для live trading

**Как использовать:**
```python
from src.strategies.base_strategy import StrategyRegistry

strategy_class = StrategyRegistry.get_strategy('ported_from_example')
strategy = strategy_class(symbol='BTCUSDT', **final_params)

# Теперь можешь запускать live
```

---

<a name="примеры"></a>
## 10. ПРИМЕРЫ использования

### Пример 1: Запуск базового WFO

```bash
cd "C:\visual projects\backtrader"
python examples/wfo_example.py --example 1
```

**Что произойдёт:**
- Загрузит `data/BTCUSDT_1m.parquet`
- Запустит Anchored WFO (5 окон)
- Выведет результаты
- Сохранит JSON

**Время выполнения:** ~15-30 минут

---

### Пример 2: Rolling WFO

```bash
python examples/wfo_example.py --example 2
```

**Что произойдёт:**
- Загрузит `data/ETHUSDT_1m.parquet`
- Запустит Rolling WFO
- Сравнение с Anchored

---

### Пример 3: Date-based validation

**Если у тебя структура:**
```
data/
  05.10.2025/
    BTCUSDT.parquet
  06.10.2025/
    BTCUSDT.parquet
  07.10.2025/
    BTCUSDT.parquet
```

```bash
python examples/wfo_example.py --example 3
```

**Что произойдёт:**
- Сканирует папки по датам
- Train на данных до 05.10
- Test на 06-07.10
- Temporal validation

---

### Пример 4: Сравнение Anchored vs Rolling

```bash
python examples/wfo_example.py --example 4
```

**Что произойдёт:**
- Запустит оба типа WFO на одном датасете
- Сравнит метрики
- Подскажет какой лучше для твоей стратегии

---

<a name="faq"></a>
## 11. FAQ - Частые вопросы

### Q1: Сколько времени занимает WFO?

**A:** Зависит от:
- Размера датасета (1M баров ≈ 20-30 мин)
- Количества trials (500 trials × 5 окон = 2500 trials total)
- CPU (n_jobs=-1 использует все ядра)

**Примерное время:**
- 1M баров, 500 trials, 5 окон, 8 CPU: **20-30 минут**
- 2M баров, 300 trials, 4 окна, 16 CPU: **15-20 минут**

**С precomputed индикаторами (твой fast_optimizer) — в 2-3x быстрее чем старый WFO!**

---

### Q2: Сколько окон нужно создавать?

**A:** Оптимально **4-6 окон**.

**Почему:**
- < 3 окна → мало данных для статистики
- > 8 окон → слишком долго, нет смысла

**Рекомендации:**
- Anchored: `step_size=0.1` → создаст 5-6 окон
- Rolling: `step_size = test_size` → не перекрываются, 4-5 окон

---

### Q3: Какой размер test окна выбрать?

**A:** **10-15% от данных**.

**Примеры:**
- 1M баров → test = 100K-150K баров
- 500K баров → test = 50K-75K баров

**Почему:**
- < 10% → мало данных для статистики
- > 20% → мало окон получится

---

### Q4: Нужен ли Gap (embargo)?

**A:** Зависит от данных и таймфрейма.

**Рекомендации:**
- **1m/5m данные:** gap = 0.01-0.03 (1-3% = 1-3 дня)
- **1h/4h данные:** gap = 0.05 (5% = неделя)
- **Daily данные:** gap = 0.1 (10% = месяц)

**Когда не нужен:**
- Если нет лагов в данных
- Если entry/exit точки не зависят от будущего

---

### Q5: Что делать если WFO Efficiency < 0.5?

**A:** Стратегия переподогнана. Действия:

1. **Упростить стратегию:**
   - Уменьшить количество параметров
   - Фиксировать некоторые параметры

2. **Изменить objective:**
   - Вместо `sharpe_ratio` использовать композитную метрику
   - Добавить penalty за drawdown

3. **Добавить constraints:**
   ```python
   min_trades=20,  # Больше сделок
   max_drawdown_threshold=30.0  # Строже по DD
   ```

4. **Проверить логику:**
   - Может быть edge слабый
   - Попробовать другую стратегию

---

### Q6: Можно ли запустить WFO на нескольких датасетах параллельно?

**A:** Да, используй multiprocessing:

```python
from multiprocessing import Pool

def run_wfo_for_dataset(data_path):
    config = WFOConfig(...)
    analyzer = WFOAnalyzer(config, data_path)
    return analyzer.run_wfo()

datasets = ['data/BTC.parquet', 'data/ETH.parquet', ...]

with Pool(processes=4) as pool:
    all_results = pool.map(run_wfo_for_dataset, datasets)
```

---

### Q7: Что делать с final_params после WFO?

**A:** Использовать для live trading!

```python
# 1. Сохранить параметры
import json
with open('final_params.json', 'w') as f:
    json.dump(results.final_params, f)

# 2. Использовать в live
from src.strategies.base_strategy import StrategyRegistry

strategy_class = StrategyRegistry.get_strategy('ported_from_example')
live_strategy = strategy_class(symbol='BTCUSDT', **results.final_params)

# 3. Запускать trade signals
# (твоя live trading инфраструктура)
```

---

### Q8: Как часто переоптимизировать?

**A:** **Раз в квартал или полгода**.

**Why:**
- Рынок меняется → параметры теряют актуальность
- Но слишком частая переоптимизация → overfitting

**Процесс:**
- Каждые 3-6 месяцев: запускать WFO на свежих данных
- Сравнивать новые параметры со старыми
- Если WFO Efficiency всё ещё > 0.5 → продолжать со старыми
- Если упала < 0.4 → обновить параметры

---

### Q9: В чём разница между wfo_example.py и wfo_analyzer.py?

**A:**

- **wfo_analyzer.py** = библиотека (класс WFOAnalyzer)
  - Ты импортируешь и используешь в своём коде

- **wfo_example.py** = примеры использования
  - Готовые скрипты для запуска
  - Обучающие примеры
  - Можешь скопировать код в свой проект

**Как использовать:**
```python
# Вариант 1: Запустить готовый пример
python examples/wfo_example.py --example 1

# Вариант 2: Использовать в своём коде
from src.optimization.wfo_analyzer import WFOAnalyzer, WFOConfig
# ... твой код
```

---

### Q10: Что лучше: Anchored или Rolling?

**A:** Зависит от рынка и стратегии.

**Anchored лучше когда:**
- Хочешь использовать весь исторический опыт
- Рынок относительно стабилен
- Много долгосрочных паттернов

**Rolling лучше когда:**
- Рынок быстро меняется (crypto)
- Старые данные теряют релевантность
- Нужна быстрая адаптация

**Совет:** Запусти оба (пример 4) и сравни WFO Efficiency!

---

## Заключение

### Чеклист перед live trading

- [ ] Запустил WFO на основном датасете
- [ ] WFO Efficiency > 0.5
- [ ] Consistency > 70%
- [ ] Hold-out test прошёл успешно
- [ ] Запустил WFO на топ-10 датасетах
- [ ] Параметры работают на большинстве символов
- [ ] Сохранил final_params
- [ ] Протестировал на paper trading
- [ ] Настроил мониторинг live performance

### Если всё ✅ → можно в live!

**Ожидания:**
- Live Sharpe ≈ Hold-out Sharpe (±20%)
- Будут периоды просадки (Consistency < 100%)
- Degradation 30-50% от backtest — это нормально

**Успехов в трейдинге! 🚀**
