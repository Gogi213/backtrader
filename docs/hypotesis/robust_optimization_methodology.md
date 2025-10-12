# Методология Робастной Оптимизации Торговых Стратегий

**Автор:** Claude Code
**Дата:** 2025-10-11
**Цель:** Избежать overfitting и построить стратегию которая работает на новых данных

---

## Содержание

1. [Проблема: Почему твой текущий подход не работает](#проблема)
2. [Walk-Forward Optimization (WFO) — Детально](#wfo-детально)
3. [Temporal Validation с разными датами](#temporal-validation)
4. [Cross-Symbol Ensemble](#cross-symbol-ensemble)
5. [Комбинированный Pipeline для 63 датасетов](#pipeline)
6. [Метрики Робастности](#метрики)
7. [Практическая Реализация](#реализация)
8. [Интерпретация Результатов](#интерпретация)
9. [Чеклист Действий](#чеклист)

---

<a name="проблема"></a>
## 1. Проблема: Почему твой текущий подход не работает

### Текущий процесс (НЕ РОБАСТНЫЙ):
```
┌─────────────────────────────────────────────────────┐
│ Шаг 1: Оптимизация на каждом датасете              │
│                                                     │
│  Dataset_1 [==================] → params_1          │
│  Dataset_2 [==================] → params_2          │
│  ...                                                │
│  Dataset_63 [=================] → params_63         │
│                                                     │
│  Используешь ВСЕ данные для оптимизации            │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│ Шаг 2: Поиск общих параметров                      │
│                                                     │
│  Сравниваешь params_1..63                          │
│  Выбираешь те что "чаще встречаются"               │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│ Шаг 3: Прогон по всем датасетам                    │
│                                                     │
│  Тестируешь найденные параметры на ТЕХ ЖЕ данных  │
└─────────────────────────────────────────────────────┘

РЕЗУЛЬТАТ: Параметры подогнаны под прошлое ❌
```

### Почему это overfitting:

1. **In-Sample Bias**: Оптимизируешь на данных 2020-2024 → параметры "заточены" под паттерны 2020-2024
2. **Look-Ahead Bias**: Используешь будущее для выбора параметров (тестируешь на тех же данных)
3. **Data Snooping**: Многократные итерации "найти-протестить" на одних данных
4. **Selection Bias**: "Общие параметры" = те что лучше работали В ПРОШЛОМ

### Реальность на новых данных (2025):
```
Backtest Sharpe:  3.5  ←  Оптимизация на 2020-2024
Live Sharpe:      0.2  ←  Реальная торговля 2025

Degradation: 94% ❌
```

---

<a name="wfo-детально"></a>
## 2. Walk-Forward Optimization (WFO) — Детально

### Концепция: Имитация Реального Трейдинга

В реальности ты:
1. Анализируешь прошлое (backtest)
2. Запускаешь стратегию в live
3. Через N месяцев — переоптимизация
4. Repeat

WFO делает ТО ЖЕ на исторических данных.

---

### 2.1 Anchored Walk-Forward

**Схема:**
```
Полный датасет: [================================================]
                 2020-01        2022-01        2024-01    2025-01

Window 1:  [Train============][Test===]
           2020-01      2022-01     2022-07

Window 2:  [Train==================][Test===]
           2020-01            2022-07      2023-01

Window 3:  [Train==========================][Test===]
           2020-01                    2023-01     2023-07

Window 4:  [Train==================================][Test===]
           2020-01                          2023-07     2024-01
```

**Особенности:**
- Train период **растёт** с каждым окном (anchored к началу)
- Test период фиксированный (обычно 3-6 месяцев)
- Имитирует accumulation of data

**Когда использовать:**
- Если данных мало в начале
- Если хочешь учитывать весь исторический опыт
- Если рынок эволюционирует медленно

---

### 2.2 Rolling Walk-Forward

**Схема:**
```
Полный датасет: [================================================]
                 2020-01        2022-01        2024-01    2025-01

Window 1:  [Train======][Test===]
           2020-01  2021-07  2022-01

Window 2:         [Train======][Test===]
                  2021-01  2022-07  2023-01

Window 3:                [Train======][Test===]
                         2022-01  2023-07  2024-01

Window 4:                       [Train======][Test===]
                                2023-01  2024-07  2025-01
```

**Особенности:**
- Train и Test периоды **фиксированного размера**
- Окна **скользят** вперёд
- Не перекрываются (или с малым overlap)

**Когда использовать:**
- Если хочешь адаптироваться к ТЕКУЩИМ условиям рынка
- Если старые данные теряют релевантность (crypto, high-freq)
- Если нужна быстрая адаптация

---

### 2.3 Пошаговый Алгоритм WFO

**Для каждого окна i:**

```python
# 1. Оптимизация на Train (In-Sample)
train_data = data[window_i_train_start : window_i_train_end]
study = optuna.create_study(direction='maximize')
study.optimize(objective(train_data), n_trials=500)
best_params_i = study.best_params
IS_sharpe_i = study.best_value

# 2. Валидация на Test (Out-of-Sample)
test_data = data[window_i_test_start : window_i_test_end]
OOS_results_i = backtest(test_data, best_params_i)
OOS_sharpe_i = OOS_results_i['sharpe_ratio']
OOS_pnl_i = OOS_results_i['net_pnl']

# 3. Сохранение результатов
wfo_results.append({
    'window': i,
    'IS_sharpe': IS_sharpe_i,
    'OOS_sharpe': OOS_sharpe_i,
    'OOS_pnl': OOS_pnl_i,
    'params': best_params_i,
    'train_period': [window_i_train_start, window_i_train_end],
    'test_period': [window_i_test_start, window_i_test_end]
})
```

**Финальная метрика:**
```python
# Склеиваем все OOS периоды в единую equity curve
full_OOS_equity = concatenate(OOS_pnl_1, OOS_pnl_2, ..., OOS_pnl_N)
WFO_sharpe = calculate_sharpe(full_OOS_equity)

# WFO Efficiency
WFO_efficiency = mean(OOS_sharpe_1..N) / mean(IS_sharpe_1..N)
```

---

### 2.4 Конкретные Параметры для Твоего Setup

#### Если датасет = 1M баров (≈ 2 года на 1m, ≈ 5 лет на 5m)

**Anchored WFO:**
```python
train_sizes = [0.40, 0.50, 0.60, 0.70, 0.80]  # % от total
test_size = 0.10  # Фиксированный тест = 10%

# Окно 1: Train 40%, Test 10% (баров 400K-500K)
# Окно 2: Train 50%, Test 10% (баров 500K-600K)
# Окно 3: Train 60%, Test 10% (баров 600K-700K)
# Окно 4: Train 70%, Test 10% (баров 700K-800K)
# Окно 5: Train 80%, Test 10% (баров 800K-900K)
# Последние 10% = Hold-out для финального теста
```

**Rolling WFO:**
```python
train_size = 400_000  # 40% баров
test_size = 100_000   # 10% баров
step = test_size      # Не перекрываются

# Окно 1: Train [0:400K], Test [400K:500K]
# Окно 2: Train [100K:500K], Test [500K:600K]
# Окно 3: Train [200K:600K], Test [600K:700K]
# ...
```

---

### 2.5 Embargo Period (Gap)

**Проблема:** В реальности между "обучением" и "торговлей" проходит время.

**Решение:** Добавить gap между Train и Test.

```
Train [==========]  GAP [...] Test [====]
```

**Рекомендации:**
- Gap = 5-10% от test_size
- Для crypto: 1-3 дня
- Для stocks: 1-5 дней

```python
gap = int(test_size * 0.05)  # 5% от test size

test_start = train_end + gap
test_end = test_start + test_size
```

**Зачем нужен:**
- Избежать information leakage (если есть лаги в данных)
- Реалистичнее имитировать live deployment
- Учесть время на переоптимизацию

---

<a name="temporal-validation"></a>
## 3. Temporal Validation с Разными Датами

### 3.1 У тебя есть датасеты с разными датами? ЭТО ЗОЛОТО! 💰

**Сценарий:**
```
Dataset_1: BTCUSDT 2020-01 → 2025-01
Dataset_2: ETHUSDT 2021-06 → 2025-01
Dataset_3: SOLUSDT 2022-03 → 2025-01
...
Dataset_63: APTUSDT 2023-01 → 2025-01
```

### 3.2 Time-Based Cross-Validation

**Идея:** Параметры должны работать на РАЗНЫХ временных периодах разных активов.

**Метод 1: Train/Test по календарным датам**

```
Все датасеты разбить по датам:

Train:  2020-01-01 → 2023-12-31  (все символы где есть эти даты)
Test:   2024-01-01 → 2024-12-31  (out-of-sample год)
```

**Пример:**
```python
train_datasets = []
test_datasets = []

for dataset in all_datasets:
    # Фильтр по дате
    train_data = dataset[dataset['time'] < '2024-01-01']
    test_data = dataset[dataset['time'] >= '2024-01-01']

    if len(train_data) > min_bars:
        train_datasets.append(train_data)
    if len(test_data) > min_bars:
        test_datasets.append(test_data)

# Оптимизация на train_datasets
# Валидация на test_datasets
```

**Профит:**
- Параметры НЕ видели 2024 год = true OOS test
- Проверка работает ли стратегия на НОВЫХ рыночных условиях

---

**Метод 2: Market Regime Validation**

**Идея:** Разные периоды = разные режимы рынка (bull/bear/sideways).

**Шаг 1: Классификация периодов**
```python
def classify_regime(data):
    """Определяет режим рынка для датасета"""
    returns = (data['close'][-1] - data['close'][0]) / data['close'][0]
    volatility = data['close'].std() / data['close'].mean()

    if returns > 0.5 and volatility < 0.1:
        return 'bull_low_vol'
    elif returns > 0.5 and volatility > 0.1:
        return 'bull_high_vol'
    elif returns < -0.2:
        return 'bear'
    else:
        return 'sideways'

# Классифицируй все датасеты
for dataset in all_datasets:
    dataset.regime = classify_regime(dataset)
```

**Шаг 2: Train на одних режимах, Test на других**
```python
# Train на bull market
train_datasets = [d for d in all_datasets if d.regime.startswith('bull')]

# Test на sideways/bear
test_datasets = [d for d in all_datasets if d.regime in ['sideways', 'bear']]

# Оптимизация
params = optimize_on_datasets(train_datasets)

# Валидация: работают ли параметры на ДРУГИХ режимах?
oos_results = test_on_datasets(test_datasets, params)
```

**Профит:**
- Проверка универсальности стратегии
- Параметры не подогнаны под один тип рынка

---

### 3.3 Combinatorial Purged Cross-Validation (Advanced)

**Проблема стандартного CV:** Временные данные имеют автокорреляцию.

**Решение:** Purging + Embargo.

```
Данные по времени: [===========================================]

Fold 1 Train: [===]    [===]    [===]
Fold 1 Test:      [X][=][X]  ← Purge/Embargo
                      ↑
                   Test period
```

**Purging:**
- Удалить из Train данные ПОСЛЕ точки Test (look-ahead bias)

**Embargo:**
- Удалить из Train данные ПЕРЕД точкой Test (leakage через overlapping trades)

**Реализация:**
```python
def purged_cv_split(data, n_splits=5, embargo_pct=0.05):
    """
    Combinatorial Purged CV для временных рядов
    """
    n = len(data)
    test_size = n // (n_splits + 1)

    for i in range(n_splits):
        test_start = (i + 1) * test_size
        test_end = test_start + test_size

        # Test
        test_indices = range(test_start, test_end)

        # Purging: убираем данные после test_end
        # Embargo: убираем данные перед test_start
        embargo_size = int(test_size * embargo_pct)

        train_indices = list(range(0, test_start - embargo_size)) + \
                       list(range(test_end + embargo_size, n))

        yield train_indices, test_indices
```

---

<a name="cross-symbol-ensemble"></a>
## 4. Cross-Symbol Ensemble

### 4.1 Концепция

**Идея:** Параметры должны работать на РАЗНЫХ символах, не только на том где оптимизировал.

### 4.2 Stratified K-Fold по символам

**Шаг 1: Стратификация**

Разбей 63 датасета на группы по характеристикам:

```python
def stratify_datasets(datasets):
    """Группировка по волатильности и тренду"""
    groups = {
        'high_vol_bull': [],
        'high_vol_bear': [],
        'low_vol_bull': [],
        'low_vol_bear': [],
        'sideways': []
    }

    for dataset in datasets:
        vol = calculate_volatility(dataset)
        trend = calculate_trend(dataset)

        if vol > 0.15:
            if trend > 0.3:
                groups['high_vol_bull'].append(dataset)
            elif trend < -0.2:
                groups['high_vol_bear'].append(dataset)
        elif vol < 0.08:
            if trend > 0.2:
                groups['low_vol_bull'].append(dataset)
            elif trend < -0.1:
                groups['low_vol_bear'].append(dataset)
        else:
            groups['sideways'].append(dataset)

    return groups
```

**Шаг 2: K-Fold Split**

```python
from sklearn.model_selection import KFold

# Разбить на 5 folds, сохраняя пропорции групп
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(all_datasets)):
    train_datasets = [all_datasets[i] for i in train_idx]  # 50 datasets
    test_datasets = [all_datasets[i] for i in test_idx]    # 13 datasets

    print(f"Fold {fold_idx}: Train {len(train_datasets)}, Test {len(test_datasets)}")

    # Оптимизация на train
    params = optimize_on_multiple_datasets(train_datasets)

    # Валидация на test
    oos_metrics = validate_on_multiple_datasets(test_datasets, params)
```

### 4.3 Ensemble параметров

**После 5 folds получишь 5 наборов параметров:**

```python
fold_1_params = {'stop_loss': 2.5, 'take_profit': 5.0, ...}
fold_2_params = {'stop_loss': 2.8, 'take_profit': 4.5, ...}
fold_3_params = {'stop_loss': 2.3, 'take_profit': 5.2, ...}
fold_4_params = {'stop_loss': 2.6, 'take_profit': 4.8, ...}
fold_5_params = {'stop_loss': 2.4, 'take_profit': 5.1, ...}
```

**Варианты Ensemble:**

**Вариант A: Median**
```python
final_params = {
    'stop_loss': np.median([2.5, 2.8, 2.3, 2.6, 2.4]) = 2.5,
    'take_profit': np.median([5.0, 4.5, 5.2, 4.8, 5.1]) = 5.0,
    ...
}
```

**Вариант B: Weighted по OOS performance**
```python
weights = [oos_sharpe_1, oos_sharpe_2, ..., oos_sharpe_5]
weights = weights / sum(weights)

final_params = {
    'stop_loss': weighted_average([2.5, 2.8, ...], weights),
    ...
}
```

**Вариант C: Intersection (консервативный)**
```python
# Только те параметры что работают на ВСЕХ folds
# Например: если stop_loss=[2.3..2.8], выбрать середину 2.5
```

---

<a name="pipeline"></a>
## 5. Комбинированный Pipeline для 63 датасетов

### 5.1 Полный Pipeline (Максимальная Робастность)

```
┌──────────────────────────────────────────────────────────────┐
│ ЭТАП 1: Подготовка данных                                   │
├──────────────────────────────────────────────────────────────┤
│ • 63 датасета загружены                                      │
│ • Классификация по regime (bull/bear/sideways)              │
│ • Стратификация по volatility                                │
│ • Резервация Hold-out (последние 10-15% всех датасетов)     │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ ЭТАП 2: Walk-Forward на каждом датасете (53 train datasets) │
├──────────────────────────────────────────────────────────────┤
│ For each dataset:                                            │
│   • Anchored WFO: 5 windows                                  │
│   • n_trials=500 per window                                  │
│   • Сохранить best_params per window                        │
│   • Рассчитать WFO Efficiency                               │
│                                                              │
│ Результат: 53 datasets × 5 windows = 265 наборов параметров │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ ЭТАП 3: Cross-Symbol Validation (5-Fold)                    │
├──────────────────────────────────────────────────────────────┤
│ Fold 1: Train 42 datasets → Test 11 datasets                │
│ Fold 2: Train 42 datasets → Test 11 datasets                │
│ Fold 3: Train 42 datasets → Test 11 datasets                │
│ Fold 4: Train 42 datasets → Test 11 datasets                │
│ Fold 5: Train 42 datasets → Test 11 datasets                │
│                                                              │
│ Для каждого fold:                                            │
│   • Усреднить лучшие параметры из WFO по train datasets     │
│   • Валидация на test datasets                              │
│   • Рассчитать OOS metrics                                  │
│                                                              │
│ Результат: 5 наборов параметров                             │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ ЭТАП 4: Ensemble финальных параметров                       │
├──────────────────────────────────────────────────────────────┤
│ • Median/Weighted average из 5 folds                         │
│ • Выбор параметров с max Consistency Score                  │
│                                                              │
│ Результат: FINAL_PARAMS                                     │
└──────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│ ЭТАП 5: Hold-out Test (Финальная проверка)                 │
├──────────────────────────────────────────────────────────────┤
│ • Прогнать FINAL_PARAMS на всех 63 hold-out периодах        │
│ • Рассчитать Sharpe, PnL, MaxDD, Win Rate                   │
│ • Сравнить с In-Sample метриками                            │
│                                                              │
│ Если OOS_Sharpe / IS_Sharpe > 0.5:                          │
│   ✅ Стратегия робастна, можно в live                       │
│ Иначе:                                                       │
│   ❌ Overfit, вернуться к упрощению стратегии               │
└──────────────────────────────────────────────────────────────┘
```

---

### 5.2 Быстрый Pipeline (Proof-of-Concept)

Если нет времени на полный пайплайн, начни с этого:

```
┌─────────────────────────────────────────────┐
│ 1. Выбрать топ-10 самых ликвидных датасетов │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ 2. Rolling WFO на каждом (3-4 окна)        │
│    n_trials=300 per window                  │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ 3. Усреднить best_params по всем окнам      │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ 4. Рассчитать WFO Efficiency                │
│    Если > 0.5: продолжить                   │
│    Иначе: упростить стратегию               │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ 5. Валидация на остальных 53 датасетах      │
│    (full data, без WFO)                     │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ 6. Hold-out test на последних 10% всех      │
└─────────────────────────────────────────────┘

Время: ~2-4 часа с precomputed индикаторами
```

---

<a name="метрики"></a>
## 6. Метрики Робастности

### 6.1 WFO Efficiency

**Формула:**
```
WFO_Efficiency = mean(OOS_Sharpe[1..N]) / mean(IS_Sharpe[1..N])
```

Где:
- `N` = количество WFO окон
- `OOS_Sharpe[i]` = Sharpe на out-of-sample периоде окна i
- `IS_Sharpe[i]` = Sharpe на in-sample периоде окна i

**Интерпретация:**
| WFO Efficiency | Оценка | Действие |
|----------------|--------|----------|
| > 0.7 | Отлично | Параметры очень робастны |
| 0.5 - 0.7 | Хорошо | Приемлемый уровень, можно использовать |
| 0.3 - 0.5 | Посредственно | Есть degradation, нужна осторожность |
| < 0.3 | Плохо | Сильный overfit, переработать стратегию |

**Пример расчёта:**
```python
wfo_results = [
    {'IS_sharpe': 2.5, 'OOS_sharpe': 1.8},
    {'IS_sharpe': 2.8, 'OOS_sharpe': 1.5},
    {'IS_sharpe': 3.0, 'OOS_sharpe': 2.0},
    {'IS_sharpe': 2.6, 'OOS_sharpe': 1.6},
]

mean_IS = (2.5 + 2.8 + 3.0 + 2.6) / 4 = 2.725
mean_OOS = (1.8 + 1.5 + 2.0 + 1.6) / 4 = 1.725

WFO_Efficiency = 1.725 / 2.725 = 0.633  ← Хорошо! ✅
```

---

### 6.2 Consistency Score

**Формула:**
```
Consistency = count(positive_OOS_periods) / total_periods
```

Где:
- `positive_OOS_periods` = количество окон/folds где OOS_Sharpe > 0 (или OOS_PnL > 0)

**Интерпретация:**
| Consistency | Оценка | Действие |
|-------------|--------|----------|
| = 1.0 | Идеально | Работает на ВСЕХ периодах |
| > 0.8 | Отлично | Высокая универсальность |
| 0.6 - 0.8 | Хорошо | Работает на большинстве периодов |
| < 0.6 | Плохо | Не универсальная стратегия |

**Пример:**
```python
oos_sharpes = [1.5, -0.2, 2.0, 1.8, 0.5]
positive_count = sum([1 for s in oos_sharpes if s > 0])  # = 4
total = len(oos_sharpes)  # = 5

Consistency = 4 / 5 = 0.8  ← Отлично! ✅
```

---

### 6.3 Degradation Ratio

**Формула:**
```
Degradation = 1 - (mean_OOS_metric / mean_IS_metric)
```

**Интерпретация:**
| Degradation | Оценка | Действие |
|-------------|--------|----------|
| < 20% | Отлично | Минимальная потеря performance |
| 20-40% | Хорошо | Приемлемый уровень |
| 40-60% | Посредственно | Значительная degradation |
| > 60% | Плохо | Сильный overfit |

**Связь с WFO Efficiency:**
```
Degradation ≈ 1 - WFO_Efficiency
```

---

### 6.4 Stability (Coefficient of Variation)

**Формула:**
```
Stability = std(OOS_Sharpe[1..N]) / mean(OOS_Sharpe[1..N])
```

**Интерпретация:**
| Stability (CV) | Оценка | Действие |
|----------------|--------|----------|
| < 0.3 | Отлично | Стабильные результаты |
| 0.3 - 0.5 | Хорошо | Умеренная вариативность |
| 0.5 - 1.0 | Посредственно | Высокая вариативность |
| > 1.0 | Плохо | Нестабильная стратегия |

**Пример:**
```python
oos_sharpes = [1.5, 1.8, 1.6, 1.7, 1.9]
mean_sharpe = 1.7
std_sharpe = 0.14

Stability = 0.14 / 1.7 = 0.082  ← Отлично! ✅
```

---

### 6.5 Composite Robustness Score

**Формула:**
```
Robustness_Score = (
    0.4 * WFO_Efficiency +
    0.3 * Consistency +
    0.2 * (1 - Stability) +
    0.1 * (1 - abs(Degradation))
)
```

**Интерпретация:**
| Score | Оценка |
|-------|--------|
| > 0.7 | Очень робастная стратегия |
| 0.5-0.7 | Робастная |
| 0.3-0.5 | Умеренно робастная |
| < 0.3 | Не робастная |

---

<a name="реализация"></a>
## 7. Практическая Реализация

### 7.1 Код: Anchored WFO на одном датасете

```python
from src.optimization.fast_optimizer import FastStrategyOptimizer
import numpy as np

def run_anchored_wfo(data_path, strategy_name, symbol):
    """
    Anchored Walk-Forward Optimization на одном датасете
    """
    # 1. Загрузка данных
    from src.data.klines_handler import VectorizedKlinesHandler
    handler = VectorizedKlinesHandler()
    full_data = handler.load_klines(data_path)
    n = len(full_data)

    # 2. Определение окон
    train_sizes = [0.40, 0.50, 0.60, 0.70, 0.80]
    test_size = 0.10
    holdout_size = 0.10

    # Резервируем hold-out
    holdout_start = int(n * (1 - holdout_size))
    data_for_wfo = full_data[:holdout_start]
    holdout_data = full_data[holdout_start:]

    wfo_results = []

    # 3. WFO Loop
    for i, train_pct in enumerate(train_sizes):
        print(f"\n{'='*60}")
        print(f"WFO Window {i+1}/{len(train_sizes)}")
        print(f"{'='*60}")

        train_end = int(len(data_for_wfo) * train_pct)
        test_start = train_end
        test_end = test_start + int(n * test_size)

        if test_end > len(data_for_wfo):
            print(f"[SKIP] Not enough data for test period")
            break

        train_data = data_for_wfo[:train_end]
        test_data = data_for_wfo[test_start:test_end]

        print(f"Train: {len(train_data):,} bars | Test: {len(test_data):,} bars")

        # 3.1 Оптимизация на Train
        optimizer = FastStrategyOptimizer(
            strategy_name=strategy_name,
            data_path=data_path,  # Будет использовать кеш
            symbol=symbol
        )

        # Хак: заменим cached_data на train_data
        optimizer.cached_data['full'] = train_data

        train_results = optimizer.optimize(
            n_trials=500,
            objective_metric='sharpe_ratio',
            min_trades=10,
            max_drawdown_threshold=50.0,
            data_slice='full'
        )

        best_params = train_results['best_params']
        is_sharpe = train_results['best_value']

        print(f"[Train] Best Sharpe: {is_sharpe:.3f}")
        print(f"[Train] Best Params: {best_params}")

        # 3.2 Валидация на Test (OOS)
        from src.strategies.base_strategy import StrategyRegistry
        strategy_class = StrategyRegistry.get_strategy(strategy_name)
        strategy = strategy_class(symbol=symbol, **best_params)

        oos_results = strategy.vectorized_process_dataset(test_data)
        oos_sharpe = oos_results.get('sharpe_ratio', 0)
        oos_pnl = oos_results.get('net_pnl', 0)
        oos_trades = oos_results.get('total', 0)

        print(f"[Test]  OOS Sharpe: {oos_sharpe:.3f}")
        print(f"[Test]  OOS PnL: {oos_pnl:.2f}")
        print(f"[Test]  OOS Trades: {oos_trades}")

        wfo_results.append({
            'window': i + 1,
            'train_bars': len(train_data),
            'test_bars': len(test_data),
            'IS_sharpe': is_sharpe,
            'OOS_sharpe': oos_sharpe,
            'OOS_pnl': oos_pnl,
            'OOS_trades': oos_trades,
            'params': best_params
        })

    # 4. WFO Метрики
    is_sharpes = [r['IS_sharpe'] for r in wfo_results]
    oos_sharpes = [r['OOS_sharpe'] for r in wfo_results]

    wfo_efficiency = np.mean(oos_sharpes) / np.mean(is_sharpes)
    consistency = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)
    stability = np.std(oos_sharpes) / np.mean(oos_sharpes) if np.mean(oos_sharpes) != 0 else 999

    print(f"\n{'='*60}")
    print(f"WFO SUMMARY")
    print(f"{'='*60}")
    print(f"WFO Efficiency: {wfo_efficiency:.3f}")
    print(f"Consistency:    {consistency:.2%}")
    print(f"Stability (CV): {stability:.3f}")

    # 5. Hold-out Test (финальный тест на unseen data)
    print(f"\n{'='*60}")
    print(f"HOLD-OUT TEST (Final Validation)")
    print(f"{'='*60}")

    # Используем median параметров из всех окон
    from statistics import median
    all_param_keys = wfo_results[0]['params'].keys()
    median_params = {}
    for key in all_param_keys:
        values = [r['params'][key] for r in wfo_results]
        if isinstance(values[0], (int, float)):
            median_params[key] = median(values)
        else:
            # Categorical: возьмём самый частый
            from collections import Counter
            median_params[key] = Counter(values).most_common(1)[0][0]

    print(f"Median Params: {median_params}")

    strategy_final = strategy_class(symbol=symbol, **median_params)
    holdout_results = strategy_final.vectorized_process_dataset(holdout_data)

    holdout_sharpe = holdout_results.get('sharpe_ratio', 0)
    holdout_pnl = holdout_results.get('net_pnl', 0)

    print(f"Hold-out Sharpe: {holdout_sharpe:.3f}")
    print(f"Hold-out PnL:    {holdout_pnl:.2f}")
    print(f"Hold-out vs IS:  {holdout_sharpe / np.mean(is_sharpes):.2%}")

    return {
        'wfo_results': wfo_results,
        'wfo_efficiency': wfo_efficiency,
        'consistency': consistency,
        'stability': stability,
        'median_params': median_params,
        'holdout_sharpe': holdout_sharpe,
        'holdout_pnl': holdout_pnl
    }

# Использование:
result = run_anchored_wfo(
    data_path='data/BTCUSDT_1m.parquet',
    strategy_name='ported_from_example',
    symbol='BTCUSDT'
)
```

---

### 7.2 Код: Cross-Symbol Validation

```python
def cross_symbol_validation(datasets, strategy_name, n_folds=5):
    """
    K-Fold Cross-Validation по разным символам
    """
    from sklearn.model_selection import KFold
    import numpy as np

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(datasets)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")

        train_datasets = [datasets[i] for i in train_idx]
        test_datasets = [datasets[i] for i in test_idx]

        print(f"Train: {len(train_datasets)} datasets")
        print(f"Test:  {len(test_datasets)} datasets")

        # 1. Оптимизация на каждом train dataset
        all_train_params = []
        for dataset in train_datasets:
            optimizer = FastStrategyOptimizer(
                strategy_name=strategy_name,
                data_path=dataset['path'],
                symbol=dataset['symbol']
            )
            results = optimizer.optimize(
                n_trials=300,  # Меньше trials для скорости
                objective_metric='sharpe_ratio'
            )
            all_train_params.append(results['best_params'])

        # 2. Усреднение параметров
        param_keys = all_train_params[0].keys()
        median_params = {}
        for key in param_keys:
            values = [p[key] for p in all_train_params]
            if isinstance(values[0], (int, float)):
                median_params[key] = np.median(values)
            else:
                from collections import Counter
                median_params[key] = Counter(values).most_common(1)[0][0]

        print(f"Fold {fold_idx+1} Median Params: {median_params}")

        # 3. Валидация на test datasets
        strategy_class = StrategyRegistry.get_strategy(strategy_name)
        test_sharpes = []
        test_pnls = []

        for dataset in test_datasets:
            handler = VectorizedKlinesHandler()
            data = handler.load_klines(dataset['path'])

            strategy = strategy_class(symbol=dataset['symbol'], **median_params)
            results = strategy.vectorized_process_dataset(data)

            test_sharpes.append(results.get('sharpe_ratio', 0))
            test_pnls.append(results.get('net_pnl', 0))

        fold_oos_sharpe = np.mean(test_sharpes)
        fold_consistency = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes)

        print(f"Fold {fold_idx+1} OOS Sharpe: {fold_oos_sharpe:.3f}")
        print(f"Fold {fold_idx+1} Consistency: {fold_consistency:.2%}")

        fold_results.append({
            'fold': fold_idx + 1,
            'median_params': median_params,
            'oos_sharpe': fold_oos_sharpe,
            'oos_pnls': test_pnls,
            'consistency': fold_consistency
        })

    # 4. Финальное усреднение по всем folds
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    all_oos_sharpes = [f['oos_sharpe'] for f in fold_results]
    avg_oos_sharpe = np.mean(all_oos_sharpes)
    avg_consistency = np.mean([f['consistency'] for f in fold_results])

    print(f"Average OOS Sharpe: {avg_oos_sharpe:.3f}")
    print(f"Average Consistency: {avg_consistency:.2%}")

    # Финальные параметры = median по всем folds
    all_fold_params = [f['median_params'] for f in fold_results]
    param_keys = all_fold_params[0].keys()
    final_params = {}
    for key in param_keys:
        values = [p[key] for p in all_fold_params]
        if isinstance(values[0], (int, float)):
            final_params[key] = np.median(values)
        else:
            from collections import Counter
            final_params[key] = Counter(values).most_common(1)[0][0]

    print(f"\nFinal Ensemble Params: {final_params}")

    return {
        'fold_results': fold_results,
        'avg_oos_sharpe': avg_oos_sharpe,
        'avg_consistency': avg_consistency,
        'final_params': final_params
    }

# Использование:
datasets = [
    {'path': 'data/BTCUSDT.parquet', 'symbol': 'BTCUSDT'},
    {'path': 'data/ETHUSDT.parquet', 'symbol': 'ETHUSDT'},
    # ... all 63 datasets
]

cv_results = cross_symbol_validation(
    datasets=datasets,
    strategy_name='ported_from_example',
    n_folds=5
)
```

---

<a name="интерпретация"></a>
## 8. Интерпретация Результатов

### 8.1 Сценарий A: Отличные результаты ✅

**Метрики:**
```
WFO Efficiency:  0.68
Consistency:     85%
Stability:       0.32
Hold-out Sharpe: 1.8 (при IS Sharpe: 2.5)
```

**Интерпретация:**
- ✅ WFO Efficiency > 0.5 → параметры робастны
- ✅ Consistency 85% → работает на большинстве периодов
- ✅ Stability < 0.5 → стабильные результаты
- ✅ Hold-out degradation 28% → приемлемо

**Действие:**
→ Стратегия готова к live trading с этими параметрами

---

### 8.2 Сценарий B: Средние результаты ⚠️

**Метрики:**
```
WFO Efficiency:  0.42
Consistency:     60%
Stability:       0.85
Hold-out Sharpe: 0.5 (при IS Sharpe: 2.8)
```

**Интерпретация:**
- ⚠️ WFO Efficiency < 0.5 → есть overfit
- ⚠️ Consistency 60% → работает не на всех периодах
- ⚠️ Stability > 0.5 → нестабильные результаты
- ❌ Hold-out degradation 82% → сильная degradation

**Действие:**
→ Упростить стратегию:
  - Уменьшить количество оптимизируемых параметров
  - Добавить constraints (min_trades, max_DD)
  - Изменить objective на более робастную метрику

---

### 8.3 Сценарий C: Плохие результаты ❌

**Метрики:**
```
WFO Efficiency:  0.18
Consistency:     35%
Stability:       1.50
Hold-out Sharpe: -0.2 (при IS Sharpe: 3.2)
```

**Интерпретация:**
- ❌ WFO Efficiency < 0.3 → сильный overfit
- ❌ Consistency < 50% → не универсальна
- ❌ Stability > 1.0 → очень нестабильна
- ❌ Hold-out отрицательный → не работает на новых данных

**Действие:**
→ Радикальные изменения:
  1. Вернуться к базовой логике стратегии
  2. Проверить есть ли edge вообще (simple buy-hold benchmark)
  3. Упростить до 2-3 параметров
  4. Переосмыслить entry/exit логику

---

### 8.4 Red Flags (Признаки Overfit)

🚩 **Red Flag #1:** IS Sharpe > 4.0
- Слишком хорошо чтобы быть правдой
- Вероятно подгонка под шум

🚩 **Red Flag #2:** OOS Sharpe / IS Sharpe < 0.2
- Massive degradation
- Параметры не переносятся на новые данные

🚩 **Red Flag #3:** Большой разброс OOS метрик по окнам
- Window 1: Sharpe 2.5
- Window 2: Sharpe -0.5
- Window 3: Sharpe 3.0
→ Нестабильность = overfit на специфичные паттерны

🚩 **Red Flag #4:** Много параметров оптимизируется
- > 7-8 параметров = риск переподгонки
- Curse of dimensionality

🚩 **Red Flag #5:** Параметры на грани диапазона
- stop_loss = 0.51 (при диапазоне 0.5-5.0)
- take_profit = 7.99 (при диапазоне 1.0-8.0)
→ Optuna "уперся" в границу = плохой sign

---

<a name="чеклист"></a>
## 9. Чеклист Действий

### Фаза 1: Подготовка (1-2 часа)

- [ ] Загрузить все 63 датасета
- [ ] Проверить качество данных (пропуски, outliers)
- [ ] Классифицировать по volatility и trend
- [ ] Зарезервировать Hold-out (последние 10-15%)
- [ ] Выбрать топ-10 датасетов для proof-of-concept

### Фаза 2: WFO на топ-10 датасетах (2-4 часа)

- [ ] Настроить параметры WFO (train/test sizes, embargo)
- [ ] Запустить Anchored WFO на каждом из 10
- [ ] Собрать результаты в таблицу
- [ ] Рассчитать WFO Efficiency для каждого
- [ ] **Decision Point:** Если avg WFO Efficiency < 0.4 → упростить стратегию

### Фаза 3: Cross-Symbol Validation (3-5 часов)

- [ ] Разбить 63 датасета на 5 folds (стратифицированно)
- [ ] Для каждого fold: оптимизация на train, валидация на test
- [ ] Собрать OOS метрики по всем folds
- [ ] Рассчитать Consistency Score
- [ ] Ensemble параметров (median/weighted)

### Фаза 4: Hold-out Test (30 мин)

- [ ] Прогнать финальные параметры на Hold-out данных
- [ ] Сравнить Hold-out metrics с IS metrics
- [ ] Рассчитать финальный Degradation
- [ ] **Decision Point:** Если Hold-out Sharpe / IS Sharpe > 0.5 → GO LIVE

### Фаза 5: Документация и Мониторинг

- [ ] Сохранить все метрики в JSON/CSV
- [ ] Записать финальные параметры
- [ ] Создать план мониторинга live performance
- [ ] Установить alerts (если live Sharpe падает на 50% от OOS)

---

## 10. Дополнительные Техники (Advanced)

### 10.1 Monte Carlo Simulation

**Идея:** Добавить случайность в entry точки и проверить стабильность PnL.

```python
def monte_carlo_validation(data, params, n_simulations=100):
    """
    Добавляет ±10% шум к entry цене и симулирует trades
    """
    pnls = []
    for i in range(n_simulations):
        # Добавляем случайный шум к entry
        noise = np.random.uniform(-0.1, 0.1, size=len(data))
        data_noisy = data.copy()
        data_noisy['entry_price'] = data['close'] * (1 + noise)

        strategy = Strategy(**params)
        results = strategy.backtest(data_noisy)
        pnls.append(results['net_pnl'])

    # Метрики
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    pct_positive = sum(1 for p in pnls if p > 0) / len(pnls)

    print(f"Monte Carlo PnL: {mean_pnl:.2f} ± {std_pnl:.2f}")
    print(f"% Positive: {pct_positive:.1%}")

    return {
        'mean_pnl': mean_pnl,
        'std_pnl': std_pnl,
        'pct_positive': pct_positive
    }
```

**Интерпретация:**
- Если `pct_positive > 80%` → стратегия робастна к timing
- Если `std_pnl / mean_pnl < 0.5` → стабильная

---

### 10.2 Parameter Sensitivity Analysis

**Идея:** Проверить насколько чувствительны результаты к изменению параметров.

```python
def sensitivity_analysis(data, base_params, param_to_test='stop_loss'):
    """
    Варьирует один параметр ±20% и смотрит на изменение Sharpe
    """
    base_value = base_params[param_to_test]
    variations = np.linspace(base_value * 0.8, base_value * 1.2, 10)

    sharpes = []
    for value in variations:
        params = base_params.copy()
        params[param_to_test] = value

        strategy = Strategy(**params)
        results = strategy.backtest(data)
        sharpes.append(results['sharpe_ratio'])

    # Визуализация
    import matplotlib.pyplot as plt
    plt.plot(variations, sharpes)
    plt.xlabel(param_to_test)
    plt.ylabel('Sharpe Ratio')
    plt.title(f'Sensitivity: {param_to_test}')
    plt.show()

    # Метрика: коэффициент вариации
    cv = np.std(sharpes) / np.mean(sharpes)
    print(f"Coefficient of Variation: {cv:.3f}")

    # Если CV < 0.2 → параметр не критичен (robust)
    # Если CV > 0.5 → параметр очень чувствительный (опасно!)

    return cv
```

---

### 10.3 Regime-Based Hold-out

**Идея:** Проверить работает ли стратегия в разных рыночных режимах.

```python
def regime_holdout_test(datasets, params, strategy_name):
    """
    Группирует hold-out данные по режимам и тестирует отдельно
    """
    regimes = {
        'bull': [],
        'bear': [],
        'sideways': []
    }

    for dataset in datasets:
        regime = classify_regime(dataset['holdout_data'])
        regimes[regime].append(dataset['holdout_data'])

    results = {}
    for regime, datas in regimes.items():
        sharpes = []
        for data in datas:
            strategy = Strategy(**params)
            res = strategy.backtest(data)
            sharpes.append(res['sharpe_ratio'])

        results[regime] = {
            'mean_sharpe': np.mean(sharpes),
            'consistency': sum(1 for s in sharpes if s > 0) / len(sharpes)
        }

        print(f"{regime.upper()}: Sharpe {results[regime]['mean_sharpe']:.2f}, Consistency {results[regime]['consistency']:.1%}")

    return results
```

**Цель:** Убедиться что стратегия работает не только в bull, но и в bear/sideways.

---

## 11. Заключение

### Ключевые Принципы Робастной Оптимизации:

1. **Never optimize on data you will test on** ← Золотое правило
2. **Time-based validation** ← Учитывай временную природу данных
3. **Multiple validation layers** ← WFO + Cross-Symbol + Hold-out
4. **Conservative metrics** ← Если сомневаешься, упрости
5. **Accept degradation** ← OOS всегда хуже IS, это нормально

### Реалистичные Ожидания:

| Этап | Sharpe Ratio |
|------|--------------|
| In-Sample (IS) | 2.5 |
| Out-of-Sample (OOS) WFO | 1.5 - 1.8 |
| Hold-out | 1.2 - 1.5 |
| Live Trading | 0.8 - 1.2 |

**Degradation 50-70% от IS к Live — это НОРМАЛЬНО для робастной стратегии.**

Если live results близки к OOS/Hold-out → ты всё сделал правильно! ✅

---

## Контакты и Ресурсы

**Дополнительная литература:**
- "Advances in Financial Machine Learning" by Marcos Lopez de Prado (глава про Cross-Validation)
- "Evidence-Based Technical Analysis" by David Aronson
- "Quantitative Trading" by Ernest Chan (глава про Walk-Forward)

**Инструменты:**
- Твой `wfo_optimizer.py` уже реализует базовый WFO
- `fast_optimizer.py` с precomputed индикаторами для скорости
- Можно добавить визуализацию WFO equity curves с `matplotlib`

---

**Версия:** 1.0
**Последнее обновление:** 2025-10-11
**Следующие шаги:** Начни с Фазы 1 чеклиста ↑
