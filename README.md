# Backtrader Optimization System

## Быстрый старт

```bash
python tui_runner.py
```

## Workflow

### Step 1: Оптимизация
1. Режим: **Оптимизация**
2. Датасет: **📁 datasets**
3. Trials: **100**
4. Нажать `r`

→ Результаты в `optimization_results/`

### Step 2: Валидация
1. Режим: **Cross-Asset Validation**
2. Датасет: **📁 datasets**
3. Нажать `r`

→ Результаты в `wfo_results/`

## Результаты

**Robust Trials**: `wfo_results/robust_trials_*.csv`
- Параметры которые работают на многих монетах
- Колонки: trial_id, positive_ratio, avg_pnl, avg_sharpe

**Validation Matrix**: `wfo_results/cross_asset_validation_*.csv`
- Полная матрица: trial × dataset → metrics

## Структура

```
backtrader/
├── tui_runner.py              # Entry point
├── src/
│   ├── optimization/          # Optimization engine
│   ├── strategies/            # Trading strategies
│   ├── tui/                   # Terminal UI
│   └── data/                  # Data handlers
├── upload/klines/datasets/    # Input data
├── optimization_results/      # Step 1 output
└── wfo_results/              # Step 2 output
```
