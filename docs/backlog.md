# Backlog изменений

## 2025-10-08: Объединение base_strategy.py и strategy_registry.py

### Изменения:
- Объединил [`base_strategy.py`](src/strategies/base_strategy.py) и [`strategy_registry.py`](src/strategies/strategy_registry.py) в один файл
- Удалил дублирование кода и ненужный файл
- Добавил методы реестра напрямую в класс [`BaseStrategy`](src/strategies/base_strategy.py:26)
- Обновил все импорты в проекте для использования новой структуры

### Технические детали:
- Перенес класс [`StrategyRegistry`](src/strategies/base_strategy.py:11) в [`BaseStrategy.py`](src/strategies/base_strategy.py:1)
- Добавил методы: [`register()`](src/strategies/base_strategy.py:95), [`get_strategy()`](src/strategies/base_strategy.py:115), [`list_strategies()`](src/strategies/base_strategy.py:125), [`create_strategy()`](src/strategies/base_strategy.py:136)
- Создал псевдонимы для обратной совместимости: `StrategyRegistry = BaseStrategy`, `StrategyFactory = BaseStrategy`
- Обновил импорты в файлах:
  - [`src/strategies/__init__.py`](src/strategies/__init__.py)
  - [`src/strategies/turbo_mean_reversion_strategy.py`](src/strategies/turbo_mean_reversion_strategy.py)
  - [`src/tui/optimization_app.py`](src/tui/optimization_app.py)
  - [`src/optimization/fast_optimizer.py`](src/optimization/fast_optimizer.py)
  - [`src/core/backtest_manager.py`](src/core/backtest_manager.py)
  - [`src/core/config_validator.py`](src/core/config_validator.py)

### Причина изменения:
- Упрощение архитектуры и уменьшение количества файлов
- Устранение дублирования кода
- Улучшение поддержки и читаемости кода
- Следование принципу KISS (Keep It Simple, Stupid)

### Результат:
- Все функциональные возможности сохранены
- Обратная совместимость поддерживается через псевдонимы
- Тестирование подтвердило работоспособность всех методов
- Код стал более лаконичным и поддерживаемым