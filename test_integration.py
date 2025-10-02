#!/usr/bin/env python3
"""
Тестирование полной интеграции системы фабрики стратегий
"""

import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_strategy_factory():
    """Тестирование фабрики стратегий"""
    print("=== Тестирование фабрики стратегий ===")
    
    try:
        from src.strategies.strategy_factory import StrategyFactory
        
        # Получение списка стратегий
        strategies = StrategyFactory.list_available_strategies()
        print(f"✓ Доступные стратегии: {strategies}")
        
        # Создание стратегии
        strategy = StrategyFactory.create('bollinger', 'BTCUSDT', period=20, std_dev=2.0)
        print(f"✓ Стратегия создана: {type(strategy).__name__}")
        
        # Получение информации о стратегии
        info = StrategyFactory.get_strategy_info('bollinger')
        print(f"✓ Информация о стратегии: {info['name']}")
        print(f"✓ Параметры по умолчанию: {info['default_params']}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в фабрике стратегий: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_config():
    """Тестирование конфигурации стратегии"""
    print("\n=== Тестирование конфигурации стратегии ===")
    
    try:
        from src.gui.config.config_models import StrategyConfig
        
        # Создание конфигурации
        config = StrategyConfig('bollinger')
        print(f"✓ Конфигурация создана со стратегией: {config.strategy_name}")
        print(f"✓ Параметры по умолчанию: {config.strategy_params}")
        
        # Получение доступных стратегий
        from src.strategies.strategy_factory import StrategyFactory
        strategies = StrategyFactory.list_available_strategies()
        print(f"✓ Доступные стратегии: {strategies}")
        
        # Обновление стратегии
        config.update_strategy('bollinger')
        print(f"✓ Обновление стратегии работает")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в конфигурации стратегии: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtest_integration():
    """Тестирование интеграции с бэктестингом"""
    print("\n=== Тестирование интеграции с бэктестингом ===")
    
    try:
        from src.data.vectorized_klines_backtest import run_vectorized_klines_backtest
        from src.strategies.strategy_factory import StrategyFactory
        
        # Создание тестовых данных
        import numpy as np
        import pandas as pd
        
        # Создаем простые тестовые данные
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        
        # Создаем OHLCV данные с правильными именами столбцов
        ohlcv = pd.DataFrame({
            'Symbol': 'TEST',
            'time': (dates.astype(np.int64) // 10**9),  # Конвертируем в Unix timestamp
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
        # Сохраняем во временный файл
        temp_file = 'temp_test_data.csv'
        ohlcv.to_csv(temp_file, index=False)
        
        # Запуск бэктестинга
        # Note: initial_capital and commission_pct are passed separately, not in strategy_params
        strategy_params = {'period': 10, 'std_dev': 2.0, 'stop_loss_pct': 1.0}
        results = run_vectorized_klines_backtest(
            csv_path=temp_file,
            symbol='TEST',
            strategy_name='bollinger',
            strategy_params=strategy_params,
            initial_capital=10000.0,
            commission_pct=0.001
        )
        
        print(f"✓ Бэктестинг выполнен успешно")
        print(f"✓ Финальный капитал: {results.get('net_pnl', 0):.2f}")
        print(f"✓ Общая доходность: {results.get('net_pnl_percentage', 0):.2f}%")
        
        # Удаляем временный файл
        os.remove(temp_file)
        
        return True
    except Exception as e:
        print(f"❌ Ошибка в интеграции с бэктестингом: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Основная функция тестирования"""
    print("Начало тестирования интеграции системы фабрики стратегий\n")
    
    results = []
    
    # Тестирование фабрики стратегий
    results.append(test_strategy_factory())
    
    # Тестирование конфигурации стратегии
    results.append(test_strategy_config())
    
    # Тестирование интеграции с бэктестингом
    results.append(test_backtest_integration())
    
    # Итоги
    print("\n=== Итоги тестирования ===")
    passed = sum(results)
    total = len(results)
    
    print(f"Пройдено тестов: {passed}/{total}")
    
    if passed == total:
        print("✅ Все тесты пройдены успешно!")
        return 0
    else:
        print("❌ Некоторые тесты не пройдены")
        return 1

if __name__ == "__main__":
    sys.exit(main())