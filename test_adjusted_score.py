"""
Тестирование новой метрики adjusted_score с использованием существующего бэктестера

Этот тест проверяет корректность работы adjusted_score метрики
в реальных условиях бэктестинга.
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# Добавляем src в путь
sys.path.append('src')

def test_adjusted_score_with_backtester():
    """Тестируем adjusted_score с реальным бэктестером"""
    print("Тестирование adjusted_score с реальным бэктестером...")
    
    try:
        # Импортируем необходимые модули
        from src.data.backtest_engine import run_vectorized_klines_backtest
        from src.strategies.strategy_registry import StrategyRegistry
        from src.optimization.metrics import calculate_adjusted_score_from_results
        
        # Проверяем доступные стратегии
        strategies = StrategyRegistry.list_strategies()
        print(f"Доступные стратегии: {strategies}")
        
        if not strategies:
            print("Ошибка: Нет доступных стратегий")
            return False
        
        # Используем первую доступную стратегию
        strategy_name = strategies[0]
        print(f"Используем стратегию: {strategy_name}")
        
        # Проверяем наличие данных
        data_dir = "upload/klines"
        if not os.path.exists(data_dir):
            print(f"Ошибка: Директория {data_dir} не существует")
            return False
        
        # Ищем CSV файлы
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"Ошибка: В директории {data_dir} нет CSV файлов")
            return False
        
        # Используем первый найденный файл
        data_file = csv_files[0]
        data_path = os.path.join(data_dir, data_file)
        symbol = data_file.split('-')[0] if '-' in data_file else data_file.split('.')[0]
        
        print(f"Используем данные: {data_file}")
        print(f"Символ: {symbol}")
        
        # Получаем параметры стратегии по умолчанию
        strategy_class = StrategyRegistry.get(strategy_name)
        default_params = strategy_class.get_default_params()
        print(f"Параметры по умолчанию: {default_params}")
        
        # Запускаем бэктест
        print("Запуск бэктеста...")
        results = run_vectorized_klines_backtest(
            csv_path=data_path,
            symbol=symbol,
            strategy_name=strategy_name,
            strategy_params=default_params,
            initial_capital=10000.0,
            position_size=1000.0,
            commission_pct=0.05
        )
        
        if 'error' in results:
            print(f"Ошибка бэктеста: {results['error']}")
            return False
        
        # Проверяем результаты
        trades = results.get('trades', [])
        total_trades = results.get('total', 0)
        
        print(f"Результаты бэктеста:")
        print(f"  Всего сделок: {total_trades}")
        print(f"  Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"  Net P&L: ${results.get('net_pnl', 0):.2f}")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
        
        # Проверяем наличие adjusted_score в результатах
        if 'adjusted_score' in results:
            print(f"  Adjusted Score: {results.get('adjusted_score', 0):.4f}")
        else:
            print("  Adjusted Score: не рассчитан")
        
        # Рассчитываем adjusted_score вручную
        print("\nРасчет adjusted_score...")
        adjusted_score = calculate_adjusted_score_from_results(results)
        print(f"  Рассчитанный Adjusted Score: {adjusted_score:.4f}")
        
        # Проверяем корректность
        if total_trades < 30:
            expected_score = -np.inf
            if adjusted_score == expected_score:
                print("[OK] Adjusted Score корректно возвращает -inf для малого количества сделок")
            else:
                print(f"[ERROR] Ожидалось {expected_score}, получено {adjusted_score}")
                return False
        else:
            if adjusted_score > -np.inf:
                print("[OK] Adjusted Score успешно рассчитан")
            else:
                print("[ERROR] Adjusted Score не должен быть -inf для достаточного количества сделок")
                return False
        
        print("\n[OK] Тест успешно пройден!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Ошибка при выполнении теста: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adjusted_score_directly():
    """Тестируем функцию adjusted_score напрямую с тестовыми данными"""
    print("\nТестирование adjusted_score с тестовыми данными...")
    
    try:
        from src.optimization.metrics import adjusted_score
        
        # Тест 1: Недостаточно сделок
        print("Тест 1: Недостаточно сделок (< 30)")
        small_trades = [
            {'pnl': 100, 'duration': 60},
            {'pnl': -50, 'duration': 30},
            {'pnl': 150, 'duration': 90}
        ]
        score = adjusted_score(small_trades)
        if score == -np.inf:
            print("✅ Верно возвращает -inf для малого количества сделок")
        else:
            print(f"❌ Ожидалось -inf, получено {score}")
            return False
        
        # Тест 2: Достаточно сделок, все прибыльные
        print("\nТест 2: 30 прибыльных сделок")
        profitable_trades = [
            {'pnl': 100, 'duration': 60} for _ in range(30)
        ]
        score = adjusted_score(profitable_trades)
        if score > 0:
            print(f"✅ Положительный скор для прибыльных сделок: {score:.4f}")
        else:
            print(f"❌ Ожидался положительный скор, получено {score}")
            return False
        
        # Тест 3: Смешанные сделки
        print("\nТест 3: 30 смешанных сделок")
        mixed_trades = []
        np.random.seed(42)  # Для воспроизводимости
        for i in range(30):
            pnl = np.random.normal(10, 50)  # Средняя прибыль 10, ст. откл. 50
            duration = np.random.randint(30, 120)  # 30-120 минут
            mixed_trades.append({'pnl': pnl, 'duration': duration})
        
        score = adjusted_score(mixed_trades)
        print(f"✅ Скор для смешанных сделок: {score:.4f}")
        
        print("\n✅ Все прямые тесты пройдены!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при прямом тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ МЕТРИКИ ADJUSTED_SCORE")
    print("=" * 60)
    
    # Запускаем тесты
    test1_passed = test_adjusted_score_directly()
    test2_passed = test_adjusted_score_with_backtester()
    
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Прямое тестирование: {'[OK] УСПЕХ' if test1_passed else '[ERROR] ПРОВАЛ'}")
    print(f"Тестирование с бэктестером: {'[OK] УСПЕХ' if test2_passed else '[ERROR] ПРОВАЛ'}")
    
    if test1_passed and test2_passed:
        print("\n[SUCCESS] ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
        print("Метрика adjusted_score готова к использованию в оптимизации")
    else:
        print("\n[WARNING] НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        print("Необходимо исправить ошибки перед использованием")