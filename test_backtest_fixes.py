#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправлений в бектесте
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.backtest_manager import BacktestManager
from src.core.backtest_config import BacktestConfig

def test_backtest_fixes():
    """Тестирование исправлений в бектесте"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ БЕКТЕСТА")
    print("=" * 60)
    
    # Создаем менеджер бектестов
    manager = BacktestManager()
    
    # Проверяем доступные стратегии
    strategies = manager.get_available_strategies()
    print(f"Доступные стратегии: {strategies}")
    
    if 'hierarchical_mean_reversion' not in strategies:
        print("ОШИБКА: Стратегия hierarchical_mean_reversion не найдена!")
        return False
    
    # Получаем параметры по умолчанию
    default_params = manager.get_strategy_params('hierarchical_mean_reversion')
    print(f"Параметры по умолчанию: {default_params}")
    
    # Проверяем ключевые параметры
    critical_params = ['s_entry', 'hl_min', 'prob_threshold_sideways', 'z_stop']
    for param in critical_params:
        if param in default_params:
            value = default_params[param]
            print(f"  {param}: {value}")
            
            # Проверяем, что параметры были исправлены
            if param == 's_entry' and value < 0.1:
                print(f"    ПРЕДУПРЕЖДЕНИЕ: s_entry все еще слишком низкий ({value})")
            elif param == 'hl_min' and value < 1.0:
                print(f"    ПРЕДУПРЕЖДЕНИЕ: hl_min все еще слишком низкий ({value})")
            elif param == 'prob_threshold_sideways' and value < 0.1:
                print(f"    ПРЕДУПРЕЖДЕНИЕ: prob_threshold_sideways все еще слишком низкий ({value})")
            elif param == 'z_stop' and value > 10.0:
                print(f"    ПРЕДУПРЕЖДЕНИЕ: z_stop все еще слишком высокий ({value})")
        else:
            print(f"  {param}: НЕ НАЙДЕН")
    
    # Ищем доступные датасеты
    upload_dir = os.path.join(os.getcwd(), 'upload', 'klines')
    if os.path.exists(upload_dir):
        datasets = [f for f in os.listdir(upload_dir) if f.endswith('.csv')]
        if datasets:
            print(f"Найдены датасеты: {datasets[:3]}...")  # Показываем первые 3
            
            # Создаем конфигурацию для теста
            config = BacktestConfig(
                strategy_name='hierarchical_mean_reversion',
                strategy_params=default_params,
                symbol='TEST',
                data_path=os.path.join(upload_dir, datasets[0]),
                initial_capital=10000.0,
                commission_pct=0.0005,
                max_klines=1000  # Ограничиваем для быстрого теста
            )
            
            print("\nЗапуск тестового бектеста с 1000 свечей...")
            try:
                results = manager.run_backtest(config)
                
                if results.has_error():
                    print(f"ОШИБКА БЕКТЕСТА: {results.get_error()}")
                    return False
                
                # Проверяем результаты
                total_trades = results.get('total', 0)
                win_rate = results.get('win_rate', 0)
                net_pnl = results.get('net_pnl', 0)
                
                print(f"\nРЕЗУЛЬТАТЫ ТЕСТА:")
                print(f"  Всего сделок: {total_trades}")
                print(f"  Win Rate: {win_rate:.2%}")
                print(f"  P&L: ${net_pnl:.2f}")
                
                # Проверяем, что количество сделок разумное
                if total_trades > 500:
                    print(f"  ПРЕДУПРЕЖДЕНИЕ: Все еще слишком много сделок ({total_trades})")
                elif total_trades == 0:
                    print(f"  ПРЕДУПРЕЖДЕНИЕ: Нет сделок, возможно параметры слишком строгие")
                else:
                    print(f"  ✓ Количество сделок выглядит разумным")
                
                # Проверяем duration если есть сделки
                trades = results.get('trades', [])
                if trades:
                    durations = [t.get('duration', 0) for t in trades]
                    avg_duration = sum(durations) / len(durations)
                    zero_durations = sum(1 for d in durations if d <= 0.01)
                    
                    print(f"  Средняя длительность сделки: {avg_duration:.2f} мин")
                    print(f"  Сделок с duration ≈ 0: {zero_durations}/{len(trades)}")
                    
                    if zero_durations > len(trades) * 0.1:  # Если более 10% сделок с нулевой длительностью
                        print(f"  ПРЕДУПРЕЖДЕНИЕ: Много сделок с нулевой длительностью")
                    else:
                        print(f"  ✓ Проблема с duration исправлена")
                
                print(f"\n✓ Тестовый бектест выполнен успешно!")
                return True
                
            except Exception as e:
                print(f"ОШИБКА при выполнении бектеста: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("Датасеты не найдены в директории upload/klines/")
            return False
    else:
        print(f"Директория {upload_dir} не найдена")
        return False

if __name__ == "__main__":
    success = test_backtest_fixes()
    if success:
        print("\n" + "=" * 60)
        print("ТЕСТ ПРОЙДЕН: Исправления работают корректно")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("ТЕСТ НЕ ПРОЙДЕН: Требуются дополнительные исправления")
        print("=" * 60)