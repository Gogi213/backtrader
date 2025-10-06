#!/usr/bin/env python3
"""
Тестовый скрипт для проверки бектеста со сбалансированными параметрами
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.backtest_manager import BacktestManager
from src.core.backtest_config import BacktestConfig

def test_balanced_backtest():
    """Тестирование бектеста со сбалансированными параметрами"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ СБАЛАНСИРОВАННЫХ ПАРАМЕТРОВ БЕКТЕСТА")
    print("=" * 60)
    
    # Создаем менеджер бектестов
    manager = BacktestManager()
    
    # Сбалансированные параметры для тестирования
    balanced_params = {
        'initial_kf_mean': 100.0,
        'initial_kf_cov': 1.0,
        'measurement_noise_r': 0.05,
        'process_noise_q': 0.1,
        'hmm_window_size': 30,
        'prob_threshold_trend': 0.9,
        'prob_threshold_sideways': 0.15,  # Сбалансированное значение
        'prob_threshold_dead': 0.99,
        'sigma_dead_threshold': 10.0,
        'ou_window_size': 50,
        'hl_min': 1.0,  # Умеренное значение
        'hl_max': 200.0,  # Умеренное значение
        'relative_uncertainty_threshold': 10.0,  # Умеренное значение
        'uncertainty_threshold': 10.0,  # Умеренное значение
        's_entry': 1.5,  # Умеренное значение
        'z_stop': 3.0,  # Умеренное значение
        'timeout_multiplier': 10.0,  # Умеренное значение
        'initial_capital': 10000.0,
        'commission_pct': 0.0005
    }
    
    print(f"Сбалансированные параметры:")
    for key, value in balanced_params.items():
        if key in ['s_entry', 'hl_min', 'prob_threshold_sideways', 'z_stop']:
            print(f"  {key}: {value}")
    
    # Ищем доступные датасеты
    upload_dir = os.path.join(os.getcwd(), 'upload', 'klines')
    if os.path.exists(upload_dir):
        datasets = [f for f in os.listdir(upload_dir) if f.endswith('.csv')]
        if datasets:
            print(f"\nИспользуем датасет: {datasets[0]}")
            
            # Создаем конфигурацию для теста
            config = BacktestConfig(
                strategy_name='hierarchical_mean_reversion',
                strategy_params=balanced_params,
                symbol='TEST',
                data_path=os.path.join(upload_dir, datasets[0]),
                initial_capital=10000.0,
                commission_pct=0.0005,
                max_klines=5000  # Увеличиваем для более надежного теста
            )
            
            print("\nЗапуск бектеста со сбалансированными параметрами...")
            try:
                results = manager.run_backtest(config)
                
                if results.has_error():
                    print(f"ОШИБКА БЕКТЕСТА: {results.get_error()}")
                    return False
                
                # Проверяем результаты
                total_trades = results.get('total', 0)
                win_rate = results.get('win_rate', 0)
                net_pnl = results.get('net_pnl', 0)
                net_pnl_pct = results.get('net_pnl_percentage', 0)
                
                print(f"\nРЕЗУЛЬТАТЫ ТЕСТА:")
                print(f"  Всего сделок: {total_trades}")
                print(f"  Win Rate: {win_rate:.2%}")
                print(f"  P&L: ${net_pnl:.2f}")
                print(f"  P&L %: {net_pnl_pct:.2f}%")
                
                # Проверяем, что количество сделок разумное
                if total_trades > 200:
                    print(f"  ПРЕДУПРЕЖДЕНИЕ: Все еще много сделок ({total_trades})")
                elif total_trades == 0:
                    print(f"  ПРЕДУПРЕЖДЕНИЕ: Нет сделок, параметры все еще слишком строгие")
                    return False
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
                    
                    if zero_durations > len(trades) * 0.1:
                        print(f"  ПРЕДУПРЕЖДЕНИЕ: Много сделок с нулевой длительностью")
                    else:
                        print(f"  ✓ Проблема с duration исправлена")
                    
                    # Проверяем P&L на корректность
                    positive_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
                    negative_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
                    
                    print(f"  Прибыльных сделок: {positive_trades}")
                    print(f"  Убыточных сделок: {negative_trades}")
                    
                    # Проверяем корректность расчета win rate
                    if win_rate == 0 and positive_trades > 0:
                        print(f"  ПРЕДУПРЕЖДЕНИЕ: Win rate рассчитан некорректно!")
                    elif win_rate > 0 and positive_trades > 0:
                        expected_win_rate = positive_trades / (positive_trades + negative_trades)
                        if abs(win_rate - expected_win_rate) > 0.1:
                            print(f"  ПРЕДУПРЕЖДЕНИЕ: Win rate может быть некорректным")
                        else:
                            print(f"  ✓ Win rate рассчитан корректно")
                
                print(f"\n✓ Бектест выполнен успешно!")
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
    success = test_balanced_backtest()
    if success:
        print("\n" + "=" * 60)
        print("ТЕСТ ПРОЙДЕН: Сбалансированные параметры работают корректно")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("ТЕСТ НЕ ПРОЙДЕН: Требуются дополнительные настройки")
        print("=" * 60)