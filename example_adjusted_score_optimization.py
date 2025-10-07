"""
Пример использования метрики adjusted_score для оптимизации стратегии

Этот пример показывает, как запустить оптимизацию с использованием
новой метрики adjusted_score.
"""

import os
import sys

# Добавляем src в путь
sys.path.append('src')

def main():
    """Основная функция примера"""
    print("=" * 60)
    print("ПРИМЕР ОПТИМИЗАЦИИ С МЕТРИКОЙ ADJUSTED_SCORE")
    print("=" * 60)
    
    try:
        # Импортируем необходимые модули
        from src.optimization.fast_optimizer import FastStrategyOptimizer
        from src.strategies.strategy_registry import StrategyRegistry
        
        # Проверяем доступные стратегии
        strategies = StrategyRegistry.list_strategies()
        print(f"Доступные стратегии: {strategies}")
        
        if not strategies:
            print("Ошибка: Нет доступных стратегий")
            return
        
        # Используем первую доступную стратегию
        strategy_name = strategies[0]
        print(f"Используем стратегию: {strategy_name}")
        
        # Проверяем наличие данных
        data_dir = "upload/klines"
        if not os.path.exists(data_dir):
            print(f"Ошибка: Директория {data_dir} не существует")
            return
        
        # Ищем CSV файлы
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"Ошибка: В директории {data_dir} нет CSV файлов")
            return
        
        # Используем первый найденный файл
        data_file = csv_files[0]
        data_path = os.path.join(data_dir, data_file)
        symbol = data_file.split('-')[0] if '-' in data_file else data_file.split('.')[0]
        
        print(f"Используем данные: {data_file}")
        print(f"Символ: {symbol}")
        
        # Создаем оптимизатор
        print("\nСоздание оптимизатора...")
        optimizer = FastStrategyOptimizer(
            strategy_name=strategy_name,
            data_path=data_path,
            symbol=symbol
        )
        
        # Запускаем оптимизацию с метрикой adjusted_score
        print("\nЗапуск оптимизации с метрикой adjusted_score...")
        print("Это может занять несколько минут...")
        
        results = optimizer.optimize(
            n_trials=20,  # Небольшое количество для примера
            objective_metric='adjusted_score',  # Используем новую метрику
            min_trades=30,  # Минимальное количество сделок
            max_drawdown_threshold=50.0,  # Максимальная просадка
            n_jobs=-1  # Используем все ядра
        )
        
        # Выводим результаты
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        print("=" * 60)
        
        print(f"Стратегия: {results.get('strategy_name', 'N/A')}")
        print(f"Символ: {results.get('symbol', 'N/A')}")
        print(f"Лучшая метрика (adjusted_score): {results.get('best_value', 0):.4f}")
        print(f"Всего испытаний: {results.get('n_trials', 0)}")
        print(f"Успешных испытаний: {results.get('successful_trials', 0)}")
        print(f"Обрезанных испытаний: {results.get('pruned_trials', 0)}")
        print(f"Время оптимизации: {results.get('optimization_time_seconds', 0):.2f} сек")
        
        # Лучшие параметры
        print("\nЛучшие параметры:")
        best_params = results.get('best_params', {})
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Результаты финального бэктеста
        final_backtest = results.get('final_backtest', {})
        if final_backtest:
            print("\nФинальный бэктест:")
            print(f"  Всего сделок: {final_backtest.get('total', 0)}")
            print(f"  Win Rate: {final_backtest.get('win_rate', 0):.2%}")
            print(f"  Net P&L: ${final_backtest.get('net_pnl', 0):.2f}")
            print(f"  Доходность: {final_backtest.get('net_pnl_percentage', 0):.2f}%")
            print(f"  Sharpe Ratio: {final_backtest.get('sharpe_ratio', 0):.2f}")
            print(f"  Profit Factor: {final_backtest.get('profit_factor', 0):.2f}")
            print(f"  Adjusted Score: {final_backtest.get('adjusted_score', 0):.4f}")
            print(f"  Макс. просадка: {final_backtest.get('max_drawdown', 0):.2f}%")
        
        print("\n✅ Оптимизация успешно завершена!")
        print("Метрика adjusted_score успешно использована для оптимизации стратегии.")
        
    except Exception as e:
        print(f"❌ Ошибка при выполнении оптимизации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()