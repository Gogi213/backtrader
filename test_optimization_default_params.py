#!/usr/bin/env python3
"""
Тест оптимизации с принудительными дефолтными параметрами
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import OptimizationManager, OptimizationConfig, BacktestConfig
from src.optimization.fast_optimizer import FastStrategyOptimizer

def get_default_params():
    """Получить дефолтные параметры стратегии"""
    return {
        'initial_kf_mean': 100.0,
        'initial_kf_cov': 1.0,
        'measurement_noise_r': 5.0,
        'process_noise_q': 0.1,
        'hmm_window_size': 30,
        'prob_threshold_trend': 0.6,
        'prob_threshold_sideways': 0.5,
        'prob_threshold_dead': 0.85,
        'sigma_dead_threshold': 1.0,
        'ou_window_size': 50,
        'hl_min': 1.0,
        'hl_max': 120.0,
        'relative_uncertainty_threshold': 0.8,
        'uncertainty_threshold': 0.8,
        's_entry': 0.05,
        'z_stop': 4.0,
        'timeout_multiplier': 3.0,
        'initial_capital': 10000.0,
        'commission_pct': 0.0005
    }

def test_optimization_with_default_params():
    """Тест оптимизации с принудительными дефолтными параметрами"""
    print("="*60)
    print("ТЕСТ ОПТИМИЗАЦИИ С ПРИНУДИТЕЛЬНЫМИ ДЕФОЛТНЫМИ ПАРАМЕТРАМИ")
    print("="*60)
    
    # Параметры
    data_path = "upload/klines/ASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv"
    symbol = "ASTERUSDT"
    default_params = get_default_params()
    
    print(f"Данные: {data_path}")
    print(f"Символ: {symbol}")
    print(f"Дефолтные параметры:")
    for param, value in default_params.items():
        print(f"  {param}: {value}")
    print("="*60)
    
    # Создаем оптимизатор напрямую
    optimizer = FastStrategyOptimizer(
        strategy_name='hierarchical_mean_reversion',
        data_path=data_path,
        symbol=symbol,
        cache_dir="optimization_cache"
    )
    
    # Создаем objective функцию
    objective = optimizer.create_objective_function(
        objective_metric='sharpe_ratio',
        min_trades=10,
        max_drawdown_threshold=50.0,
        use_adaptive=False  # Используем полные данные
    )
    
    # Создаем mock trial для тестирования с дефолтными параметрами
    class MockTrial:
        def __init__(self, params):
            self.number = 0
            self._params = params
            self._reported_values = {}
            
        def suggest_float(self, name, low, high, *args):
            return self._params.get(name, (low + high) / 2)
            
        def suggest_int(self, name, low, high, *args):
            return int(self._params.get(name, (low + high) / 2))
            
        def suggest_categorical(self, name, choices):
            return self._params.get(name, choices[0])
            
        def report(self, value, step):
            self._reported_values[step] = value
            
        def should_prune(self):
            return False
    
    # Запускаем с дефолтными параметрами
    mock_trial = MockTrial(default_params)
    try:
        objective_value = objective(mock_trial)
        print("\nРЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ С ДЕФОЛТНЫМИ ПАРАМЕТРАМИ:")
        print("-" * 40)
        print(f"Objective (Sharpe Ratio): {objective_value:.4f}")
        
        # Сравнение с результатами стандартного бектеста
        standard_sharpe = 8.7657  # Из предыдущего теста
        diff = abs(objective_value - standard_sharpe)
        print(f"Стандартный Sharpe Ratio: {standard_sharpe:.4f}")
        print(f"Разница: {diff:.4f}")
        print(f"Совпадение: {(1 - diff/max(objective_value, 0.01)):.1%}")
        
        print("="*60)
        return objective_value
    except Exception as e:
        print(f"Ошибка в прогоне: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_real_optimization():
    """Запуск реальной оптимизации с 3 trials"""
    print("\n" + "="*60)
    print("РЕАЛЬНАЯ ОПТИМИЗАЦИЯ (3 TRIALS)")
    print("="*60)
    
    # Параметры
    data_path = "upload/klines/ASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv"
    symbol = "ASTERUSDT"
    
    # Создаем оптимизатор
    optimizer = FastStrategyOptimizer(
        strategy_name='hierarchical_mean_reversion',
        data_path=data_path,
        symbol=symbol,
        cache_dir="optimization_cache"
    )
    
    # Запускаем оптимизацию
    results = optimizer.optimize(
        n_trials=3,
        objective_metric='sharpe_ratio',
        min_trades=10,
        max_drawdown_threshold=50.0,
        timeout=None,
        n_jobs=1,
        use_adaptive=False,
        sampler=None,
        pruner=None
    )
    
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print("-" * 40)
    print(f"Лучший Sharpe Ratio: {results.get('best_value', 0):.4f}")
    print(f"Всего trials: {results.get('n_trials', 0)}")
    print(f"Успешных trials: {results.get('successful_trials', 0)}")
    print(f"Время оптимизации: {results.get('optimization_time_seconds', 0):.2f} сек")
    
    # Показываем лучшие параметры
    best_params = results.get('best_params', {})
    print("\nЛУЧШИЕ ПАРАМЕТРЫ:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Показываем результаты финального бектеста
    final_backtest = results.get('final_backtest')
    if final_backtest:
        print("\nФИНАЛЬНЫЙ БЕКТЕСТ:")
        print(f"Всего сделок: {final_backtest.get('total', 0)}")
        print(f"Win Rate: {final_backtest.get('win_rate', 0):.2%}")
        print(f"Чистая P&L: ${final_backtest.get('net_pnl', 0):.2f}")
        print(f"Доходность: {final_backtest.get('net_pnl_percentage', 0):.2f}%")
        print(f"Sharpe Ratio: {final_backtest.get('sharpe_ratio', 0):.2f}")
        print(f"Profit Factor: {final_backtest.get('profit_factor', 0):.2f}")
        print(f"Макс. просадка: {final_backtest.get('max_drawdown', 0):.2f}%")
    
    print("="*60)
    return results

def main():
    """Основная функция"""
    print("ТЕСТИРОВАНИЕ ОПТИМИЗАЦИИ С ДЕФОЛТНЫМИ ПАРАМЕТРАМИ")
    print("="*60)
    
    # 1. Тест с принудительными дефолтными параметрами
    default_result = test_optimization_with_default_params()
    
    # 2. Реальная оптимизация
    real_results = run_real_optimization()
    
    # 3. Сравнение результатов
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    if default_result:
        print(f"Оптимизация с дефолтными параметрами: {default_result:.4f}")
    
    if real_results:
        best_value = real_results.get('best_value', 0)
        final_backtest = real_results.get('final_backtest', {})
        final_sharpe = final_backtest.get('sharpe_ratio', 0)
        
        print(f"Лучший результат оптимизации: {best_value:.4f}")
        print(f"Финальный Sharpe Ratio: {final_backtest.get('sharpe_ratio', 0):.4f}")
        
        if default_result:
            diff1 = abs(best_value - default_result)
            diff2 = abs(final_sharpe - default_result)
            print(f"Разница лучшего с дефолтным: {diff1:.4f}")
            print(f"Разница финального с дефолтным: {diff2:.4f}")
    
    print("="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")

if __name__ == "__main__":
    main()