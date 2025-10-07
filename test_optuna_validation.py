#!/usr/bin/env python3
"""
Валидация Optuna оптимизации с дефолтными параметрами
Сравнение результатов Optuna с обычным бектестером
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import BacktestManager, BacktestConfig
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

def run_standard_backtest(data_path, symbol, params):
    """Запуск стандартного бектеста"""
    print("="*60)
    print("СТАНДАРТНЫЙ БЕКТЕСТ")
    print("="*60)
    
    config = BacktestConfig(
        strategy_name='hierarchical_mean_reversion',
        symbol=symbol,
        data_path=data_path,
        initial_capital=10000.0,
        commission_pct=0.05,
        position_size_dollars=1000.0,
        strategy_params=params,
        enable_turbo_mode=True,
        verbose=True
    )
    
    manager = BacktestManager()
    results = manager.run_backtest(config)
    
    if results.is_successful():
        results_dict = results.to_dict()
        print("РЕЗУЛЬТАТЫ СТАНДАРТНОГО БЕКТЕСТА:")
        print("-" * 40)
        print(f"Всего сделок: {results_dict.get('total', 0)}")
        print(f"Win Rate: {results_dict.get('win_rate', 0):.2%}")
        print(f"Чистая P&L: ${results_dict.get('net_pnl', 0):.2f}")
        print(f"Доходность: {results_dict.get('net_pnl_percentage', 0):.2f}%")
        print(f"Sharpe Ratio: {results_dict.get('sharpe_ratio', 0):.2f}")
        print(f"Profit Factor: {results_dict.get('profit_factor', 0):.2f}")
        print(f"Макс. просадка: {results_dict.get('max_drawdown', 0):.2f}%")
        print(f"Выигрышных сделок: {results_dict.get('total_winning_trades', 0)}")
        print(f"Проигрышных сделок: {results_dict.get('total_losing_trades', 0)}")
        print(f"Средний выигрыш: ${results_dict.get('average_win', 0):.2f}")
        print(f"Средний проигрыш: ${results_dict.get('average_loss', 0):.2f}")
        print(f"Крупнейший выигрыш: ${results_dict.get('largest_win', 0):.2f}")
        print(f"Крупнейший проигрыш: ${results_dict.get('largest_loss', 0):.2f}")
        print(f"Макс. серия стоп-лоссов: {results_dict.get('loose_streak', 0)}")
        print("="*60)
        return results_dict
    else:
        print(f"Ошибка: {results.get_error()}")
        return None

def run_optuna_single_trial(data_path, symbol, params):
    """Запуск одиночного прогона Optuna с дефолтными параметрами"""
    print("="*60)
    print("OPTUNA ОДИНОЧНЫЙ ПРОГОН (ДЕФОЛТНЫЕ ПАРАМЕТРЫ)")
    print("="*60)
    
    # Создаем оптимизатор
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
    mock_trial = MockTrial(params)
    try:
        objective_value = objective(mock_trial)
        print("РЕЗУЛЬТАТЫ OPTUNA ОДИНОЧНОГО ПРОГОНА:")
        print("-" * 40)
        print(f"Objective (Sharpe Ratio): {objective_value:.4f}")
        print("="*60)
        return objective_value
    except Exception as e:
        print(f"Ошибка в Optuna прогоне: {e}")
        return None

def run_optuna_optimization(data_path, symbol, n_trials=5):
    """Запуск полноценной Optuna оптимизации"""
    print("="*60)
    print(f"OPTUNA ОПТИМИЗАЦИЯ ({n_trials} TRIALS)")
    print("="*60)
    
    # Создаем оптимизатор
    optimizer = FastStrategyOptimizer(
        strategy_name='hierarchical_mean_reversion',
        data_path=data_path,
        symbol=symbol,
        cache_dir="optimization_cache"
    )
    
    # Запускаем оптимизацию
    results = optimizer.optimize(
        n_trials=n_trials,
        objective_metric='sharpe_ratio',
        min_trades=10,
        max_drawdown_threshold=50.0,
        timeout=None,
        n_jobs=1,  # Используем 1 ядро для детального сравнения
        use_adaptive=False,  # Используем полные данные
        sampler=None,  # Дефолтный сэмплер
        pruner=None   # Без прунинга для первых тестов
    )
    
    print("РЕЗУЛЬТАТЫ OPTUNA ОПТИМИЗАЦИИ:")
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
        print("\nФИНАЛЬНЫЙ БЕКТЕСТ С ЛУЧШИМИ ПАРАМЕТРАМИ:")
        print(f"Всего сделок: {final_backtest.get('total', 0)}")
        print(f"Win Rate: {final_backtest.get('win_rate', 0):.2%}")
        print(f"Чистая P&L: ${final_backtest.get('net_pnl', 0):.2f}")
        print(f"Доходность: {final_backtest.get('net_pnl_percentage', 0):.2f}%")
        print(f"Sharpe Ratio: {final_backtest.get('sharpe_ratio', 0):.2f}")
        print(f"Profit Factor: {final_backtest.get('profit_factor', 0):.2f}")
        print(f"Макс. просадка: {final_backtest.get('max_drawdown', 0):.2f}%")
    
    print("="*60)
    return results

def compare_results(standard_results, optuna_single, optuna_optimization):
    """Сравнение результатов"""
    print("="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    if standard_results:
        print("СТАНДАРТНЫЙ БЕКТЕСТ:")
        print(f"  Sharpe Ratio: {standard_results.get('sharpe_ratio', 0):.4f}")
        print(f"  Всего сделок: {standard_results.get('total', 0)}")
        print(f"  Win Rate: {standard_results.get('win_rate', 0):.2%}")
        print(f"  Чистая P&L: ${standard_results.get('net_pnl', 0):.2f}")
    
    if optuna_single is not None:
        print("\nOPTUNA ОДИНОЧНЫЙ ПРОГОН:")
        print(f"  Sharpe Ratio: {optuna_single:.4f}")
        
        if standard_results:
            diff = abs(optuna_single - standard_results.get('sharpe_ratio', 0))
            print(f"  Разница со стандартным: {diff:.4f}")
            print(f"  Совпадение: {(1 - diff/max(optuna_single, 0.01)):.1%}")
    
    if optuna_optimization:
        best_value = optuna_optimization.get('best_value', 0)
        final_backtest = optuna_optimization.get('final_backtest', {})
        
        print(f"\nOPTUNA ОПТИМИЗАЦИЯ:")
        print(f"  Лучший Sharpe Ratio: {best_value:.4f}")
        print(f"  Финальный Sharpe Ratio: {final_backtest.get('sharpe_ratio', 0):.4f}")
        print(f"  Всего сделок: {final_backtest.get('total', 0)}")
        print(f"  Win Rate: {final_backtest.get('win_rate', 0):.2%}")
        print(f"  Чистая P&L: ${final_backtest.get('net_pnl', 0):.2f}")
        
        if standard_results:
            diff = abs(final_backtest.get('sharpe_ratio', 0) - standard_results.get('sharpe_ratio', 0))
            print(f"  Разница финального со стандартным: {diff:.4f}")
            print(f"  Совпадение: {(1 - diff/max(final_backtest.get('sharpe_ratio', 0.01), 0.01)):.1%}")
    
    print("="*60)

def main():
    """Основная функция"""
    print("ВАЛИДАЦИЯ OPTUNA ОПТИМИЗАЦИИ")
    print("Сравнение результатов Optuna с обычным бектестером")
    print("="*60)
    
    # Параметры
    data_path = "upload/klines/ASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv"
    symbol = "ASTERUSDT"
    default_params = get_default_params()
    
    print(f"Данные: {data_path}")
    print(f"Символ: {symbol}")
    print(f"Дефолтные параметры: {len(default_params)}")
    print("="*60)
    
    # 1. Запуск стандартного бектеста
    standard_results = run_standard_backtest(data_path, symbol, default_params)
    
    # 2. Запуск одиночного прогона Optuna с дефолтными параметрами
    optuna_single = run_optuna_single_trial(data_path, symbol, default_params)
    
    # 3. Запуск короткой оптимизации Optuna
    optuna_optimization = run_optuna_optimization(data_path, symbol, n_trials=5)
    
    # 4. Сравнение результатов
    compare_results(standard_results, optuna_single, optuna_optimization)
    
    print("\nВАЛИДАЦИЯ ЗАВЕРШЕНА")

if __name__ == "__main__":
    main()