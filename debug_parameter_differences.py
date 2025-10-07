#!/usr/bin/env python3
"""
Отладка различий в параметрах между стандартным бектестом и Optuna
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import BacktestManager, BacktestConfig
from src.optimization.fast_optimizer import FastStrategyOptimizer
from src.data.backtest_engine import run_vectorized_klines_backtest

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

def test_standard_backtest(data_path, symbol, params):
    """Тест стандартного бектеста"""
    print("="*60)
    print("СТАНДАРТНЫЙ БЕКТЕСТ (BacktestManager)")
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
        print("РЕЗУЛЬТАТЫ:")
        print(f"Всего сделок: {results_dict.get('total', 0)}")
        print(f"Sharpe Ratio: {results_dict.get('sharpe_ratio', 0):.4f}")
        print(f"Win Rate: {results_dict.get('win_rate', 0):.2%}")
        print(f"Чистая P&L: ${results_dict.get('net_pnl', 0):.2f}")
        print(f"Profit Factor: {results_dict.get('profit_factor', 0):.2f}")
        return results_dict
    else:
        print(f"Ошибка: {results.get_error()}")
        return None

def test_direct_backtest(data_path, symbol, params):
    """Тест прямого вызова бектест движка"""
    print("\n" + "="*60)
    print("ПРЯМОЙ ВЫЗОВ БЕКТЕСТ ДВИЖКА (run_vectorized_klines_backtest)")
    print("="*60)
    
    # Подготовка параметров
    strategy_params = params.copy()
    # Удаляем параметры, которые передаются отдельно
    strategy_params.pop('initial_capital', None)
    strategy_params.pop('commission_pct', None)
    
    print(f"Параметры стратегии: {strategy_params}")
    print(f"Initial Capital: {10000.0}")
    print(f"Commission: {0.05}")
    
    results = run_vectorized_klines_backtest(
        csv_path=data_path,
        symbol=symbol,
        strategy_name='hierarchical_mean_reversion',
        strategy_params=strategy_params,
        initial_capital=10000.0,
        commission_pct=0.05
    )
    
    if 'error' not in results:
        print("РЕЗУЛЬТАТЫ:")
        print(f"Всего сделок: {results.get('total', 0)}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Чистая P&L: ${results.get('net_pnl', 0):.2f}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        return results
    else:
        print(f"Ошибка: {results['error']}")
        return None

def test_optuna_direct(data_path, symbol, params):
    """Тест прямого вызова Optuna с дефолтными параметрами"""
    print("\n" + "="*60)
    print("ПРЯМОЙ ВЫЗОВ OPTUNA С ДЕФОЛТНЫМИ ПАРАМЕТРАМИ")
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
        use_adaptive=False
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
        print(f"Параметры: {params}")
        objective_value = objective(mock_trial)
        print("\nРЕЗУЛЬТАТЫ:")
        print(f"Objective (Sharpe Ratio): {objective_value:.4f}")
        return objective_value
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_optuna_with_modified_params(data_path, symbol, params):
    """Тест Optuna с модифицированными параметрами"""
    print("\n" + "="*60)
    print("ТЕСТ OPTUNA С МОДИФИЦИРОВАННЫМИ ПАРАМЕТРАМИ")
    print("="*60)
    
    # Модифицируем параметры для теста
    modified_params = params.copy()
    modified_params['commission_pct'] = 0.0005  # Изменяем комиссию
    
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
        use_adaptive=False
    )
    
    # Создаем mock trial
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
    
    # Запускаем с модифицированными параметрами
    mock_trial = MockTrial(modified_params)
    try:
        print(f"Параметры: {modified_params}")
        objective_value = objective(mock_trial)
        print("\nРЕЗУЛЬТАТЫ:")
        print(f"Objective (Sharpe Ratio): {objective_value:.4f}")
        return objective_value
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Основная функция"""
    print("ОТЛАДКА РАЗЛИЧИЙ В ПАРАМЕТРАХ")
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
    
    # 1. Стандартный бектест
    standard_results = test_standard_backtest(data_path, symbol, default_params)
    
    # 2. Прямой вызов бектест движка
    direct_results = test_direct_backtest(data_path, symbol, default_params)
    
    # 3. Прямой вызов Optuna
    optuna_results = test_optuna_direct(data_path, symbol, default_params)
    
    # 4. Тест с модифицированными параметрами
    modified_results = test_optuna_with_modified_params(data_path, symbol, default_params)
    
    # Сравнение результатов
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    if standard_results:
        print(f"Стандартный бектест:")
        print(f"  Sharpe Ratio: {standard_results.get('sharpe_ratio', 0):.4f}")
        print(f"  Всего сделок: {standard_results.get('total', 0)}")
    
    if direct_results:
        print(f"\nПрямой вызов бектест движка:")
        print(f"  Sharpe Ratio: {direct_results.get('sharpe_ratio', 0):.4f}")
        print(f"  Всего сделок: {direct_results.get('total', 0)}")
        
        if standard_results:
            diff = abs(direct_results.get('sharpe_ratio', 0) - standard_results.get('sharpe_ratio', 0))
            print(f"  Разница со стандартным: {diff:.4f}")
    
    if optuna_results is not None:
        print(f"\nПрямой вызов Optuna:")
        print(f"  Sharpe Ratio: {optuna_results:.4f}")
        
        if standard_results:
            diff = abs(optuna_results - standard_results.get('sharpe_ratio', 0))
            print(f"  Разница со стандартным: {diff:.4f}")
    
    if modified_results is not None:
        print(f"\nOptuna с модифицированными параметрами:")
        print(f"  Sharpe Ratio: {modified_results:.4f}")
        
        if optuna_results is not None:
            diff = abs(modified_results - optuna_results)
            print(f"  Разница с Optuna дефолт: {diff:.4f}")
    
    print("\n" + "="*60)
    print("АНАЛИЗ ПАРАМЕТРОВ")
    print("="*60)
    
    print("Проверка обработки параметров:")
    print("1. BacktestManager передает параметры в run_vectorized_klines_backtest")
    print("2. run_vectorized_klines_backtest удаляет initial_capital и commission_pct из strategy_params")
    print("3. Optuna использует все параметры, включая initial_capital и commission_pct")
    print("4. Различия в результатах могут быть из-за разной обработки комиссии")
    
    print("\nВозможные причины различий:")
    print("1. Разная обработка параметра commission_pct")
    print("2. Разная обработка параметра initial_capital")
    print("3. Разные пути выполнения кода")
    
    print("="*60)
    print("ОТЛАДКА ЗАВЕРШЕНА")

if __name__ == "__main__":
    main()