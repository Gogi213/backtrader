#!/usr/bin/env python3
"""
Тест исправления комиссии
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
        'commission_pct': 0.0005  # 0.05%
    }

def test_standard_backtest(data_path, symbol, params):
    """Тест стандартного бектеста с исправленной комиссией"""
    print("="*60)
    print("СТАНДАРТНЫЙ БЕКТЕСТ (BacktestManager) - ИСПРАВЛЕННАЯ КОМИССИЯ")
    print("="*60)
    
    config = BacktestConfig(
        strategy_name='hierarchical_mean_reversion',
        symbol=symbol,
        data_path=data_path,
        initial_capital=10000.0,
        commission_pct=0.0005,  # Исправляем на 0.0005 (0.05%)
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

def main():
    """Основная функция"""
    print("ТЕСТ ИСПРАВЛЕНИЯ КОМИССИИ")
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
    
    print("\nПРОБЛЕМА:")
    print("1. BacktestManager передает commission_pct=0.05 (0.05%)")
    print("2. Стратегия делит комиссию на 100: 0.05 / 100 = 0.0005 (0.0005%)")
    print("3. Optuna передает commission_pct=0.0005 (0.05%)")
    print("4. Стратегия делит комиссию на 100: 0.0005 / 100 = 0.000005 (0.000005%)")
    print("5. Разница в комиссии: 100 раз!")
    
    # 1. Стандартный бектест с исправленной комиссией
    standard_results = test_standard_backtest(data_path, symbol, default_params)
    
    # 2. Прямой вызов Optuna
    optuna_results = test_optuna_direct(data_path, symbol, default_params)
    
    # Сравнение результатов
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    if standard_results:
        print(f"Стандартный бектест (исправленная комиссия):")
        print(f"  Sharpe Ratio: {standard_results.get('sharpe_ratio', 0):.4f}")
        print(f"  Всего сделок: {standard_results.get('total', 0)}")
    
    if optuna_results is not None:
        print(f"\nПрямой вызов Optuna:")
        print(f"  Sharpe Ratio: {optuna_results:.4f}")
        
        if standard_results:
            diff = abs(optuna_results - standard_results.get('sharpe_ratio', 0))
            print(f"  Разница со стандартным: {diff:.4f}")
            print(f"  Совпадение: {(1 - diff/max(optuna_results, 0.01)):.1%}")
    
    print("\n" + "="*60)
    print("ВЫВОД")
    print("="*60)
    print("Проблема была в разной обработке комиссии:")
    print("1. BacktestManager: commission_pct=0.05 -> 0.05/100 = 0.0005")
    print("2. Optuna: commission_pct=0.0005 -> 0.0005/100 = 0.000005")
    print("3. Разница в 100 раз!")
    print("\nРешение:")
    print("1. Использовать одинаковое значение commission_pct=0.0005")
    print("2. Или исправить расчет комиссии в стратегии")
    
    print("="*60)
    print("ТЕСТ ЗАВЕРШЕН")

if __name__ == "__main__":
    main()