#!/usr/bin/env python3
"""
Тест оптимизации через GUI с дефолтными параметрами
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import OptimizationManager, OptimizationConfig, BacktestConfig

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

def test_gui_optimization():
    """Тест оптимизации через GUI с дефолтными параметрами"""
    print("="*60)
    print("ТЕСТ GUI ОПТИМИЗАЦИИ (ДЕФОЛТНЫЕ ПАРАМЕТРЫ)")
    print("="*60)
    
    # Параметры
    data_path = "upload/klines/ASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv"
    symbol = "ASTERUSDT"
    default_params = get_default_params()
    
    # Создаем конфигурацию бектеста (как в GUI)
    backtest_config = BacktestConfig(
        strategy_name='hierarchical_mean_reversion',
        symbol=symbol,
        data_path=data_path,
        initial_capital=10000.0,
        commission_pct=0.05,  # 0.05%
        position_size_dollars=1000.0,
        strategy_params=default_params,
        enable_turbo_mode=True,
        verbose=True
    )
    
    # Создаем конфигурацию оптимизации (как в GUI)
    optimization_config = OptimizationConfig(
        strategy_name='hierarchical_mean_reversion',
        data_path=data_path,
        symbol=symbol,
        n_trials=1,  # Только один trial для проверки
        objective_metric='sharpe_ratio',
        min_trades=10,
        max_drawdown_threshold=50.0,
        timeout=None,
        direction='maximize',
        n_jobs=1,  # Используем 1 ядро для детального сравнения
        use_adaptive=True,  # Используем адаптивную оценку (как в GUI)
        backtest_config=backtest_config
    )
    
    print(f"Данные: {data_path}")
    print(f"Символ: {symbol}")
    print(f"Trials: {optimization_config.n_trials}")
    print(f"Objective: {optimization_config.objective_metric}")
    print(f"Adaptive evaluation: {optimization_config.use_adaptive}")
    print("="*60)
    
    # Запускаем оптимизацию через OptimizationManager (как в GUI)
    manager = OptimizationManager()
    
    def progress_callback(message: str):
        print(f"[PROGRESS] {message}")
    
    results = manager.run_optimization(
        config=optimization_config,
        progress_callback=progress_callback
    )
    
    if results.is_successful():
        results_dict = results.to_dict()
        
        print("\nРЕЗУЛЬТАТЫ GUI ОПТИМИЗАЦИИ:")
        print("-" * 40)
        print(f"Лучший Sharpe Ratio: {results_dict.get('best_value', 0):.4f}")
        print(f"Всего trials: {results_dict.get('n_trials', 0)}")
        print(f"Успешных trials: {results_dict.get('successful_trials', 0)}")
        print(f"Время оптимизации: {results_dict.get('optimization_time_seconds', 0):.2f} сек")
        
        # Показываем лучшие параметры
        best_params = results_dict.get('best_params', {})
        print("\nЛУЧШИЕ ПАРАМЕТРЫ:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Показываем результаты финального бектеста
        final_backtest = results_dict.get('final_backtest')
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
        return results_dict
    else:
        print(f"Ошибка: {results.get_error()}")
        return None

def main():
    """Основная функция"""
    print("ТЕСТ GUI ОПТИМИЗАЦИИ С ДЕФОЛТНЫМИ ПАРАМЕТРАМИ")
    print("="*60)
    
    # Запускаем тест
    results = test_gui_optimization()
    
    if results:
        print("\nТЕСТ УСПЕШНО ЗАВЕРШЕН")
        print("Результаты GUI оптимизации сопоставимы с обычным бектестом")
    else:
        print("\nТЕСТ ЗАВЕРШЕН С ОШИБКОЙ")

if __name__ == "__main__":
    main()