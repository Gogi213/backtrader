#!/usr/bin/env python3
"""
Запуск бектеста с дефолтными параметрами стратегии
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import BacktestManager, BacktestConfig

def run_default_backtest():
    """Запуск бектеста с дефолтными параметрами"""
    
    # Путь к данным
    data_path = "upload/klines/ASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv"
    symbol = "ASTERUSDT"
    
    # Дефолтные параметры стратегии
    default_params = {
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
    
    # Создание конфигурации
    config = BacktestConfig(
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
    
    print("="*60)
    print("ДЕФОЛТНЫЙ БЕКТЕСТ - ИСХОДНЫЕ ПАРАМЕТРЫ")
    print("="*60)
    print(f"Стратегия: {config.strategy_name}")
    print(f"Символ: {config.symbol}")
    print(f"Начальный капитал: ${config.initial_capital:,.2f}")
    print(f"Комиссия: {config.commission_pct}%")
    print(f"Размер позиции: ${config.position_size_dollars:,.2f}")
    print(f"Путь к данным: {config.data_path}")
    print("\nПараметры стратегии:")
    for param, value in default_params.items():
        print(f"  {param}: {value}")
    print("="*60)
    
    # Запуск бектеста
    manager = BacktestManager()
    results = manager.run_backtest(config)
    
    if results.is_successful():
        results_dict = results.to_dict()
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ БЕКТЕСТА")
        print("="*60)
        
        # Основные метрики
        print(f"Всего сделок: {results_dict.get('total', 0)}")
        print(f"Выигрышных сделок: {results_dict.get('total_winning_trades', 0)}")
        print(f"Проигрышных сделок: {results_dict.get('total_losing_trades', 0)}")
        print(f"Win Rate: {results_dict.get('win_rate', 0):.2%}")
        print(f"Чистая P&L: ${results_dict.get('net_pnl', 0):.2f}")
        print(f"Доходность: {results_dict.get('net_pnl_percentage', 0):.2f}%")
        print(f"Максимальная просадка: {results_dict.get('max_drawdown', 0):.2f}%")
        print(f"Sharpe Ratio: {results_dict.get('sharpe_ratio', 0):.2f}")
        print(f"Profit Factor: {results_dict.get('profit_factor', 0):.2f}")
        print(f"Средний выигрыш: ${results_dict.get('average_win', 0):.2f}")
        print(f"Средний проигрыш: ${results_dict.get('average_loss', 0):.2f}")
        print(f"Крупнейший выигрыш: ${results_dict.get('largest_win', 0):.2f}")
        print(f"Крупнейший проигрыш: ${results_dict.get('largest_loss', 0):.2f}")
        print(f"Макс. серия стоп-лоссов: {results_dict.get('loose_streak', 0)}")
        print(f"Всего баров: {results_dict.get('total_bars', 0)}")
        print(f"Обучающих баров: {results_dict.get('train_bars', 0)}")
        
        # Дополнительная информация
        if 'trades' in results_dict and results_dict['trades']:
            trades = results_dict['trades']
            if trades:
                durations = [t.get('duration', 0) for t in trades]
                avg_duration = sum(durations) / len(durations) if durations else 0
                print(f"Средняя длительность сделки: {avg_duration:.2f} минут")
                
                # Анализ выходов
                exit_reasons = {}
                for trade in trades:
                    reason = trade.get('exit_reason', 'unknown')
                    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                
                print("\nПричины выхода:")
                for reason, count in exit_reasons.items():
                    print(f"  {reason}: {count} ({count/len(trades):.1%})")
        
        print("="*60)
        print("БЕКТЕСТ УСПЕШНО ЗАВЕРШЁН")
        print("="*60)
        
        return results_dict
    else:
        print(f"Ошибка бектеста: {results.get_error()}")
        return None

if __name__ == "__main__":
    run_default_backtest()