#!/usr/bin/env python3
"""
Тест бектеста через GUI с дефолтными параметрами
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import BacktestManager, BacktestConfig

def test_gui_backtest():
    """Тест бектеста через GUI с дефолтными параметрами"""
    
    # Создаем конфигурацию с дефолтными параметрами (как в GUI)
    config = BacktestConfig(
        strategy_name='hierarchical_mean_reversion',
        symbol='ASTERUSDT',
        data_path='upload/klines/ASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv',
        initial_capital=10000.0,
        commission_pct=0.05,  # 0.05%
        position_size_dollars=1000.0,
        strategy_params={
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
        },
        enable_turbo_mode=True,
        verbose=True
    )
    
    print("="*60)
    print("GUI БЕКТЕСТ - ДЕФОЛТНЫЕ ПАРАМЕТРЫ")
    print("="*60)
    
    # Запускаем бектест через BacktestManager (как в GUI)
    manager = BacktestManager()
    results = manager.run_backtest(config)
    
    if results.is_successful():
        results_dict = results.to_dict()
        
        print("\nРЕЗУЛЬТАТЫ GUI БЕКТЕСТА:")
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
        print(f"Всего баров: {results_dict.get('total_bars', 0)}")
        print(f"Обучающих баров: {results_dict.get('train_bars', 0)}")
        
        # Анализ причин выхода
        if 'trades' in results_dict and results_dict['trades']:
            trades = results_dict['trades']
            exit_reasons = {}
            for trade in trades:
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            print("\nПричины выхода:")
            for reason, count in exit_reasons.items():
                print(f"  {reason}: {count} ({count/len(trades):.1%})")
        
        print("="*60)
        return results_dict
    else:
        print(f"Ошибка бектеста: {results.get_error()}")
        return None

if __name__ == "__main__":
    test_gui_backtest()