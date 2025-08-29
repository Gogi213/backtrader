"""
Простой пример запуска бэктеста через VectorBT.
Основной интерфейс - через panel_app.py
"""
import pandas as pd
import os
from panel_app.strategies.core import run_vbt_strategy
from panel_app.data_utils.loader import load_ohlcv

def run_backtest(symbol='BTCUSDT', timeframe='1h', start='2024-08-01', end='2024-08-15', 
                 cash=10000, commission=0.05, leverage=1):
    """
    Простой запуск бэктеста Z-Score SMA стратегии.
    """
    # Загружаем данные
    df = load_ohlcv(symbol, timeframe, start, end)
    
    # Запускаем стратегию
    pf, signals = run_vbt_strategy(
        df, 
        'ZScoreSMA',
        fee=commission/100,
        init_cash=cash,
        leverage=leverage
    )
    
    # Выводим основные метрики
    try:
        stats = pf.stats()
        print(f'Symbol: {symbol}')
        print(f'Period: {start} to {end}')
        print(f'Final Portfolio Value: {stats.get("End Value", "N/A")}')
        print(f'Total Return [%]: {stats.get("Total Return [%]", "N/A")}')
        print(f'Sharpe Ratio: {stats.get("Sharpe Ratio", "N/A")}')
        print(f'Max Drawdown [%]: {stats.get("Max Drawdown [%]", "N/A")}')
        print(f'Trade Count: {len(pf.trades.records)}')
    except Exception as e:
        print(f'Ошибка расчета метрик: {e}')
        print(f'Final Portfolio Value: {pf.value().iloc[-1]}')
    
    return pf

if __name__ == '__main__':
    # Пример использования
    run_backtest(
        symbol='BTCUSDT',
        timeframe='1h', 
        start='2024-08-01',
        end='2024-08-15'
    )
