import numpy as np
import pandas as pd
from typing import Dict, Any
from trading_simulator import run_trading_simulation


def trading_strategy_objective_high_win_rate(data: pd.DataFrame, params: Dict[str, Any], min_trades_threshold: int = 10) -> float:
    """
    Целевая функция, которая приоритизирует поиск максимального Win Rate.

    Args:
        data: DataFrame с рыночными данными.
        params: Словарь с параметрами стратегии.
        min_trades_threshold: Минимальное количество сделок для получения значимого результата.

    Returns:
        Комплексная метрика, где Win Rate является основным компонентом.
    """
    simulation_params = params.copy()

    # Установка значений по умолчанию для неоптимизируемых параметров
    defaults = {
        'position_size': 10.0, 'commission': 0.1, 'stop_loss_pct': 2.0,
        'take_profit_pct': 4.0, 'entry_logic_mode': "Принты и HLdir", 'hldir_window': 10, 'hldir_offset': 0,
        'enable_additional_filters': False, 'aggressive_mode': False
    }
    for key, value in defaults.items():
        if key not in simulation_params:
            simulation_params[key] = value

    # Запускаем симуляцию в обычном (не агрессивном) режиме для реалистичной оценки
    results = run_trading_simulation(data, simulation_params)

    total_trades = results.get('total_trades', 0)
    win_rate = results.get('win_rate', 0.0)
    total_pnl = results.get('total_pnl', 0.0)
    profit_factor = results.get('profit_factor', 0.0)

    # 1. Штраф за малое количество сделок
    if total_trades < min_trades_threshold:
        # Сильный штраф, который немного уменьшается по мере приближения к порогу
        return -100.0 + total_trades

    # 2. Штраф за убыточность
    if total_pnl <= 0 or profit_factor < 1.0:
        # Возвращаем отрицательное значение, чтобы отсечь эти результаты,
        # но оно все еще зависит от win_rate, чтобы оптимизатор мог двигаться в правильном направлении.
        return win_rate - 1.0

    # 3. Основная метрика: Win Rate как главный компонент
    # Умножаем на 100, чтобы получить значения в диапазоне [0, 100]
    score = win_rate * 100

    # 4. Добавляем небольшие бонусы для различения стратегий с одинаковым Win Rate
    
    # Бонус за количество сделок (логарифмический, чтобы не доминировать)
    # Например, для 10 сделок бонус ~1, для 100 ~2.
    trade_bonus = np.log10(total_trades)
    
    # Бонус за качество прибыли (Profit Factor)
    # Ограничиваем PF, чтобы избежать аномальных значений
    capped_profit_factor = min(profit_factor, 5.0)
    # PF > 1 дает положительный бонус
    pf_bonus = (capped_profit_factor - 1.0)

    # Итоговая оценка
    final_score = score + trade_bonus + pf_bonus

    # Ограничиваем итоговое значение для стабильности
    clipped_score = np.clip(final_score, -100.0, 110.0)

    return clipped_score