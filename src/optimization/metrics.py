"""
Advanced Performance Metrics for Strategy Optimization

This module provides sophisticated performance metrics for trading strategy
optimization, including the adjusted_score metric with comprehensive
risk-adjusted evaluation.

Author: HFT System
"""
import numpy as np
from typing import List, Dict, Any


def adjusted_score(trades: List[Dict], annualization_factor: float = 98280) -> float:
    """
    Расчет скорректированной оценки стратегии с учетом множества факторов
    
    Args:
        trades: Список сделок с полями 'pnl' и 'duration'
        annualization_factor: Коэффициент для годовой доходности (98280 = 252×6.5×60)
        
    Returns:
        Скорректированная оценка стратегии
        
    Формула учитывает:
    1. Доходность за единицу времени в рынке (annualized)
    2. Риск – ожидаемое максимальное падение (bootstrapped)
    3. Стат-значимость – bootstrap p-value доходности
    4. Quality – Profit Factor + % winners + avg win/loss ratio
    """
    if len(trades) < 30:
        return -np.inf

    pnls = np.array([t['pnl'] for t in trades])
    durations = np.array([t['duration'] for t in trades])   # в минутах
    total_minutes = durations.sum()

    # 1. Annualized return в рынке (а не просто в год)
    ret = pnls.sum()
    annualized = ret / total_minutes * annualization_factor

    # 2. Bootstrap-95% Expected Shortfall (модельный риск)
    boot = np.random.choice(pnls, size=(10000, len(pnls)), replace=True)
    equity_curves = boot.cumsum(axis=1)
    running_max = np.maximum.accumulate(equity_curves, axis=1)
    drawdowns = (running_max - equity_curves) / (running_max + 1e-8)
    es95 = np.percentile(drawdowns.max(axis=1), 95)   # 95% worst DD

    # 3. Bootstrap p-value (доходность > 0)
    boot_ret = boot.sum(axis=1)
    p_value = (boot_ret <= 0).mean()

    # 4. Quality
    win_rate = (pnls > 0).mean()
    avg_win = pnls[pnls > 0].mean() if win_rate > 0 else 0
    avg_loss = abs(pnls[pnls < 0].mean()) if win_rate < 1 else 0
    profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss + 1e-8)
    quality = profit_factor * np.sqrt(win_rate * (1 - win_rate))   # tuned PF

    # 5. Штрафы
    risk_penalty = es95 * 5.0                                    # 10% ES → -0.5
    p_penalty = 0.0 if p_value < 0.05 else 3.0                   # не значимо → -3
    low_trades_penalty = 0.0 if len(trades) >= 60 else (60 - len(trades)) * 0.05

    # 6. Итог
    score = (annualized / (es95 + 1e-8)) + quality - p_penalty - low_trades_penalty
    return float(score)


def calculate_adjusted_score_from_results(results: Dict[str, Any]) -> float:
    """
    Расчет adjusted_score из результатов бэктеста
    
    Args:
        results: Результаты бэктеста с полем 'trades'
        
    Returns:
        Значение adjusted_score
    """
    trades = results.get('trades', [])
    if not trades:
        return -np.inf
    
    # Добавляем duration если отсутствует (расчитываем из времени входа/выхода)
    for trade in trades:
        if 'duration' not in trade:
            # Если есть время входа и выхода, считаем длительность
            if 'entry_time' in trade and 'exit_time' in trade:
                trade['duration'] = (trade['exit_time'] - trade['entry_time']) / 60  # в минутах
            else:
                # Иначе используем среднюю длительность сделки (например, 60 минут)
                trade['duration'] = 60
    
    return adjusted_score(trades)


def create_adjusted_score_objective() -> callable:
    """
    Создает objective функцию для Optuna на основе adjusted_score
    
    Returns:
        Objective функция для оптимизации
    """
    def objective(results: Dict[str, Any]) -> float:
        return calculate_adjusted_score_from_results(results)
    
    return objective


# Дополнительные метрики для будущего использования
def calmar_ratio(trades: List[Dict], initial_capital: float = 10000) -> float:
    """
    Расчет коэффициента Калмара (годовая доходность / максимальная просадка)
    """
    if not trades:
        return 0
    
    total_pnl = sum(t['pnl'] for t in trades)
    annual_return = (total_pnl / initial_capital) * 100  # Упрощенная годовая доходность
    
    # Расчет максимальной просадки
    equity_curve = [initial_capital]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade.get('pnl', 0))
    
    peak = initial_capital
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    return annual_return / (max_dd * 100) if max_dd > 0 else 0


def sterling_ratio(trades: List[Dict], initial_capital: float = 10000) -> float:
    """
    Расчет коэффициента Стерлинга (годовая доходность / средняя просадка)
    """
    if not trades:
        return 0
    
    total_pnl = sum(t['pnl'] for t in trades)
    annual_return = (total_pnl / initial_capital) * 100  # Упрощенная годовая доходность
    
    # Расчет средней просадки через bootstrap
    pnls = np.array([t['pnl'] for t in trades])
    boot = np.random.choice(pnls, size=(1000, len(pnls)), replace=True)
    equity_curves = boot.cumsum(axis=1)
    running_max = np.maximum.accumulate(equity_curves, axis=1)
    drawdowns = (running_max - equity_curves) / (running_max + 1e-8)
    avg_dd = np.mean(drawdowns.max(axis=1))
    
    return annual_return / (avg_dd * 100) if avg_dd > 0 else 0