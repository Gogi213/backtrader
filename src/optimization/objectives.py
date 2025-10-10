"""
Модуль с кастомными целевыми функциями (objectives) для оптимизации в Optuna.
"""
import numpy as np
from typing import Dict, Any

def sharpe_with_drawdown_penalty(results: Dict[str, Any]) -> float:
    """
    Целевая функция: Sharpe Ratio с штрафом за максимальную просадку.
    """
    sharpe = results.get('sharpe_ratio', 0.0)
    max_drawdown = results.get('max_drawdown', 0.0)
    penalty = max_drawdown / 10.0
    return sharpe - penalty

def sharpe_multiplied_by_profit_factor(results: Dict[str, Any]) -> float:
    """
    Целевая функция: Sharpe Ratio, умноженный на логарифм Profit Factor.
    """
    sharpe = results.get('sharpe_ratio', 0.0)
    profit_factor = results.get('profit_factor', 0.0)
    multiplier = np.log(1 + profit_factor)
    return sharpe * multiplier

def sharpe_ratio(results: Dict[str, Any]) -> float:
    """
    Целевая функция: Sharpe Ratio.
    """
    return results.get('sharpe_ratio', 0.0)

def pnl(results: Dict[str, Any]) -> float:
    """
    Целевая функция: Total PnL (Profit and Loss).
    """
    return results.get('net_pnl', 0.0)

def sharpe_profit_factor_trades_score(results: Dict[str, Any]) -> float:
    """
    Целевая функция: Sharpe * log(Profit Factor) * log(Total Trades).
    """
    sharpe = results.get('sharpe_ratio', 0.0)
    profit_factor = results.get('profit_factor', 0.0)
    total_trades = results.get('total', 0)
    if total_trades == 0:
        return -1e6
    pf_multiplier = np.log(1 + profit_factor)
    trades_multiplier = np.log(1 + total_trades)
    return sharpe * pf_multiplier * trades_multiplier

# Реестр доступных кастомных целевых функций
OBJECTIVE_FUNCTIONS = {
    "sharpe_ratio": sharpe_ratio,
    "pnl": pnl,
    "sharpe_pf_trades_score": sharpe_profit_factor_trades_score,
    "sharpe_with_drawdown_penalty": sharpe_with_drawdown_penalty,
    "sharpe_x_profit_factor": sharpe_multiplied_by_profit_factor,
}
