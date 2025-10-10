"""
Strategy ported from the EXAMPLE project.
This strategy is based on the logic from signal_generator.py and trading_simulator.py.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from numba import njit, prange

from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry
from ..data.klines_handler import NumpyKlinesData
from .signal_generators import generate_signals # Import the new generator

@njit
def _get_prints_direction(long_prints, short_prints, i, prints_analysis_period, prints_threshold_ratio):
    """Определяет направление на основе анализа 'принтов'."""
    start_idx = max(0, i - prints_analysis_period + 1)
    end_idx = i + 1
    
    long_sum = np.sum(long_prints[start_idx:end_idx])
    short_sum = np.sum(short_prints[start_idx:end_idx])

    if short_sum > 0:
        ratio = long_sum / short_sum
        if ratio > prints_threshold_ratio:
            return 1  # Long
        elif ratio < (1 / prints_threshold_ratio):
            return -1 # Short
    elif long_sum > 0:
        return 1 # Long, если есть только покупки
        
    return 0 # Нейтрально

@njit
def _get_hldir_direction(hldir, i, hldir_window, hldir_offset):
    """Определяет направление на основе анализа 'HLdir'."""
    # Смещение окна анализа в прошлое. offset=0 включает текущую свечу i.
    end_idx = i - hldir_offset + 1
    start_idx = max(0, end_idx - hldir_window)
    
    if start_idx >= end_idx:
        return 0 # Недостаточно данных

    window_slice = hldir[start_idx:end_idx]
    
    # Ручной расчет среднего для совместимости с Numba, игнорируя NaN
    sum_val = 0.0
    count = 0
    for val in window_slice:
        if not np.isnan(val):
            sum_val += val
            count += 1
            
    if count == 0:
        return 0 # Нейтрально

    avg_hldir = sum_val / count
    
    if avg_hldir > 0.5:
        return 1 # Long
    else: # avg_hldir <= 0.5
        return -1 # Short

@njit
def _find_exit_optimized(
    entry_idx, entry_price, direction, stop_loss_price, take_profit_price,
    highs, lows, opens, closes, signal_mask, aggressive_mode
):
    """
    Оптимизированный итеративный поиск точки выхода.
    Проверяет свечи одну за другой, чтобы избежать создания больших срезов.
    """
    n = len(highs)
    for i in range(entry_idx + 1, n):
        # 1. Проверка агрессивного выхода (высший приоритет)
        # Выход происходит на свече ПЕРЕД новым сигналом.
        if aggressive_mode and signal_mask[i]:
            # Выходим на предыдущей свече (i-1).
            # Убедимся, что i-1 это не свеча входа.
            if i - 1 > entry_idx:
                return i - 1, closes[i - 1], 3  # Aggressive exit
            # Если сигнал на первой же свече после входа, игнорируем его для выхода,
            # но он может сработать как SL/TP на этой же свече.

        # 2. Проверка SL/TP на текущей свече `i`
        sl_triggered = False
        tp_triggered = False

        if direction == 1:  # Long
            if lows[i] <= stop_loss_price:
                sl_triggered = True
            if highs[i] >= take_profit_price:
                tp_triggered = True
        else:  # Short
            if highs[i] >= stop_loss_price:
                sl_triggered = True
            if lows[i] <= take_profit_price:
                tp_triggered = True

        # Логика одновременного срабатывания SL и TP на одной свече:
        # В HFT-системах часто предполагается, что SL имеет приоритет,
        # так как он исполняется по худшей цене (рыночный ордер).
        if sl_triggered and tp_triggered:
            # Если оба сработали, отдаем приоритет SL
            return i, stop_loss_price, 1 # Stop Loss
        elif sl_triggered:
            return i, stop_loss_price, 1 # Stop Loss
        elif tp_triggered:
            return i, take_profit_price, 2 # Take Profit

    # Если выход не найден, выходим на последней свече
    return n - 1, opens[n - 1], 4 # End of data

@njit
def _build_trades_vectorized(
    times, opens, highs, lows, closes, signal_mask,
    hldir, long_prints, short_prints,
    stop_loss_pct, take_profit_pct,
    position_size_dollars, commission_pct,
    prints_analysis_period, prints_threshold_ratio,
    hldir_window, hldir_offset,
    entry_logic_mode,
    aggressive_mode
) -> List[tuple]:
    """
    ВЕКТОРИЗОВАННЫЙ бэктестинг - итерация только по сигналам!
    """
    trades = []

    # Собираем все индексы сигналов
    signal_indices = []
    for i in range(len(signal_mask)):
        if signal_mask[i]:
            signal_indices.append(i)

    if len(signal_indices) == 0:
        return trades

    current_idx = 0  # Индекс в данных (не в сигналах)

    # Итерируемся только по сигналам!
    for sig_idx in signal_indices:
        # Пропускаем сигналы до текущей позиции
        if sig_idx <= current_idx:
            continue

        # Проверяем, что можем войти на следующей свече
        if sig_idx >= len(times) - 1:
            break

        # Определяем направление на свече сигнала
        prints_dir = 0
        hldir_dir = 0

        if entry_logic_mode == 0 or entry_logic_mode == 1:
            prints_dir = _get_prints_direction(long_prints, short_prints, sig_idx, prints_analysis_period, prints_threshold_ratio)

        if entry_logic_mode == 0 or entry_logic_mode == 2:
            hldir_dir = _get_hldir_direction(hldir, sig_idx, hldir_window, hldir_offset)

        direction = 0
        if entry_logic_mode == 0:
            if prints_dir != 0 and prints_dir == hldir_dir:
                direction = prints_dir
        elif entry_logic_mode == 1:
            direction = prints_dir
        elif entry_logic_mode == 2:
            direction = hldir_dir

        if direction == 0:
            continue

        # Вход на следующей свече
        entry_idx = sig_idx + 1
        entry_price = opens[entry_idx]
        entry_time = times[entry_idx]

        # Расчет SL/TP
        if direction == 1:
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100.0)
            take_profit_price = entry_price * (1 + take_profit_pct / 100.0)
        else:
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100.0)
            take_profit_price = entry_price * (1 - take_profit_pct / 100.0)

        # Оптимизированный итеративный поиск выхода
        exit_idx, exit_price, exit_reason = _find_exit_optimized(
            entry_idx, entry_price, direction, stop_loss_price, take_profit_price,
            highs, lows, opens, closes, signal_mask, aggressive_mode
        )

        # Расчет PnL
        if direction == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        pnl = pnl_pct * position_size_dollars
        pnl -= (position_size_dollars * commission_pct) * 2

        trades.append((
            entry_time, times[exit_idx], direction,
            entry_price, exit_price, position_size_dollars, pnl,
            exit_reason, entry_idx, exit_idx
        ))

        # Обновляем текущую позицию для следующего сигнала
        current_idx = exit_idx

    return trades

@StrategyRegistry.register('ported_from_example')
class PortedFromExampleStrategy(BaseStrategy):
    """
    Порт стратегии из директории EXAMPLE/TUHv3, приведенный в соответствие с signals.txt.
    Использует модульный генератор сигналов и Numba-jitted цикл бэктестинга.
    """
    def __init__(self, symbol: str, **kwargs):
        default_params = self.get_default_params()
        default_params.update(kwargs)
        super().__init__(symbol, signal_generator=generate_signals, **default_params)
        for param_name, param_value in self.params.items():
            setattr(self, param_name, param_value)

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Возвращает параметры по умолчанию, как описано в signals.txt."""
        return {
            # Этап 1: Генерация сигнала-кандидата
            'vol_period': 20, 'vol_pctl': 1.0, 'range_period': 20, 'rng_pctl': 1.0,
            'natr_period': 10, 'natr_min': 0.35, 'lookback_period': 20, 'min_growth_pct': 1.0,
            
            # Этап 2: Подтверждение направления
            'entry_logic_mode': "Принты и HLdir", # Варианты: "Принты и HLdir", "Только по принтам", "Только по HLdir"
            'prints_analysis_period': 2, 'prints_threshold_ratio': 1.5,
            'hldir_window': 3, 'hldir_offset': 0,
            
            # Этап 3: Управление позицией
            'stop_loss_pct': 2.0, 'take_profit_pct': 4.0,
            'aggressive_mode': False,
            
            # Параметры симуляции
            'initial_capital': 10000.0, 'position_size_dollars': 1000.0, 'commission_pct': 0.001
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, tuple]:
        """Возвращает пространство поиска для оптимизации."""
        return {
            # Этап 1
            'vol_period': ('int', 10, 100), 'vol_pctl': ('float', 0.1, 10.0),
            'range_period': ('int', 10, 100), 'rng_pctl': ('float', 0.1, 10.0),
            # natr_period, natr_min, lookback_period - используются значения по умолчанию
            'min_growth_pct': ('float', 0.1, 5.0),

            # Этап 2
            'entry_logic_mode': ('categorical', ["Принты и HLdir", "Только по принтам", "Только по HLdir"]),
            'prints_analysis_period': ('int', 1, 10), 'prints_threshold_ratio': ('float', 1.1, 5.0),
            'hldir_window': ('int', 2, 20), 'hldir_offset': ('int', 0, 5),

            # Этап 3
            'stop_loss_pct': ('float', 0.5, 10.0), 'take_profit_pct': ('float', 1.0, 20.0),
            'aggressive_mode': ('categorical', [True, False]),
        }

    def vectorized_process_dataset(self, data: 'NumpyKlinesData') -> Dict[str, Any]:
        signal_conditions = self._generate_signals(data)

        times = data['time']
        opens = data['open']
        highs = data['high']
        lows = data['low']
        closes = data['close']
        hldir = data.get('hldir')
        long_prints = data.get('long_prints')
        short_prints = data.get('short_prints')

        # Преобразование строкового режима в int для Numba
        mode_map = {"Принты и HLdir": 0, "Только по принтам": 1, "Только по HLdir": 2}
        entry_logic_mode_int = mode_map.get(self.entry_logic_mode, 0)

        raw_trades = _build_trades_vectorized(
            times, opens, highs, lows, closes, signal_conditions,
            hldir, long_prints, short_prints,
            self.stop_loss_pct, self.take_profit_pct,
            self.position_size_dollars, self.commission_pct,
            self.prints_analysis_period, self.prints_threshold_ratio,
            self.hldir_window, self.hldir_offset,
            entry_logic_mode_int,
            self.aggressive_mode
        )

        trades = []
        exit_reason_map = {1: 'stop_loss', 2: 'take_profit', 3: 'aggressive_exit', 4: 'end_of_data'}
        side_map = {1: 'long', -1: 'short'}
        for t in raw_trades:
            pnl = t[6]
            trades.append({
                'timestamp': pd.to_datetime(t[0], unit='ms'),
                'exit_timestamp': pd.to_datetime(t[1], unit='ms'),
                'symbol': self.symbol,
                'side': side_map[t[2]],
                'entry_price': t[3],
                'exit_price': t[4],
                'size': t[5],
                'pnl': pnl,
                'pnl_percentage': (pnl / t[5]) * 100 if t[5] != 0 else 0,
                'duration': (t[1] - t[0]),
                'exit_reason': exit_reason_map.get(t[7], 'unknown'),
                'entry_idx': t[8],
                'exit_idx': t[9],
            })

        metrics = self.calculate_performance_metrics(trades, self.initial_capital)

        return {
            'trades': trades,
            'symbol': self.symbol,
            'total_bars': len(times),
            'train_bars': 0,
            'indicator_data': {
                'times': times,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'prices': closes,  # 'prices' является псевдонимом для 'close'
                'signal_mask': signal_conditions,
            },
            **metrics
        }