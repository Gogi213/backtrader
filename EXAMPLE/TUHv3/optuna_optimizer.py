"""
Модуль для оптимизации гиперпараметров с использованием Optuna.

Этот модуль предоставляет функции для запуска стандартной оптимизации торговых стратегий,
включая проверку робастности найденных параметров.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Union
import optuna
from optuna import Trial
import sys
import streamlit as st
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from signal_generator import generate_signals
from trading_simulator import run_trading_simulation


def objective(
    trial: Trial,
    data: pd.DataFrame,
    strategy_func: Callable[[pd.DataFrame, Dict[str, Any]], float],
    param_space: Dict[str, Tuple[str, ...]]
) -> float:
    """
    Целевая функция для Optuna. Включает проверку на робастность (стабильность).
    Implements parameter rounding to reduce search space and improve optimization speed.

    Args:
        trial: Optuna trial object
        data: In-sample data for optimization
        strategy_func: User-defined strategy function that takes data and params, returns metric
        param_space: Dictionary defining parameter space

    Returns:
        Metric value to optimize
    """
    # Sample parameters based on param_space
    params = {}
    for param_name, param_config in param_space.items():
        param_type = param_config[0]
        if param_type == "int":
            low, high = param_config[1], param_config[2]
            step = param_config[3] if len(param_config) > 3 else 1
            params[param_name] = trial.suggest_int(param_name, low, high, step=step)
        elif param_type == "float":
            low, high = param_config[1], param_config[2]
            log = param_config[3] if len(param_config) > 3 else False
            # Для float параметров используем шаг 0.01 для уменьшения пространства поиска
            suggested_value = trial.suggest_float(param_name, low, high, log=log, step=0.01)
            params[param_name] = suggested_value
        elif param_type == "categorical":
            choices = param_config[1]
            params[param_name] = trial.suggest_categorical(param_name, choices)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    # Calculate metrics using strategy function
    metric = strategy_func(data, params)
    
    # Возвращаем "чистую" метрику без проверки на робастность для максимального ускорения.
    # Проверку на робастность можно будет провести вручную для лучших найденных параметров.
    return metric


def get_trading_strategy_param_space():
    """
    Возвращает стандартное пространство параметров для торговой стратегии.

    Returns:
        Словарь с определением пространства параметров
    """
    return {
        "vol_period": ("int", 5, 50),
        "vol_pctl": ("float", 0.1, 30.0),
        "range_period": ("int", 5, 50),
        "rng_pctl": ("float", 0.1, 30.0),
        "natr_period": ("int", 5, 50),
        "natr_min": ("float", 0.01, 2.0),
        "lookback_period": ("int", 5, 100),
        "min_growth_pct": ("float", -2.0, 5.0),
        "prints_analysis_period": ("int", 1, 10),
        "prints_threshold_ratio": ("float", 0.1, 5.0),  # Используется при фиксированном соотношении
        "stop_loss_pct": ("float", 0.5, 10.0),
        "take_profit_pct": ("float", 1.0, 20.0),
        "hldir_window": ("int", 1, 20),  # Параметр для усреднения HLdir
        "enable_additional_filters": ("categorical", [False, True])  # По умолчанию дополнительные фильтры отключены
    }


def get_discretized_param_space(param_space, float_precision=2):
    """
    Преобразует пространство параметров, чтобы использовать только дискретные значения для float параметров.
    
    Args:
        param_space: Оригинальное пространство параметров
        float_precision: Количество знаков после запятой для float параметров
        
    Returns:
        Новое пространство параметров с дискретными значениями для float параметров
    """
    discretized_space = {}
    step_size = 10 ** (-float_precision)  # Например, 0.01 для 2 знаков после запятой
    
    for param_name, param_config in param_space.items():
        param_type = param_config[0]
        if param_type == "float":
            low, high = param_config[1], param_config[2]
            # Создаем дискретный диапазон значений с заданной точностью
            # Вычисляем количество шагов
            steps = int((high - low) / step_size) + 1
            # Генерируем значения
            values = [round(low + i * step_size, float_precision) for i in range(steps)]
            values = [v for v in values if low <= v <= high]  # Убедимся, что все значения в диапазоне
            discretized_space[param_name] = ("categorical", values)
        else:
            discretized_space[param_name] = param_config
    
    return discretized_space


def run_optimization(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Основная функция для запуска оптимизации параметров стратегии.
    
    Args:
        params: Словарь с параметрами оптимизации, включая:
            - data: DataFrame с рыночными данными
            - param_space: пространство параметров для оптимизации
            - n_trials: количество проб
            - direction: направление оптимизации ('maximize' или 'minimize')
            - position_size: размер позиции (не оптимизируется, но передается в симуляцию)
            - commission: комиссия (не оптимизируется, но передается в симуляцию)
            - stop_loss_pct: стоп-лосс (не оптимизируется, но передается в симуляцию)
            - take_profit_pct: тейк-профит (не оптимизируется, но передается в симуляцию)
            - use_discretized_param_space: использовать ли дискретизированное пространство параметров (по умолчанию False)
    
    Returns:
        Словарь с результатами оптимизации
    """
    data = params.get('data')
    param_space = params.get('param_space', get_trading_strategy_param_space())
    n_trials = params.get('n_trials', 50)
    direction = params.get('direction', 'maximize')
    optimization_type = 'optuna' # Оставляем только стандартную оптимизацию
    
    # Получаем дополнительные параметры, которые не оптимизируются, но необходимы для симуляции
    position_size = params.get('position_size', 100.0)
    commission = params.get('commission', 0.1)
    stop_loss_pct = params.get('stop_loss_pct', 2.0)
    take_profit_pct = params.get('take_profit_pct', 4.0)
    use_discretized_param_space = params.get('use_discretized_param_space', False)
    # Используем единственную целевую функцию
    from strategy_objectives import trading_strategy_objective_high_win_rate
    strategy_func = trading_strategy_objective_high_win_rate
    
    # Преобразуем param_space в формат, который принимает objective функция
    optuna_param_space = {}
    for key, value in param_space.items():
        if isinstance(value, tuple) and len(value) >= 3:  # Проверяем, что это (type, min, max)
            param_type = value[0]
            if param_type == "int":
                min_val, max_val = value[1], value[2]
                optuna_param_space[key] = ("int", int(float(min_val)), int(float(max_val)))
            elif param_type == "float":
                min_val, max_val = value[1], value[2]
                optuna_param_space[key] = ("float", float(min_val), float(max_val))
            elif param_type == "categorical":
                choices = value[1] if len(value) > 1 else []
                optuna_param_space[key] = ("categorical", [item if isinstance(item, (int, float, str, bool)) else str(item) for item in choices])
        elif isinstance(value, tuple) and len(value) >= 2:  # Это диапазон (min, max) без типа
            if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                # Предполагаем, что это float диапазон
                min_val, max_val = value[0], value[1]
                if isinstance(min_val, int) and isinstance(max_val, int):
                    optuna_param_space[key] = ("int", int(float(min_val)), int(float(max_val)))
                else:
                    optuna_param_space[key] = ("float", float(min_val), float(max_val))
        elif isinstance(value, list) and len(value) > 0:
            # Это категориальные значения
            optuna_param_space[key] = ("categorical", [item if isinstance(item, (int, float, str, bool)) else str(item) for item in value])
        else:
            # Если формат не определен, пропускаем
            continue
    
    # Создаем обертку для целевой функции, чтобы передавать дополнительные параметры.
    def create_wrapped_strategy_objective(pos_size, comm, sl_pct, tp_pct):
        def wrapped_objective(data: pd.DataFrame, strategy_params: Dict[str, Any]) -> float:
            simulation_params = strategy_params.copy()
            simulation_params['position_size'] = round(pos_size, 2)
            simulation_params['commission'] = round(comm, 3)
            simulation_params['stop_loss_pct'] = round(sl_pct, 2)
            simulation_params['take_profit_pct'] = round(tp_pct, 2)
            
            if 'enable_additional_filters' not in simulation_params:
                simulation_params['enable_additional_filters'] = False
            
            return strategy_func(data, simulation_params)
        return wrapped_objective
    
    # Создаем обертку с конкретными значениями параметров
    wrapped_strategy_objective = create_wrapped_strategy_objective(position_size, commission, stop_loss_pct, take_profit_pct)
    
    # Запускаем обычную оптимизацию Optuna
    sample_data = data.copy()
    
    def optimization_objective(trial):
        return objective(trial, sample_data, wrapped_strategy_objective, optuna_param_space)
    
    if direction == 'multi':
        study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'minimize'])
    else:
        study = optuna.create_study(direction=direction)
    
    try:
        # Запускаем оптимизацию в одном процессе (n_jobs=1),
        # Теперь используем все ядра (n_jobs=-1)
        study.optimize(
            optimization_objective,
            n_trials=n_trials,
            show_progress_bar=True, # Можно включить для отладки в консоли
            n_jobs=-1
        )
    except Exception as e:
        print(f"Ошибка при выполнении оптимизации: {e}")
        return {
            'best_params': {},
            'best_value': None,
            'n_trials': n_trials,
            'direction': direction,
            'optimization_type': 'optuna',
            'top_10_results': []
        }

    if not (len(study.trials) > 0 and any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)):
        print("Ошибка при запуске Optuna оптимизации: No trials are completed yet.")
        return {
            'best_params': {}, 'best_value': None, 'n_trials': n_trials,
            'direction': direction, 'optimization_type': 'optuna', 'top_10_results': []
        }

    # Сбор и сортировка результатов
    completed_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=(direction == 'maximize')
    )

    # Создаем топ-10 результатов
    top_10_results = []
    for trial in completed_trials[:10]:
        try:
            rounded_params = {k: round(v, 2) if isinstance(v, float) else v for k, v in trial.params.items()}
            simulation_params = {
                **rounded_params,
                'position_size': position_size, 'commission': commission,
                'stop_loss_pct': stop_loss_pct, 'take_profit_pct': take_profit_pct
            }
            if 'enable_additional_filters' not in simulation_params:
                simulation_params['enable_additional_filters'] = False

            sim_results = run_trading_simulation(sample_data, simulation_params)
            
            result = {
                'value': trial.value,
                'trial_number': trial.number,
                'total_pnl': sim_results.get('total_pnl', 0),
                'win_rate': sim_results.get('win_rate', 0),
                'total_trades': sim_results.get('total_trades', 0),
                'max_drawdown': sim_results.get('max_drawdown', 0),
                'profit_factor': sim_results.get('profit_factor', 0),
                **rounded_params
            }
            top_10_results.append(result)
        except Exception as e:
            print(f"Ошибка при вычислении метрик для пробы {trial.number}: {e}")

    if not hasattr(study, 'best_params') or not study.best_params:
        print("Ошибка: Не найдено лучшее значение в результате оптимизации.")
        return {
            'best_params': {}, 'best_value': None, 'n_trials': n_trials,
            'direction': direction, 'optimization_type': 'optuna', 'top_10_results': top_10_results
        }

    # Сохранение и вывод результатов
    symbol = params.get('symbol', 'UNKNOWN')
    rounded_best_params = {k: round(v, 2) if isinstance(v, float) else v for k, v in study.best_params.items()}

    results_to_save = {
        'best_params': rounded_best_params,
        'best_value': study.best_value,
        'n_trials': n_trials,
        'direction': direction,
        'optimization_type': 'optuna',
        'top_10_results': top_10_results,
        'symbol': symbol
    }
    
    save_optimization_results(results_to_save, symbol, 'optuna')
    
    print("\nТоп-10 результатов оптимизации:")
    print("(PnL - общая прибыль/убыток, WR - WinRate, DD - Max Drawdown, PF - Profit Factor)")
    print(f"{'#':<4} {'Trial':<7} {'Value':<10} {'PnL($)':<10} {'WR':<8} {'Trades':<8} {'DD':<10} {'PF':<6} {'Params'}")
    print("-" * 120)
    for i, res in enumerate(top_10_results):
        param_keys = [k for k in res if k not in ['value', 'trial_number', 'total_pnl', 'win_rate', 'total_trades', 'max_drawdown', 'profit_factor']]
        params_str = ", ".join([f"{k}={res[k]}" for k in param_keys])
        print(f"{i+1:<4} #{res.get('trial_number', 'N/A'):<6} {res['value']:<10.4f} {res['total_pnl']:<10.2f} {res['win_rate']:<8.2%} {res['total_trades']:<8} {res['max_drawdown']:<10.2%} {res['profit_factor']:<6.2f} {params_str}")
    
    return results_to_save

def save_optimization_results(results, symbol, optimization_type='optuna', run_name=None):
   """
   Сохраняет результаты оптимизации в JSON файл
   """
   import json
   from datetime import datetime
   import os
   
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   if run_name:
       filename = f"run_{timestamp}_{symbol}_{optimization_type}_{run_name}.json"
   else:
       best_value = results.get('best_value', 0)
       # Добавляем информацию о том, что результаты содержат номера триалов
       filename = f"run_{timestamp}_{symbol}_{optimization_type}_${best_value:.0f}_OPTUNA.json"
   
   filepath = os.path.join("optimization_runs", filename)
   
   # Создаем директорию, если она не существует
   os.makedirs(os.path.dirname(filepath), exist_ok=True)
   
   # Округляем float параметры в результатах перед сохранением
   def round_floats(obj):
       if isinstance(obj, float):
           return round(obj, 2)
       elif isinstance(obj, dict):
           return {key: round_floats(value) for key, value in obj.items()}
       elif isinstance(obj, list):
           return [round_floats(item) for item in obj]
       else:
           return obj

   # Применяем округление ко всем результатам
   rounded_results = round_floats(results)

   # Сохраняем результаты в JSON файл
   with open(filepath, 'w', encoding='utf-8') as f:
       json.dump(rounded_results, f, indent=2, ensure_ascii=False, default=str)
   
   print(f"Результаты оптимизации сохранены в {filepath}")
   return filepath

# Example usage
if __name__ == "__main__":
    import numpy as np

    # --- 1. Загрузка рабочего CSV файла ---
    data_file = "HEMIUSDT-klines-1s-2025-09-01_to_2025-09-02.csv"
    data_path = os.path.join("dataCSV", data_file)
    try:
        data = pd.read_csv(data_path)
        # Переименовываем столбец 'Volume' в 'volume' для совместимости
        if 'Volume' in data.columns:
            data = data.rename(columns={'Volume': 'volume'})
        data['datetime'] = pd.to_datetime(data['time'], unit='s')
        data = data.set_index('datetime')
        print(f"Загружен рабочий датасет: {data_file}, {len(data)} строк.")
    except FileNotFoundError:
        print(f"Ошибка: Файл данных не найден по пути {data_path}")
        print("Пожалуйста, убедитесь, что файл находится в папке 'dataCSV'.")
        sys.exit(1)

    # --- 2. Использование рабочих диапазонов параметров ---
    # Диапазоны взяты из файла 'Загрузить диапазоны 1многоV7.json'
    param_space = {
        "vol_period": ("int", 15, 20),
        "vol_pctl": ("float", 1.0, 1.0),
        "range_period": ("int", 1, 1),
        "rng_pctl": ("float", 1.0, 1.0),
        "natr_period": ("int", 1, 1),
        "natr_min": ("float", 0.01, 0.05),
        "lookback_period": ("int", 50, 100),
        "min_growth_pct": ("float", 1.0, 2.0),
        "prints_analysis_period": ("int", 1, 3),
        "prints_threshold_ratio": ("float", 1.0, 2.5),
        "stop_loss_pct": ("float", 1.0, 1.0),
        "take_profit_pct": ("float", 1.0, 3.0),
        "hldir_window": ("int", 1, 1),
        "enable_additional_filters": ("categorical", [False])
    }

    # Для примера и проверки сгенерируем сигналы с параметрами из середины диапазонов
    example_params = {}
    for name, config in param_space.items():
        if config[0] == "int":
            example_params[name] = (config[1] + config[2]) // 2
        elif config[0] == "float":
            example_params[name] = (config[1] + config[2]) / 2
        elif config[0] == "categorical":
            example_params[name] = config[1][0]

    signals = generate_signals(data, example_params)
    print(f"Сгенерировано {len(signals)} сигналов с тестовыми параметрами.")

    # Запускаем обычную оптимизацию Optuna для тестирования и профилирования
    print("\n" + "="*50)
    print("Запуск стандартной оптимизации Optuna...")
    opt_params = {
        'data': data,
        'param_space': param_space,
        'n_trials': 10,  # Уменьшено для быстрого теста
        'direction': 'maximize',
        'optimization_type': 'optuna',
        'position_size': 100.0,
        'commission': 0.1,
        'stop_loss_pct': 2.0,
        'take_profit_pct': 4.0
    }
    results = run_optimization(opt_params)

    print("Результаты стандартной оптимизации:")
    if results and results.get('best_value') is not None:
        print(f"Лучшее значение: {results['best_value']:.4f}")
        print(f"Лучшие параметры: {results['best_params']}")
    else:
        print("Оптимизация не дала результатов.")
    
    # Проверяем, есть ли результаты
    if results and results.get('best_value') is not None:
        print("Оптимизация завершена успешно.")
    else:
        print("Оптимизация не нашла подходящих решений.")
        
    print("\n" + "="*50)
    print("Профилирование одной симуляции:")
    print("Чтобы найти узкие места, запустим симуляцию под профилировщиком.")

    import cProfile
    import pstats

    # Создаем объект профилировщика
    profiler = cProfile.Profile()

    # Включаем профилировщик
    profiler.enable()

    # Запускаем функцию, которую хотим измерить
    # (один вызов симуляции с тестовыми параметрами)
    print("Запуск оптимизации для профилирования...")
    _ = run_optimization(opt_params)
    print("Симуляция завершена.")

    # Отключаем профилировщик
    profiler.disable()

    # Сохраняем результаты профилирования в файл
    stats_file = "simulation_profile.prof"
    profiler.dump_stats(stats_file)
    print(f"Результаты профилирования сохранены в файл: {stats_file}")
    print("\nДля анализа результатов установите snakeviz: pip install snakeviz")
    print(f"И выполните в терминале команду: snakeviz {stats_file}")