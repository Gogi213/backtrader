#!/usr/bin/env python3
"""
CLI Optimizer for HFT System

Simple command-line interface for running strategy optimizations without any GUI.
This is the fastest way to run optimizations with Optuna.

Usage:
    python cli_optimizer.py --strategy turbo_mean_reversion --data upload/klines/BTCUSDT_1m.csv --trials 100

Author: HFT System
"""
import argparse
import sys
import os
import time
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.optimization.fast_optimizer import FastStrategyOptimizer
    import src.strategies  # This will trigger the dynamic loading
    from src.strategies import StrategyRegistry
    
    # Import profiling capabilities
    try:
        from src.profiling import OptunaProfiler, StrategyProfiler, ProfileReport
        PROFILING_AVAILABLE = True
    except ImportError:
        PROFILING_AVAILABLE = False
        
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все зависимости установлены:")
    print("pip install optuna numpy pandas")
    sys.exit(1)


def main():
    """Main CLI function"""
    # Get available strategies for help text and default value
    available_strategies = StrategyRegistry.list_strategies()
    if not available_strategies:
        print("Ошибка: Ни одна стратегия не найдена. Убедитесь, что стратегии определены в 'src/strategies'.")
        sys.exit(1)

    default_strategy = available_strategies[0]
    
    parser = argparse.ArgumentParser(
        description="Быстрая оптимизация HFT стратегий через командную строку",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Примеры использования:
  %(prog)s --strategy {default_strategy} --data upload/klines/BTCUSDT_1m.csv --trials 50
  %(prog)s --strategy {default_strategy} --data upload/klines/BTCUSDT_1m.csv --trials 100 --jobs 4
  %(prog)s --strategy {default_strategy} --data upload/klines/BTCUSDT_1m.csv --trials 200 --metric net_pnl
        """
    )
    
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default=default_strategy,
        choices=available_strategies,
        help=f"Название стратегии для оптимизации (по умолчанию: {default_strategy})"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Путь к CSV файлу с данными (по умолчанию: первый файл в upload/klines)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Торговый символ (по умолчанию: извлекается из имени файла)"
    )
    
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=100,
        help="Количество испытаний оптимизации (по умолчанию: 100)"
    )
    
    parser.add_argument(
        "--metric", "-m",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "net_pnl", "profit_factor", "win_rate", "net_pnl_percentage"],
        help="Метрика оптимизации (по умолчанию: sharpe_ratio)"
    )
    
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=-1,
        help="Количество параллельных заданий (-1 для всех ядер, по умолчанию: -1)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Лимит времени в секундах (по умолчанию: без ограничения)"
    )
    
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Минимальное количество сделок (по умолчанию: 10)"
    )
    
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=50.0,
        help="Максимальная просадка в процентах (по умолчанию: 50.0)"
    )
    
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Начальный капитал (по умолчанию: 10000.0)"
    )
    
    parser.add_argument(
        "--position-size",
        type=float,
        default=1000.0,
        help="Размер позиции (по умолчанию: 1000.0)"
    )
    
    parser.add_argument(
        "--commission",
        type=float,
        default=0.05,
        help="Комиссия в процентах (по умолчанию: 0.05)"
    )
    
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Показать доступные стратегии и выйти"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Включить профилирование производительности"
    )
    
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="profiling_reports",
        help="Директория для отчетов профилирования (по умолчанию: profiling_reports)"
    )
    
    args = parser.parse_args()
    
    # List strategies if requested
    if args.list_strategies:
        print("Доступные стратегии:")
        strategies = StrategyRegistry.list_strategies()
        for strategy in strategies:
            print(f"  - {strategy}")
        return
    
    # Handle dataset selection
    if args.data is None:
        # Find first data file in upload/klines
        klines_dir = "upload/klines"
        if os.path.exists(klines_dir):
            data_files = [f for f in os.listdir(klines_dir) if f.endswith(('.csv', '.parquet'))]
            if data_files:
                args.data = os.path.join(klines_dir, data_files[0])
                print(f"Используется датасет по умолчанию: {data_files[0]}")
            else:
                print(f"Ошибка: Файлы данных (.csv или .parquet) не найдены в {klines_dir}")
                sys.exit(1)
        else:
            print(f"Ошибка: Директория {klines_dir} не найдена")
            sys.exit(1)
    
    # Extract symbol from filename if not provided
    if args.symbol is None:
        filename = os.path.basename(args.data)
        args.symbol = filename.split('-')[0] if '-' in filename else filename.split('.')[0]
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"Ошибка: Файл данных не найден: {args.data}")
        sys.exit(1)
    
    if args.trials <= 0:
        print("Ошибка: Количество испытаний должно быть положительным")
        sys.exit(1)
    
    # Print configuration
    print("=" * 60)
    print("HFT OPTIMIZATION - CLI MODE")
    print("=" * 60)
    print(f"Стратегия:         {args.strategy}")
    print(f"Файл данных:       {args.data}")
    print(f"Символ:            {args.symbol}")
    print(f"Испытаний:         {args.trials}")
    print(f"Метрика:           {args.metric}")
    print(f"Параллельных:      {args.jobs}")
    print(f"Мин. сделок:       {args.min_trades}")
    print(f"Макс. просадка:    {args.max_drawdown}%")
    print(f"Начальный капитал: ${args.initial_capital:,.2f}")
    print(f"Размер позиции:    ${args.position_size:,.2f}")
    print(f"Комиссия:          {args.commission}%")
    if args.timeout:
        print(f"Лимит времени:      {args.timeout} сек")
    print("=" * 60)
    
    try:
        # Create optimizer with backtest config
        print("Создание оптимизатора...")
        start_time = time.time()
        
        from src.core.backtest_config import BacktestConfig
        backtest_config = BacktestConfig(
            strategy_name=args.strategy,
            symbol=args.symbol,
            data_path=args.data,
            initial_capital=args.initial_capital,
            commission_pct=args.commission / 100.0,  # Convert from percentage
            position_size_dollars=args.position_size
        )
        
        # Check if profiling is requested
        if args.profile and not PROFILING_AVAILABLE:
            print("Warning: Профилирование запрошено, но недоступно. Установите зависимости:")
            print("pip install line_profiler memory_profiler psutil matplotlib seaborn")
            args.profile = False
        
        optimizer = FastStrategyOptimizer(
            strategy_name=args.strategy,
            data_path=args.data,
            symbol=args.symbol,
            backtest_config=backtest_config,
            enable_profiling=args.profile,
            profiling_output_dir=args.profile_dir
        )
        
        if args.profile:
            print(f"Профилирование включено - отчеты будут сохранены в {args.profile_dir}")
        
        print("Запуск оптимизации...")
        print("-" * 60)
        
        # Run optimization
        results = optimizer.optimize(
            n_trials=args.trials,
            objective_metric=args.metric,
            min_trades=args.min_trades,
            max_drawdown_threshold=args.max_drawdown,
            timeout=args.timeout,
            n_jobs=args.jobs
        )
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Print results
        print("-" * 60)
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print("-" * 60)
        
        if 'error' in results:
            print(f"Ошибка: {results['error']}")
            sys.exit(1)
        
        print(f"Лучшее значение ({args.metric}): {results.get('best_value', 0):.4f}")
        print(f"Всего испытаний:              {results.get('n_trials', 0)}")
        print(f"Успешных испытаний:           {results.get('successful_trials', 0)}")
        print(f"Обрезанных испытаний:         {results.get('pruned_trials', 0)}")
        print(f"Время оптимизации:            {optimization_time:.2f} сек")
        print(f"Параллельных заданий:         {results.get('parallel_jobs', 1)}")
        
        # Best parameters
        if 'best_params' in results and results['best_params']:
            print("\nЛучшие параметры:")
            for param, value in results['best_params'].items():
                print(f"  {param}: {value}")
        
        # Final backtest results
        if 'final_backtest' in results and results['final_backtest']:
            print("\nФинальный бэктест:")
            backtest = results['final_backtest']
            print(f"  Всего сделок:     {backtest.get('total', 0)}")
            print(f"  Sharpe Ratio:    {backtest.get('sharpe_ratio', 0):.2f}")
            print(f"  P&L:             ${backtest.get('net_pnl', 0):,.2f}")
            print(f"  Доходность:      {backtest.get('net_pnl_percentage', 0):.2f}%")
            print(f"  Макс. просадка:  {backtest.get('max_drawdown', 0):.2f}%")
        
        print("-" * 60)
        print(f"Оптимизация завершена за {optimization_time:.2f} секунд")
        
    except KeyboardInterrupt:
        print("\nОптимизация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка оптимизации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()