"""
OptimizationManager — минимальный менеджер
Без валидации, без GUI, без дублирования
Author: HFT System (optimized)
"""
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from .backtest_config import BacktestConfig
from .backtest_results import BacktestResults


@dataclass
class OptimizationConfig:
    """Configuration for optimization process"""
    strategy_name: str = 'hierarchical_mean_reversion'
    data_path: str = ''
    symbol: str = 'BTCUSDT'
    study_name: Optional[str] = None
    direction: str = 'maximize'
    storage: Optional[str] = None
    cache_dir: str = "optimization_cache"
    n_trials: int = 100
    objective_metric: str = 'sharpe_ratio'
    min_trades: int = 10
    max_drawdown_threshold: float = 50.0
    timeout: Optional[float] = 600
    n_jobs: int = -1


@dataclass
class OptimizationResults:
    """Results from optimization process"""
    strategy_name: str
    symbol: str
    study_name: str
    objective_metric: str
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    successful_trials: int
    pruned_trials: int
    optimization_time_seconds: float
    optimization_completed_at: str
    parallel_jobs: int
    final_backtest: Dict[str, Any]


class OptimizationManager:
    def __init__(self):
        self._stats = {'total': 0, 'ok': 0, 'fail': 0, 'time': 0.0}

    def run(self, config) -> BacktestResults:
        self._stats['total'] += 1
        t0 = time.time()

        try:
            from ..optimization.fast_optimizer import FastStrategyOptimizer
            optimizer = FastStrategyOptimizer(
                strategy_name=config.strategy_name,
                data_path=config.data_path,
                symbol=config.symbol,
                study_name=config.study_name,
                direction=config.direction,
                storage=config.storage,
                cache_dir=config.cache_dir,
                enable_debug=False
            )
            raw = optimizer.optimize(
                n_trials=config.n_trials,
                objective_metric=config.objective_metric,
                min_trades=config.min_trades,
                max_drawdown_threshold=config.max_drawdown_threshold,
                timeout=config.timeout,
                n_jobs=config.n_jobs
            )
            self._stats['ok'] += 1
            return BacktestResults(raw)

        except Exception as e:
            self._stats['fail'] += 1
            return BacktestResults({'error': str(e)})

        finally:
            self._stats['time'] += time.time() - t0

    def stats(self) -> dict:
        return self._stats.copy()