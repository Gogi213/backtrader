"""
High-Performance Strategy Optimizer with Caching
Optimized version: removed duplication, unified strategy interface, cleaned logs

Author: HFT System (optimized)
"""
import os
import optuna
import numpy as np
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
import json
import pickle
import hashlib
import multiprocessing as mp
from dataclasses import dataclass

from ..data.backtest_engine import run_vectorized_klines_backtest
from ..strategies.base_strategy import StrategyRegistry

# Import advanced metrics
try:
    from .metrics import create_adjusted_score_objective
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

# Import profiling capabilities
try:
    from ..profiling import OptunaProfiler, StrategyProfiler
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False


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


class FastStrategyOptimizer:
    """
    High-performance optimizer for trading strategies with caching
    Optimized: unified interface, no duplication, clean logs
    """

    def __init__(self,
                 strategy_name: str,
                 data_path: str,
                 symbol: str = 'BTCUSDT',
                 study_name: Optional[str] = None,
                 direction: str = 'maximize',
                 storage: Optional[str] = None,
                 cache_dir: str = "optimization_cache",
                 backtest_config: Optional[Any] = None,
                 enable_profiling: bool = False,
                 enable_debug: bool = False,
                 profiling_output_dir: str = "optimization_profiling"):

        self.strategy_name = strategy_name
        self.data_path = data_path
        self.symbol = symbol
        self.direction = direction
        self.storage = storage
        self.enable_debug = enable_debug

        strategy_class = StrategyRegistry.get_strategy(strategy_name)
        if not strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry")
        self.strategy_class = strategy_class
        self.param_space = strategy_class.get_param_space()

        if not study_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"{strategy_name}_{symbol}_{timestamp}"
        self.study_name = study_name

        if backtest_config:
            self.backtest_config = backtest_config
        else:
            from ..core.backtest_config import BacktestConfig
            self.backtest_config = BacktestConfig(
                strategy_name=strategy_name,
                symbol=symbol,
                data_path=data_path,
                initial_capital=100.0,
                commission_pct=0.0005,
                position_size_dollars=50.0
            )

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_key = self._generate_cache_key()

        self.study = None
        self.best_params = None
        self.best_value = None
        self.optimization_history = []

        self.enable_profiling = enable_profiling and PROFILING_AVAILABLE
        self.profiling_output_dir = profiling_output_dir
        self.profiler = None
        if self.enable_profiling:
            self.profiler = OptunaProfiler(enable_resource_monitoring=True)

        self._preprocess_data()

    def _generate_cache_key(self) -> str:
        file_hash = hashlib.md5()
        with open(self.data_path, 'rb') as f:
            file_hash.update(f.read())
        key_data = f"{self.strategy_name}_{self.symbol}_{file_hash.hexdigest()[:8]}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _preprocess_data(self):
        cache_file = os.path.join(self.cache_dir, f"data_{self.cache_key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.cached_data = pickle.load(f)
            return

        from ..data.klines_handler import VectorizedKlinesHandler
        handler = VectorizedKlinesHandler()
        klines_data = handler.load_klines(self.data_path)
        
        full_size = len(klines_data)
        self.cached_data = {
            'full': klines_data,
            'p50': klines_data[int(full_size * 0.5):],
            'p25': klines_data[int(full_size * 0.75):],
            'p10': klines_data[int(full_size * 0.9):],
            'full_size': full_size
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(self.cached_data, f)


    def create_objective_function(self,
                                 objective_metric: str = 'sharpe_ratio',
                                 custom_objective: Optional[Callable] = None,
                                 min_trades: int = 10,
                                 max_drawdown_threshold: float = 50.0) -> Callable:
        def objective(trial):
            try:
                params = {}
                for param_name, (param_type, *bounds) in self.param_space.items():
                    if param_type == 'float':
                        params[param_name] = trial.suggest_float(param_name, *bounds)
                    elif param_type == 'int':
                        params[param_name] = trial.suggest_int(param_name, *bounds)
                    elif param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, *bounds)

                if self.enable_debug:
                    print(f"[Trial {trial.number}] params: {params}")
                
                # Проверка параметров
                print(f"[PARAMS] s_entry={params.get('s_entry')}, hl_min={params.get('hl_min')}, hl_max={params.get('hl_max')}")

                klines_data = self.cached_data['full']
                strategy = self.strategy_class(symbol=self.symbol, **params)
                results = strategy.vectorized_process_dataset(klines_data)

                if 'error' in results:
                    return -float('inf') if self.direction == 'maximize' else float('inf')

                total_trades = results.get('total', 0)
                max_drawdown = abs(results.get('max_drawdown', 0))

                # Отладочный вывод для диагностики 0 сделок
                if self.enable_debug or True:  # временно всегда вкл
                    print(f"[Trial {trial.number}] trades={results.get('total', 0)}")

                # Быстрый патч для диагностики 0 сделок
                if results.get('total', 0) == 0:
                    print(f"[ZERO-TRADES] Trial {trial.number}: 0 trades")
                    # Попробуем получить дополнительную информацию из стратегии
                    if hasattr(strategy, '_last_debug_info'):
                        debug_info = strategy._last_debug_info
                        print(f"[ZERO-TRADES] entry_mask.sum() = {debug_info.get('entry_mask_sum', 'N/A')}")
                        print(f"[ZERO-TRADES] regime_ok.sum() = {debug_info.get('regime_ok_sum', 'N/A')}")
                        print(f"[ZERO-TRADES] z_entry_ok.sum() = {debug_info.get('z_entry_ok_sum', 'N/A')}")
                        print(f"[ZERO-TRADES] hl_ok.sum() = {debug_info.get('hl_ok_sum', 'N/A')}")
                    return -1e6  # штраф за 0 сделок

                if total_trades < min_trades:
                    penalty = min_trades - total_trades
                    return -penalty if self.direction == 'maximize' else penalty

                if max_drawdown > max_drawdown_threshold:
                    penalty = max_drawdown - max_drawdown_threshold
                    return -penalty if self.direction == 'maximize' else penalty

                if custom_objective:
                    value = custom_objective(results)
                elif objective_metric == 'adjusted_score' and ADVANCED_METRICS_AVAILABLE:
                    value = create_adjusted_score_objective()(results)
                elif objective_metric in results:
                    value = results[objective_metric]
                else:
                    raise ValueError(f"Metric '{objective_metric}' not found in results")

                trial.report(value, trial.number)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                return value

            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                if self.enable_debug:
                    print(f"[Trial {trial.number}] error: {e}")
                return -float('inf') if self.direction == 'maximize' else float('inf')

        return objective

    def optimize(self,
                 n_trials: int = 100,
                 objective_metric: str = 'sharpe_ratio',
                 custom_objective: Optional[Callable] = None,
                 min_trades: int = 10,
                 max_drawdown_threshold: float = 50.0,
                 timeout: Optional[float] = 600,
                 n_jobs: int = -1,
                 sampler: Optional[optuna.samplers.BaseSampler] = None,
                 pruner: Optional[optuna.pruners.BasePruner] = None) -> Dict[str, Any]:

        print(f"Starting FAST optimization for {self.strategy_name} on {self.symbol}")
        print(f"Objective: {objective_metric}, Trials: {n_trials}, Jobs: {n_jobs}")
        print("-" * 60)

        if sampler is None:
            sampler = optuna.samplers.TPESampler(
                seed=42,
                multivariate=True,
                group=True,
                n_startup_trials=10,
                warn_independent_sampling=False
            )
        if pruner is None:
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource='auto',
                reduction_factor=3
            )

        objective = self.create_objective_function(
            objective_metric=objective_metric,
            custom_objective=custom_objective,
            min_trades=min_trades,
            max_drawdown_threshold=max_drawdown_threshold
        )

        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        n_jobs = min(n_jobs, mp.cpu_count())

        if self.storage:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage,
                sampler=sampler,
                pruner=pruner
            )
        else:
            study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                sampler=sampler,
                pruner=pruner
            )

        self.study = study

        start_time = datetime.now()

        if self.enable_profiling and self.profiler:
            objective = self.profiler.wrap_objective_function(objective)
            with self.profiler.profile_optimization(self.study_name):
                study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
        else:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

        end_time = datetime.now()
        self.best_params = study.best_params
        self.best_value = study.best_value

        # Final backtest
        klines_data = self.cached_data['full']
        strategy = self.strategy_class(symbol=self.symbol, **self.best_params)
        final_results = strategy.vectorized_process_dataset(klines_data)

        results = {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'study_name': self.study_name,
            'objective_metric': objective_metric,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(study.trials),
            'successful_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'optimization_time_seconds': (end_time - start_time).total_seconds(),
            'optimization_completed_at': end_time.isoformat(),
            'parallel_jobs': n_jobs,
            'final_backtest': final_results
        }

        self.optimization_history.append(results)

        print(f"Optimization completed in {results['optimization_time_seconds']:.2f}s")
        print(f"Best {objective_metric}: {self.best_value:.4f}")

        if self.enable_profiling and self.profiler:
            self._generate_profiling_report(study, results['optimization_time_seconds'])

        return results

    def get_parameter_importance(self) -> Dict[str, float]:
        if not self.study:
            raise ValueError("No optimization study available. Run optimize() first.")
        try:
            return optuna.importance.get_param_importances(self.study)
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")
            return {}

    def save_results(self, filepath: str) -> None:
        if not self.optimization_history:
            raise ValueError("No optimization results to save")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'optimization_history': self.optimization_history,
                'parameter_importance': self.get_parameter_importance(),
                'saved_at': datetime.now().isoformat()
            }, f, indent=2)
        print(f"Results saved to {filepath}")

    def load_results(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            results = json.load(f)
        self.optimization_history = results.get('optimization_history', [])
        if self.optimization_history:
            latest = self.optimization_history[-1]
            self.best_params = latest.get('best_params')
            self.best_value = latest.get('best_value')
        print(f"Results loaded from {filepath}")
        return results


# Convenience function
def quick_fast_optimize(strategy_name: str,
                       data_path: str,
                       symbol: str = 'BTCUSDT',
                       n_trials: int = 50,
                       objective_metric: str = 'sharpe_ratio',
                       n_jobs: int = -1,
                       initial_capital: float = 100.0,
                       position_size: float = 50.0) -> Dict[str, Any]:
    from ..core.backtest_config import BacktestConfig
    backtest_config = BacktestConfig(
        strategy_name=strategy_name,
        symbol=symbol,
        data_path=data_path,
        initial_capital=initial_capital,
        commission_pct=0.0005,
        position_size_dollars=position_size
    )
    
    optimizer = FastStrategyOptimizer(
        strategy_name=strategy_name,
        data_path=data_path,
        symbol=symbol,
        backtest_config=backtest_config
    )
    return optimizer.optimize(
        n_trials=n_trials,
        objective_metric=objective_metric,
        n_jobs=n_jobs
    )


def create_composite_objective(weights: Dict[str, float]) -> Callable:
    def composite_objective(results: Dict[str, Any]) -> float:
        score = 0.0
        for metric, weight in weights.items():
            if metric in results:
                score += results[metric] * weight
        return score
    return composite_objective