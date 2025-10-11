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

# Import advanced metrics & objectives
from .objectives import OBJECTIVE_FUNCTIONS
try:
    from .metrics import create_adjusted_score_objective
    ADVANCED_METRICS_AVAILABLE = True
    # Добавляем старую метрику в новый реестр для совместимости
    OBJECTIVE_FUNCTIONS['adjusted_score'] = create_adjusted_score_objective()
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

# Import profiling capabilities
try:
    from ..profiling import OptunaProfiler, StrategyProfiler, ProfileReport
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
            safe_symbol = symbol.replace('/', '-')
            study_name = f"{strategy_name}_{safe_symbol}_{timestamp}"
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
                position_size_dollars=100.0
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
            self.profiler = OptunaProfiler()

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

    def _sample_params_conditional(self, trial) -> Dict[str, Any]:
        """
        Conditional parameter sampling for 'ported_from_example' strategy.
        Only samples parameters relevant to the chosen entry_logic_mode.

        This reduces the search space dimension from 7 to 5-6 parameters,
        speeding up TPE sampler by ~2x.
        """
        default_params = self.strategy_class.get_default_params()
        params = {}

        # Step 1: Sample entry_logic_mode FIRST
        entry_logic_mode = trial.suggest_categorical(
            'entry_logic_mode',
            ["Принты и HLdir", "Только по принтам", "Только по HLdir"]
        )
        params['entry_logic_mode'] = entry_logic_mode

        # Step 2: Conditionally sample prints_* parameters
        if entry_logic_mode in ["Принты и HLdir", "Только по принтам"]:
            params['prints_analysis_period'] = trial.suggest_int('prints_analysis_period', 1, 10)
            params['prints_threshold_ratio'] = trial.suggest_float('prints_threshold_ratio', 1.1, 5.0, step=0.01)
            trial.set_user_attr('prints_params_sampled', True)
        else:
            # Use defaults for "Только по HLdir" mode
            params['prints_analysis_period'] = default_params['prints_analysis_period']
            params['prints_threshold_ratio'] = default_params['prints_threshold_ratio']
            trial.set_user_attr('prints_params_sampled', False)

        # Step 3: Conditionally sample hldir_* parameters
        if entry_logic_mode in ["Принты и HLdir", "Только по HLdir"]:
            params['hldir_window'] = trial.suggest_int('hldir_window', 2, 20)
            params['hldir_offset'] = trial.suggest_int('hldir_offset', 0, 5)
            trial.set_user_attr('hldir_params_sampled', True)
        else:
            # Use defaults for "Только по принтам" mode
            params['hldir_window'] = default_params['hldir_window']
            params['hldir_offset'] = default_params['hldir_offset']
            trial.set_user_attr('hldir_params_sampled', False)

        # Step 4: Always sample stop/take profit parameters
        params['stop_loss_pct'] = trial.suggest_float('stop_loss_pct', 0.5, 5.0, step=0.01)
        params['take_profit_pct'] = trial.suggest_float('take_profit_pct', 1.0, 8.0, step=0.01)

        return params

    def _sample_params_standard(self, trial) -> Dict[str, Any]:
        """
        Standard parameter sampling (samples all parameters from param_space).
        Used for strategies that don't have conditional dependencies.
        """
        params = {}
        for param_name, (param_type, *bounds) in self.param_space.items():
            if param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1], step=0.01)
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, *bounds)
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, *bounds)
        return params


    def create_objective_function(self,
                                 objective_metric: str = 'sharpe_ratio',
                                 custom_objective: Optional[Callable] = None,
                                 min_trades: int = 10,
                                 max_drawdown_threshold: float = 50.0,
                                 data_slice: str = 'full') -> Callable:
        def objective(trial):
            try:
                # Use conditional sampling for 'ported_from_example' strategy
                # This reduces search space and speeds up TPE by ~2x
                if self.strategy_name == 'ported_from_example':
                    params = self._sample_params_conditional(trial)
                else:
                    params = self._sample_params_standard(trial)

                if self.enable_debug:
                    sampled = []
                    if trial.user_attrs.get('prints_params_sampled', True):
                        sampled.append('prints')
                    if trial.user_attrs.get('hldir_params_sampled', True):
                        sampled.append('hldir')
                    print(f"[Trial {trial.number}] mode={params.get('entry_logic_mode', 'N/A')}, sampled={sampled}, params={params}")
                
                klines_data = self.cached_data[data_slice]
                strategy = self.strategy_class(symbol=self.symbol, **params)
                results = strategy.vectorized_process_dataset(klines_data)

                if 'error' in results:
                    return -float('inf') if self.direction == 'maximize' else float('inf')

                total_trades = results.get('total', 0)
                max_drawdown = abs(results.get('max_drawdown', 0))

                # Store all metrics in trial user_attrs for later analysis
                trial.set_user_attr('pnl', results.get('net_pnl', 0))
                trial.set_user_attr('winrate', results.get('win_rate', 0))
                trial.set_user_attr('trades', total_trades)
                trial.set_user_attr('sharpe', results.get('sharpe_ratio', 0))
                trial.set_user_attr('pf', results.get('profit_factor', 0))
                trial.set_user_attr('max_dd', max_drawdown)
                trial.set_user_attr('sortino', results.get('sortino_ratio', 0))
                trial.set_user_attr('avg_win', results.get('average_win', 0))
                trial.set_user_attr('avg_loss', results.get('average_loss', 0))
                trial.set_user_attr('winrate_long', results.get('winrate_long', 0))
                trial.set_user_attr('winrate_short', results.get('winrate_short', 0))
                trial.set_user_attr('consecutive_stops', results.get('consecutive_stops', 0))

                # Проверка ограничений
                if results.get('total', 0) == 0:
                    return -1e6

                if total_trades < min_trades:
                    return -1e6

                if max_drawdown > max_drawdown_threshold:
                    return -1e6

                # Calculate objective value
                if custom_objective:
                    value = custom_objective(results)
                # Сначала проверяем новые комплексные метрики
                elif objective_metric in OBJECTIVE_FUNCTIONS:
                    value = OBJECTIVE_FUNCTIONS[objective_metric](results)
                # Затем стандартные метрики из результатов бэктеста
                elif objective_metric in results:
                    value = results[objective_metric]
                else:
                    raise ValueError(f"Metric or objective '{objective_metric}' not found in results or registered objectives.")

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
                 pruner: Optional[optuna.pruners.BasePruner] = None,
                 callbacks: Optional[List[Callable]] = None,
                 data_slice: str = 'full') -> Dict[str, Any]:

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
            max_drawdown_threshold=max_drawdown_threshold,
            data_slice=data_slice
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

        # Pre-warm Numba cache to prevent re-compilation in parallel processes
        print("\nPre-warming Numba JIT cache...")
        try:
            # Use default parameters to create a dummy strategy instance
            default_params = self.strategy_class.get_default_params()
            dummy_strategy = self.strategy_class(symbol=self.symbol, **default_params)
            
            # Use a small slice of data to trigger compilation of all @njit functions
            if len(self.cached_data['full']) > 200:
                warmup_data_slice = self.cached_data['full'][:200]
                dummy_strategy.vectorized_process_dataset(warmup_data_slice)
                print("Numba JIT cache pre-warmed successfully.")
            else:
                print("Warning: Not enough data to pre-warm Numba cache.")
        except Exception as e:
            # Catching a broad exception to ensure optimization doesn't fail if pre-warming does
            print(f"Warning: Failed to pre-warm Numba cache: {e}")
        print("-" * 60)

        start_time = datetime.now()

        if self.enable_profiling and self.profiler:
            self.profiler.profile_study(
                study,
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                callbacks=callbacks
            )
        else:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, callbacks=callbacks)

        end_time = datetime.now()
        self.best_params = study.best_params
        self.best_value = study.best_value

        # Final backtest
        klines_data = self.cached_data['full']
        strategy = self.strategy_class(symbol=self.symbol, **self.best_params)
        final_results = strategy.vectorized_process_dataset(klines_data)

        # Create composite best_value string: pnl|winrate|trades|sharpe*pf
        pnl = final_results.get('net_pnl', 0)
        winrate = final_results.get('win_rate', 0)
        total_trades = final_results.get('total', 0)
        sharpe = final_results.get('sharpe_ratio', 0)
        pf = final_results.get('profit_factor', 0)
        sharpe_x_pf = sharpe * pf

        composite_value = f"{pnl:.2f}|{winrate:.2%}|{total_trades}|{sharpe_x_pf:.2f}"

        results = {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'study_name': self.study_name,
            'objective_metric': objective_metric,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'best_value_composite': composite_value,
            'n_trials': len(study.trials),
            'successful_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'optimization_time_seconds': (end_time - start_time).total_seconds(),
            'optimization_completed_at': end_time.isoformat(),
            'parallel_jobs': n_jobs,
            'final_backtest': final_results,
            'positive_trials': [] # Будет заполнено ниже
        }

        self.optimization_history.append(results)

        print(f"Optimization completed in {results['optimization_time_seconds']:.2f}s")
        print(f"Best {objective_metric}: {self.best_value:.4f}")
        print(f"Composite Value: {composite_value}")
        print("-" * 80)

        # Collect all trials with positive PnL
        positive_trials = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pnl = trial.user_attrs.get('pnl', 0)
                if pnl > 0:
                    positive_trials.append({
                        'trial': trial.number,
                        'pnl': pnl,
                        'winrate': trial.user_attrs.get('winrate', 0),
                        'trades': trial.user_attrs.get('trades', 0),
                        'sharpe': trial.user_attrs.get('sharpe', 0),
                        'pf': trial.user_attrs.get('pf', 0),
                        'max_dd': trial.user_attrs.get('max_dd', 0),
                        'sortino': trial.user_attrs.get('sortino', 0),
                        'avg_win': trial.user_attrs.get('avg_win', 0),
                        'avg_loss': trial.user_attrs.get('avg_loss', 0),
                        'winrate_long': trial.user_attrs.get('winrate_long', 0),
                        'winrate_short': trial.user_attrs.get('winrate_short', 0),
                        'consecutive_stops': trial.user_attrs.get('consecutive_stops', 0),
                        'params': trial.params
                    })

        # Sort by PnL descending
        positive_trials.sort(key=lambda x: x['pnl'], reverse=True)

        results['positive_trials'] = positive_trials

        if positive_trials:
            print(f"\n🟢 Trials with Positive PnL ({len(positive_trials)} found):")
            print("=" * 160)
            # Header
            print(f"{'Trial':<8} {'PnL':<12} {'WinRate':<10} {'Trades':<8} {'Sharpe':<10} {'PF':<10} {'MaxDD%':<10} {'WRLong':<10} {'WRShort':<10} {'ConsSL':<8}")
            print("-" * 160)
            # Rows
            for t in positive_trials:
                print(f"{t['trial']:<8} {t['pnl']:<12.2f} {t['winrate']:<10.2%} {t['trades']:<8} {t['sharpe']:<10.2f} {t['pf']:<10.2f} {t['max_dd']:<10.2f} {t['winrate_long']:<10.2%} {t['winrate_short']:<10.2%} {t['consecutive_stops']:<8}")
            print("=" * 160)
        else:
            print("\n❌ No trials with positive PnL found")
            print("-" * 80)

        if self.enable_profiling and self.profiler:
            self._generate_profiling_report(study)

        return results

    def _generate_profiling_report(self, study: optuna.Study) -> None:
        """
        Generate and save the profiling report.
        """
        if not self.profiler:
            return

        os.makedirs(self.profiling_output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.profiling_output_dir,
            f"optuna_profile_{self.study_name}_{timestamp}.txt"
        )

        self.profiler.save_optimization_report(report_path)

        # Export metrics to JSON
        metrics_path = os.path.join(
            self.profiling_output_dir,
            f"optuna_metrics_{self.study_name}_{timestamp}.json"
        )
        self.profiler.export_metrics(metrics_path)

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
                       position_size: float = 100.0) -> Dict[str, Any]:
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