"""
High-Performance Strategy Optimizer with Caching

This module provides a fast optimizer for trading strategies using Optuna
with caching and parallel processing for 10x+ speedup.

Author: HFT System
"""
import os
import optuna
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Tuple
from datetime import datetime
import json
import warnings
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from ..data.backtest_engine import run_vectorized_klines_backtest
from ..strategies.strategy_registry import StrategyRegistry

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


class FastStrategyOptimizer:
    """
    High-performance optimizer for trading strategies with caching
    
    Features:
    - Data preprocessing and caching
    - Parallel optimization trials
    - Adaptive evaluation with reduced data for early trials
    - Smart pruning
    - Result caching across optimization sessions
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
                 profiling_output_dir: str = "optimization_profiling"):
        """
        Initialize the fast optimizer
        
        Args:
            strategy_name: Name of the strategy to optimize
            data_path: Path to the CSV data file
            symbol: Trading symbol
            study_name: Name for the Optuna study (auto-generated if None)
            direction: Optimization direction ('maximize' or 'minimize')
            storage: Database URL for persistent storage (None for in-memory)
            cache_dir: Directory for caching data and results
            backtest_config: Backtest configuration
            enable_profiling: Enable performance profiling
            profiling_output_dir: Directory for profiling reports
        """
        self.strategy_name = strategy_name
        self.data_path = data_path
        self.symbol = symbol
        
        # Validate strategy exists
        strategy_class = StrategyRegistry.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry")
        
        self.strategy_class = strategy_class
        self.param_space = strategy_class.get_param_space()
        
        # Generate study name if not provided
        if not study_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"{strategy_name}_{symbol}_{timestamp}"
        
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        
        # Use provided backtest config or create a default one
        if backtest_config:
            self.backtest_config = backtest_config
        else:
            from ..core.backtest_config import BacktestConfig
            self.backtest_config = BacktestConfig(
                strategy_name=strategy_name,
                symbol=symbol,
                data_path=data_path,
                initial_capital=10000.0,
                commission_pct=0.05,
                position_size_dollars=1000.0
            )
        
        # Caching
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_key = self._generate_cache_key()
        
        # Initialize study
        self.study = None
        self.best_params = None
        self.best_value = None
        self.optimization_history = []
        
        # Profiling setup
        self.enable_profiling = enable_profiling and PROFILING_AVAILABLE
        self.profiling_output_dir = profiling_output_dir
        self.profiler = None
        
        if self.enable_profiling:
            self.profiler = OptunaProfiler(enable_resource_monitoring=True)
            print(f"Profiling enabled - reports will be saved to {profiling_output_dir}")
        
        # Preprocess and cache data
        self._preprocess_data()
        
    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on strategy and data"""
        # Get file hash
        file_hash = hashlib.md5()
        with open(self.data_path, 'rb') as f:
            file_hash.update(f.read())
        
        # Create unique key
        key_data = f"{self.strategy_name}_{self.symbol}_{file_hash.hexdigest()[:8]}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _preprocess_data(self):
        """Preprocess and cache data for faster optimization using numpy arrays"""
        cache_file = os.path.join(self.cache_dir, f"data_{self.cache_key}.pkl")
        
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # DEBUG: Check what we loaded from cache
            print(f"DEBUG: Loaded cached_data type: {type(loaded_data)}")
            print(f"DEBUG: cached_data keys: {list(loaded_data.keys())}")
            for key in ['full', 'p50', 'p25', 'p10']:
                if key in loaded_data:
                    subset = loaded_data[key]
                    print(f"DEBUG: {key} type: {type(subset)}")
                    if hasattr(subset, 'keys'):
                        print(f"DEBUG: {key} keys: {list(subset.keys())}")
                    if hasattr(subset, 'columns'):
                        print(f"DEBUG: {key} columns: {list(subset.columns)}")
            
            # Convert DataFrames to numpy arrays if needed
            self.cached_data = {}
            for key in ['full', 'p50', 'p25', 'p10']:
                if key in loaded_data:
                    subset = loaded_data[key]
                    if hasattr(subset, 'columns'):  # pandas DataFrame
                        self.cached_data[key] = {
                            'times': subset['time'].values,
                            'opens': subset['open'].values,
                            'highs': subset['high'].values,
                            'lows': subset['low'].values,
                            'closes': subset['close'].values,
                            'volumes': subset['Volume'].values
                        }
                    else:  # dict with numpy arrays
                        self.cached_data[key] = subset
            
            # Copy full_size
            if 'full_size' in loaded_data:
                self.cached_data['full_size'] = loaded_data['full_size']
            
            return
        
        print("Preprocessing data for caching...")
        from ..data.klines_handler import VectorizedKlinesHandler
        
        # Load and preprocess data
        handler = VectorizedKlinesHandler()
        klines_data = handler.load_klines(self.data_path)
        
        # DEBUG: Check what handler returned
        print(f"DEBUG: handler.load_klines returned type: {type(klines_data)}")
        if hasattr(klines_data, 'columns'):
            print(f"DEBUG: klines_data columns: {list(klines_data.columns)}")
        if hasattr(klines_data, 'data'):
            print(f"DEBUG: klines_data.data keys: {list(klines_data.data.keys())}")
        
        # Extract all data to numpy arrays at once (no pandas after this)
        # Handle both NumpyKlinesData and DataFrame
        if hasattr(klines_data, 'data'):  # NumpyKlinesData
            times = klines_data.data['time']
            opens = klines_data.data['open']
            highs = klines_data.data['high']
            lows = klines_data.data['low']
            closes = klines_data.data['close']
            volumes = klines_data.data['volume']
        else:  # pandas DataFrame
            times = klines_data['time'].values
            opens = klines_data['open'].values
            highs = klines_data['high'].values
            lows = klines_data['low'].values
            closes = klines_data['close'].values
            volumes = klines_data['Volume'].values
        
        # DEBUG: Log extracted data
        print(f"DEBUG: _preprocess_data extracted data")
        print(f"DEBUG: times type={type(times)}, shape={times.shape}")
        print(f"DEBUG: closes type={type(closes)}, shape={closes.shape}")
        print(f"DEBUG: full_size={len(times)}")
        
        # Create different data sizes for adaptive evaluation (using numpy slicing)
        full_size = len(times)
        
        # Create data subsets (10%, 25%, 50%, 100%)
        self.cached_data = {
            'full': {
                'times': times,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'volumes': volumes
            },
            'p50': {
                'times': times[int(full_size * 0.5):],
                'opens': opens[int(full_size * 0.5):] if opens is not None else None,
                'highs': highs[int(full_size * 0.5):] if highs is not None else None,
                'lows': lows[int(full_size * 0.5):] if lows is not None else None,
                'closes': closes[int(full_size * 0.5):],
                'volumes': volumes[int(full_size * 0.5):] if volumes is not None else None
            },
            'p25': {
                'times': times[int(full_size * 0.75):],
                'opens': opens[int(full_size * 0.75):] if opens is not None else None,
                'highs': highs[int(full_size * 0.75):] if highs is not None else None,
                'lows': lows[int(full_size * 0.75):] if lows is not None else None,
                'closes': closes[int(full_size * 0.75):],
                'volumes': volumes[int(full_size * 0.75):] if volumes is not None else None
            },
            'p10': {
                'times': times[int(full_size * 0.9):],
                'opens': opens[int(full_size * 0.9):] if opens is not None else None,
                'highs': highs[int(full_size * 0.9):] if highs is not None else None,
                'lows': lows[int(full_size * 0.9):] if lows is not None else None,
                'closes': closes[int(full_size * 0.9):],
                'volumes': volumes[int(full_size * 0.9):] if volumes is not None else None
            },
            'full_size': full_size
        }
        
        # DEBUG: Log cache structure
        print(f"DEBUG: Cache structure created")
        for key in ['full', 'p50', 'p25', 'p10']:
            subset = self.cached_data[key]
            print(f"DEBUG: {key} - times shape={subset['times'].shape}, closes shape={subset['closes'].shape}")
        
        # DEBUG: Check what we're saving to cache
        print(f"DEBUG: Saving cached_data type: {type(self.cached_data)}")
        print(f"DEBUG: cached_data keys: {list(self.cached_data.keys())}")
        for key in ['full', 'p50', 'p25', 'p10']:
            if key in self.cached_data:
                subset = self.cached_data[key]
                print(f"DEBUG: {key} type: {type(subset)}")
                if hasattr(subset, 'keys'):
                    print(f"DEBUG: {key} keys: {list(subset.keys())}")
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cached_data, f)
        
        print(f"Data cached to {cache_file}")
    
    def create_objective_function(self, 
                                 objective_metric: str = 'sharpe_ratio',
                                 custom_objective: Optional[Callable] = None,
                                 min_trades: int = 10,
                                 max_drawdown_threshold: float = 50.0,
                                 use_adaptive: bool = True) -> Callable:
        """
        Create the objective function for optimization with adaptive evaluation
        
        Args:
            objective_metric: Metric to optimize
            custom_objective: Custom objective function
            min_trades: Minimum number of trades required
            max_drawdown_threshold: Maximum allowed drawdown percentage
            use_adaptive: Use adaptive evaluation with reduced data for early trials
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            try:
                print(f"DEBUG: Starting trial {trial.number}")
                
                # Suggest parameters based on strategy's parameter space
                params = {}
                for param_name, (param_type, *bounds) in self.param_space.items():
                    if param_type == 'float':
                        if len(bounds) == 2:
                            params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1], step=bounds[2])
                    elif param_type == 'int':
                        if len(bounds) == 2:
                            params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1])
                        else:
                            params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1], step=bounds[2])
                    elif param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, bounds)
                
                print(f"DEBUG: Generated params: {params}")
                
                # Always use full data (adaptive evaluation disabled)
                data_key = 'full'
                data_size_pct = 1.0
                
                print(f"DEBUG: Using full data (adaptive evaluation disabled)")
                
                # Get cached data (numpy arrays, no pandas)
                print(f"DEBUG: Accessing cached_data with key '{data_key}'")
                print(f"DEBUG: Available keys in cached_data: {list(self.cached_data.keys())}")
                data_arrays = self.cached_data[data_key]
                print(f"DEBUG: Retrieved data_arrays, type={type(data_arrays)}")
                
                # DEBUG: Log data structure
                print(f"DEBUG: data_key={data_key}")
                print(f"DEBUG: data_arrays keys={list(data_arrays.keys())}")
                print(f"DEBUG: 'times' in data_arrays={'times' in data_arrays}")
                if 'times' in data_arrays:
                    print(f"DEBUG: times type={type(data_arrays['times'])}, shape={data_arrays['times'].shape if hasattr(data_arrays['times'], 'shape') else len(data_arrays['times'])}")
                print(f"DEBUG: 'closes' in data_arrays={'closes' in data_arrays}")
                if 'closes' in data_arrays:
                    print(f"DEBUG: closes type={type(data_arrays['closes'])}, shape={data_arrays['closes'].shape if hasattr(data_arrays['closes'], 'shape') else len(data_arrays['closes'])}")
                
                # Create strategy instance with parameters
                strategy = self.strategy_class(symbol=self.symbol, **params)
                
                # For hierarchical_mean_reversion strategy, pass numpy arrays directly
                if self.strategy_name == 'hierarchical_mean_reversion':
                    print("DEBUG: Calling turbo_process_dataset with numpy arrays...")
                    # DEBUG: Log before calling strategy
                    print(f"DEBUG: About to call strategy.turbo_process_dataset")
                    print(f"DEBUG: times available: {'times' in data_arrays and data_arrays['times'] is not None}")
                    print(f"DEBUG: closes available: {'closes' in data_arrays and data_arrays['closes'] is not None}")
                    
                    # Run backtest with suggested parameters using numpy arrays directly
                    try:
                        # Handle both DataFrame and dict formats
                        if hasattr(data_arrays, 'columns'):  # pandas DataFrame
                            times = data_arrays['time'].values
                            opens = data_arrays['open'].values
                            highs = data_arrays['high'].values
                            lows = data_arrays['low'].values
                            closes = data_arrays['close'].values
                        else:  # dict with numpy arrays
                            times = data_arrays['times']
                            opens = data_arrays['opens']
                            highs = data_arrays['highs']
                            lows = data_arrays['lows']
                            closes = data_arrays['closes']
                        
                        results = strategy.turbo_process_dataset(
                            times=times,
                            prices=closes,
                            opens=opens,
                            highs=highs,
                            lows=lows,
                            closes=closes
                        )
                        print("DEBUG: turbo_process_dataset completed successfully")
                    except KeyError as e:
                        print(f"DEBUG: KeyError in turbo_process_dataset: {e}")
                        print(f"DEBUG: Available keys: {list(data_arrays.keys())}")
                        raise
                    except Exception as e:
                        print(f"DEBUG: Exception in turbo_process_dataset: {type(e).__name__}: {e}")
                        raise
                    
                    # Convert to standard format
                    results['symbol'] = self.symbol
                else:
                    # For non-turbo strategies, pass numpy arrays directly to backtest engine
                    # The backtest engine will handle strategy-specific processing
                    results = run_vectorized_klines_backtest(
                        csv_path=None,  # No CSV file
                        times=data_arrays['times'],
                        prices=data_arrays['closes'],
                        opens=data_arrays['opens'],
                        highs=data_arrays['highs'],
                        lows=data_arrays['lows'],
                        closes=data_arrays['closes'],
                        volumes=data_arrays['volumes'],
                        symbol=self.symbol,
                        strategy_name=self.strategy_name,
                        strategy_params=params,
                        initial_capital=self.backtest_config.initial_capital,
                        position_size=self.backtest_config.position_size_dollars,
                        commission_pct=self.backtest_config.commission_pct
                    )
                
                # Check if backtest failed
                if 'error' in results:
                    print(f"Trial failed: {results['error']}")
                    return -float('inf') if self.direction == 'maximize' else float('inf')
                
                # Extract metrics
                total_trades = results.get('total', 0)
                max_drawdown = abs(results.get('max_drawdown', 0))
                
                # Apply constraints
                if total_trades < min_trades:
                    penalty = min_trades - total_trades
                    return -penalty if self.direction == 'maximize' else penalty
                
                if max_drawdown > max_drawdown_threshold:
                    penalty = max_drawdown - max_drawdown_threshold
                    return -penalty if self.direction == 'maximize' else penalty
                
                # Get objective value
                if custom_objective:
                    value = custom_objective(results)
                elif objective_metric == 'adjusted_score' and ADVANCED_METRICS_AVAILABLE:
                    # Use advanced adjusted_score metric
                    adjusted_objective = create_adjusted_score_objective()
                    value = adjusted_objective(results)
                elif objective_metric in results:
                    value = results[objective_metric]
                else:
                    raise ValueError(f"Metric '{objective_metric}' not found in backtest results")
                
                # Data size adjustment removed - always using full data
                
                # Report intermediate value for pruning
                trial.report(value, trial.number)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                return value
                    
            except optuna.exceptions.TrialPruned:
                # Re-raise pruning exception
                raise
            except KeyError as e:
                print(f"DEBUG: KeyError in trial {trial.number}: {e}")
                print(f"DEBUG: This is likely the 'times' error we're looking for!")
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                return -float('inf') if self.direction == 'maximize' else float('inf')
            except Exception as e:
                print(f"DEBUG: Trial {trial.number} error: {type(e).__name__}: {e}")
                print(f"DEBUG: This might be the 'times' error!")
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                return -float('inf') if self.direction == 'maximize' else float('inf')
        
        return objective
    
    def optimize(self,
                 n_trials: int = 100,
                 objective_metric: str = 'sharpe_ratio',
                 custom_objective: Optional[Callable] = None,
                 min_trades: int = 10,
                 max_drawdown_threshold: float = 50.0,
                 timeout: Optional[float] = 600,  # 10 minutes default from GUI
                 n_jobs: int = -1,
                 use_adaptive: bool = True,
                 sampler: Optional[optuna.samplers.BaseSampler] = None,
                 pruner: Optional[optuna.pruners.BasePruner] = None) -> Dict[str, Any]:
        """
        Run the optimization with parallel processing and adaptive evaluation
        
        Args:
            n_trials: Number of optimization trials
            objective_metric: Metric to optimize
            custom_objective: Custom objective function
            min_trades: Minimum number of trades required
            max_drawdown_threshold: Maximum allowed drawdown percentage
            timeout: Time limit in seconds
            n_jobs: Number of parallel jobs (-1 for all cores)
            use_adaptive: Use adaptive evaluation with reduced data for early trials
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: HyperbandPruner)
            
        Returns:
            Dictionary with optimization results
        """
        print(f"Starting FAST optimization for {self.strategy_name} on {self.symbol}")
        print(f"Data: {self.data_path}")
        print(f"Objective: {objective_metric}")
        print(f"Trials: {n_trials}")
        print(f"Parallel jobs: {n_jobs}")
        print(f"Adaptive evaluation: {use_adaptive} (disabled - always using full data)")
        print(f"Direction: {self.direction}")
        print("-" * 60)
        
        # Set default sampler and pruner with advanced settings
        if sampler is None:
            # Use TPESampler with multivariate mode for better parameter relationships
            sampler = optuna.samplers.TPESampler(
                seed=42,
                multivariate=True,  # Consider parameter relationships
                group=True,  # Group parameters for better sampling
                n_startup_trials=10,  # More trials before model kicks in
                warn_independent_sampling=False  # Suppress warnings for independent sampling
            )
        if pruner is None:
            # Use HyperbandPruner for more aggressive pruning
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,  # Minimum resource allocated to a trial
                max_resource='auto',  # Automatically determined based on n_trials
                reduction_factor=3  # Aggressive pruning
            )
        
        # Create objective function
        objective = self.create_objective_function(
            objective_metric=objective_metric,
            custom_objective=custom_objective,
            min_trades=min_trades,
            max_drawdown_threshold=max_drawdown_threshold,
            use_adaptive=use_adaptive
        )
        
        # Determine number of parallel jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        n_jobs = min(n_jobs, mp.cpu_count())
        
        # Store n_jobs for profiling analysis
        self._last_n_jobs = n_jobs
        
        # Create or load study
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
        
        # Run optimization with parallel processing
        start_time = datetime.now()
        
        # Setup profiling if enabled
        if self.enable_profiling and self.profiler:
            # Wrap objective function for profiling
            objective = self.profiler.wrap_objective_function(objective)
            
            # Profile optimization
            with self.profiler.profile_optimization(self.study_name):
                if n_jobs > 1:
                    # Parallel optimization
                    print(f"Running optimization with {n_jobs} parallel jobs...")
                    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
                else:
                    # Sequential optimization
                    study.optimize(objective, n_trials=n_trials, timeout=timeout)
        else:
            # Regular optimization without profiling
            if n_jobs > 1:
                # Parallel optimization
                print(f"Running optimization with {n_jobs} parallel jobs...")
                study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
            else:
                # Sequential optimization
                study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        end_time = datetime.now()
        
        # Store results
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        # Calculate optimization statistics
        trials_df = study.trials_dataframe()
        successful_trials = trials_df[trials_df['state'] == 'COMPLETE']
        pruned_trials = trials_df[trials_df['state'] == 'PRUNED']
        
        optimization_time = (end_time - start_time).total_seconds()
        
        results = {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'study_name': self.study_name,
            'objective_metric': objective_metric,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(study.trials),
            'successful_trials': len(successful_trials),
            'pruned_trials': len(pruned_trials),
            'optimization_time_seconds': optimization_time,
            'optimization_completed_at': end_time.isoformat(),
            'parallel_jobs': n_jobs,
            'adaptive_evaluation': use_adaptive
        }
        
        # Run final backtest with best parameters on full data
        print("Running final backtest with best parameters...")
        
        # Get cached data (numpy arrays, no pandas)
        data_arrays = self.cached_data['full']
        
        # For hierarchical_mean_reversion strategy, pass numpy arrays directly
        if self.strategy_name == 'hierarchical_mean_reversion':
            # Create strategy instance with best parameters
            strategy = self.strategy_class(symbol=self.symbol, **self.best_params)
            
            # Run backtest with best parameters using numpy arrays directly
            final_results = strategy.turbo_process_dataset(
                times=data_arrays['times'],
                prices=data_arrays['closes'],
                opens=data_arrays['opens'],
                highs=data_arrays['highs'],
                lows=data_arrays['lows'],
                closes=data_arrays['closes']
            )
            
            # Convert to standard format
            final_results['symbol'] = self.symbol
        else:
            # For non-turbo strategies, pass numpy arrays directly to backtest engine
            # The backtest engine will handle strategy-specific processing
            final_results = run_vectorized_klines_backtest(
                csv_path=None,  # No CSV file
                times=data_arrays['times'],
                prices=data_arrays['closes'],
                opens=data_arrays['opens'],
                highs=data_arrays['highs'],
                lows=data_arrays['lows'],
                closes=data_arrays['closes'],
                volumes=data_arrays['volumes'],
                symbol=self.symbol,
                strategy_name=self.strategy_name,
                strategy_params=self.best_params,
                initial_capital=self.backtest_config.initial_capital,
                position_size=self.backtest_config.position_size_dollars,
                commission_pct=self.backtest_config.commission_pct
            )
        
        if 'error' not in final_results:
            results['final_backtest'] = final_results
            print(f"Final backtest completed with {final_results.get('total', 0)} trades")
            print(f"Sharpe Ratio: {final_results.get('sharpe_ratio', 0):.2f}")
            print(f"Net P&L: ${final_results.get('net_pnl', 0):,.2f}")
            print(f"Return: {final_results.get('net_pnl_percentage', 0):.2f}%")
        else:
            results['final_backtest_error'] = final_results['error']
            print(f"Final backtest failed: {final_results['error']}")
        
        self.optimization_history.append(results)
        
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best {objective_metric}: {self.best_value:.4f}")
        print(f"Speedup: ~{n_jobs}x with parallel processing")
        print(f"Adaptive evaluation disabled - using full data for all trials")
        
        # Generate profiling report if enabled
        if self.enable_profiling and self.profiler:
            self._generate_profiling_report(study, optimization_time)
        
        return results
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all optimizations run"""
        return self.optimization_history
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Get parameter importance based on the optimization study
        
        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if not self.study:
            raise ValueError("No optimization study available. Run optimize() first.")
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")
            return {}
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to a JSON file
        
        Args:
            filepath: Path to save the results
        """
        if not self.optimization_history:
            raise ValueError("No optimization results to save")
        
        results = {
            'optimization_history': self.optimization_history,
            'parameter_importance': self.get_parameter_importance(),
            'saved_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load optimization results from a JSON file
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Loaded results dictionary
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.optimization_history = results.get('optimization_history', [])
        
        # Restore latest optimization state
        if self.optimization_history:
            latest = self.optimization_history[-1]
            self.best_params = latest.get('best_params')
            self.best_value = latest.get('best_value')
        
        print(f"Results loaded from {filepath}")
        return results
    
    def _generate_profiling_report(self, study, optimization_time: float) -> None:
        """
        Generate comprehensive profiling report
        
        Args:
            study: Completed Optuna study
            optimization_time: Total optimization time
        """
        if not self.enable_profiling or not self.profiler:
            return
        
        try:
            # Create output directory
            os.makedirs(self.profiling_output_dir, exist_ok=True)
            
            # Analyze study results
            study_analysis = self.profiler.analyze_study_results(study)
            
            # Get parallel efficiency
            parallel_efficiency = self.profiler.get_parallel_efficiency(
                getattr(self, '_last_n_jobs', 1)
            )
            
            # Generate comprehensive report
            from ..profiling.profiling_utils import ProfileReport
            report_generator = ProfileReport(self.profiling_output_dir)
            
            # Prepare profiling data for report
            profiling_data = {
                'strategy_name': self.strategy_name,
                'optimization_stats': self.profiler.optimization_stats,
                'trial_times': self.profiler.trial_times,
                'resource_usage': self.profiler.resource_usage,
                'parameter_importance': self.profiler.parameter_importance,
                'optimization_analysis': study_analysis,
                'parallel_efficiency': parallel_efficiency,
                'bottlenecks': self.profiler.get_optimization_bottlenecks(),
                'cprofile_stats': self.profiler.cprofile_stats
            }
            
            # Generate report
            report_path = report_generator.generate_optimization_report(
                profiling_data, self.study_name
            )
            
            # Export metrics for further analysis
            metrics_path = os.path.join(
                self.profiling_output_dir,
                f"{self.study_name}_metrics.json"
            )
            self.profiler.export_metrics(metrics_path)
            
            print(f"\nProfiling report generated:")
            print(f"  Report: {report_path}")
            print(f"  Metrics: {metrics_path}")
            
            # Print key insights
            self._print_profiling_insights(study_analysis, parallel_efficiency)
            
        except Exception as e:
            print(f"Error generating profiling report: {e}")
    
    def _print_profiling_insights(self, study_analysis: Dict[str, Any],
                                parallel_efficiency: Dict[str, Any]) -> None:
        """
        Print key profiling insights to console
        
        Args:
            study_analysis: Study analysis results
            parallel_efficiency: Parallel efficiency analysis
        """
        print("\n" + "="*50)
        print("PROFILING INSIGHTS")
        print("="*50)
        
        # Trial statistics
        total_trials = study_analysis.get('n_trials', 0)
        successful_trials = study_analysis.get('n_complete', 0)
        pruned_trials = study_analysis.get('n_pruned', 0)
        
        print(f"Trial Statistics:")
        print(f"  Total trials: {total_trials}")
        print(f"  Successful: {successful_trials} ({successful_trials/total_trials*100:.1f}%)")
        print(f"  Pruned: {pruned_trials} ({pruned_trials/total_trials*100:.1f}%)")
        
        # Performance metrics
        if 'avg_trial_time' in study_analysis:
            print(f"\nPerformance Metrics:")
            print(f"  Average trial time: {study_analysis['avg_trial_time']:.4f}s")
            print(f"  Median trial time: {study_analysis['median_trial_time']:.4f}s")
        
        # Pruning effectiveness
        if 'pruning_efficiency' in study_analysis:
            print(f"\nPruning Analysis:")
            print(f"  Pruning efficiency: {study_analysis['pruning_efficiency']:.1f}%")
            if study_analysis['pruning_efficiency'] > 20:
                print("  ✅ Pruning is effectively saving time")
            else:
                print("  ⚠️ Pruning may need adjustment")
        
        # Parallel efficiency
        if parallel_efficiency.get('n_jobs', 1) > 1:
            print(f"\nParallel Processing:")
            print(f"  Jobs used: {parallel_efficiency.get('n_jobs', 1)}")
            print(f"  Speedup achieved: {parallel_efficiency.get('actual_speedup', 1):.2f}x")
            print(f"  Parallel efficiency: {parallel_efficiency.get('parallel_efficiency', 0):.1f}%")
            
            if parallel_efficiency.get('parallel_efficiency', 0) > 70:
                print("  ✅ Good parallel efficiency")
            elif parallel_efficiency.get('parallel_efficiency', 0) > 40:
                print("  ⚠️ Moderate parallel efficiency")
            else:
                print("  🐌 Low parallel efficiency - consider reducing parallelism")
        
        # Resource usage
        if self.profiler.resource_usage:
            resource_summary = self.profiler.get_resource_summary()
            if resource_summary:
                print(f"\nResource Usage:")
                print(f"  Average CPU: {resource_summary['cpu']['avg']:.1f}%")
                print(f"  Peak CPU: {resource_summary['cpu']['max']:.1f}%")
                print(f"  Average Memory: {resource_summary['memory']['avg_mb']:.1f}MB")
                print(f"  Peak Memory: {resource_summary['memory']['max_mb']:.1f}MB")
        
        # Top bottlenecks
        bottlenecks = self.profiler.get_optimization_bottlenecks(top_n=3)
        if bottlenecks:
            print(f"\nTop Bottlenecks:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"  {i}. {bottleneck['type'].replace('_', ' ').title()}")
                if 'execution_time' in bottleneck:
                    print(f"     Time: {bottleneck['execution_time']:.4f}s")
                if 'trial_number' in bottleneck:
                    print(f"     Trial: {bottleneck['trial_number']}")
        
        print("="*50)


# Convenience function for quick fast optimization
def quick_fast_optimize(strategy_name: str,
                       data_path: str,
                       symbol: str = 'BTCUSDT',
                       n_trials: int = 50,
                       objective_metric: str = 'sharpe_ratio',
                       n_jobs: int = -1) -> Dict[str, Any]:
    """
    Quick fast optimization with default parameters
    
    Args:
        strategy_name: Name of the strategy to optimize
        data_path: Path to the CSV data file
        symbol: Trading symbol
        n_trials: Number of trials
        objective_metric: Metric to optimize
        n_jobs: Number of parallel jobs
        
    Returns:
        Optimization results
    """
    optimizer = FastStrategyOptimizer(
        strategy_name=strategy_name,
        data_path=data_path,
        symbol=symbol
    )
    
    return optimizer.optimize(
        n_trials=n_trials,
        objective_metric=objective_metric,
        n_jobs=n_jobs
    )


def create_composite_objective(weights: Dict[str, float]) -> Callable:
    """
    Create a composite objective function that combines multiple metrics

    Args:
        weights: Dictionary mapping metric names to weights
        
    Returns:
        Composite objective function
    """
    def composite_objective(results: Dict[str, Any]) -> float:
        score = 0.0
        for metric, weight in weights.items():
            if metric in results:
                score += results[metric] * weight
        return score

    return composite_objective