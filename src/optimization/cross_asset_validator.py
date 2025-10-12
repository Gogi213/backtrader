"""
Cross-Asset Validator

Берет положительные trials из multi-dataset оптимизации
и тестирует их на ВСЕХ датасетах чтобы найти robust параметры.

Workflow:
1. Загружает результаты оптимизации (все positive trials)
2. Для каждого trial тестирует на ВСЕХ датасетах
3. Создает матрицу: trial × dataset -> metrics
4. Находит trials которые работают на многих датасетах

Author: Claude Code
Date: 2025-10-12
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..data.klines_handler import VectorizedKlinesHandler
from ..strategies.base_strategy import StrategyRegistry


@dataclass
class PositiveTrial:
    """Положительный trial из оптимизации"""
    source_symbol: str
    trial_number: int
    params: Dict[str, Any]
    source_sharpe: float
    source_pnl: float
    source_trades: int


@dataclass
class CrossAssetTestResult:
    """Результат теста trial на одном датасете"""
    trial_id: str  # "BTC_trial5"
    test_symbol: str
    params: Dict[str, Any]
    sharpe: float
    pnl: float
    trades: int
    win_rate: float
    max_dd: float


class CrossAssetValidator:
    """
    Валидатор trials на кросс-ассет основе
    Supports cluster-based validation for finding robustness within similar markets
    """

    def __init__(self, datasets_dir: str = "upload/klines/datasets", cluster_mapping: Dict[str, str] = None):
        self.datasets_dir = datasets_dir
        self.datasets = self._load_all_datasets()
        self.positive_trials: List[PositiveTrial] = []
        self.test_results: List[CrossAssetTestResult] = []
        self.cluster_mapping = cluster_mapping or {}  # {symbol: cluster_name}

    def _load_all_datasets(self) -> Dict[str, Any]:
        """Загружает все датасеты из папки"""
        datasets = {}
        data_path = Path(self.datasets_dir)

        if not data_path.exists():
            print(f"[WARNING]  Datasets directory not found: {data_path}")
            return datasets

        handler = VectorizedKlinesHandler()

        for file_path in data_path.glob("*.parquet"):
            symbol = file_path.name.split('-')[0] if '-' in file_path.name else file_path.stem

            try:
                data = handler.load_klines(str(file_path))
                datasets[symbol] = {
                    'data': data,
                    'path': str(file_path),
                    'bars': len(data)
                }
                print(f"  [OK] Loaded {symbol}: {len(data):,} bars")
            except Exception as e:
                print(f"  [FAIL] Failed to load {symbol}: {e}")

        print(f"\n[SUCCESS] Loaded {len(datasets)} datasets\n")
        return datasets

    def load_positive_trials(
        self,
        results_dir: str = "optimization_results",
        results_file: str = None,
        min_pnl: float = 0.0,
        min_sharpe: float = 0.0,
        min_trades: int = 10
    ) -> List[PositiveTrial]:
        """
        Загружает positive trials из сохраненных результатов оптимизации

        Args:
            results_dir: Папка с результатами оптимизации
            results_file: Конкретный файл для загрузки (если None, берет последний)
            min_pnl: Минимальный PnL
            min_sharpe: Минимальный Sharpe
            min_trades: Минимальное количество трейдов
        """
        positive_trials = []
        results_path = Path(results_dir)

        if not results_path.exists():
            print(f"[WARNING]  Results directory not found: {results_path}")
            print(f"[INFO] Run multi-dataset optimization first via TUI!")
            return positive_trials

        # Если конкретный файл не указан, берем самый свежий
        if results_file is None:
            json_files = list(results_path.glob("multi_dataset_*.json"))
            if not json_files:
                print(f"[WARNING]  No optimization results found in {results_path}")
                print(f"[INFO] Run multi-dataset optimization first via TUI!")
                return positive_trials

            # Берем самый свежий по времени модификации
            results_file = max(json_files, key=lambda p: p.stat().st_mtime)
            print(f"Loading latest results file: {results_file.name}")
        else:
            results_file = results_path / results_file

        # Загружаем JSON
        try:
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [FAIL] Corrupted JSON file: {results_file.name}")
            print(f"  [INFO] JSON error: {e}")
            print(f"  [INFO] Delete this file and re-run optimization")
            return positive_trials

        try:
            positive_trials_data = results.get('positive_trials', [])

            if not positive_trials_data:
                print(f"[WARNING]  No positive trials found in {results_file.name}")
                return positive_trials

            print(f"Loading positive trials from {results_file.name}...")
            print(f"Found {len(positive_trials_data)} trials in file")

            for trial in positive_trials_data:
                pnl = trial.get('pnl', 0)
                sharpe = trial.get('sharpe', 0)
                trades = trial.get('trades', 0)
                dataset = trial.get('dataset', 'UNKNOWN')

                # Извлекаем символ из имени датасета
                symbol = dataset.split('-')[0] if '-' in dataset else dataset.split('.')[0]

                if pnl >= min_pnl and sharpe >= min_sharpe and trades >= min_trades:
                    positive_trial = PositiveTrial(
                        source_symbol=symbol,
                        trial_number=trial.get('trial', 0),
                        params=trial.get('params', {}),
                        source_sharpe=sharpe,
                        source_pnl=pnl,
                        source_trades=trades
                    )
                    positive_trials.append(positive_trial)

        except Exception as e:
            print(f"  [FAIL] Error loading {results_file}: {e}")
            import traceback
            traceback.print_exc()
            return positive_trials

        self.positive_trials = positive_trials
        print(f"[SUCCESS] Found {len(positive_trials)} positive trials (after filters)\n")
        return positive_trials

    def test_trial_on_datasets(
        self,
        trial: PositiveTrial,
        strategy_name: str,
        cluster_mode: bool = False
    ) -> List[CrossAssetTestResult]:
        """
        Тестирует trial на датасетах

        Args:
            trial: Trial для тестирования
            strategy_name: Имя стратегии
            cluster_mode: Если True, тестирует только на датасетах из того же кластера

        Returns:
            Список результатов
        """
        results = []
        strategy_class = StrategyRegistry.get_strategy(strategy_name)
        trial_id = f"{trial.source_symbol}_trial{trial.trial_number}"

        # Determine which datasets to test on
        if cluster_mode and self.cluster_mapping:
            source_cluster = self.cluster_mapping.get(trial.source_symbol, 'unknown')
            test_symbols = [s for s, c in self.cluster_mapping.items() if c == source_cluster]
        else:
            test_symbols = list(self.datasets.keys())

        for symbol in test_symbols:
            if symbol not in self.datasets:
                continue

            dataset_info = self.datasets[symbol]
            try:
                strategy = strategy_class(symbol=symbol, **trial.params)
                backtest_results = strategy.vectorized_process_dataset(dataset_info['data'])

                result = CrossAssetTestResult(
                    trial_id=trial_id,
                    test_symbol=symbol,
                    params=trial.params,
                    sharpe=backtest_results.get('sharpe_ratio', 0),
                    pnl=backtest_results.get('net_pnl', 0),
                    trades=backtest_results.get('total', 0),
                    win_rate=backtest_results.get('win_rate', 0),
                    max_dd=abs(backtest_results.get('max_drawdown', 0))
                )
                results.append(result)

            except Exception as e:
                print(f"  [FAIL] Error testing {trial_id} on {symbol}: {e}")
                continue

        return results

    def validate_all_trials(
        self,
        strategy_name: str,
        trials: List[PositiveTrial] = None,
        progress_callback: callable = None,
        cluster_mode: bool = False
    ) -> pd.DataFrame:
        """
        Валидирует positive trials на датасетах

        Args:
            strategy_name: Имя стратегии
            trials: Список trials (если None, использует self.positive_trials)
            progress_callback: Функция для прогресса (trial_idx, total_trials, trial_id)
            cluster_mode: Если True, тестирует только внутри кластеров

        Returns:
            DataFrame с результатами
        """
        if trials is None:
            trials = self.positive_trials

        if not trials:
            print("[WARNING]  No trials to validate!")
            return pd.DataFrame()

        mode_str = "cluster-based" if cluster_mode else "all datasets"
        print(f"Validating {len(trials)} trials ({mode_str})...")
        print()

        all_results = []

        for idx, trial in enumerate(trials, 1):
            trial_id = f"{trial.source_symbol}_trial{trial.trial_number}"
            print(f"[{idx}/{len(trials)}] Testing {trial_id}...")

            if progress_callback:
                progress_callback(idx, len(trials), trial_id)

            trial_results = self.test_trial_on_datasets(trial, strategy_name, cluster_mode)
            all_results.extend(trial_results)

            # Краткая статистика по этому trial
            positive_count = sum(1 for r in trial_results if r.pnl > 0)
            avg_pnl = np.mean([r.pnl for r in trial_results])
            print(f"  -> Positive on {positive_count}/{len(trial_results)} datasets, Avg PnL: {avg_pnl:.2f}\n")

        self.test_results = all_results

        # Создаем DataFrame
        df_data = []
        for result in all_results:
            row = {
                'trial_id': result.trial_id,
                'test_symbol': result.test_symbol,
                'sharpe': result.sharpe,
                'pnl': result.pnl,
                'trades': result.trades,
                'win_rate': result.win_rate,
                'max_dd': result.max_dd
            }
            # Добавляем параметры
            for param_name, value in result.params.items():
                row[f'param_{param_name}'] = value

            df_data.append(row)

        df = pd.DataFrame(df_data)
        return df

    def find_robust_trials(
        self,
        min_positive_ratio: float = 0.5,
        min_avg_pnl: float = 10.0,
        group_by_cluster: bool = False
    ) -> pd.DataFrame:
        """
        Находит robust trials

        Args:
            min_positive_ratio: Минимальная доля положительных датасетов (0.5 = 50%)
            min_avg_pnl: Минимальный средний PnL
            group_by_cluster: Группировать результаты по кластерам

        Returns:
            DataFrame с robust trials
        """
        if not self.test_results:
            print("[WARNING]  No test results! Run validate_all_trials() first")
            return pd.DataFrame()

        # Группируем по trial_id
        trials_summary = {}

        for result in self.test_results:
            if result.trial_id not in trials_summary:
                trials_summary[result.trial_id] = {
                    'pnls': [],
                    'sharpes': [],
                    'positive_count': 0,
                    'total_count': 0,
                    'params': result.params,
                    'cluster': self.cluster_mapping.get(result.test_symbol, 'unknown') if group_by_cluster else None
                }

            trials_summary[result.trial_id]['pnls'].append(result.pnl)
            trials_summary[result.trial_id]['sharpes'].append(result.sharpe)
            trials_summary[result.trial_id]['total_count'] += 1

            if result.pnl > 0:
                trials_summary[result.trial_id]['positive_count'] += 1

        # Фильтруем robust
        robust_trials = []

        for trial_id, summary in trials_summary.items():
            positive_ratio = summary['positive_count'] / summary['total_count']
            avg_pnl = np.mean(summary['pnls'])
            avg_sharpe = np.mean(summary['sharpes'])

            if positive_ratio >= min_positive_ratio and avg_pnl >= min_avg_pnl:
                trial_data = {
                    'trial_id': trial_id,
                    'positive_ratio': positive_ratio,
                    'positive_count': summary['positive_count'],
                    'total_datasets': summary['total_count'],
                    'avg_pnl': avg_pnl,
                    'avg_sharpe': avg_sharpe,
                    'params': summary['params']
                }

                if group_by_cluster:
                    trial_data['cluster'] = summary['cluster']

                robust_trials.append(trial_data)

        df = pd.DataFrame(robust_trials)

        if not df.empty:
            df = df.sort_values('positive_ratio', ascending=False)

        return df

    def export_results(
        self,
        all_results_df: pd.DataFrame,
        robust_trials_df: pd.DataFrame,
        output_dir: str = "wfo_results"
    ):
        """Экспортирует результаты в CSV"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Все результаты
        all_results_path = Path(output_dir) / f"cross_asset_validation_{timestamp}.csv"
        all_results_df.to_csv(all_results_path, index=False)
        print(f"[SUCCESS] All results saved: {all_results_path}")

        # Robust trials
        robust_path = Path(output_dir) / f"robust_trials_{timestamp}.csv"
        robust_trials_df.to_csv(robust_path, index=False)
        print(f"[SUCCESS] Robust trials saved: {robust_path}")


# CLI для быстрого запуска
if __name__ == "__main__":
    print("="*80)
    print("CROSS-ASSET VALIDATOR")
    print("="*80)
    print()

    validator = CrossAssetValidator(datasets_dir="upload/klines/datasets")

    # Загружаем positive trials
    positive_trials = validator.load_positive_trials(
        cache_dir="optimization_cache",
        min_pnl=0.0,
        min_sharpe=0.0,
        min_trades=10
    )

    if not positive_trials:
        print("[ERROR] No positive trials found!")
        exit(1)

    # Валидируем на всех датасетах
    strategy_name = "ported_from_example"  # TODO: получить из config
    all_results_df = validator.validate_all_trials(strategy_name, positive_trials)

    # Находим robust
    robust_df = validator.find_robust_trials(
        min_positive_ratio=0.5,  # Работает минимум на 50% датасетов
        min_avg_pnl=10.0
    )

    print()
    print("="*80)
    print("ROBUST TRIALS (work on >= 50% of datasets)")
    print("="*80)
    print(robust_df[['trial_id', 'positive_ratio', 'positive_count', 'total_datasets', 'avg_pnl', 'avg_sharpe']])

    # Экспорт
    validator.export_results(all_results_df, robust_df)
