"""
Dataset Cluster Analyzer

Groups datasets by volatility to find robustness within similar market conditions.

Author: Claude Code
Date: 2025-10-12
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from ..data.klines_handler import VectorizedKlinesHandler


@dataclass
class DatasetMetrics:
    """Metrics for a single dataset"""
    symbol: str
    volatility: float
    avg_return: float
    bars_count: int


class DatasetClusterAnalyzer:
    """Cluster datasets by market characteristics"""

    def __init__(self, datasets_dir: str = "upload/klines/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.metrics: List[DatasetMetrics] = []
        self.clusters: Dict[str, List[str]] = {}
        self.symbol_to_cluster: Dict[str, str] = {}

    def analyze_datasets(self) -> List[DatasetMetrics]:
        """Analyze all datasets and compute metrics"""
        handler = VectorizedKlinesHandler()
        metrics = []

        if not self.datasets_dir.exists():
            print(f"[ERROR] Directory not found: {self.datasets_dir}")
            return metrics

        for file_path in self.datasets_dir.glob("*.parquet"):
            symbol = file_path.name.split('-')[0] if '-' in file_path.name else file_path.stem

            try:
                data = handler.load_klines(str(file_path))

                # Calculate returns
                close_prices = data['close'].values
                returns = np.diff(close_prices) / close_prices[:-1]

                # Metrics
                volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized %
                avg_return = np.mean(returns) * 100

                metrics.append(DatasetMetrics(
                    symbol=symbol,
                    volatility=volatility,
                    avg_return=avg_return,
                    bars_count=len(data)
                ))

            except Exception as e:
                print(f"[FAIL] Error analyzing {symbol}: {e}")
                continue

        self.metrics = sorted(metrics, key=lambda x: x.volatility)
        return self.metrics

    def create_clusters(self, n_clusters: int = 3) -> Dict[str, List[str]]:
        """Create clusters based on volatility quantiles"""
        if not self.metrics:
            print("[ERROR] No metrics available. Run analyze_datasets() first.")
            return {}

        volatilities = [m.volatility for m in self.metrics]

        if n_clusters == 3:
            q33 = np.percentile(volatilities, 33)
            q66 = np.percentile(volatilities, 66)

            clusters = {
                'low_volatility': [],
                'medium_volatility': [],
                'high_volatility': []
            }

            for metric in self.metrics:
                if metric.volatility <= q33:
                    cluster_name = 'low_volatility'
                elif metric.volatility <= q66:
                    cluster_name = 'medium_volatility'
                else:
                    cluster_name = 'high_volatility'

                clusters[cluster_name].append(metric.symbol)
                self.symbol_to_cluster[metric.symbol] = cluster_name

        else:
            # Simple equal-size clustering
            cluster_size = len(self.metrics) // n_clusters
            clusters = {}

            for i in range(n_clusters):
                cluster_name = f'cluster_{i+1}'
                clusters[cluster_name] = []

                start_idx = i * cluster_size
                end_idx = start_idx + cluster_size if i < n_clusters - 1 else len(self.metrics)

                for metric in self.metrics[start_idx:end_idx]:
                    clusters[cluster_name].append(metric.symbol)
                    self.symbol_to_cluster[metric.symbol] = cluster_name

        self.clusters = clusters
        return clusters

    def get_cluster_for_symbol(self, symbol: str) -> str:
        """Get cluster name for a symbol"""
        return self.symbol_to_cluster.get(symbol, 'unknown')

    def get_symbols_in_cluster(self, cluster_name: str) -> List[str]:
        """Get all symbols in a cluster"""
        return self.clusters.get(cluster_name, [])

    def print_clusters(self):
        """Print cluster information"""
        print("\n" + "="*80)
        print("DATASET CLUSTERS (by volatility)")
        print("="*80)

        for cluster_name, symbols in self.clusters.items():
            # Get volatility range for this cluster
            vols = [m.volatility for m in self.metrics if m.symbol in symbols]
            min_vol = min(vols) if vols else 0
            max_vol = max(vols) if vols else 0

            print(f"\n{cluster_name.upper()}: {len(symbols)} datasets")
            print(f"Volatility range: {min_vol:.1f}% - {max_vol:.1f}%")
            print(f"Symbols: {', '.join(symbols)}")

    def save_clusters(self, output_path: str = "optimization/clusters.json"):
        """Save cluster mapping to JSON"""
        import json
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'clusters': self.clusters,
            'symbol_to_cluster': self.symbol_to_cluster,
            'metrics': [
                {
                    'symbol': m.symbol,
                    'volatility': m.volatility,
                    'avg_return': m.avg_return,
                    'bars': m.bars_count
                }
                for m in self.metrics
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n[SUCCESS] Clusters saved to {output_file}")

    @staticmethod
    def load_clusters(input_path: str = "optimization/clusters.json") -> Dict[str, Any]:
        """Load cluster mapping from JSON"""
        import json

        input_file = Path(input_path)
        if not input_file.exists():
            return {}

        with open(input_file, 'r') as f:
            return json.load(f)
