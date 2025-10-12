"""
Optimization Results Database (SQLite)

Author: Claude Code
Date: 2025-10-12
"""
import sqlite3
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class OptimizationResultsDB:
    """SQLite database for optimization results"""

    def __init__(self, db_path: str = "optimization/db/results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    datasets_count INTEGER NOT NULL,
                    total_trials INTEGER NOT NULL,
                    positive_trials_count INTEGER NOT NULL,
                    best_params TEXT,
                    final_backtest TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS positive_trials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    trial_number INTEGER NOT NULL,
                    dataset TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    sharpe REAL NOT NULL,
                    trades INTEGER NOT NULL,
                    winrate REAL,
                    winrate_long REAL,
                    winrate_short REAL,
                    profit_factor REAL,
                    max_dd REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    consecutive_stops INTEGER,
                    params TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES optimization_runs(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_id
                ON positive_trials(run_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pnl
                ON positive_trials(pnl DESC)
            """)

    def _clean_value(self, value: Any) -> Any:
        """Clean NaN/Infinity for storage"""
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        return value

    def save_optimization_run(self, results: Dict[str, Any]) -> int:
        """Save optimization run and return run_id"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert optimization run
            cursor.execute("""
                INSERT INTO optimization_runs (
                    strategy_name, timestamp, datasets_count,
                    total_trials, positive_trials_count,
                    best_params, final_backtest
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                results.get('strategy_name', 'unknown'),
                datetime.now().isoformat(),
                results.get('datasets_count', 1),
                results.get('final_backtest', {}).get('total', 0),
                len(results.get('positive_trials', [])),
                json.dumps(results.get('best_params', {})),
                json.dumps({
                    k: self._clean_value(v)
                    for k, v in results.get('final_backtest', {}).items()
                    if k not in ('trades', 'indicator_data', 'times', 'ohlcv_data')
                })
            ))

            run_id = cursor.lastrowid

            # Insert positive trials
            for trial in results.get('positive_trials', []):
                cursor.execute("""
                    INSERT INTO positive_trials (
                        run_id, trial_number, dataset, pnl, sharpe, trades,
                        winrate, winrate_long, winrate_short, profit_factor,
                        max_dd, avg_win, avg_loss, consecutive_stops, params
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    trial.get('trial', 0),
                    trial.get('dataset', 'unknown'),
                    self._clean_value(trial.get('pnl', 0)),
                    self._clean_value(trial.get('sharpe', 0)),
                    trial.get('trades', 0),
                    self._clean_value(trial.get('winrate', 0)),
                    self._clean_value(trial.get('winrate_long', 0)),
                    self._clean_value(trial.get('winrate_short', 0)),
                    self._clean_value(trial.get('pf', 0)),
                    self._clean_value(trial.get('max_dd', 0)),
                    self._clean_value(trial.get('avg_win', 0)),
                    self._clean_value(trial.get('avg_loss', 0)),
                    trial.get('consecutive_stops', 0),
                    json.dumps(trial.get('params', {}))
                ))

            conn.commit()
            return run_id

    def get_latest_run(self, strategy_name: Optional[str] = None) -> Optional[int]:
        """Get latest run_id for strategy"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if strategy_name:
                cursor.execute("""
                    SELECT id FROM optimization_runs
                    WHERE strategy_name = ?
                    ORDER BY created_at DESC LIMIT 1
                """, (strategy_name,))
            else:
                cursor.execute("""
                    SELECT id FROM optimization_runs
                    ORDER BY created_at DESC LIMIT 1
                """)

            row = cursor.fetchone()
            return row[0] if row else None

    def load_positive_trials(
        self,
        run_id: Optional[int] = None,
        min_pnl: float = 0.0,
        min_sharpe: float = 0.0,
        min_trades: int = 10,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Load positive trials from database"""
        if run_id is None:
            run_id = self.get_latest_run()
            if run_id is None:
                return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT * FROM positive_trials
                WHERE run_id = ?
                AND pnl >= ?
                AND sharpe >= ?
                AND trades >= ?
                ORDER BY pnl DESC
            """

            params = [run_id, min_pnl, min_sharpe, min_trades]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)

            trials = []
            for row in cursor.fetchall():
                trials.append({
                    'trial': row['trial_number'],
                    'dataset': row['dataset'],
                    'pnl': row['pnl'],
                    'sharpe': row['sharpe'],
                    'trades': row['trades'],
                    'winrate': row['winrate'],
                    'winrate_long': row['winrate_long'],
                    'winrate_short': row['winrate_short'],
                    'pf': row['profit_factor'],
                    'max_dd': row['max_dd'],
                    'avg_win': row['avg_win'],
                    'avg_loss': row['avg_loss'],
                    'consecutive_stops': row['consecutive_stops'],
                    'params': json.loads(row['params'])
                })

            return trials

    def get_run_info(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get run metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM optimization_runs WHERE id = ?
            """, (run_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return {
                'id': row['id'],
                'strategy_name': row['strategy_name'],
                'timestamp': row['timestamp'],
                'datasets_count': row['datasets_count'],
                'total_trials': row['total_trials'],
                'positive_trials_count': row['positive_trials_count'],
                'best_params': json.loads(row['best_params']),
                'final_backtest': json.loads(row['final_backtest']),
                'created_at': row['created_at']
            }
