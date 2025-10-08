"""
FULLY VECTORIZED Hierarchical Mean Reversion Strategy
Complete numpy/batch processing implementation - cleaned and runnable

Author: HFT System (cleaned)
"""
import numpy as np
import time
from typing import Dict, Any, List
import warnings
from numba import njit

try:
    from sklearn.mixture import GaussianMixture
    import scipy.stats as stats
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    warnings.warn(f"ML dependencies required: {e}")

# Import fast Kalman implementation
from .fast_kalman import fast_kalman_1d

from numpy.lib.stride_tricks import sliding_window_view

from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry


def create_rolling_windows(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Safe rolling windows using sliding_window_view.
    Returns shape (n - window_size + 1, window_size) or empty array if insufficient length.
    """
    arr = np.ascontiguousarray(arr)
    n = len(arr)
    if n < window_size:
        return np.empty((0, window_size), dtype=arr.dtype)
    return sliding_window_view(arr, window_shape=window_size)


@njit(cache=True, fastmath=True)
def _fast_1d_gmm_proba(x, w, mu, sig):
    """x: 1-D array, w/mu/sig: 1-D len=3"""
    n = len(x)
    probs = np.empty((n, 3))
    for i in range(n):
        z0 = (x[i] - mu[0]) / sig[0]
        z1 = (x[i] - mu[1]) / sig[1]
        z2 = (x[i] - mu[2]) / sig[2]
        e0 = np.exp(-0.5 * z0 * z0) / sig[0]
        e1 = np.exp(-0.5 * z1 * z1) / sig[1]
        e2 = np.exp(-0.5 * z2 * z2) / sig[2]
        den = w[0]*e0 + w[1]*e1 + w[2]*e2 + 1e-300
        probs[i, 0] = w[0] * e0 / den
        probs[i, 1] = w[1] * e1 / den
        probs[i, 2] = w[2] * e2 / den
    return probs


def vectorized_ou_half_life(gaps: np.ndarray, window_size: int) -> tuple:
    """
    Vectorized OU half-life. Uses sliding_window_view for windows.
    Returns arrays length = len(gaps) with -1/999 defaults for invalids.
    """
    n = len(gaps)
    if n < window_size:
        return np.full(n, -1.0), np.full(n, 999.0)

    windows = create_rolling_windows(gaps, window_size)  # shape (n_windows, window_size)
    n_windows = windows.shape[0]

    half_lives = np.full(n, -1.0)
    hl_std_errors = np.full(n, 999.0)

    x_all = windows[:, :-1]                       # (n_windows, window_size-1)
    y_all = np.diff(windows, axis=1)              # (n_windows, window_size-1)
    n_points = x_all.shape[1]

    sum_x = np.sum(x_all, axis=1)
    sum_y = np.sum(y_all, axis=1)
    sum_xy = np.sum(x_all * y_all, axis=1)
    sum_x2 = np.sum(x_all * x_all, axis=1)

    denominator = n_points * sum_x2 - sum_x * sum_x
    valid_mask = denominator != 0
    slopes = np.full(n_windows, np.nan)
    slopes[valid_mask] = (n_points * sum_xy[valid_mask] - sum_x[valid_mask] * sum_y[valid_mask]) / denominator[valid_mask]

    mr_mask = (-1 < slopes) & (slopes < 0)
    valid_indices = np.where(valid_mask & mr_mask)[0]

    if len(valid_indices) > 0:
        slopes_valid = slopes[valid_indices]
        theta = -slopes_valid
        hl = np.log(2) / theta

        x_valid = x_all[valid_indices]
        y_valid = y_all[valid_indices]
        x_mean = np.mean(x_valid, axis=1)
        intercept = (sum_y[valid_indices] - slopes_valid * sum_x[valid_indices]) / n_points
        y_pred = slopes_valid[:, None] * x_valid + intercept[:, None]
        residuals = y_valid - y_pred

        if n_points > 2:
            mse = np.sum(residuals**2, axis=1) / (n_points - 2)
            # Оптимизация: var = E[x²] - (E[x])²
            # Убираем лишний временный массив x_mean[:, None]
            n = x_valid.shape[1]
            denom = np.einsum('ij,ij->i', x_valid, x_valid) - n * x_mean**2
            # protect division by zero
            denom[denom == 0] = np.nan
            std_err = np.sqrt(mse) / np.sqrt(denom)
            hl_std_error = std_err / np.abs(slopes_valid)
        else:
            hl_std_error = np.full(len(valid_indices), 999.0)

        output_indices = valid_indices + window_size - 1
        half_lives[output_indices] = hl
        hl_std_errors[output_indices] = hl_std_error

    return half_lives, hl_std_errors



@StrategyRegistry.register('hierarchical_mean_reversion')
class HierarchicalMeanReversionStrategy(BaseStrategy):
    def __init__(self,
                 symbol: str,
                 **kwargs):

        if not ML_AVAILABLE:
            raise ImportError(
                "HierarchicalMeanReversionStrategy requires ML dependencies.\n"
                "Install with: pip install scikit-learn scipy numba"
            )

        # Получаем параметры по умолчанию и обновляем их переданными значениями
        default_params = self.get_default_params()
        default_params.update(kwargs)
        
        # Устанавливаем все параметры как атрибуты
        for param_name, param_value in default_params.items():
            setattr(self, param_name, param_value)
        
        super().__init__(symbol, **default_params)
        
        # GMM параметры для быстрой 1-D аналитики
        self._gmm_w = None
        self._gmm_mu = None
        self._gmm_sig = None

    def _validate_prices(self, prices):
        """
        Валидация цен на наличие NaN/inf значений
        
        Args:
            prices: массив цен для валидации
            
        Raises:
            ValueError: если в ценах есть NaN или inf значения
        """
        if not np.isfinite(prices).all():
            raise ValueError("test_prices contains NaN/inf values")

    def _process_dataset_core(self,
                             times: np.ndarray,
                             prices: np.ndarray,
                             opens: np.ndarray = None,
                             highs: np.ndarray = None,
                             lows: np.ndarray = None,
                             closes: np.ndarray = None,
                             total_bars: int = None) -> Dict[str, Any]:
        """
        Основная логика обработки набора данных, используемая обеими публичными методами.
        
        Args:
            times: массив временных меток
            prices: массив цен закрытия
            opens: массив цен открытия (опционально)
            highs: массив максимальных цен (опционально)
            lows: массив минимальных цен (опционально)
            closes: массив цен закрытия (опционально, может совпадать с prices)
            total_bars: общее количество баров (для отчетности)
            
        Returns:
            Dict[str, Any]: результаты обработки с сделками и метриками
        """
        if total_bars is None:
            total_bars = len(times)
            
        print(f"[ULTRA-FAST VECTORIZED] Processing {total_bars} bars directly with numpy arrays...")

        train_size = max(int(total_bars * 0.3), self.hmm_window_size + 100)
        # ensure at least some test bars
        if train_size >= total_bars:
            train_size = max(total_bars - 2, 1)
        train_prices = prices[:train_size]
        test_times = times[train_size:]
        test_prices = prices[train_size:]
        test_opens = opens[train_size:] if opens is not None else None
        test_highs = highs[train_size:] if highs is not None else None
        test_lows = lows[train_size:] if lows is not None else None
        test_closes = closes[train_size:] if closes is not None else None  # RESTORED: used in indicator_data

        # STEP 1: HMM on diffs (kept original behavior)
        price_changes_train = np.diff(train_prices)
        hmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        if len(price_changes_train) > 0:
            hmm_model.fit(price_changes_train.reshape(-1, 1))
            # Извлекаем параметры GMM для быстрой 1-D аналитики
            self._gmm_w = hmm_model.weights_
            self._gmm_mu = hmm_model.means_.ravel()
            self._gmm_sig = np.sqrt(hmm_model.covariances_.ravel())

        # STEP 2: Fast Kalman filter with Numba JIT compilation
        if len(test_prices) == 0:
            raise ValueError("No test data after train/test split")

        # VALIDATION: Check for NaN/inf in test_prices
        if not np.isfinite(test_prices).all():
            raise ValueError("test_prices contains NaN/inf values")

        # OPTIMIZATION: Use fast Numba-compiled 1D Kalman filter
        initial_state = test_prices[0] if len(test_prices) > 0 else self.initial_kf_mean
        
        # PERFORMANCE: Track execution time
        t0 = time.perf_counter()
        
        fair_values, fair_value_stds = fast_kalman_1d(
            test_prices,
            initial_state,
            self.initial_kf_cov,
            self.process_noise_q,
            self.measurement_noise_r
        )
        
        # PERFORMANCE: Log execution time
        kalman_time = time.perf_counter() - t0
        if hasattr(self, 'debug') and self.debug:
            print(f"Kalman filter execution time: {kalman_time:.6f}s")
        
        # VALIDATION: Check dimensions
        assert fair_values.shape == test_prices.shape, f"fair_values shape {fair_values.shape} != test_prices shape {test_prices.shape}"
        assert fair_value_stds.shape == test_prices.shape, f"fair_value_stds shape {fair_value_stds.shape} != test_prices shape {test_prices.shape}"
        
        gaps = test_prices - fair_values
        z_scores = gaps / fair_value_stds

        # STEP 3: HMM regime detection (kept original per-diff-on-windows predict approach)
        price_changes_test = np.diff(test_prices)
        n = len(price_changes_test)
        # OPTIMIZATION: Use uint8 instead of object for regimes (0=DISABLED, 1=ENABLED)
        regimes = np.zeros(len(test_prices), dtype=np.uint8)   # 0=DISABLED, 1=ENABLED

        if n >= self.hmm_window_size and len(price_changes_train) > 0:
            # OPTIMIZATION: Extract only last diffs instead of predicting on all window elements
            # This reduces complexity from O(n * window_size) to O(n)
            try:
                # Get only the last difference from each window (O(n) instead of O(n*window_size))
                last_diffs = price_changes_test[self.hmm_window_size-1:]
                # Используем быструю 1-D аналитику вместо predict_proba
                regime_probs_last = _fast_1d_gmm_proba(last_diffs, self._gmm_w, self._gmm_mu, self._gmm_sig)
                
                p_trend = regime_probs_last[:, 0]
                p_sideways = regime_probs_last[:, 1]

                trend_mask = p_trend > self.prob_threshold_trend
                sideways_mask = p_sideways > self.prob_threshold_sideways
                
                # Отладочный вывод для режимов
                print(f"[DEBUG] p_trend.min/max: {p_trend.min():.4f}/{p_trend.max():.4f}, threshold: {self.prob_threshold_trend}")
                print(f"[DEBUG] p_sideways.min/max: {p_sideways.min():.4f}/{p_sideways.max():.4f}, threshold: {self.prob_threshold_sideways}")
                print(f"[DEBUG] trend_mask.sum(): {trend_mask.sum()}, sideways_mask.sum(): {sideways_mask.sum()}")
                
                # OPTIMIZATION: Combine masks and assign once
                mask = trend_mask | sideways_mask
                # Исправляем индексацию - mask имеет размер (n - hmm_window_size + 1)
                # а нам нужно присвоить значения regimes[hmm_window_size-1:][mask]
                start_idx = self.hmm_window_size - 1
                end_idx = start_idx + len(mask)
                regimes[start_idx:end_idx][mask] = np.where(sideways_mask[mask], 1, 0)
            except Exception as e:
                # Fall back: leave regimes as default (0=DISABLED)
                print(f"[DEBUG] Exception in regime detection: {e}")
                pass

        # STEP 4: OU half-lives
        half_lives, hl_std_errors = vectorized_ou_half_life(gaps, self.ou_window_size)

        # STEP 5: Signals
        regime_ok = (regimes == 1)   # 1=ENABLED, optimized comparison with uint8
        z_entry_ok = np.abs(z_scores) >= self.s_entry
        hl_ok = (half_lives > 0) & (half_lives >= self.hl_min) & (half_lives < self.hl_max)
        # OPTIMIZATION: half_lives already filtered with > 0, so np.maximum is redundant
        uncertainty_ok = (half_lives > 0) & ((hl_std_errors / half_lives) <= self.relative_uncertainty_threshold)

        entry_mask = regime_ok & z_entry_ok & hl_ok
        
        # Отладочный вывод для проверки масок
        print(f"[DEBUG] entry_mask.sum() = {entry_mask.sum()} / {len(entry_mask)}")
        print(f"[DEBUG] regime_ok.sum() = {regime_ok.sum()}")
        print(f"[DEBUG] z_entry_ok.sum() = {z_entry_ok.sum()}")
        print(f"[DEBUG] hl_ok.sum() = {hl_ok.sum()}")
        
        # Сохраняем отладочную информацию для доступа из оптимизатора
        self._last_debug_info = {
            'entry_mask_sum': entry_mask.sum(),
            'regime_ok_sum': regime_ok.sum(),
            'z_entry_ok_sum': z_entry_ok.sum(),
            'hl_ok_sum': hl_ok.sum()
        }

        entry_long = entry_mask & (z_scores < -self.s_entry)
        entry_short = entry_mask & (z_scores > self.s_entry)
        
        # Отладочная статистика по входам
        print(f"[DEBUG] Long entries: {entry_long.sum()}, Short entries: {entry_short.sum()}")
        if entry_short.sum() > 0:
            short_z_scores = z_scores[entry_short]
            print(f"[DEBUG] Short z_scores - min: {short_z_scores.min():.4f}, max: {short_z_scores.max():.4f}, mean: {short_z_scores.mean():.4f}")

        trades = self._build_trades_vectorized(
            test_times, test_prices, z_scores,
            entry_long, entry_short, half_lives, regimes
        )

        metrics = self._calculate_performance_metrics(trades)

        indicator_data = {
            'times': test_times,
            'prices': test_prices,
            'fair_values': fair_values,
            'z_scores': z_scores,
            'half_lives': half_lives,
            'regimes': regimes
        }

        if all(arr is not None for arr in [test_opens, test_highs, test_lows, test_closes]):
            indicator_data.update({
                'open': test_opens,
                'high': test_highs,
                'low': test_lows,
                'close': test_closes
            })

        return {
            'trades': trades,
            'symbol': self.symbol,
            'total': len(trades),
            'total_bars': total_bars,
            'train_bars': train_size,
            'indicator_data': indicator_data,
            **metrics
        }

    def vectorized_process_dataset(self, df) -> Dict[str, Any]:
        """
        Обрабатывает DataFrame или словарь, извлекая данные и вызывая основную логику обработки.
        
        Args:
            df: DataFrame или словарь с данными OHLCV
            
        Returns:
            Dict[str, Any]: результаты обработки с сделками и метриками
        """
        if hasattr(df, 'data'):
            # NumpyKlinesData
            times = df['time']
            closes = df['close']
            opens = df['open'] if 'open' in df.columns else None
            highs = df['high'] if 'high' in df.columns else None
            lows = df['low'] if 'low' in df.columns else None
            total_bars = len(df)
        elif hasattr(df, 'columns'):
            # pandas DataFrame
            required_cols = ['time', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            times = df['time'].values
            closes = df['close'].values
            opens = df['open'].values if 'open' in df.columns else None
            highs = df['high'].values if 'high' in df.columns else None
            lows = df['low'].values if 'low' in df.columns else None
            total_bars = len(df)
        else:
            # Словарь с numpy массивами
            required_cols = ['time', 'close']
            missing_cols = [col for col in required_cols if col not in df]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            times = df['time']
            closes = df['close']
            opens = df.get('open')
            highs = df.get('high')
            lows = df.get('low')
            total_bars = len(times)

        return self._process_dataset_core(
            times=times,
            prices=closes,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            total_bars=total_bars
        )

    def _build_trades_vectorized(self, times, prices, z_scores, entry_long, entry_short, half_lives, regimes):
        trades = []
        n = len(prices)

        entry_points = np.where(entry_long | entry_short)[0]
        if len(entry_points) == 0:
            return trades

        position = None
        entry_idx = None
        entry_time = None
        entry_price_val = None
        entry_hl = None
        max_floating_profit = 0.0
        max_floating_loss = 0.0

        # helper to compute minutes safely
        def duration_minutes(t_exit, t_entry):
            try:
                diff = t_exit - t_entry
                # numpy timedelta64
                if isinstance(diff, np.timedelta64):
                    secs = diff.astype('timedelta64[s]').astype(float)
                    return secs / 60.0
                return float(diff) / 60.0
            except Exception:
                return 0.0

        # iterate only over indices where entries or potential exits exist
        for i in range(n):
            if position is None:
                if entry_long[i]:
                    position = 'long'
                    entry_idx = i
                    entry_time = times[i]
                    entry_price_val = prices[i]
                    entry_hl = half_lives[i]
                    max_floating_profit = 0.0
                    max_floating_loss = 0.0
                elif entry_short[i]:
                    position = 'short'
                    entry_idx = i
                    entry_time = times[i]
                    entry_price_val = prices[i]
                    entry_hl = half_lives[i]
                    max_floating_profit = 0.0
                    max_floating_loss = 0.0
            else:
                # Отслеживаем максимальную плавающую прибыль/убыток во время сделки
                position_size = getattr(self, 'position_size_dollars', 50.0)
                if position == 'long':
                    current_floating_pnl = (prices[i] - entry_price_val) * (position_size / entry_price_val)
                    max_floating_profit = max(max_floating_profit, current_floating_pnl)
                    max_floating_loss = min(max_floating_loss, current_floating_pnl)
                else:  # short
                    current_floating_pnl = (entry_price_val - prices[i]) * (position_size / entry_price_val)
                    max_floating_profit = max(max_floating_profit, current_floating_pnl)
                    max_floating_loss = min(max_floating_loss, current_floating_pnl)
                exit_reason = None

                if (position == 'long' and z_scores[i] > 0) or (position == 'short' and z_scores[i] < 0):
                    exit_reason = 'take_profit'
                elif (position == 'long' and z_scores[i] < -self.z_stop) or (position == 'short' and z_scores[i] > self.z_stop):
                    exit_reason = 'stop_loss_hypothesis_failed'
                elif entry_time is not None and (times[i] - entry_time) > self.timeout_multiplier * entry_hl:
                    exit_reason = 'stop_loss_timeout'
                elif regimes[i] != 1:  # 1=ENABLED
                    exit_reason = 'stop_loss_regime_change'
                elif entry_hl > 0 and (half_lives[i] / entry_hl) > self.relative_uncertainty_threshold:
                    exit_reason = 'stop_loss_uncertainty_spike'
                
                # Отладочная информация для шортов
                if position == 'short' and exit_reason:
                    print(f"[SHORT DEBUG] Exit: {exit_reason}, z_score: {z_scores[i]:.4f}, entry_z: {z_scores[entry_idx]:.4f}, "
                          f"entry_price: {entry_price_val:.2f}, exit_price: {prices[i]:.2f}, "
                          f"price_change: {(prices[i] - entry_price_val):.2f}, pnl: {current_floating_pnl:.2f}")

                if exit_reason:
                    exit_price = prices[i]
                    # Используем размер позиции из параметров стратегии
                    position_size = getattr(self, 'position_size_dollars', 50.0)
                    # commission_pct is fraction (e.g. 0.0005) -> multiply by size and two sides
                    commission = position_size * self.commission_pct * 2

                    if position == 'long':
                        # Для лонгов: (цена выхода - цена входа) * размер позиции / цена входа - комиссия
                        pnl = (exit_price - entry_price_val) * (position_size / entry_price_val) - commission
                        pnl_pct = (exit_price - entry_price_val) / entry_price_val * 100
                    else:
                        # Для шортов: (цена входа - цена выхода) * размер позиции / цена входа - комиссия
                        pnl = (entry_price_val - exit_price) * (position_size / entry_price_val) - commission
                        pnl_pct = (entry_price_val - exit_price) / entry_price_val * 100

                    trade = {
                        'timestamp': entry_time,
                        'exit_timestamp': times[i],
                        'symbol': self.symbol,
                        'side': position,
                        'entry_price': entry_price_val,
                        'exit_price': exit_price,
                        'size': position_size,
                        'pnl': pnl,
                        'pnl_percentage': pnl_pct,
                        'duration': duration_minutes(times[i], entry_time),
                        'exit_reason': exit_reason,
                        'capital_before': 0,
                        'capital_after': 0,
                        'max_floating_profit': max_floating_profit,
                        'max_floating_loss': max_floating_loss
                    }
                    trades.append(trade)
                    position = None

        if trades:
            capital = self.initial_capital
            pnls = np.array([trade['pnl'] for trade in trades], dtype=float)
            cumulative_pnls = np.cumsum(pnls)
            for i, trade in enumerate(trades):
                trade['capital_before'] = capital + (cumulative_pnls[i - 1] if i > 0 else 0.0)
                trade['capital_after'] = capital + cumulative_pnls[i]

        return trades

    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        metrics = BaseStrategy.calculate_performance_metrics(trades, self.initial_capital)
        metrics.pop('total', None)
        return metrics


    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            'initial_kf_mean': 100.0,
            'initial_kf_cov': 1.0,
            'measurement_noise_r': 5.0,
            'process_noise_q': 0.1,
            'hmm_window_size': 30,
            'prob_threshold_trend': 0.6,
            'prob_threshold_sideways': 0.5,
            'prob_threshold_dead': 0.85,
            'sigma_dead_threshold': 1.0,
            'ou_window_size': 35,
            'hl_min': 1.0,
            'hl_max': 120.0,
            'relative_uncertainty_threshold': 0.8,
            's_entry': 0.05,
            'z_stop': 4.0,
            'timeout_multiplier': 3.0,
            'initial_capital': 100.0,
            'position_size_dollars': 50.0,
            'commission_pct': 0.0005
        }

    @classmethod
    def get_param_space(cls) -> Dict[str, tuple]:
        return {
            'measurement_noise_r': ('float', 0.1, 2.0),
            'process_noise_q': ('float', 0.01, 0.2),
            'hmm_window_size': ('int', 20, 100),
            'prob_threshold_trend': ('float', 0.5, 0.95),
            'prob_threshold_sideways': ('float', 0.3, 0.9),
            'prob_threshold_dead': ('float', 0.7, 0.99),
            'ou_window_size': ('int', 30, 200),
            'hl_min': ('float', 0.5, 5.0),
            'hl_max': ('float', 50.0, 200.0),
            'relative_uncertainty_threshold': ('float', 0.2, 1.0),
            's_entry': ('float', 0.01, 0.2),
            'z_stop': ('float', 2.0, 6.0),
            'timeout_multiplier': ('float', 1.0, 5.0),
            'initial_capital': ('float', 10.0, 10000.0),
            'position_size_dollars': ('float', 10.0, 1000.0),
            'commission_pct': ('float', 0.0001, 0.001)
        }
