"""
FULLY VECTORIZED Hierarchical Mean Reversion Strategy
Complete numpy/batch processing implementation - cleaned and runnable

Author: HFT System (cleaned)
"""
import numpy as np
from typing import Dict, Any, List
import warnings
from numba import njit

try:
    from pykalman import KalmanFilter
    from sklearn.mixture import GaussianMixture
    import scipy.stats as stats
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    warnings.warn(f"ML dependencies required: {e}")

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


@njit
def _create_rolling_windows_numba(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Numba-compatible rolling windows (manual copy).
    """
    n = len(arr)
    if n < window_size:
        return np.empty((0, window_size), dtype=arr.dtype)

    n_windows = n - window_size + 1
    out = np.empty((n_windows, window_size), dtype=arr.dtype)
    for i in range(n_windows):
        for j in range(window_size):
            out[i, j] = arr[i + j]
    return out


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
            denom = np.sum((x_valid - x_mean[:, None])**2, axis=1)
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


@njit
def _vectorized_ou_half_life_numba(gaps: np.ndarray, window_size: int) -> tuple:
    """
    Numba fallback: manual rolling-regression per window.
    """
    n = len(gaps)
    if n < window_size:
        return np.full(n, -1.0), np.full(n, 999.0)

    half_lives = np.full(n, -1.0)
    hl_std_errors = np.full(n, 999.0)

    n_windows = n - window_size + 1
    for i in range(n_windows):
        # build window
        window = np.empty(window_size, dtype=gaps.dtype)
        for j in range(window_size):
            window[j] = gaps[i + j]
        x = window[:-1]
        m = len(x)
        if m <= 1:
            continue
        y = np.empty(m, dtype=gaps.dtype)
        for j in range(m):
            y[j] = window[j + 1] - window[j]

        n_points = m
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        for j in range(m):
            xv = x[j]
            yv = y[j]
            sum_x += xv
            sum_y += yv
            sum_xy += xv * yv
            sum_x2 += xv * xv

        denominator = n_points * sum_x2 - sum_x * sum_x
        if denominator == 0:
            continue
        slope = (n_points * sum_xy - sum_x * sum_y) / denominator
        if not (-1 < slope < 0):
            continue

        theta = -slope
        half_life = np.log(2.0) / theta

        # residuals and std error
        intercept = (sum_y - slope * sum_x) / n_points
        sse = 0.0
        xmean = sum_x / n_points
        denom_var = 0.0
        for j in range(m):
            ypred = slope * x[j] + intercept
            resid = y[j] - ypred
            sse += resid * resid
            denom_var += (x[j] - xmean) * (x[j] - xmean)
        if n_points > 2 and denom_var != 0:
            mse = sse / (n_points - 2)
            std_err = np.sqrt(mse) / np.sqrt(denom_var)
            hl_std_error = std_err / abs(slope)
        else:
            hl_std_error = 999.0

        idx = i + window_size - 1
        half_lives[idx] = half_life
        hl_std_errors[idx] = hl_std_error

    return half_lives, hl_std_errors


@StrategyRegistry.register('hierarchical_mean_reversion')
class HierarchicalMeanReversionStrategy(BaseStrategy):
    def __init__(self,
                 symbol: str,
                 initial_kf_mean: float = 100.0,
                 initial_kf_cov: float = 1.0,
                 measurement_noise_r: float = 5.0,
                 process_noise_q: float = 0.1,
                 hmm_window_size: int = 30,
                 prob_threshold_trend: float = 0.6,
                 prob_threshold_sideways: float = 0.5,
                 prob_threshold_dead: float = 0.85,
                 sigma_dead_threshold: float = 1.0,
                 ou_window_size: int = 50,
                 hl_min: float = 1.0,
                 hl_max: float = 120.0,
                 relative_uncertainty_threshold: float = 0.8,
                 uncertainty_threshold: float = 0.8,
                 s_entry: float = 0.05,
                 z_stop: float = 4.0,
                 timeout_multiplier: float = 3.0,
                 initial_capital: float = 10000.0,
                 commission_pct: float = 0.0005):

        if not ML_AVAILABLE:
            raise ImportError(
                "HierarchicalMeanReversionStrategy requires ML dependencies.\n"
                "Install with: pip install pykalman scikit-learn scipy"
            )

        params = locals().copy()
        params.pop('self')
        params.pop('symbol')
        super().__init__(symbol, **params)

        self.initial_kf_mean = initial_kf_mean
        self.initial_kf_cov = initial_kf_cov
        self.measurement_noise_r = measurement_noise_r
        self.process_noise_q = process_noise_q
        self.hmm_window_size = hmm_window_size
        self.prob_threshold_trend = prob_threshold_trend
        self.prob_threshold_sideways = prob_threshold_sideways
        self.prob_threshold_dead = prob_threshold_dead
        self.sigma_dead_threshold = sigma_dead_threshold
        self.ou_window_size = ou_window_size
        self.hl_min = hl_min
        self.hl_max = hl_max
        self.relative_uncertainty_threshold = relative_uncertainty_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.s_entry = s_entry
        self.z_stop = z_stop
        self.timeout_multiplier = timeout_multiplier
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

    def vectorized_process_dataset(self, df) -> Dict[str, Any]:
        if hasattr(df, 'data'):
            times = df['time']
            closes = df['close']
            opens = df['open'] if 'open' in df.columns else None
            highs = df['high'] if 'high' in df.columns else None
            lows = df['low'] if 'low' in df.columns else None
            total_bars = len(df)
        else:
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

        print(f"[ULTRA-FAST VECTORIZED] Processing {total_bars} bars directly with numpy arrays...")

        train_size = max(int(total_bars * 0.3), self.hmm_window_size + 100)
        # ensure at least some test bars
        if train_size >= total_bars:
            train_size = max(total_bars - 2, 1)
        train_prices = closes[:train_size]
        test_times = times[train_size:]
        test_prices = closes[train_size:]
        test_opens = opens[train_size:] if opens is not None else None
        test_highs = highs[train_size:] if highs is not None else None
        test_lows = lows[train_size:] if lows is not None else None
        test_closes = closes[train_size:]

        # STEP 1: HMM on diffs (kept original behavior)
        price_changes_train = np.diff(train_prices)
        hmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        if len(price_changes_train) > 0:
            hmm_model.fit(price_changes_train.reshape(-1, 1))

        # STEP 2: Kalman filter
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=test_prices[0] if len(test_prices) > 0 else self.initial_kf_mean,
            initial_state_covariance=self.initial_kf_cov,
            observation_covariance=self.measurement_noise_r,
            transition_covariance=self.process_noise_q
        )

        if len(test_prices) == 0:
            raise ValueError("No test data after train/test split")

        state_means, state_covs = kf.filter(test_prices)
        fair_values = np.asarray(state_means)[:, 0]
        fair_value_stds = np.sqrt(np.asarray(state_covs)[:, 0, 0])

        gaps = test_prices - fair_values
        # safe z-score with division protection
        z_scores = np.zeros_like(gaps, dtype=float)
        nonzero = fair_value_stds != 0
        z_scores[nonzero] = gaps[nonzero] / fair_value_stds[nonzero]

        # STEP 3: HMM regime detection (kept original per-diff-on-windows predict approach)
        price_changes_test = np.diff(test_prices)
        n = len(price_changes_test)
        regimes = np.full(len(test_prices), 'TRADE_DISABLED', dtype=object)

        if n >= self.hmm_window_size and len(price_changes_train) > 0:
            windows = create_rolling_windows(price_changes_test, self.hmm_window_size)
            if windows.size > 0:
                # original code predicted per-element; keep the same reshape/predict_proba behavior
                try:
                    regime_probs = hmm_model.predict_proba(windows.reshape(-1, 1))
                    # regime_probs shape (n_windows*window_size, n_components)
                    # restore to (n_windows, window_size, n_components) if possible
                    n_windows = windows.shape[0]
                    n_comp = regime_probs.shape[1]
                    regime_probs = regime_probs.reshape(n_windows, self.hmm_window_size, n_comp)
                    regime_probs_last = regime_probs[:, -1, :]  # last diff in each window

                    p_trend = regime_probs_last[:, 0]
                    p_sideways = regime_probs_last[:, 1]
                    p_dead = regime_probs_last[:, 2]

                    trend_mask = p_trend > self.prob_threshold_trend
                    sideways_mask = p_sideways > self.prob_threshold_sideways
                    # indices align to test_prices positions
                    indices = np.arange(len(regime_probs_last)) + self.hmm_window_size
                    regimes[indices[trend_mask]] = 'TRADE_DISABLED'
                    regimes[indices[sideways_mask]] = 'TRADE_ENABLED'
                except Exception:
                    # Fall back: leave regimes as default
                    pass

        # STEP 4: OU half-lives
        half_lives, hl_std_errors = vectorized_ou_half_life(gaps, self.ou_window_size)

        # STEP 5: Signals
        regime_ok = (regimes != 'DEAD')   # kept original semantics (no 'DEAD' assigned elsewhere)
        z_entry_ok = np.abs(z_scores) >= self.s_entry
        hl_ok = (half_lives > 0) & (half_lives >= self.hl_min) & (half_lives < self.hl_max)
        uncertainty_ok = (half_lives > 0) & ((hl_std_errors / np.maximum(half_lives, 1)) <= self.relative_uncertainty_threshold)

        entry_mask = regime_ok & z_entry_ok & hl_ok

        entry_long = entry_mask & (z_scores < -self.s_entry)
        entry_short = entry_mask & (z_scores > self.s_entry)

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
                if position == 'long':
                    current_floating_pnl = (prices[i] - entry_price_val) * (100.0 / entry_price_val)
                    max_floating_profit = max(max_floating_profit, current_floating_pnl)
                    max_floating_loss = min(max_floating_loss, current_floating_pnl)
                else:  # short
                    current_floating_pnl = (entry_price_val - prices[i]) * (100.0 / entry_price_val)
                    max_floating_profit = max(max_floating_profit, current_floating_pnl)
                    max_floating_loss = min(max_floating_loss, current_floating_pnl)
                exit_reason = None

                if (position == 'long' and z_scores[i] > 0) or (position == 'short' and z_scores[i] < 0):
                    exit_reason = 'take_profit'
                elif (position == 'long' and z_scores[i] < -self.z_stop) or (position == 'short' and z_scores[i] > self.z_stop):
                    exit_reason = 'stop_loss_hypothesis_failed'
                elif entry_time is not None and (times[i] - entry_time) > self.timeout_multiplier * entry_hl:
                    exit_reason = 'stop_loss_timeout'
                elif regimes[i] != 'TRADE_ENABLED':
                    exit_reason = 'stop_loss_regime_change'
                elif entry_hl > 0 and (half_lives[i] / entry_hl) > self.uncertainty_threshold:
                    exit_reason = 'stop_loss_uncertainty_spike'

                if exit_reason:
                    exit_price = prices[i]
                    position_size = 100.0
                    # commission_pct is fraction (e.g. 0.0005) -> multiply by size and two sides
                    commission = position_size * self.commission_pct * 2 / 100

                    if position == 'long':
                        pnl = (exit_price - entry_price_val) * (position_size / entry_price_val) - commission
                        pnl_pct = (exit_price - entry_price_val) / entry_price_val * 100
                    else:
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

    def turbo_process_dataset(self,
                             times: np.ndarray,
                             prices: np.ndarray,
                             opens: np.ndarray = None,
                             highs: np.ndarray = None,
                             lows: np.ndarray = None,
                             closes: np.ndarray = None) -> Dict[str, Any]:

        if times is None or len(times) == 0:
            raise ValueError("Empty or None times array")
        if closes is None or len(closes) == 0:
            raise ValueError("Empty or None closes array")

        train_size = max(int(len(times) * 0.3), self.hmm_window_size + 100)
        if train_size >= len(times):
            train_size = max(len(times) - 2, 1)

        train_prices = closes[:train_size]
        test_times = times[train_size:]
        test_prices = closes[train_size:]
        test_opens = opens[train_size:] if opens is not None else None
        test_highs = highs[train_size:] if highs is not None else None
        test_lows = lows[train_size:] if lows is not None else None
        test_closes = closes[train_size:]

        price_changes_train = np.diff(train_prices)
        hmm_model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        if len(price_changes_train) > 0:
            hmm_model.fit(price_changes_train.reshape(-1, 1))

        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=test_prices[0] if len(test_prices) > 0 else self.initial_kf_mean,
            initial_state_covariance=self.initial_kf_cov,
            observation_covariance=self.measurement_noise_r,
            transition_covariance=self.process_noise_q
        )

        state_means, state_covs = kf.filter(test_prices)
        fair_values = np.asarray(state_means)[:, 0]
        fair_value_stds = np.sqrt(np.asarray(state_covs)[:, 0, 0])

        gaps = test_prices - fair_values
        z_scores = np.zeros_like(gaps, dtype=float)
        nonzero = fair_value_stds != 0
        z_scores[nonzero] = gaps[nonzero] / fair_value_stds[nonzero]

        price_changes_test = np.diff(test_prices)
        n = len(price_changes_test)
        regimes = np.full(len(test_prices), 'TRADE_DISABLED', dtype=object)

        if n >= self.hmm_window_size and len(price_changes_train) > 0:
            windows = create_rolling_windows(price_changes_test, self.hmm_window_size)
            if windows.size > 0:
                try:
                    regime_probs = hmm_model.predict_proba(windows.reshape(-1, 1))
                    n_windows = windows.shape[0]
                    n_comp = regime_probs.shape[1]
                    regime_probs = regime_probs.reshape(n_windows, self.hmm_window_size, n_comp)
                    regime_probs_last = regime_probs[:, -1, :]
                    p_trend = regime_probs_last[:, 0]
                    p_sideways = regime_probs_last[:, 1]
                    indices = np.arange(len(regime_probs_last)) + self.hmm_window_size
                    trend_mask = p_trend > self.prob_threshold_trend
                    sideways_mask = p_sideways > self.prob_threshold_sideways
                    regimes[indices[trend_mask]] = 'TRADE_DISABLED'
                    regimes[indices[sideways_mask]] = 'TRADE_ENABLED'
                except Exception:
                    pass

        half_lives, hl_std_errors = vectorized_ou_half_life(gaps, self.ou_window_size)

        regime_ok = (regimes != 'DEAD')
        z_entry_ok = np.abs(z_scores) >= self.s_entry
        hl_ok = (half_lives > 0) & (half_lives >= self.hl_min) & (half_lives < self.hl_max)
        uncertainty_ok = (half_lives > 0) & ((hl_std_errors / np.maximum(half_lives, 1)) <= self.relative_uncertainty_threshold)

        entry_mask = regime_ok & z_entry_ok & hl_ok

        entry_long = entry_mask & (z_scores < -self.s_entry)
        entry_short = entry_mask & (z_scores > self.s_entry)

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

        if all(arr is not None for arr in [opens, highs, lows, closes]):
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
            'total_bars': len(test_times),
            'train_bars': train_size,
            'indicator_data': indicator_data,
            **metrics
        }

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
            'ou_window_size': 50,
            'hl_min': 1.0,
            'hl_max': 120.0,
            'relative_uncertainty_threshold': 0.8,
            'uncertainty_threshold': 0.8,
            's_entry': 0.05,
            'z_stop': 4.0,
            'timeout_multiplier': 3.0,
            'initial_capital': 10000.0,
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
            'uncertainty_threshold': ('float', 0.2, 1.0),
            's_entry': ('float', 0.01, 0.2),
            'z_stop': ('float', 2.0, 6.0),
            'timeout_multiplier': ('float', 1.0, 5.0),
            'initial_capital': ('float', 1000.0, 100000.0),
            'commission_pct': ('float', 0.0001, 0.001)
        }
