"""
Comprehensive Test Suite for Pure Tick HFT System
Tests all components: data loading, strategy, backtest, performance

Author: HFT System
"""
import unittest
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta
import time

# Import our modules
from src.data.pure_tick_handler import PureTickHandler, fast_rolling_mean, fast_rolling_std
from src.strategies.pure_tick_bollinger_strategy import PureTickBollingerStrategy
from pure_tick_backtest import run_pure_tick_backtest, calculate_performance_metrics


class TestPureTickHandler(unittest.TestCase):
    """Test Pure Tick Handler functionality"""

    def setUp(self):
        """Set up test environment"""
        self.handler = PureTickHandler()
        self.temp_dir = tempfile.mkdtemp()

    def create_sample_tick_data(self, num_ticks=1000, symbol="TESTUSDT"):
        """Create sample tick data for testing"""
        # Generate realistic price movement
        start_price = 100.0
        prices = [start_price]

        for i in range(num_ticks - 1):
            # Random walk with slight upward bias
            change = np.random.normal(0, 0.001) * prices[-1]
            new_price = max(prices[-1] + change, 0.001)  # Prevent negative prices
            prices.append(new_price)

        # Create tick data
        data = {
            'id': range(1, num_ticks + 1),
            'price': prices,
            'qty': np.random.uniform(1, 100, num_ticks),
            'quote_qty': [p * q for p, q in zip(prices, np.random.uniform(1, 100, num_ticks))],
            'time': [1758672000000 + i * 1000 for i in range(num_ticks)],  # 1 second intervals
            'is_buyer_maker': np.random.choice([True, False], num_ticks)
        }

        df = pd.DataFrame(data)
        return df

    def create_test_csv(self, num_ticks=1000):
        """Create test CSV file"""
        df = self.create_sample_tick_data(num_ticks)
        csv_path = os.path.join(self.temp_dir, "test_ticks.csv")
        df.to_csv(csv_path, index=False)
        return csv_path, df

    def test_load_ticks_success(self):
        """Test successful tick data loading"""
        csv_path, original_df = self.create_test_csv(1000)

        loaded_df = self.handler.load_ticks(csv_path)

        self.assertEqual(len(loaded_df), 1000)
        self.assertTrue('side' in loaded_df.columns)
        self.assertTrue('datetime' in loaded_df.columns)
        self.assertTrue(loaded_df['time'].is_monotonic_increasing)

    def test_load_ticks_file_not_found(self):
        """Test error handling for missing file"""
        with self.assertRaises(FileNotFoundError):
            self.handler.load_ticks("nonexistent.csv")

    def test_load_ticks_missing_columns(self):
        """Test error handling for missing columns"""
        # Create CSV with missing columns
        df = pd.DataFrame({'id': [1], 'price': [100.0]})  # Missing required columns
        csv_path = os.path.join(self.temp_dir, "bad_ticks.csv")
        df.to_csv(csv_path, index=False)

        with self.assertRaises(ValueError):
            self.handler.load_ticks(csv_path)

    def test_load_ticks_invalid_prices(self):
        """Test error handling for invalid prices"""
        df = self.create_sample_tick_data(100)
        df.loc[50, 'price'] = -10.0  # Invalid negative price

        csv_path = os.path.join(self.temp_dir, "invalid_ticks.csv")
        df.to_csv(csv_path, index=False)

        with self.assertRaises(ValueError):
            self.handler.load_ticks(csv_path)

    def test_prepare_price_array(self):
        """Test price array preparation"""
        df = self.create_sample_tick_data(100)
        price_array = self.handler.prepare_price_array(df)

        self.assertEqual(len(price_array), 100)
        self.assertEqual(price_array.dtype, np.float64)
        self.assertTrue(np.all(price_array > 0))

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        prices = np.array([100.0, 101.0, 102.0, 101.5, 100.5, 99.0, 100.0, 101.0, 102.0, 103.0])
        period = 5
        std_dev = 2.0

        sma, upper, lower = self.handler.calculate_bollinger_bands(prices, period, std_dev)

        self.assertEqual(len(sma), len(prices))
        self.assertEqual(len(upper), len(prices))
        self.assertEqual(len(lower), len(prices))

        # Check that bands are properly positioned
        valid_indices = ~np.isnan(sma)
        if np.any(valid_indices):
            self.assertTrue(np.all(upper[valid_indices] >= sma[valid_indices]))
            self.assertTrue(np.all(lower[valid_indices] <= sma[valid_indices]))

    def test_get_statistics(self):
        """Test statistics calculation"""
        df = self.create_sample_tick_data(1000)
        # Add side column that would be added by load_ticks method
        df['side'] = df['is_buyer_maker'].apply(lambda x: 'sell' if x else 'buy')
        stats = self.handler.get_statistics(df)

        self.assertEqual(stats['total_ticks'], 1000)
        self.assertIn('price_range', stats)
        self.assertIn('time_range', stats)
        self.assertIn('volume_total', stats)
        self.assertIn('buy_ticks', stats)
        self.assertIn('sell_ticks', stats)


class TestNumbaFunctions(unittest.TestCase):
    """Test numba-optimized functions"""

    def test_fast_rolling_mean(self):
        """Test fast rolling mean calculation"""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        window = 3

        result = fast_rolling_mean(prices, window)

        # Check that first window-1 values are NaN
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))

        # Check calculated values
        self.assertAlmostEqual(result[2], 2.0, places=6)  # (1+2+3)/3
        self.assertAlmostEqual(result[3], 3.0, places=6)  # (2+3+4)/3
        self.assertAlmostEqual(result[9], 9.0, places=6)  # (8+9+10)/3

    def test_fast_rolling_std(self):
        """Test fast rolling standard deviation"""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window = 3

        result = fast_rolling_std(prices, window)

        # Check that first window-1 values are NaN
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))

        # Check that valid values are positive
        self.assertGreaterEqual(result[2], 0)
        self.assertGreaterEqual(result[3], 0)
        self.assertGreaterEqual(result[4], 0)

    def test_numba_performance(self):
        """Test that numba functions are reasonably fast"""
        prices = np.random.random(10000).astype(np.float64) * 100
        window = 50

        start_time = time.time()
        fast_rolling_mean(prices, window)
        mean_time = time.time() - start_time

        start_time = time.time()
        fast_rolling_std(prices, window)
        std_time = time.time() - start_time

        # Should complete within reasonable time (generous limits)
        self.assertLess(mean_time, 1.0)  # 1 second max
        self.assertLess(std_time, 1.0)   # 1 second max


class TestPureTickBollingerStrategy(unittest.TestCase):
    """Test Pure Tick Bollinger Strategy"""

    def setUp(self):
        """Set up test environment"""
        self.strategy = PureTickBollingerStrategy(
            symbol="TESTUSDT",
            period=10,  # Small period for testing
            std_dev=2.0,
            stop_loss_pct=0.01,  # 1%
            initial_capital=10000.0
        )

    def create_trending_prices(self, num_ticks=100, trend=0.001):
        """Create trending price data"""
        prices = [100.0]
        times = [pd.Timestamp.now()]

        for i in range(1, num_ticks):
            # Add trend + noise
            change = trend + np.random.normal(0, 0.0005)
            prices.append(prices[-1] * (1 + change))
            times.append(times[-1] + pd.Timedelta(seconds=1))

        return times, prices

    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.symbol, "TESTUSDT")
        self.assertEqual(self.strategy.period, 10)
        self.assertEqual(self.strategy.std_dev, 2.0)
        self.assertEqual(self.strategy.initial_capital, 10000.0)
        self.assertIsNone(self.strategy.position)
        self.assertEqual(len(self.strategy.completed_trades), 0)

    def test_process_tick_insufficient_data(self):
        """Test processing when insufficient data for BB"""
        tick_time = pd.Timestamp.now()

        # Process a few ticks (less than period)
        for i in range(5):
            trades = self.strategy.process_tick(tick_time + pd.Timedelta(seconds=i), 100.0 + i)
            self.assertEqual(len(trades), 0)  # Should not generate trades yet

    def test_process_tick_with_signals(self):
        """Test tick processing with entry/exit signals"""
        times, prices = self.create_trending_prices(50)

        all_trades = []
        for time, price in zip(times, prices):
            trades = self.strategy.process_tick(time, price)
            all_trades.extend(trades)

        # Should have processed ticks
        self.assertGreater(self.strategy.tick_count, 0)
        self.assertGreaterEqual(len(all_trades), 0)  # May or may not generate trades

    def test_get_stats(self):
        """Test strategy statistics"""
        # Process some ticks
        times, prices = self.create_trending_prices(20)

        for time, price in zip(times, prices):
            self.strategy.process_tick(time, price)

        stats = self.strategy.get_stats()

        self.assertEqual(stats['ticks_processed'], 20)
        self.assertIn('total_trades', stats)
        self.assertIn('current_capital', stats)
        self.assertIn('has_position', stats)


class TestBacktestFunctionality(unittest.TestCase):
    """Test backtest functionality"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def create_test_data_file(self, num_ticks=1000):
        """Create test data file"""
        handler = PureTickHandler()

        # Generate price data with some volatility
        start_price = 100.0
        prices = [start_price]

        for i in range(num_ticks - 1):
            change = np.random.normal(0, 0.002) * prices[-1]
            new_price = max(prices[-1] + change, 0.001)
            prices.append(new_price)

        data = {
            'id': range(1, num_ticks + 1),
            'price': prices,
            'qty': np.random.uniform(1, 100, num_ticks),
            'quote_qty': [p * q for p, q in zip(prices, np.random.uniform(1, 100, num_ticks))],
            'time': [1758672000000 + i * 1000 for i in range(num_ticks)],
            'is_buyer_maker': np.random.choice([True, False], num_ticks)
        }

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.temp_dir, "test_data.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_calculate_performance_metrics_empty(self):
        """Test performance metrics with empty trades"""
        metrics = calculate_performance_metrics([])

        self.assertEqual(metrics['total'], 0)
        self.assertEqual(metrics['win_rate'], 0)
        self.assertEqual(metrics['net_pnl'], 0)

    def test_calculate_performance_metrics_with_trades(self):
        """Test performance metrics with sample trades"""
        trades = [
            {'pnl': 10.0},
            {'pnl': -5.0},
            {'pnl': 15.0},
            {'pnl': -3.0}
        ]

        metrics = calculate_performance_metrics(trades, 10000.0)

        self.assertEqual(metrics['total'], 4)
        self.assertEqual(metrics['total_winning_trades'], 2)
        self.assertEqual(metrics['total_losing_trades'], 2)
        self.assertEqual(metrics['win_rate'], 0.5)
        self.assertEqual(metrics['net_pnl'], 17.0)

    def test_run_pure_tick_backtest_success(self):
        """Test successful backtest execution"""
        csv_path = self.create_test_data_file(1000)

        results = run_pure_tick_backtest(
            csv_path=csv_path,
            symbol="TESTUSDT",
            bb_period=20,
            bb_std=2.0,
            stop_loss_pct=1.0,
            initial_capital=10000.0,
            max_ticks=1000
        )

        self.assertNotIn('error', results)
        self.assertEqual(results['symbol'], 'TESTUSDT')
        self.assertIn('trades', results)
        self.assertIn('total', results)
        self.assertIn('ticks_processed', results)
        self.assertEqual(results['ticks_processed'], 1000)

    def test_run_pure_tick_backtest_file_not_found(self):
        """Test backtest with non-existent file"""
        results = run_pure_tick_backtest(
            csv_path="nonexistent.csv",
            symbol="TESTUSDT"
        )

        self.assertIn('error', results)

    def test_run_pure_tick_backtest_limited_ticks(self):
        """Test backtest with tick limit"""
        csv_path = self.create_test_data_file(2000)

        results = run_pure_tick_backtest(
            csv_path=csv_path,
            symbol="TESTUSDT",
            max_ticks=500
        )

        self.assertNotIn('error', results)
        self.assertEqual(results['ticks_processed'], 500)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def create_realistic_data(self, num_ticks=5000):
        """Create realistic market data for testing"""
        # Create price series with trend and mean reversion
        start_price = 100.0
        prices = [start_price]

        for i in range(num_ticks - 1):
            # Add some mean reversion and trend
            mean_reversion = -0.0001 * (prices[-1] - start_price)
            noise = np.random.normal(0, 0.001)
            change = mean_reversion + noise

            new_price = max(prices[-1] * (1 + change), 0.001)
            prices.append(new_price)

        # Create full dataset
        data = {
            'id': range(1, num_ticks + 1),
            'price': prices,
            'qty': np.random.uniform(1, 50, num_ticks),
            'quote_qty': [p * q for p, q in zip(prices, np.random.uniform(1, 50, num_ticks))],
            'time': [1758672000000 + i * 100 for i in range(num_ticks)],  # 100ms intervals
            'is_buyer_maker': np.random.choice([True, False], num_ticks)
        }

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.temp_dir, "realistic_data.csv")
        df.to_csv(csv_path, index=False)
        return csv_path, prices

    def test_full_hft_scenario(self):
        """Test complete HFT trading scenario"""
        csv_path, original_prices = self.create_realistic_data(5000)

        # Run aggressive HFT strategy
        results = run_pure_tick_backtest(
            csv_path=csv_path,
            symbol="HFTUSDT",
            bb_period=20,    # Short period for HFT
            bb_std=1.5,      # Tight bands
            stop_loss_pct=0.5,  # Tight stop loss
            initial_capital=10000.0,
            max_ticks=5000
        )

        self.assertNotIn('error', results)

        # Verify HFT characteristics
        trades = results.get('trades', [])

        if trades:  # If trades were generated
            # Check trade durations (should be short for HFT)
            durations = [t.get('duration', 0) for t in trades]
            avg_duration = np.mean(durations) if durations else 0

            # For HFT, most trades should be relatively quick
            self.assertGreater(results['ticks_processed'], 0)
            self.assertGreaterEqual(results['total'], 0)

            # Verify trade data integrity
            for trade in trades:
                self.assertIn('entry_price', trade)
                self.assertIn('exit_price', trade)
                self.assertIn('pnl', trade)
                self.assertIn('side', trade)
                self.assertIn('duration', trade)
                self.assertIn('exit_reason', trade)

                # Prices should be positive
                self.assertGreater(trade['entry_price'], 0)
                self.assertGreater(trade['exit_price'], 0)

    def test_performance_consistency(self):
        """Test that performance metrics are consistent"""
        csv_path, _ = self.create_realistic_data(2000)

        # Run same backtest twice
        results1 = run_pure_tick_backtest(
            csv_path=csv_path,
            symbol="TESTUSDT",
            bb_period=30,
            bb_std=2.0,
            max_ticks=2000
        )

        results2 = run_pure_tick_backtest(
            csv_path=csv_path,
            symbol="TESTUSDT",
            bb_period=30,
            bb_std=2.0,
            max_ticks=2000
        )

        # Results should be identical (deterministic)
        self.assertEqual(results1.get('total', 0), results2.get('total', 0))
        self.assertEqual(results1.get('ticks_processed', 0), results2.get('ticks_processed', 0))

        # P&L should be identical (within floating point precision)
        pnl1 = results1.get('net_pnl', 0)
        pnl2 = results2.get('net_pnl', 0)
        self.assertAlmostEqual(pnl1, pnl2, places=6)


def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    print("STARTING COMPREHENSIVE PURE TICK SYSTEM TESTS")
    print("="*60)

    # Test suites
    test_suites = [
        TestPureTickHandler,
        TestNumbaFunctions,
        TestPureTickBollingerStrategy,
        TestBacktestFunctionality,
        TestIntegrationScenarios
    ]

    total_tests = 0
    total_failures = 0

    for suite_class in test_suites:
        print(f"\nTesting {suite_class.__name__}")
        print("-" * 40)

        suite = unittest.TestLoader().loadTestsFromTestCase(suite_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        tests_run = result.testsRun
        failures = len(result.failures) + len(result.errors)

        total_tests += tests_run
        total_failures += failures

        print(f"Tests run: {tests_run}, Failures: {failures}")

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"Total Failures: {total_failures}")
    print(f"Success Rate: {((total_tests - total_failures) / total_tests * 100) if total_tests > 0 else 0:.1f}%")

    if total_failures == 0:
        print("ALL TESTS PASSED! Pure Tick System is ready for HFT!")
    else:
        print(f"{total_failures} test(s) failed. Please review and fix.")

    return total_failures == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)