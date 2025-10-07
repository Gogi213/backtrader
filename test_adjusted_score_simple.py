"""
Simple test for adjusted_score metric with existing backtester

This test checks the correctness of the adjusted_score metric
in real backtesting conditions.
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.append('src')

def test_adjusted_score_with_backtester():
    """Test adjusted_score with real backtester"""
    print("Testing adjusted_score with real backtester...")
    
    try:
        # Import necessary modules
        from src.data.backtest_engine import run_vectorized_klines_backtest
        from src.strategies.strategy_registry import StrategyRegistry
        from src.optimization.metrics import calculate_adjusted_score_from_results
        
        # Check available strategies
        strategies = StrategyRegistry.list_strategies()
        print(f"Available strategies: {strategies}")
        
        if not strategies:
            print("Error: No available strategies")
            return False
        
        # Use first available strategy
        strategy_name = strategies[0]
        print(f"Using strategy: {strategy_name}")
        
        # Check data availability
        data_dir = "upload/klines"
        if not os.path.exists(data_dir):
            print(f"Error: Directory {data_dir} does not exist")
            return False
        
        # Find CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"Error: No CSV files in {data_dir}")
            return False
        
        # Use first found file
        data_file = csv_files[0]
        data_path = os.path.join(data_dir, data_file)
        symbol = data_file.split('-')[0] if '-' in data_file else data_file.split('.')[0]
        
        print(f"Using data: {data_file}")
        print(f"Symbol: {symbol}")
        
        # Get default strategy parameters
        strategy_class = StrategyRegistry.get(strategy_name)
        default_params = strategy_class.get_default_params()
        print(f"Default parameters: {default_params}")
        
        # Remove initial_capital and commission_pct from default_params to avoid conflicts
        params_for_backtest = default_params.copy()
        params_for_backtest.pop('initial_capital', None)
        params_for_backtest.pop('commission_pct', None)
        
        # Run backtest
        print("Running backtest...")
        results = run_vectorized_klines_backtest(
            csv_path=data_path,
            symbol=symbol,
            strategy_name=strategy_name,
            strategy_params=params_for_backtest,
            initial_capital=10000.0,
            commission_pct=0.05
        )
        
        if 'error' in results:
            print(f"Backtest error: {results['error']}")
            return False
        
        # Check results
        trades = results.get('trades', [])
        total_trades = results.get('total', 0)
        
        print(f"Backtest results:")
        print(f"  Total trades: {total_trades}")
        print(f"  Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"  Net P&L: ${results.get('net_pnl', 0):.2f}")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
        
        # Check adjusted_score in results
        if 'adjusted_score' in results:
            print(f"  Adjusted Score: {results.get('adjusted_score', 0):.4f}")
        else:
            print("  Adjusted Score: not calculated")
        
        # Calculate adjusted_score manually
        print("\nCalculating adjusted_score...")
        adjusted_score = calculate_adjusted_score_from_results(results)
        print(f"  Calculated Adjusted Score: {adjusted_score:.4f}")
        
        # Validate correctness
        if total_trades < 30:
            expected_score = -np.inf
            if adjusted_score == expected_score:
                print("[OK] Adjusted Score correctly returns -inf for insufficient trades")
            else:
                print(f"[ERROR] Expected {expected_score}, got {adjusted_score}")
                return False
        else:
            if adjusted_score > -np.inf:
                print("[OK] Adjusted Score successfully calculated")
            else:
                print("[ERROR] Adjusted Score should not be -inf for sufficient trades")
                return False
        
        print("\n[OK] Test passed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during test execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adjusted_score_directly():
    """Test adjusted_score function directly with test data"""
    print("\nTesting adjusted_score with test data...")
    
    try:
        from src.optimization.metrics import adjusted_score
        
        # Test 1: Insufficient trades
        print("Test 1: Insufficient trades (< 30)")
        small_trades = [
            {'pnl': 100, 'duration': 60},
            {'pnl': -50, 'duration': 30},
            {'pnl': 150, 'duration': 90}
        ]
        score = adjusted_score(small_trades)
        if score == -np.inf:
            print("[OK] Correctly returns -inf for insufficient trades")
        else:
            print(f"[ERROR] Expected -inf, got {score}")
            return False
        
        # Test 2: Sufficient trades, all profitable
        print("\nTest 2: 30 profitable trades")
        profitable_trades = [
            {'pnl': 100, 'duration': 60} for _ in range(30)
        ]
        score = adjusted_score(profitable_trades)
        if score > 0:
            print(f"[OK] Positive score for profitable trades: {score:.4f}")
        else:
            print(f"[ERROR] Expected positive score, got {score}")
            return False
        
        # Test 3: Mixed trades
        print("\nTest 3: 30 mixed trades")
        mixed_trades = []
        np.random.seed(42)  # For reproducibility
        for i in range(30):
            pnl = np.random.normal(10, 50)  # Mean profit 10, std 50
            duration = np.random.randint(30, 120)  # 30-120 minutes
            mixed_trades.append({'pnl': pnl, 'duration': duration})
        
        score = adjusted_score(mixed_trades)
        print(f"[OK] Score for mixed trades: {score:.4f}")
        
        print("\n[OK] All direct tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during direct testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ADJUSTED_SCORE METRIC")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_adjusted_score_directly()
    test2_passed = test_adjusted_score_with_backtester()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Direct testing: {'[OK] SUCCESS' if test1_passed else '[ERROR] FAILED'}")
    print(f"Testing with backtester: {'[OK] SUCCESS' if test2_passed else '[ERROR] FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("Adjusted_score metric is ready for use in optimization")
    else:
        print("\n[WARNING] SOME TESTS FAILED")
        print("Need to fix errors before use")