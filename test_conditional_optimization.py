"""
Test script for Conditional Parameter Space optimization
This runs a small optimization to verify that conditional sampling works correctly
"""
import time
from src.optimization.fast_optimizer import FastStrategyOptimizer

def test_conditional_sampling():
    """
    Test conditional parameter sampling with small trial count
    """
    print("="*80)
    print("TESTING CONDITIONAL PARAMETER SPACE OPTIMIZATION")
    print("="*80)

    # Configuration
    strategy_name = 'ported_from_example'
    data_path = r'data\klines\AIAUSDT_1m_2024-11-01_2024-12-01.pkl'  # Adjust path as needed
    symbol = 'AIAUSDT'
    n_trials = 10  # Small number for quick test

    print(f"\nStrategy: {strategy_name}")
    print(f"Symbol: {symbol}")
    print(f"Trials: {n_trials}")
    print(f"Data: {data_path}")
    print("-"*80)

    # Create optimizer with debug enabled to see parameter sampling
    optimizer = FastStrategyOptimizer(
        strategy_name=strategy_name,
        data_path=data_path,
        symbol=symbol,
        enable_debug=True  # This will print which params are sampled
    )

    print("\n⏱️  Starting optimization with CONDITIONAL parameter space...")
    start_time = time.time()

    results = optimizer.optimize(
        n_trials=n_trials,
        objective_metric='sharpe_ratio',
        min_trades=5,  # Relaxed constraint
        max_drawdown_threshold=50.0,
        n_jobs=1  # Single process for easier debugging
    )

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"✅ Optimization completed in {elapsed:.2f}s")
    print(f"   Average time per trial: {elapsed/n_trials:.2f}s")
    print(f"\n📊 Best parameters:")
    for k, v in results['best_params'].items():
        print(f"   {k}: {v}")
    print(f"\n🎯 Best {results['objective_metric']}: {results['best_value']:.4f}")
    print(f"   Composite value: {results['best_value_composite']}")

    # Check trial attributes to verify conditional sampling worked
    print("\n🔍 Verifying conditional sampling...")
    study = optimizer.study
    modes_count = {'Принты и HLdir': 0, 'Только по принтам': 0, 'Только по HLdir': 0}

    for trial in study.trials:
        if trial.state.name == 'COMPLETE':
            mode = trial.params.get('entry_logic_mode', 'unknown')
            modes_count[mode] = modes_count.get(mode, 0) + 1

            # Check if conditional sampling worked correctly
            prints_sampled = trial.user_attrs.get('prints_params_sampled', None)
            hldir_sampled = trial.user_attrs.get('hldir_params_sampled', None)

            if mode == 'Только по HLdir':
                assert prints_sampled == False, f"Trial {trial.number}: prints should NOT be sampled for 'Только по HLdir'"
                assert hldir_sampled == True, f"Trial {trial.number}: hldir should be sampled for 'Только по HLdir'"
            elif mode == 'Только по принтам':
                assert prints_sampled == True, f"Trial {trial.number}: prints should be sampled for 'Только по принтам'"
                assert hldir_sampled == False, f"Trial {trial.number}: hldir should NOT be sampled for 'Только по принтам'"
            elif mode == 'Принты и HLdir':
                assert prints_sampled == True, f"Trial {trial.number}: prints should be sampled for 'Принты и HLdir'"
                assert hldir_sampled == True, f"Trial {trial.number}: hldir should be sampled for 'Принты и HLdir'"

    print(f"   ✅ Conditional sampling verified!")
    print(f"   Entry mode distribution: {modes_count}")
    print("="*80)

    return results

if __name__ == "__main__":
    try:
        results = test_conditional_sampling()
        print("\n✅ TEST PASSED - Conditional parameter space optimization works correctly!")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
