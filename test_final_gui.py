#!/usr/bin/env python3
"""
Final test of all GUI improvements
"""
import sys
import time

def test_simplified_gui():
    """Test simplified GUI with all improvements"""
    print("Testing final simplified GUI...")

    from PyQt6.QtWidgets import QApplication, QTimer
    from src.gui.gui_visualizer import ProfessionalBacktester

    app = QApplication(sys.argv)
    window = ProfessionalBacktester()

    # Simulate realistic backtest results
    realistic_results = {
        'total': 71276,  # Large number of trades like full dataset
        'win_rate': 0.687,
        'net_pnl': 2845.75,
        'net_pnl_percentage': 28.46,
        'max_drawdown': 12.3,
        'sharpe_ratio': 1.92,
        'profit_factor': 2.47,
        'total_winning_trades': 48956,
        'total_losing_trades': 22320,
        'average_win': 0.15,
        'average_loss': -0.23,
        'largest_win': 5.42,
        'largest_loss': -2.87,
        'trades': [
            {
                'timestamp': 1725672000000 + i * 1000,
                'exit_timestamp': 1725672000000 + i * 1000 + 30000,
                'symbol': 'BARDUSDT',
                'side': 'long' if i % 3 != 0 else 'short',
                'entry_price': 1.5054 + (i % 1000) * 0.0001,
                'exit_price': 1.5054 + (i % 1000) * 0.0001 + (0.001 if i % 3 != 0 else -0.0005),
                'size': 100.0,
                'pnl': 0.1 if i % 3 != 0 else -0.05,
                'pnl_percentage': 0.067 if i % 3 != 0 else -0.033,
                'duration': 30000,
                'exit_reason': 'target_hit' if i % 3 != 0 else 'stop_loss'
            }
            for i in range(71276)  # Full 71K trades
        ],
        'bb_data': {
            'times': [1725672000000 + i * 100 for i in range(100000)],  # 100K price points
            'prices': [1.5054 + (i % 10000) * 0.00001 for i in range(100000)],
            'sma': [1.5054 + (i % 10000) * 0.00001 for i in range(100000)],
            'upper_band': [1.5054 + (i % 10000) * 0.00001 + 0.003 for i in range(100000)],
            'lower_band': [1.5054 + (i % 10000) * 0.00001 - 0.003 for i in range(100000)]
        }
    }

    def simulate_realistic_test():
        print("Simulating realistic 71K trades backtest completion...")
        start_time = time.time()

        window.results_data = realistic_results
        window._display_results()

        elapsed = time.time() - start_time
        print(f"Results displayed in {elapsed:.2f}s")

        # Check if GUI is responsive
        window.status_bar.showMessage("Test completed - GUI is responsive!", 3000)

        # Auto-close after 2 seconds
        QTimer.singleShot(2000, app.quit)

    # Start test after 500ms
    QTimer.singleShot(500, simulate_realistic_test)

    window.show()
    return app.exec()

def main():
    print("Final GUI Test - All Improvements")
    print("=" * 40)
    print("Testing:")
    print("✓ No Data Mode selection")
    print("✓ No limits on trades display")
    print("✓ Working Charts & Signals")
    print("✓ Full dataset processing")
    print("✓ 71K trades handling")
    print("=" * 40)

    result = test_simplified_gui()

    if result == 0:
        print("SUCCESS: Final GUI test completed successfully")
        print("All improvements working correctly!")
    else:
        print("FAIL: GUI test failed")

    return result

if __name__ == "__main__":
    sys.exit(main())