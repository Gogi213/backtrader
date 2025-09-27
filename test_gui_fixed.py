#!/usr/bin/env python3
"""
Test fixed GUI - simulated backtest completion without matplotlib blocking
"""
import sys
import os
from datetime import datetime

def test_gui_with_simulated_backtest():
    """Test GUI with simulated backtest to verify no freezing"""
    print("Testing fixed GUI with simulated backtest completion...")

    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer
        from src.gui.gui_visualizer import ProfessionalBacktester

        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        print("Creating GUI window...")
        window = ProfessionalBacktester()

        # Simulate backtest completion with fake results
        fake_results = {
            'total': 100,
            'win_rate': 0.65,
            'net_pnl': 1250.50,
            'net_pnl_percentage': 12.51,
            'max_drawdown': 8.5,
            'sharpe_ratio': 1.85,
            'profit_factor': 2.1,
            'total_winning_trades': 65,
            'total_losing_trades': 35,
            'average_win': 35.20,
            'average_loss': -18.75,
            'largest_win': 125.50,
            'largest_loss': -45.25,
            'trades': [
                {
                    'timestamp': 1725672000000,
                    'exit_timestamp': 1725672060000,
                    'symbol': 'TESTUSDT',
                    'side': 'long',
                    'entry_price': 1.5054,
                    'exit_price': 1.5084,
                    'size': 100.0,
                    'pnl': 30.0,
                    'pnl_percentage': 2.0,
                    'duration': 60000,
                    'exit_reason': 'target_hit'
                }
            ]
        }

        def simulate_backtest_complete():
            print("Simulating backtest completion...")
            window.results_data = fake_results
            window._display_results()
            print("Results displayed - checking if GUI is responsive...")

            # Test if GUI is still responsive by updating status
            window.status_bar.showMessage("GUI is responsive after backtest completion!", 5000)

            # Auto-close after 3 seconds for testing
            QTimer.singleShot(3000, app.quit)

        # Simulate backtest completion after 1 second
        QTimer.singleShot(1000, simulate_backtest_complete)

        print("Starting GUI event loop...")
        window.show()

        return app.exec()

    except Exception as e:
        print(f"GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    print("Fixed GUI Test - No Matplotlib Blocking")
    print("=" * 45)

    result = test_gui_with_simulated_backtest()

    if result == 0:
        print("SUCCESS: GUI remained responsive after simulated backtest")
    else:
        print("FAIL: GUI test failed")

    return result

if __name__ == "__main__":
    sys.exit(main())