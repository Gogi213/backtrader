#!/usr/bin/env python3
"""
Quick GUI test without infinite loop
"""
import sys
import os

def test_gui_init():
    """Test GUI initialization without running event loop"""
    print("Testing GUI initialization...")

    try:
        from PyQt6.QtWidgets import QApplication
        from src.gui.gui_visualizer import ProfessionalBacktester

        # Create application but don't exec()
        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        print("Creating GUI window...")
        window = ProfessionalBacktester()

        print("GUI window created successfully")

        # Don't call app.exec() - just verify creation works
        return True

    except Exception as e:
        print(f"GUI initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("HFT GUI Launch Test")
    print("=" * 30)

    if test_gui_init():
        print("SUCCESS: GUI can be initialized")
        return 0
    else:
        print("FAIL: GUI initialization failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())