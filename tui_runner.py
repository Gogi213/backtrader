#!/usr/bin/env python3
"""
TUI Runner for HFT Optimization System

Simple script to run the terminal user interface for strategy optimization.
This provides a lightweight alternative to the full GUI.

Usage:
    python tui_runner.py

Author: HFT System
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.tui.optimization_app import run_tui_app
    
    if __name__ == "__main__":
        print("Запуск TUI интерфейса для оптимизации HFT стратегий...")
        print("Нажмите Ctrl+C для выхода")
        print("-" * 50)
        
        try:
            run_tui_app()
        except KeyboardInterrupt:
            print("\nВыход из программы...")
        except Exception as e:
            print(f"Ошибка запуска TUI: {e}")
            sys.exit(1)

except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все зависимости установлены:")
    print("pip install textual rich optuna numpy pandas")
    sys.exit(1)