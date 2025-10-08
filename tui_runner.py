#!/usr/bin/env python3
"""
TUI Runner for HFT Optimization

A simple script to run the TUI optimization interface.
Author: HFT System
"""
import sys
import os

from src.tui.optimization_app import run_tui_app


def main():
    """Main entry point"""
    print("Запуск TUI интерфейса для оптимизации HFT стратегий...")
    print("Нажмите Ctrl+C для выхода")
    print("-" * 50)
    
    try:
        run_tui_app()
    except KeyboardInterrupt:
        print("\nВыход из программы...")
        sys.exit(0)
    except Exception as e:
        print(f"Ошибка запуска: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()