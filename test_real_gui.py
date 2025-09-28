#!/usr/bin/env python3
"""
Тест РЕАЛЬНОЙ GUI с исправленным autoRange
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from PyQt6.QtWidgets import QApplication
from src.gui.gui_visualizer import ProfessionalBacktester

def test_real_gui():
    """Запуск реальной GUI для визуального тестирования"""
    print("="*60)
    print("ЗАПУСК РЕАЛЬНОЙ GUI С ИСПРАВЛЕНИЕМ AUTORANGE")
    print("="*60)

    try:
        app = QApplication(sys.argv)

        # Create main GUI
        gui = ProfessionalBacktester()
        gui.show()

        print("\n*** GUI запущен! ***")
        print("Инструкции для тестирования:")
        print("1. Нажмите 'START BACKTEST'")
        print("2. Проверьте, отображается ли график с ценой")
        print("3. Проверьте, видны ли торговые сигналы (треугольники)")
        print("4. Закройте окно для завершения")

        # Run GUI
        sys.exit(app.exec())

    except Exception as e:
        print(f"[ERROR] ОШИБКА запуска GUI: {e}")
        return False

if __name__ == "__main__":
    test_real_gui()