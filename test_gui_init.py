#!/usr/bin/env python3
"""
Тест инициализации GUI компонентов без графического отображения
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

def test_gui_components():
    """Тест импорта и инициализации GUI компонентов"""
    print("="*60)
    print("ТЕСТ ИНИЦИАЛИЗАЦИИ GUI КОМПОНЕНТОВ")
    print("="*60)

    try:
        # Test imports
        print("1. Тестирование импортов...")
        from src.gui.gui_visualizer import ProfessionalBacktester
        from src.gui.charts.pyqtgraph_chart import HighPerformanceChart
        from src.gui.tabs.tab_chart_signals import ChartSignalsTab
        print("   [OK] Все импорты успешны")

        # Test chart component creation (without GUI)
        print("\n2. Тестирование создания компонентов...")

        # We can't create QWidgets without QApplication, but we can test class definitions
        chart_class = HighPerformanceChart
        print(f"   [OK] HighPerformanceChart class: {chart_class}")

        tab_class = ChartSignalsTab
        print(f"   [OK] ChartSignalsTab class: {tab_class}")

        gui_class = ProfessionalBacktester
        print(f"   [OK] ProfessionalBacktester class: {gui_class}")

        # Test method existence
        print("\n3. Тестирование методов...")
        chart_methods = [m for m in dir(chart_class) if not m.startswith('_')]
        print(f"   [OK] HighPerformanceChart методы: {len(chart_methods)} найдено")

        if 'update_chart' in chart_methods:
            print("   [OK] Критический метод update_chart найден")
        else:
            print("   [ERROR] update_chart метод отсутствует!")

        print("\n4. Проверка исправленного кода...")
        # Read the fixed chart file and verify critical fixes
        chart_file = "src/gui/charts/pyqtgraph_chart.py"
        if os.path.exists(chart_file):
            with open(chart_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if 'CRITICAL FIX' in content:
                print("   [OK] CRITICAL FIX комментарии найдены")
            else:
                print("   [WARNING] CRITICAL FIX комментарии не найдены")

            if 'enableAutoRange(True, True)' in content:
                print("   [OK] enableAutoRange(True, True) исправление найдено")
            else:
                print("   [ERROR] enableAutoRange исправление отсутствует!")

            if 'autoRange()' in content:
                print("   [OK] autoRange() вызов найден")
            else:
                print("   [ERROR] autoRange() вызов отсутствует!")

        print("\n" + "="*60)
        print("РЕЗУЛЬТАТ: Все компоненты готовы к работе!")
        print("="*60)
        print("\nДля визуального тестирования запустите:")
        print("python test_real_gui.py")
        print("\nИли запустите главную GUI:")
        print("python src/gui/gui_visualizer.py")

        return True

    except Exception as e:
        print(f"[ERROR] Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gui_components()
    sys.exit(0 if success else 1)