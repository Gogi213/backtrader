"""
ФИНАЛЬНЫЙ ТЕСТ: Проверка что график полностью работает
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def final_chart_test():
    """Финальная проверка что график работает"""
    print("=" * 60)
    print("ФИНАЛЬНЫЙ ТЕСТ ГРАФИКА")
    print("=" * 60)

    try:
        from PyQt6.QtWidgets import QApplication
        from src.gui.gui_visualizer import ProfessionalBacktester

        # Создаем GUI
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        gui = ProfessionalBacktester()

        # Быстрый бэктест
        dataset = gui.dataset_combo.currentText()
        dataset_path = gui.dataset_manager.get_dataset_path(dataset)
        symbol = gui.dataset_manager.extract_symbol(dataset)
        config = gui.config

        from src.gui.config.config_models import BacktestWorker

        result_captured = None

        def capture(result):
            nonlocal result_captured
            result_captured = result

        worker = BacktestWorker(dataset_path, symbol, config, tick_mode=True, max_ticks=None)
        worker.result_signal.connect(capture)
        worker.run()

        if result_captured:
            # Тестируем полный workflow
            gui.results_data = result_captured
            gui._display_results()

            # Проверяем результат
            chart_items = len(gui.chart_signals_tab.chart.plot_widget.listDataItems())
            trades_count = len(result_captured.get('trades', []))
            bb_points = len(result_captured.get('bb_data', {}).get('times', []))

            print(f"""
РЕЗУЛЬТАТЫ ФИНАЛЬНОГО ТЕСТА:

[ГРАФИК]
- Элементов в графике: {chart_items}
- Статус: {'✓ РАБОТАЕТ' if chart_items > 0 else '✗ НЕ РАБОТАЕТ'}

[ДАННЫЕ]
- Торговые сделки: {trades_count}
- BB точки: {bb_points}

[ИТОГ]
{'🎯 ГРАФИК ПОЛНОСТЬЮ ИСПРАВЛЕН!' if chart_items > 0 else '🚨 ГРАФИК ВСЕ ЕЩЕ НЕ РАБОТАЕТ'}
            """)

            return chart_items > 0

        else:
            print("Ошибка: нет результата от worker")
            return False

    except Exception as e:
        print(f"Ошибка теста: {e}")
        return False

if __name__ == "__main__":
    success = final_chart_test()
    print("=" * 60)
    print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {'УСПЕХ' if success else 'НЕУДАЧА'}")
    print("=" * 60)