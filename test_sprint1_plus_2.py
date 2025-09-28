#!/usr/bin/env python3
"""
ULTRA TEST: Проверка реального GUI workflow для поиска проблемы с графиком
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_real_gui_chart_flow():
    """Тест реального GUI workflow с фокусом на график"""
    print("="*80)
    print("[REAL GUI TEST] ПОИСК ПРОБЛЕМЫ С ГРАФИКОМ В РЕАЛЬНОМ GUI")
    print("="*80)

    try:
        from PyQt6.QtWidgets import QApplication
        from src.gui.gui_visualizer import ProfessionalBacktester

        # Создаем реальное GUI приложение
        print("\n[STEP 1] Создание реального GUI...")
        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        gui = ProfessionalBacktester()
        print("   [OK] GUI создан")

        # Проверяем chart компонент
        print("\n[STEP 2] Проверка chart компонента в GUI...")
        chart_tab = gui.chart_signals_tab
        print(f"   chart_signals_tab: {chart_tab}")
        print(f"   chart_signals_tab тип: {type(chart_tab)}")

        if hasattr(chart_tab, 'chart'):
            chart = chart_tab.chart
            print(f"   chart: {chart}")
            print(f"   chart тип: {type(chart)}")

            if hasattr(chart, 'plot_widget'):
                plot_widget = chart.plot_widget
                print(f"   plot_widget: {plot_widget}")
                print(f"   plot_widget тип: {type(plot_widget)}")
            else:
                print("   [ERROR] chart не имеет plot_widget!")
        else:
            print("   [ERROR] chart_signals_tab не имеет chart!")

        # Получаем данные как в реальном GUI
        print("\n[STEP 3] Получение данных как в реальном GUI...")

        # Используем dataset manager
        dataset = gui.dataset_combo.currentText()
        dataset_path = gui.dataset_manager.get_dataset_path(dataset)
        symbol = gui.dataset_manager.extract_symbol(dataset)

        print(f"   Датасет: {dataset}")
        print(f"   Путь: {dataset_path}")
        print(f"   Символ: {symbol}")

        # Конфиг как в GUI
        config = gui.config
        print(f"   BB период: {config.bb_period}")
        print(f"   BB std: {config.bb_std}")

        # Симуляция бэктеста как в GUI
        print("\n[STEP 4] Симуляция бэктеста как в GUI...")
        from src.gui.config.config_models import BacktestWorker

        # Результат для захвата
        captured_result = None

        def capture_result(result):
            nonlocal captured_result
            captured_result = result
            print(f"   [CALLBACK] Результат получен: {len(result.get('trades', []))} сделок")

        # Создаем worker как в GUI
        worker = BacktestWorker(
            csv_path=dataset_path,
            symbol=symbol,
            config=config,
            tick_mode=True,
            max_ticks=None
        )

        # Подключаем сигнал
        worker.result_signal.connect(capture_result)

        # Запускаем
        print("   Запуск worker...")
        worker.run()

        if captured_result:
            trades = captured_result.get('trades', [])
            bb_data = captured_result.get('bb_data', {})
            print(f"   [OK] Получено: {len(trades)} сделок, {len(bb_data.get('times', []))} BB точек")

            # КРИТИЧЕСКИЙ ТЕСТ: Симуляция _on_complete() как в GUI
            print("\n[STEP 5] Симуляция _on_complete() как в GUI...")

            # Устанавливаем results_data как в GUI
            gui.results_data = captured_result
            print("   [OK] results_data установлен в GUI")

            # Вызываем _display_results() как в GUI
            print("   Вызов gui._display_results()...")

            # Добавляем логирование в chart_tab
            original_update = chart_tab.update_chart

            def debug_chart_update(results_data):
                print(f"   [CHART_TAB] update_chart() вызван!")
                print(f"   [CHART_TAB] results_data тип: {type(results_data)}")
                if results_data:
                    print(f"   [CHART_TAB] results_data ключи: {list(results_data.keys())}")
                    bb_data = results_data.get('bb_data', {})
                    if bb_data:
                        print(f"   [CHART_TAB] bb_data содержит {len(bb_data.get('times', []))} точек")
                    else:
                        print("   [CHART_TAB] bb_data пустой!")

                # Вызываем оригинальную функцию
                result = original_update(results_data)
                print(f"   [CHART_TAB] update_chart() завершен")
                return result

            # Патчим функцию
            chart_tab.update_chart = debug_chart_update

            # Вызываем как в GUI
            gui._display_results()

            print("   [OK] _display_results() завершен")

            # Проверяем состояние графика
            print("\n[STEP 6] Проверка состояния графика после обновления...")
            if hasattr(chart_tab, 'chart') and hasattr(chart_tab.chart, 'plot_widget'):
                items = chart_tab.chart.plot_widget.listDataItems()
                print(f"   График содержит: {len(items)} элементов")

                if len(items) > 0:
                    print("   [SUCCESS] График содержит элементы!")
                    for i, item in enumerate(items):
                        print(f"     Элемент {i}: {type(item).__name__}")
                else:
                    print("   [ERROR] График пустой!")
                    return False
            else:
                print("   [ERROR] Не могу проверить график!")
                return False

            return True

        else:
            print("   [ERROR] Worker не вернул результат!")
            return False

    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_debug_logging():
    """Добавить debug логирование в реальный GUI код"""
    print("\n" + "="*80)
    print("[GUI DEBUG] ДОБАВЛЕНИЕ ЛОГИРОВАНИЯ В РЕАЛЬНЫЙ GUI")
    print("="*80)

    try:
        # Проверяем ключевые файлы GUI
        files_to_check = [
            'src/gui/gui_visualizer.py',
            'src/gui/tabs/tab_chart_signals.py',
            'src/gui/charts/pyqtgraph_chart.py'
        ]

        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"   [OK] {file_path} найден")
            else:
                print(f"   [ERROR] {file_path} не найден!")

        # Рекомендации для отладки
        print("\n[РЕКОМЕНДАЦИИ] Для дальнейшей отладки:")
        print("1. Добавить print() в gui_visualizer.py в методе _display_results()")
        print("2. Добавить print() в tab_chart_signals.py в методе update_chart()")
        print("3. Добавить print() в pyqtgraph_chart.py в методе update_chart()")
        print("4. Запустить реальное GUI и проследить логи")

        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    print("REAL GUI CHART DEBUGGING")

    success1 = test_real_gui_chart_flow()
    success2 = test_gui_debug_logging()

    print("\n" + "="*80)
    if success1:
        print("[SUCCESS] РЕАЛЬНЫЙ GUI WORKFLOW РАБОТАЕТ!")
        print("График должен отображаться корректно")
    else:
        print("[ERROR] НАЙДЕНА ПРОБЛЕМА В РЕАЛЬНОМ GUI!")
        print("Нужно добавить debug логирование в GUI файлы")

    print("="*80)