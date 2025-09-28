"""
GUI Debug Test - найти проблему с отображением графиков
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gui_components():
    """Тест компонентов GUI без полного запуска"""
    print("=" * 60)
    print("ТЕСТ КОМПОНЕНТОВ GUI")
    print("=" * 60)

    try:
        # 1. Тест импортов GUI
        print("\n1. Тестирование импортов GUI...")

        from PyQt6.QtWidgets import QApplication
        print("   [OK] PyQt6 импортирован")

        from src.gui.config.config_models import StrategyConfig, BacktestWorker
        print("   [OK] config_models импортирован")

        from src.gui.data.dataset_manager import DatasetManager
        print("   [OK] dataset_manager импортирован")

        from src.gui.tabs.tab_chart_signals import ChartSignalsTab
        print("   [OK] tab_chart_signals импортирован")

        from src.gui.charts.pyqtgraph_chart import HighPerformanceChart
        print("   [OK] pyqtgraph_chart импортирован")

        # 2. Тест создания компонентов
        print("\n2. Тестирование создания компонентов...")

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        config = StrategyConfig()
        print(f"   [OK] StrategyConfig создан: BB period = {config.bb_period}")

        chart_tab = ChartSignalsTab()
        print("   [OK] ChartSignalsTab создан")

        # 3. Тест DatasetManager
        print("\n3. Тестирование DatasetManager...")

        # Mock объекты для тестирования
        class MockCombo:
            def __init__(self):
                self.items = []
                self.current_index = 0
            def addItems(self, items):
                self.items.extend(items)
            def setCurrentIndex(self, index):
                self.current_index = index
            def currentText(self):
                return self.items[self.current_index] if self.items else ""

        class MockLabel:
            def __init__(self):
                self.text = ""
            def setText(self, text):
                self.text = text

        class MockLogger:
            def log(self, msg):
                # Fix encoding issues by replacing special characters
                msg_clean = msg.replace('→', '->')
                print(f"   [LOG] {msg_clean}")

        mock_combo = MockCombo()
        mock_label = MockLabel()
        mock_logger = MockLogger()

        dataset_manager = DatasetManager(mock_combo, mock_label, mock_logger)
        dataset_manager.load_datasets()

        if mock_combo.items:
            print(f"   [OK] Найдено {len(mock_combo.items)} датасетов")
            first_dataset = mock_combo.currentText()
            print(f"   [OK] Первый датасет: {first_dataset}")

            dataset_path = dataset_manager.get_dataset_path(first_dataset)
            print(f"   [OK] Путь к датасету: {dataset_path}")

            if os.path.exists(dataset_path):
                print("   [OK] Файл датасета существует")
            else:
                print("   [ERROR] Файл датасета НЕ существует!")
                return False
        else:
            print("   [ERROR] Датасеты не найдены!")
            return False

        # 4. Тест BacktestWorker
        print("\n4. Тестирование BacktestWorker...")

        first_dataset = mock_combo.currentText()
        dataset_path = dataset_manager.get_dataset_path(first_dataset)
        symbol = dataset_manager.extract_symbol(first_dataset)

        print(f"   Датасет: {first_dataset}")
        print(f"   Символ: {symbol}")
        print(f"   Путь: {dataset_path}")
        print(f"   BB период: {config.bb_period}")

        # Создаем BacktestWorker но не запускаем
        worker = BacktestWorker(
            csv_path=dataset_path,
            symbol=symbol,
            config=config,
            tick_mode=True,
            max_ticks=1000  # Ограничиваем для теста
        )
        print("   [OK] BacktestWorker создан")

        # 5. Проверка функции бэктеста
        print("\n5. Тестирование функции бэктеста...")

        from src.data.vectorized_klines_backtest import run_vectorized_klines_backtest
        print("   [OK] run_vectorized_klines_backtest импортирован")

        # Быстрый тест бэктеста с ограниченными данными
        print("   Запуск быстрого бэктеста...")
        results = run_vectorized_klines_backtest(
            csv_path=dataset_path,
            symbol=symbol,
            bb_period=config.bb_period,
            bb_std=config.bb_std,
            stop_loss_pct=config.stop_loss_pct,
            initial_capital=config.initial_capital,
            max_klines=100  # Очень мало для быстрого теста
        )

        print(f"   [OK] Бэктест завершен: {len(results.get('trades', []))} сделок")
        print(f"   [OK] BB данные: {len(results.get('bb_data', {}).get('times', []))} точек")

        # 6. Тест обновления графика
        print("\n6. Тестирование обновления графика...")

        print("   Вызываем chart_tab.update_chart()...")
        chart_tab.update_chart(results)
        print("   [OK] График обновлен без ошибок")

        print("\n" + "=" * 60)
        print("ВСЕ КОМПОНЕНТЫ GUI РАБОТАЮТ КОРРЕКТНО!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n[ERROR] Ошибка в компонентах GUI: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gui_components()
    print(f"\nРезультат теста: {'УСПЕШНО' if success else 'ОШИБКА'}")
    sys.exit(0 if success else 1)