"""
Test Full GUI Workflow - симуляция полного процесса GUI без ограничений
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_full_gui_workflow():
    """Тест полного GUI workflow с полным датасетом"""
    print("=" * 60)
    print("ТЕСТ ПОЛНОГО GUI WORKFLOW")
    print("=" * 60)

    try:
        # Подготовка
        from PyQt6.QtWidgets import QApplication
        from src.gui.config.config_models import StrategyConfig, BacktestWorker
        from src.gui.data.dataset_manager import DatasetManager
        from src.gui.tabs.tab_chart_signals import ChartSignalsTab

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        # Mock компоненты
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
                msg_clean = msg.replace('→', '->')
                print(f"   [LOG] {msg_clean}")

        # Настройка
        config = StrategyConfig()
        mock_combo = MockCombo()
        mock_label = MockLabel()
        mock_logger = MockLogger()

        dataset_manager = DatasetManager(mock_combo, mock_label, mock_logger)
        dataset_manager.load_datasets()

        dataset = mock_combo.currentText()
        dataset_path = dataset_manager.get_dataset_path(dataset)
        symbol = dataset_manager.extract_symbol(dataset)

        print(f"\n1. ПАРАМЕТРЫ ТЕСТИРОВАНИЯ:")
        print(f"   Датасет: {dataset}")
        print(f"   Символ: {symbol}")
        print(f"   BB период: {config.bb_period}")
        print(f"   BB std: {config.bb_std}")
        print(f"   Stop loss: {config.stop_loss_pct}%")

        # Тест 1: Полный датасет без ограничений (как в реальном GUI)
        print(f"\n2. ТЕСТ ПОЛНОГО ДАТАСЕТА (без ограничений):")

        from src.data.vectorized_klines_backtest import run_vectorized_klines_backtest

        results_full = run_vectorized_klines_backtest(
            csv_path=dataset_path,
            symbol=symbol,
            bb_period=config.bb_period,
            bb_std=config.bb_std,
            stop_loss_pct=config.stop_loss_pct,
            initial_capital=config.initial_capital,
            max_klines=None  # БЕЗ ОГРАНИЧЕНИЙ - как в реальном GUI
        )

        trades_full = results_full.get('trades', [])
        bb_data_full = results_full.get('bb_data', {})

        print(f"   Результат: {len(trades_full)} сделок, {len(bb_data_full.get('times', []))} BB точек")

        if len(trades_full) > 0:
            print(f"   [OK] СДЕЛКИ СГЕНЕРИРОВАНЫ! Первые 3:")
            for i, trade in enumerate(trades_full[:3]):
                pnl = trade.get('pnl', 0)
                side = trade.get('side', 'N/A')
                entry_price = trade.get('entry_price', 0)
                print(f"      {i+1}. {side} @ {entry_price:.4f} -> P&L: ${pnl:.2f}")
        else:
            print("   [ERROR] НЕТ СДЕЛОК! Возможна проблема с параметрами")

        # Тест 2: Симуляция GUI worker thread
        print(f"\n3. ТЕСТ GUI WORKER THREAD:")

        # Создаем WorkerThread точно как в GUI
        worker = BacktestWorker(
            csv_path=dataset_path,
            symbol=symbol,
            config=config,
            tick_mode=True,  # HFT режим как в GUI
            max_ticks=None   # Без ограничений как в GUI
        )

        # Имитируем запуск worker.run() без реального threading
        print("   Имитируем worker.run()...")
        worker.run()

        print("   Worker завершен без ошибок")

        # Тест 3: Симуляция полного GUI workflow
        print(f"\n4. ТЕСТ ПОЛНОГО GUI WORKFLOW:")

        chart_tab = ChartSignalsTab()

        # Обновляем график с полными данными
        print("   Обновляем график с полными данными...")
        chart_tab.update_chart(results_full)

        print("   [OK] График обновлен успешно")

        # Финальный отчет
        print(f"\n" + "=" * 60)
        print("ФИНАЛЬНЫЙ ОТЧЕТ:")
        print(f"Всего сделок: {len(trades_full)}")
        print(f"BB точек: {len(bb_data_full.get('times', []))}")
        print(f"Net P&L: ${results_full.get('net_pnl', 0):.2f}")
        print(f"Win rate: {results_full.get('win_rate', 0)*100:.1f}%")

        if len(trades_full) > 0:
            print("[SUCCESS] ПРОБЛЕМА РЕШЕНА: Сделки генерируются корректно!")
            print("   -> График должен отображаться с торговыми сигналами")
        else:
            print("[ERROR] ПРОБЛЕМА ОСТАЕТСЯ: Сделки не генерируются")
            print("   -> Возможно нужно изменить параметры BB или данные")

        print("=" * 60)

        return len(trades_full) > 0

    except Exception as e:
        print(f"\n[ERROR] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_gui_workflow()
    print(f"\nРезультат: {'УСПЕШНО' if success else 'ТРЕБУЕТ ДАЛЬНЕЙШЕЙ ДИАГНОСТИКИ'}")
    sys.exit(0 if success else 1)