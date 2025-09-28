"""
Точная симуляция GUI workflow - найти различия с рабочей версией
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_exact_gui_simulation():
    """Симулируем точно тот же workflow что в GUI"""
    print("=" * 60)
    print("ТОЧНАЯ СИМУЛЯЦИЯ GUI WORKFLOW")
    print("=" * 60)

    try:
        from PyQt6.QtWidgets import QApplication
        from src.gui.gui_visualizer import ProfessionalBacktester

        print("\n1. СОЗДАНИЕ НАСТОЯЩЕГО GUI ОБЪЕКТА:")

        app = QApplication.instance()
        if app is None:
            app = QApplication([])

        # Создаем реальный GUI объект
        gui = ProfessionalBacktester()
        print("   [OK] ProfessionalBacktester создан")

        # Получаем реальные настройки из GUI
        config = gui.config
        print(f"\n2. РЕАЛЬНЫЕ НАСТРОЙКИ GUI:")
        print(f"   BB period: {config.bb_period}")
        print(f"   BB std: {config.bb_std}")
        print(f"   Stop loss: {config.stop_loss_pct}%")
        print(f"   Initial capital: ${config.initial_capital}")
        print(f"   Max ticks GUI: {getattr(config, 'max_ticks_gui', 'НЕТ')}")
        print(f"   Max ticks unlimited: {getattr(config, 'max_ticks_unlimited', 'НЕТ')}")

        # Получаем текущий датасет из GUI
        dataset = gui.dataset_combo.currentText()
        if not dataset:
            print("   [ERROR] Датасет не выбран в GUI!")
            return False

        dataset_path = gui.dataset_manager.get_dataset_path(dataset)
        symbol = gui.dataset_manager.extract_symbol(dataset)

        print(f"\n3. ДАННЫЕ ИЗ GUI:")
        print(f"   Датасет: {dataset}")
        print(f"   Путь: {dataset_path}")
        print(f"   Символ: {symbol}")
        print(f"   Файл существует: {os.path.exists(dataset_path)}")

        # Симулируем нажатие кнопки "Start Backtest"
        print(f"\n4. СИМУЛЯЦИЯ НАЖАТИЯ 'START BACKTEST':")

        # Проверяем что происходит в _start_backtest()
        print("   Проверяем готовность к запуску...")

        if not dataset:
            print("   [ERROR] Датасет не выбран")
            return False

        if not os.path.exists(dataset_path):
            print("   [ERROR] Файл данных не найден")
            return False

        # Обновляем конфиг из GUI (симулируем _update_config())
        gui._update_config()
        print("   [OK] Конфиг обновлен из GUI")

        # Симулируем создание BacktestWorker с теми же параметрами что в GUI
        print(f"\n5. СОЗДАНИЕ BACKTEST WORKER (как в GUI):")

        from src.gui.config.config_models import BacktestWorker

        tick_mode = True  # Как в GUI
        max_ticks = None  # Как в GUI - полный датасет

        worker = BacktestWorker(
            csv_path=dataset_path,
            symbol=symbol,
            config=gui.config,  # Используем конфиг из GUI
            tick_mode=tick_mode,
            max_ticks=max_ticks
        )

        print(f"   Worker создан с параметрами:")
        print(f"   - csv_path: {dataset_path}")
        print(f"   - symbol: {symbol}")
        print(f"   - tick_mode: {tick_mode}")
        print(f"   - max_ticks: {max_ticks}")
        print(f"   - BB period: {worker.config.bb_period}")
        print(f"   - BB std: {worker.config.bb_std}")

        # Запускаем worker.run() напрямую (без threading)
        print(f"\n6. ЗАПУСК WORKER.RUN() (без threading):")

        # Добавляем захват результата
        result_captured = None

        def capture_result(result):
            nonlocal result_captured
            result_captured = result

        worker.result_signal.connect(capture_result)

        # Запускаем
        worker.run()

        # Проверяем результат
        if result_captured:
            trades = result_captured.get('trades', [])
            bb_data = result_captured.get('bb_data', {})

            print(f"   [OK] Worker завершен успешно:")
            print(f"   - Сделок: {len(trades)}")
            print(f"   - BB точек: {len(bb_data.get('times', []))}")
            print(f"   - Net P&L: ${result_captured.get('net_pnl', 0):.2f}")
            print(f"   - Win rate: {result_captured.get('win_rate', 0)*100:.1f}%")

            if len(trades) > 0:
                print(f"   [SUCCESS] Сделки генерируются в GUI workflow!")

                # Проверяем GUI обновление
                print(f"\n7. СИМУЛЯЦИЯ ОБНОВЛЕНИЯ GUI:")

                # Симулируем _on_complete()
                gui.results_data = result_captured
                gui._display_results()

                print("   [OK] GUI компоненты обновлены")

                return True
            else:
                print(f"   [ERROR] Нет сделок в GUI workflow!")
                return False
        else:
            print("   [ERROR] Worker не вернул результат!")
            return False

    except Exception as e:
        print(f"\n[ERROR] Ошибка симуляции: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_exact_gui_simulation()
    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТ: {'ВСЕ РАБОТАЕТ!' if success else 'НАЙДЕНА ПРОБЛЕМА!'}")
    print(f"{'='*60}")
    sys.exit(0 if success else 1)