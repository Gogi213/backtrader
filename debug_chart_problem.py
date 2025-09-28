#!/usr/bin/env python3
"""
ULTRA DEBUG: Максимально детальная диагностика проблемы с отображением графика
Широкое логирование каждого шага от данных до рендеринга
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def log_section(title):
    """Красивое логирование секций"""
    print("\n" + "="*80)
    print(f"[DEBUG] {title}")
    print("="*80)

def log_step(step, msg):
    """Логирование шагов"""
    print(f"   [{step}] {msg}")

def log_data(name, data, max_items=5):
    """Детальное логирование данных"""
    try:
        if hasattr(data, '__len__') and not isinstance(data, (type, str)):
            print(f"   [DATA] {name}: тип={type(data).__name__}, длина={len(data)}")
            if len(data) > 0:
                if hasattr(data, 'dtype'):
                    print(f"       dtype={data.dtype}")
                if isinstance(data, (list, np.ndarray)) and len(data) > 0:
                    sample = data[:max_items] if len(data) > max_items else data
                    print(f"       образец: {sample}")
                    if hasattr(data, 'min') and hasattr(data, 'max'):
                        print(f"       диапазон: {data.min():.6f} - {data.max():.6f}")
        else:
            print(f"   [DATA] {name}: {data}")
    except Exception as e:
        print(f"   [DATA] {name}: ОШИБКА ЛОГИРОВАНИЯ - {e}")

def ultra_debug_chart():
    """ULTRA DEBUG диагностика проблемы с графиком"""
    log_section("НАЧАЛО ULTRA DEBUG ДИАГНОСТИКИ")

    try:
        # ЭТАП 1: Подготовка PyQt
        log_section("ЭТАП 1: ИНИЦИАЛИЗАЦИЯ PyQt")
        log_step("1.1", "Импорт PyQt6...")
        from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
        log_step("1.2", "PyQt6 импортирован успешно")

        log_step("1.3", "Создание QApplication...")
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        log_step("1.4", "QApplication создан")

        # ЭТАП 2: Импорт и создание компонентов графика
        log_section("ЭТАП 2: КОМПОНЕНТЫ ГРАФИКА")
        log_step("2.1", "Импорт pyqtgraph...")
        import pyqtgraph as pg
        log_step("2.2", f"PyQtGraph версия: {pg.__version__ if hasattr(pg, '__version__') else 'неизвестно'}")

        log_step("2.3", "Проверка OpenGL...")
        try:
            import OpenGL
            log_step("2.4", f"OpenGL доступен: {OpenGL.__version__}")
        except ImportError:
            log_step("2.4", "OpenGL НЕ доступен")

        log_step("2.5", "Импорт HighPerformanceChart...")
        from src.gui.charts.pyqtgraph_chart import HighPerformanceChart
        log_step("2.6", "HighPerformanceChart импортирован")

        # ЭТАП 3: Генерация тестовых данных
        log_section("ЭТАП 3: ГЕНЕРАЦИЯ ДАННЫХ")
        log_step("3.1", "Импорт backtest функции...")
        from src.data.vectorized_klines_backtest import run_vectorized_klines_backtest
        log_step("3.2", "Функция импортирована")

        log_step("3.3", "Запуск бэктеста...")
        csv_path = "upload/klines/ASTERUSDT-klines-10s-2025-09-20_to_2025-09-21.csv"
        results = run_vectorized_klines_backtest(
            csv_path=csv_path,
            symbol="ASTERUSDT",
            bb_period=50,  # ИСПРАВЛЕННОЕ ЗНАЧЕНИЕ
            bb_std=2.0,    # ИСПРАВЛЕННОЕ ЗНАЧЕНИЕ
            stop_loss_pct=1.0,
            initial_capital=10000.0,
            max_klines=1000  # Ограничиваем для дебага
        )
        log_step("3.4", "Бэктест завершен")

        # ДЕТАЛЬНАЯ ДИАГНОСТИКА ДАННЫХ
        log_section("ЭТАП 4: АНАЛИЗ ДАННЫХ")

        bb_data = results.get('bb_data', {})
        trades = results.get('trades', [])

        log_data("bb_data keys", list(bb_data.keys()) if bb_data else [])
        log_data("trades count", len(trades))

        if bb_data:
            for key in ['times', 'prices', 'bb_upper', 'bb_middle', 'bb_lower']:
                if key in bb_data:
                    log_data(f"bb_data['{key}']", bb_data[key])
                else:
                    log_step("ERROR", f"Отсутствует ключ: {key}")

        # ПРОВЕРКА НА NaN И НЕКОРРЕКТНЫЕ ЗНАЧЕНИЯ
        log_section("ЭТАП 5: ВАЛИДАЦИЯ ДАННЫХ")

        validation_passed = True

        if bb_data and 'times' in bb_data and 'prices' in bb_data:
            times = np.array(bb_data['times'])
            prices = np.array(bb_data['prices'])

            # Проверка NaN
            times_nan = np.isnan(times).sum()
            prices_nan = np.isnan(prices).sum()
            log_step("5.1", f"NaN в times: {times_nan}")
            log_step("5.2", f"NaN в prices: {prices_nan}")

            if times_nan > 0 or prices_nan > 0:
                log_step("ERROR", "Обнаружены NaN значения!")
                validation_passed = False

            # Проверка бесконечности
            times_inf = np.isinf(times).sum()
            prices_inf = np.isinf(prices).sum()
            log_step("5.3", f"Inf в times: {times_inf}")
            log_step("5.4", f"Inf в prices: {prices_inf}")

            if times_inf > 0 or prices_inf > 0:
                log_step("ERROR", "Обнаружены Inf значения!")
                validation_passed = False

            # Проверка диапазонов
            log_step("5.5", f"Times диапазон: {times.min():.0f} - {times.max():.0f}")
            log_step("5.6", f"Prices диапазон: {prices.min():.6f} - {prices.max():.6f}")

            # Проверка монотонности времени
            times_sorted = np.all(times[:-1] <= times[1:])
            log_step("5.7", f"Times отсортированы: {times_sorted}")

            if not times_sorted:
                log_step("WARNING", "Времена не отсортированы по возрастанию!")

        else:
            log_step("ERROR", "bb_data не содержит необходимые ключи!")
            validation_passed = False

        log_step("5.8", f"Валидация данных: {'ПРОЙДЕНА' if validation_passed else 'ПРОВАЛЕНА'}")

        # ЭТАП 6: СОЗДАНИЕ И НАСТРОЙКА ГРАФИКА
        log_section("ЭТАП 6: СОЗДАНИЕ ГРАФИКА")

        log_step("6.1", "Создание HighPerformanceChart...")
        chart = HighPerformanceChart()
        log_step("6.2", "График создан")

        log_step("6.3", "Проверка plot_widget...")
        plot_widget = chart.plot_widget
        log_data("plot_widget", f"тип={type(plot_widget).__name__}, существует={plot_widget is not None}")

        # ЭТАП 7: ДЕТАЛЬНАЯ ДИАГНОСТИКА ОБНОВЛЕНИЯ ГРАФИКА
        log_section("ЭТАП 7: ОБНОВЛЕНИЕ ГРАФИКА С ДЕТАЛЬНЫМ ЛОГИРОВАНИЕМ")

        log_step("7.1", "Вызов chart.update_chart()...")

        # Временно патчим функцию для максимального логирования
        original_update = chart.update_chart

        def debug_update_chart(results_data):
            log_step("7.2", "Внутри update_chart()")
            log_data("results_data type", type(results_data))
            log_data("results_data keys", list(results_data.keys()) if isinstance(results_data, dict) else "не словарь")

            if not results_data:
                log_step("ERROR", "results_data пустая!")
                return

            bb_data = results_data.get('bb_data')
            log_step("7.3", f"bb_data получен: {bb_data is not None}")

            if not bb_data or 'times' not in bb_data:
                log_step("ERROR", "bb_data не содержит times!")
                return

            # Подробная диагностика данных перед отрисовкой
            times_ms = np.array(bb_data['times'], dtype=np.float64)
            prices = np.array(bb_data['prices'], dtype=np.float32)

            log_data("times_ms for chart", times_ms)
            log_data("prices for chart", prices)

            # Преобразование в секунды для PyQtGraph
            times_sec = times_ms / 1000.0
            log_data("times_sec for chart", times_sec)

            # Проверка данных перед plot
            log_step("7.4", "Очистка предыдущих данных...")
            chart.plot_widget.clear()
            log_step("7.5", "График очищен")

            # Фильтрация валидных данных
            valid_mask = ~(np.isnan(prices) | np.isnan(times_sec))
            times_clean = times_sec[valid_mask]
            prices_clean = prices[valid_mask]

            log_data("times_clean", times_clean)
            log_data("prices_clean", prices_clean)

            if len(times_clean) == 0:
                log_step("ERROR", "Нет валидных данных после фильтрации!")
                return

            log_step("7.6", f"Готов к отрисовке: {len(times_clean)} точек")

            # КРИТИЧЕСКИЙ МОМЕНТ: СОЗДАНИЕ ЛИНИИ ЦЕНЫ
            log_step("7.7", "Создание линии цены...")
            try:
                # Подробное логирование создания pen
                pen = pg.mkPen(color='#00aaff', width=1.5)
                log_step("7.8", f"Pen создан: {pen}")

                # Попытка создать plot
                log_step("7.9", "Вызов plot_widget.plot()...")
                price_curve = chart.plot_widget.plot(
                    times_clean, prices_clean,
                    pen=pen,
                    name='Price',
                    antialias=False
                )
                log_step("7.10", f"plot() вернул: {price_curve}")
                log_step("7.11", f"price_curve тип: {type(price_curve)}")

                # Проверка что линия добавлена
                items = chart.plot_widget.listDataItems()
                log_step("7.12", f"Элементов в графике: {len(items)}")
                for i, item in enumerate(items):
                    log_step("7.13", f"  Элемент {i}: {type(item).__name__}")

            except Exception as e:
                log_step("ERROR", f"Ошибка при создании линии цены: {e}")
                import traceback
                traceback.print_exc()
                return

            # Настройка диапазонов
            log_step("7.14", "Настройка диапазонов...")
            try:
                chart.plot_widget.setXRange(times_clean.min(), times_clean.max(), padding=0.02)
                chart.plot_widget.setYRange(prices_clean.min(), prices_clean.max(), padding=0.02)
                log_step("7.15", "Диапазоны установлены")

                chart.plot_widget.autoRange()
                log_step("7.16", "autoRange() вызван")

            except Exception as e:
                log_step("ERROR", f"Ошибка при настройке диапазонов: {e}")

            log_step("7.17", "update_chart() завершен")

        # Заменяем функцию на debug версию
        chart.update_chart = debug_update_chart

        # Вызываем обновление с полным логированием
        chart.update_chart(results)

        # ЭТАП 8: ПРОВЕРКА РЕЗУЛЬТАТА
        log_section("ЭТАП 8: ПРОВЕРКА РЕЗУЛЬТАТА")

        log_step("8.1", "Проверка содержимого графика...")
        items = chart.plot_widget.listDataItems()
        log_step("8.2", f"Всего элементов в графике: {len(items)}")

        if len(items) == 0:
            log_step("ERROR", "ГРАФИК ПУСТОЙ! Элементы не добавлены!")
        else:
            log_step("SUCCESS", f"График содержит {len(items)} элементов")
            for i, item in enumerate(items):
                log_step("8.3", f"  Элемент {i}: {type(item).__name__}")

                # Проверяем данные элемента
                if hasattr(item, 'xData') and hasattr(item, 'yData'):
                    x_data = item.xData
                    y_data = item.yData
                    if x_data is not None and y_data is not None:
                        log_step("8.4", f"    X данные: {len(x_data)} точек")
                        log_step("8.5", f"    Y данные: {len(y_data)} точек")
                    else:
                        log_step("WARNING", "    Данные элемента None!")

        # Проверка видимости
        log_step("8.6", "Проверка настроек видимости...")
        view_box = chart.plot_widget.getViewBox()
        if view_box:
            state = view_box.getState()
            log_step("8.7", f"ViewBox состояние: {state}")

        # ЭТАП 9: ТЕСТ ПРОСТЕЙШЕГО ГРАФИКА
        log_section("ЭТАП 9: ТЕСТ ПРОСТЕЙШЕГО ГРАФИКА")

        log_step("9.1", "Создание простейшего тестового графика...")

        # Создаем новый простой график для сравнения
        simple_widget = pg.PlotWidget()

        # Простые тестовые данные
        test_x = np.array([1, 2, 3, 4, 5])
        test_y = np.array([1, 4, 2, 3, 5])

        log_step("9.2", "Добавление тестовых данных...")
        test_curve = simple_widget.plot(test_x, test_y, pen='r')

        test_items = simple_widget.listDataItems()
        log_step("9.3", f"Тестовый график содержит: {len(test_items)} элементов")

        if len(test_items) > 0:
            log_step("SUCCESS", "Простейший график работает!")
        else:
            log_step("ERROR", "Даже простейший график НЕ работает!")

        # ФИНАЛЬНЫЙ ОТЧЕТ
        log_section("ФИНАЛЬНЫЙ ОТЧЕТ")

        chart_items = len(chart.plot_widget.listDataItems())
        test_items_count = len(test_items)

        print(f"""
[РЕЗУЛЬТАТЫ] ULTRA DEBUG ДИАГНОСТИКИ:

[ДАННЫЕ]:
   - BB точек: {len(bb_data.get('times', []))}
   - Сделок: {len(trades)}
   - Валидация: {'ПРОЙДЕНА' if validation_passed else 'ПРОВАЛЕНА'}

[ГРАФИК]:
   - Основной график: {chart_items} элементов
   - Тестовый график: {test_items_count} элементов
   - PyQtGraph работает: {'ДА' if test_items_count > 0 else 'НЕТ'}

[ПРОБЛЕМА]:
   {'ГРАФИК ПУСТОЙ - элементы не добавляются!' if chart_items == 0 else 'График содержит элементы'}
        """)

        return chart_items > 0

    except Exception as e:
        log_step("CRITICAL ERROR", f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = ultra_debug_chart()
    print(f"\n{'[SUCCESS] ДИАГНОСТИКА ЗАВЕРШЕНА' if success else '[ERROR] ПРОБЛЕМА ОБНАРУЖЕНА'}")
    sys.exit(0 if success else 1)