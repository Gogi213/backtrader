"""
ГЛУБОЧАЙШИЙ DEBUG VERSION pyqtgraph_chart.py с максимальными логами
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from datetime import datetime


class HighPerformanceChart(QWidget):
    """
    Ultra-fast chart component optimized for 500k+ data points
    Uses PyQtGraph for maximum rendering performance
    """

    def __init__(self):
        super().__init__()
        self.plot_widget = None
        self.price_curve = None
        self.bb_upper_curve = None
        self.bb_middle_curve = None
        self.bb_lower_curve = None
        self.bb_fill = None
        self.buy_scatter = None
        self.sell_scatter = None

        # Performance optimizations
        self.max_display_points = 500000  # Maximum points to display
        self.downsample_threshold = 100000  # Start downsampling above this

        self._init_ui()
        self._setup_performance_optimizations()

    def _init_ui(self):
        """Initialize high-performance chart UI"""
        layout = QVBoxLayout(self)

        # Chart info bar
        info_layout = QHBoxLayout()
        self.info_label = QLabel("HFT Chart - Ready")
        self.info_label.setFont(QFont("Consolas", 9))
        self.info_label.setStyleSheet("color: #888888; padding: 5px;")

        self.performance_label = QLabel("Performance: -")
        self.performance_label.setFont(QFont("Consolas", 9))
        self.performance_label.setStyleSheet("color: #888888; padding: 5px;")

        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        info_layout.addWidget(self.performance_label)

        layout.addLayout(info_layout)

        # Create high-performance plot widget with time axis
        axis = pg.DateAxisItem(orientation='bottom')
        self.plot_widget = pg.PlotWidget(
            title="HFT Price Chart - Bollinger Bands Strategy",
            labels={'left': 'Price (USDT)', 'bottom': 'Time'},
            axisItems={'bottom': axis}
        )

        # Configure plot for maximum performance
        self.plot_widget.setBackground('#2b2b2b')
        self.plot_widget.getAxis('left').setTextPen('#ffffff')
        self.plot_widget.getAxis('bottom').setTextPen('#ffffff')
        self.plot_widget.getAxis('left').setPen('#555555')
        self.plot_widget.getAxis('bottom').setPen('#555555')

        # Enable OpenGL for maximum performance
        self.plot_widget.setAntialiasing(False)  # Disable for performance

        layout.addWidget(self.plot_widget)

    def _setup_performance_optimizations(self):
        """Setup performance optimizations for large datasets"""
        # Enable OpenGL rendering for maximum speed
        try:
            pg.setConfigOptions(useOpenGL=True)
            pg.setConfigOptions(enableExperimental=True)
            pg.setConfigOptions(antialias=False)  # Disable for performance
        except:
            pass  # OpenGL might not be available

        # Configure plot widget optimizations
        self.plot_widget.setClipToView(True)  # Only render visible data
        self.plot_widget.setDownsampling(mode='peak')  # Smart downsampling

        # Disable automatic range updates for performance
        self.plot_widget.enableAutoRange(enable=False)

        # Set view limits for large datasets
        self.plot_widget.setLimits(maxXRange=self.max_display_points)

    def update_chart(self, results_data):
        """
        ГЛУБОЧАЙШИЙ DEBUG VERSION: Update chart with максимальными логами
        """
        print("\n" + "="*100)
        print("🔥 ГЛУБОЧАЙШИЙ DEBUG: update_chart() НАЧАТ!")
        print("="*100)

        # ГЛУБОЧАЙШИЙ ЛОГ 1: Проверка входных данных
        print(f"[DEBUG-001] results_data type: {type(results_data)}")
        print(f"[DEBUG-002] results_data is None: {results_data is None}")
        print(f"[DEBUG-003] results_data bool: {bool(results_data)}")

        if not results_data:
            print("❌ [DEBUG-004] КРИТИЧЕСКАЯ ОШИБКА: results_data пустая!")
            self.info_label.setText("NO DATA - results_data empty")
            return

        print(f"✅ [DEBUG-005] results_data содержит {len(results_data)} ключей")
        print(f"[DEBUG-006] results_data ключи: {list(results_data.keys())}")

        # ГЛУБОЧАЙШИЙ ЛОГ 2: Извлечение bb_data
        bb_data = results_data.get('bb_data')
        print(f"[DEBUG-007] bb_data извлечен: {bb_data is not None}")
        print(f"[DEBUG-008] bb_data type: {type(bb_data)}")

        if not bb_data:
            print("❌ [DEBUG-009] КРИТИЧЕСКАЯ ОШИБКА: bb_data отсутствует!")
            self.info_label.setText("NO BB DATA")
            return

        print(f"✅ [DEBUG-010] bb_data содержит {len(bb_data)} ключей")
        print(f"[DEBUG-011] bb_data ключи: {list(bb_data.keys())}")

        # ГЛУБОЧАЙШИЙ ЛОГ 3: Проверка обязательных ключей
        required_keys = ['times', 'prices']
        for key in required_keys:
            if key not in bb_data:
                print(f"❌ [DEBUG-012] КРИТИЧЕСКАЯ ОШИБКА: Отсутствует ключ '{key}'!")
                self.info_label.setText(f"MISSING KEY: {key}")
                return
            print(f"✅ [DEBUG-013] Ключ '{key}' найден")

        # ГЛУБОЧАЙШИЙ ЛОГ 4: Анализ данных времени и цен
        times_raw = bb_data['times']
        prices_raw = bb_data['prices']

        print(f"[DEBUG-014] times_raw type: {type(times_raw)}")
        print(f"[DEBUG-015] times_raw length: {len(times_raw)}")
        print(f"[DEBUG-016] times_raw first 3: {times_raw[:3] if len(times_raw) > 0 else 'EMPTY'}")

        print(f"[DEBUG-017] prices_raw type: {type(prices_raw)}")
        print(f"[DEBUG-018] prices_raw length: {len(prices_raw)}")
        print(f"[DEBUG-019] prices_raw first 3: {prices_raw[:3] if len(prices_raw) > 0 else 'EMPTY'}")

        if len(times_raw) == 0 or len(prices_raw) == 0:
            print("❌ [DEBUG-020] КРИТИЧЕСКАЯ ОШИБКА: Пустые массивы данных!")
            self.info_label.setText("EMPTY ARRAYS")
            return

        # ГЛУБОЧАЙШИЙ ЛОГ 5: Преобразование данных
        print(f"[DEBUG-021] Преобразование times в numpy array...")
        times_ms = np.array(times_raw, dtype=np.float64)
        print(f"[DEBUG-022] times_ms shape: {times_ms.shape}")
        print(f"[DEBUG-023] times_ms dtype: {times_ms.dtype}")
        print(f"[DEBUG-024] times_ms range: {times_ms.min():.0f} - {times_ms.max():.0f}")

        print(f"[DEBUG-025] Преобразование prices в numpy array...")
        prices = np.array(prices_raw, dtype=np.float32)
        print(f"[DEBUG-026] prices shape: {prices.shape}")
        print(f"[DEBUG-027] prices dtype: {prices.dtype}")
        print(f"[DEBUG-028] prices range: {prices.min():.6f} - {prices.max():.6f}")

        # ГЛУБОЧАЙШИЙ ЛОГ 6: Преобразование времени в секунды
        print(f"[DEBUG-029] Конвертация milliseconds -> seconds...")
        times_sec = times_ms / 1000.0
        print(f"[DEBUG-030] times_sec shape: {times_sec.shape}")
        print(f"[DEBUG-031] times_sec range: {times_sec.min():.6f} - {times_sec.max():.6f}")
        print(f"[DEBUG-032] times_sec first 3: {times_sec[:3]}")

        # ГЛУБОЧАЙШИЙ ЛОГ 7: Проверка на NaN/Inf
        times_nan_count = np.isnan(times_sec).sum()
        times_inf_count = np.isinf(times_sec).sum()
        prices_nan_count = np.isnan(prices).sum()
        prices_inf_count = np.isinf(prices).sum()

        print(f"[DEBUG-033] times_sec NaN: {times_nan_count}")
        print(f"[DEBUG-034] times_sec Inf: {times_inf_count}")
        print(f"[DEBUG-035] prices NaN: {prices_nan_count}")
        print(f"[DEBUG-036] prices Inf: {prices_inf_count}")

        if times_nan_count > 0 or times_inf_count > 0 or prices_nan_count > 0 or prices_inf_count > 0:
            print("⚠️  [DEBUG-037] ПРЕДУПРЕЖДЕНИЕ: Обнаружены недопустимые значения!")

        # ГЛУБОЧАЙШИЙ ЛОГ 8: Фильтрация валидных данных
        print(f"[DEBUG-038] Фильтрация валидных данных...")
        valid_mask = ~(np.isnan(prices) | np.isnan(times_sec))
        times_clean = times_sec[valid_mask]
        prices_clean = prices[valid_mask]

        print(f"[DEBUG-039] После фильтрации: {len(times_clean)} валидных точек из {len(times_sec)}")
        print(f"[DEBUG-040] times_clean range: {times_clean.min():.6f} - {times_clean.max():.6f}")
        print(f"[DEBUG-041] prices_clean range: {prices_clean.min():.6f} - {prices_clean.max():.6f}")

        if len(times_clean) == 0:
            print("❌ [DEBUG-042] КРИТИЧЕСКАЯ ОШИБКА: Нет валидных данных после фильтрации!")
            self.info_label.setText("NO VALID DATA")
            return

        # ГЛУБОЧАЙШИЙ ЛОГ 9: Очистка графика
        print(f"[DEBUG-043] Очистка plot_widget...")
        print(f"[DEBUG-044] plot_widget type: {type(self.plot_widget)}")
        print(f"[DEBUG-045] plot_widget до очистки items: {len(self.plot_widget.listDataItems())}")

        self.plot_widget.clear()

        print(f"[DEBUG-046] plot_widget после очистки items: {len(self.plot_widget.listDataItems())}")

        # ГЛУБОЧАЙШИЙ ЛОГ 10: Создание pen для линии цены
        print(f"[DEBUG-047] Создание pen для цены...")
        try:
            price_pen = pg.mkPen(color='#00aaff', width=2)  # Увеличиваем толщину для лучшей видимости
            print(f"[DEBUG-048] price_pen создан: {price_pen}")
            print(f"[DEBUG-049] price_pen color: {price_pen.color().name()}")
            print(f"[DEBUG-050] price_pen width: {price_pen.width()}")
        except Exception as e:
            print(f"❌ [DEBUG-051] ОШИБКА создания pen: {e}")
            return

        # 🔥 ГЛУБОЧАЙШИЙ ЛОГ 11: КРИТИЧЕСКИЙ МОМЕНТ - СОЗДАНИЕ ЛИНИИ ЦЕНЫ
        print(f"\n🔥 [DEBUG-052] КРИТИЧЕСКИЙ МОМЕНТ: Создание линии цены...")
        print(f"[DEBUG-053] Параметры для plot():")
        print(f"[DEBUG-054]   x (times_clean): shape={times_clean.shape}, dtype={times_clean.dtype}")
        print(f"[DEBUG-055]   y (prices_clean): shape={prices_clean.shape}, dtype={prices_clean.dtype}")
        print(f"[DEBUG-056]   pen: {price_pen}")
        print(f"[DEBUG-057]   name: 'Price'")
        print(f"[DEBUG-058]   antialias: False")

        try:
            print(f"[DEBUG-059] Вызов plot_widget.plot()...")

            # МАКСИМАЛЬНО ПОДРОБНЫЙ ВЫЗОВ
            self.price_curve = self.plot_widget.plot(
                times_clean,
                prices_clean,
                pen=price_pen,
                name='DEEP_DEBUG_Price',
                antialias=False,
                connect='all',
                stepMode=None,
                fillLevel=None,
                brush=None
            )

            print(f"✅ [DEBUG-060] plot() вернул: {self.price_curve}")
            print(f"[DEBUG-061] price_curve type: {type(self.price_curve)}")
            print(f"[DEBUG-062] price_curve имеет xData: {hasattr(self.price_curve, 'xData')}")
            print(f"[DEBUG-063] price_curve имеет yData: {hasattr(self.price_curve, 'yData')}")

            if hasattr(self.price_curve, 'xData') and hasattr(self.price_curve, 'yData'):
                x_data = self.price_curve.xData
                y_data = self.price_curve.yData
                print(f"[DEBUG-064] price_curve.xData: {x_data[:3] if x_data is not None and len(x_data) > 0 else 'None/Empty'}")
                print(f"[DEBUG-065] price_curve.yData: {y_data[:3] if y_data is not None and len(y_data) > 0 else 'None/Empty'}")
                print(f"[DEBUG-066] xData length: {len(x_data) if x_data is not None else 'None'}")
                print(f"[DEBUG-067] yData length: {len(y_data) if y_data is not None else 'None'}")

        except Exception as e:
            print(f"❌ [DEBUG-068] КРИТИЧЕСКАЯ ОШИБКА при plot(): {e}")
            import traceback
            traceback.print_exc()
            return

        # ГЛУБОЧАЙШИЙ ЛОГ 12: Проверка состояния графика после добавления линии
        print(f"\n[DEBUG-069] Проверка состояния графика после добавления цены...")
        current_items = self.plot_widget.listDataItems()
        print(f"[DEBUG-070] Количество элементов в графике: {len(current_items)}")

        for i, item in enumerate(current_items):
            print(f"[DEBUG-071] Элемент {i}: {type(item).__name__}")
            if hasattr(item, 'name'):
                print(f"[DEBUG-072]   name: {item.name}")
            if hasattr(item, 'xData') and hasattr(item, 'yData'):
                x_data = item.xData
                y_data = item.yData
                x_len = len(x_data) if x_data is not None else 0
                y_len = len(y_data) if y_data is not None else 0
                print(f"[DEBUG-073]   data: x={x_len} points, y={y_len} points")

        if len(current_items) == 0:
            print("❌ [DEBUG-074] КРИТИЧЕСКАЯ ОШИБКА: График пустой после добавления цены!")
        else:
            print(f"✅ [DEBUG-075] График содержит {len(current_items)} элементов")

        # ГЛУБОЧАЙШИЙ ЛОГ 13: Установка диапазонов
        print(f"\n[DEBUG-076] Установка диапазонов графика...")
        print(f"[DEBUG-077] X диапазон: {times_clean.min():.6f} - {times_clean.max():.6f}")
        print(f"[DEBUG-078] Y диапазон: {prices_clean.min():.6f} - {prices_clean.max():.6f}")

        try:
            print(f"[DEBUG-079] Вызов setXRange()...")
            self.plot_widget.setXRange(times_clean.min(), times_clean.max(), padding=0.05)
            print(f"[DEBUG-080] setXRange() выполнен")

            print(f"[DEBUG-081] Вызов setYRange()...")
            self.plot_widget.setYRange(prices_clean.min(), prices_clean.max(), padding=0.05)
            print(f"[DEBUG-082] setYRange() выполнен")

            print(f"[DEBUG-083] Вызов autoRange()...")
            self.plot_widget.autoRange()
            print(f"[DEBUG-084] autoRange() выполнен")

        except Exception as e:
            print(f"❌ [DEBUG-085] ОШИБКА установки диапазонов: {e}")

        # ГЛУБОЧАЙШИЙ ЛОГ 14: Финальная проверка состояния
        print(f"\n[DEBUG-086] ФИНАЛЬНАЯ ПРОВЕРКА...")
        final_items = self.plot_widget.listDataItems()
        print(f"[DEBUG-087] Финальное количество элементов: {len(final_items)}")

        # Проверка ViewBox
        view_box = self.plot_widget.getViewBox()
        if view_box:
            print(f"[DEBUG-088] ViewBox получен: {type(view_box)}")
            try:
                state = view_box.getState()
                print(f"[DEBUG-089] ViewBox state получен")
                print(f"[DEBUG-090] viewRange: {state.get('viewRange', 'N/A')}")
                print(f"[DEBUG-091] targetRange: {state.get('targetRange', 'N/A')}")
                print(f"[DEBUG-092] autoRange: {state.get('autoRange', 'N/A')}")
            except Exception as e:
                print(f"[DEBUG-093] Ошибка получения ViewBox state: {e}")
        else:
            print(f"❌ [DEBUG-094] ViewBox не найден!")

        # Информация для пользователя
        self.info_label.setText(f"DEEP DEBUG: {len(final_items)} items, {len(times_clean)} points")

        print(f"\n" + "="*100)
        print(f"🔥 ГЛУБОЧАЙШИЙ DEBUG: update_chart() ЗАВЕРШЕН!")
        print(f"ИТОГ: {len(final_items)} элементов в графике из {len(times_clean)} точек данных")
        print("="*100)

        return len(final_items) > 0

    def clear(self):
        """Clear chart data efficiently"""
        if self.plot_widget:
            self.plot_widget.clear()
            self.info_label.setText("Chart cleared")

    def get_widget(self):
        """Get the widget for integration"""
        return self