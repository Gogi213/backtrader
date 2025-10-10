import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import numpy as np
from signal_generator import generate_signals, calculate_natr
from trading_simulator import run_trading_simulation, TradingSimulator
import optuna_optimizer as wfo_optuna # Импортируем модуль оптимизации
import strategy_objectives # Импортируем модуль с дополнительными целевыми функциями
# Настройка страницы
st.set_page_config(
    page_title="Streamlit Backtester",
    page_icon="📈",
    layout="wide"
)

# Создание директорий для хранения профилей и прогонов
os.makedirs("profiles", exist_ok=True)
os.makedirs("optimization_runs", exist_ok=True)

# Заголовок приложения
st.title("📈 Streamlit Backtester")

# Навигация
st.sidebar.header("Навигация")

# Создаем три кнопки в одной строке для навигации с подсветкой активной
col1, col2, col3 = st.sidebar.columns(3)

# Устанавливаем текущую страницу из session_state или по умолчанию
current_page = st.session_state.get("page", "Аналитика")

with col1:
    # Определяем, является ли эта кнопка активной
    is_active = current_page == "Анализ сигналов"
    btn_type = "secondary" if not is_active else "primary"
    if st.button("Анализ", key="nav_analyze", use_container_width=True, type=btn_type):
        st.session_state.page = "Анализ сигналов"
        st.rerun()

with col2:
    # Определяем, является ли эта кнопка активной
    is_active = current_page == "Оптимизация"
    btn_type = "secondary" if not is_active else "primary"
    if st.button("Оптимизация", key="nav_optimize", use_container_width=True, type=btn_type):
        st.session_state.page = "Оптимизация"
        st.rerun()

with col3:
    # Определяем, является ли эта кнопка активной
    is_active = current_page == "Аналитика"
    btn_type = "secondary" if not is_active else "primary"
    if st.button("Аналитика", key="nav_analytics", use_container_width=True, type=btn_type):
        st.session_state.page = "Аналитика"
        st.rerun()

# Функции для работы с профилями
def get_profile_directory(module):
    """Получить директорию для профилей по модулю"""
    # Проверяем, что модуль один из допустимых
    if module not in ["analysis", "optimization"]:
        raise ValueError(f"Неподдерживаемый модуль: {module}")
    return f"profiles/{module}"

def save_profile(profile_name, data, module):
    """Сохранить профиль в JSON-файл"""
    # Валидация входных данных
    if not profile_name or not isinstance(profile_name, str):
        st.error("Название профиля должно быть непустой строкой")
        return False
    
    if not data or not isinstance(data, dict):
        st.error("Данные профиля должны быть непустым словарем")
        return False
    
    try:
        directory = get_profile_directory(module)
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{profile_name}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        st.success(f"Профиль '{profile_name}' сохранён!")
        return True
    except Exception as e:
        st.error(f"Ошибка при сохранении профиля: {str(e)}")
        return False

@st.cache_data(ttl=300)  # Кэшируем на 5 минут
def load_profile(profile_name, module):
    """Загрузить профиль из JSON-файла"""
    # Валидация входных данных
    if not profile_name or not isinstance(profile_name, str):
        st.error("Название профиля должно быть непустой строкой")
        return None
    
    try:
        directory = get_profile_directory(module)
        file_path = os.path.join(directory, f"{profile_name}.json")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Проверяем, что загруженный файл содержит допустимые данные
        if not isinstance(data, dict):
            st.error(f"Файл профиля '{profile_name}' содержит некорректные данные")
            return None
            
        return data
    except FileNotFoundError:
        st.error(f"Профиль '{profile_name}' не найден!")
        return None
    except json.JSONDecodeError:
        st.error(f"Файл профиля '{profile_name}' содержит некорректный JSON")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке профиля: {str(e)}")
        return None

@st.cache_data(ttl=10)  # Кэшируем на 60 секунд
def get_profiles(module):
    """Получить список доступных профилей"""
    try:
        directory = get_profile_directory(module)
        profiles = [f.replace('.json', '') for f in os.listdir(directory) if f.endswith('.json')]
        return profiles
    except FileNotFoundError:
        # Директория может не существовать, если нет профилей
        return []
    except Exception as e:
        st.error(f"Ошибка при получении списка профилей: {str(e)}")
        return []

# Функция для загрузки профиля в session_state
def load_profile_to_session_state(profile_data, module):
    """Загрузить данные профиля в session_state для указанного модуля"""
    # Валидация входных данных
    if not profile_data or not isinstance(profile_data, dict):
        st.error("Данные профиля должны быть непустым словарем")
        return False
    
    try:
        # Обработка выбранных файлов
        if 'selected_files' in profile_data:
            selected_files_from_profile = profile_data['selected_files']
            
            # Получаем список всех доступных CSV-файлов
            all_csv_files = []
            try:
                all_csv_files = [f for f in os.listdir("dataCSV") if f.endswith(".csv")]
            except FileNotFoundError:
                st.warning("Папка dataCSV не найдена, не удалось восстановить выбор файлов.")

            # Устанавливаем состояние каждого чекбокса
            for csv_file in all_csv_files:
                is_selected = csv_file in selected_files_from_profile
                st.session_state[f"csv_{module}_{csv_file}"] = is_selected

        for key, value in profile_data.items():
            # Проверяем, что ключ - строка
            if not isinstance(key, str):
                st.warning(f"Пропуск недопустимого ключа: {key}")
                continue
            
            # Преобразуем строку даты в datetime.date, если это дата
            if key in ["start_date", "end_date"] and isinstance(value, str):
                try:
                    # Преобразуем строку в формат datetime.date
                    date_obj = datetime.strptime(value.split()[0], "%Y-%m-%d").date()
                    st.session_state[f"{key}_{module}"] = date_obj
                except ValueError:
                    # Если не удается преобразовать, оставляем как есть
                    st.session_state[f"{key}_{module}"] = value
            # Обработка специального случая для enable_additional_filters
            elif key == "enable_additional_filters" and isinstance(value, bool):
                st.session_state[f"enable_additional_filters_{module}"] = value
            # Обработка параметров enable_additional_filters_min и enable_additional_filters_max для оптимизации
            elif key == "enable_additional_filters_min" and isinstance(value, bool):
                # Определяем значение для radio-кнопки на основе min и max значений
                max_value = profile_data.get("enable_additional_filters_max", not value)  # используем соответствующее max значение
                if value == False and max_value == False:
                    st.session_state["enable_additional_filters_option"] = "False"
                elif value == True and max_value == True:
                    st.session_state["enable_additional_filters_option"] = "True"
                else:  # value == False and max_value == True (или другие комбинации, которые означают "Both")
                    st.session_state["enable_additional_filters_option"] = "Both"
            # Обработка параметров для анализа принтов
            elif key in ["prints_threshold_ratio", "prints_analysis_period",
                         "prints_threshold_ratio_min", "prints_threshold_ratio_max"]:
                st.session_state[f"{key}_{module}"] = value
            # Обработка параметров hldir_window
            elif key in ["hldir_window", "hldir_window_min", "hldir_window_max", "use_hldir"]:
                st.session_state[f"{key}_{module}"] = value
            # Обработка параметра enable_additional_filters_max (уже обработан в предыдущем условии, но добавим для полноты)
            elif key == "enable_additional_filters_max":
                # Этот параметр уже обработан в предыдущем условии при обработке min
                pass
            # Обработка aggressive_mode
            elif key == "aggressive_mode":
                pass
            # Обработка параметров enable_additional_filters_min и enable_additional_filters_max для сохранения в session_state
            elif key in ["enable_additional_filters_min", "enable_additional_filters_max", "entry_logic_mode", "hldir_offset_min", "hldir_offset_max"]:
                # Эти параметры обрабатываются специальным образом или передаются как есть
                st.session_state[f"{key}_optimization"] = value
            else:
                st.session_state[f"{key}_{module}"] = value
             
        return True
    except Exception as e:
        st.error(f"Ошибка при загрузке профиля в session_state: {str(e)}")
        return False

# Функция для обработки одного файла данных
def process_single_file(file, module=None):
    """Обработка одного CSV-файла"""
    file_path = os.path.join("dataCSV", file)
    with open(file_path, 'rb') as f:
        uploaded_file = f
        
        try:
            # Парсинг CSV-файла с учетом заголовков
            df = pd.read_csv(uploaded_file, header=0)
            
            # Проверяем, что файл имеет ожидаемые столбцы
            expected_columns = ['Symbol', 'time', 'open', 'high', 'low', 'close', 'Volume', 'HLdir', 'long_prints', 'short_prints']
            if not all(col in df.columns for col in expected_columns):
                st.error(f"Файл {file} не содержит ожидаемые столбцы. Найдены: {list(df.columns)}")
                return None
            
            # Проверяем, что все столбцы содержат допустимые значения
            required_numeric_cols = ['time', 'open', 'high', 'low', 'close', 'Volume']
            
            # Проверяем, что столбцы содержат только числовые значения, используя векторизованные операции
            numeric_series_dict = {}
            for col in required_numeric_cols:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.notna().all():
                    # Находим индексы строк с некорректными значениями
                    invalid_indices = df[numeric_series.isna()].index.tolist()
                    st.error(f"Файл {file} содержит некорректные данные в столбце {col} на строках: {invalid_indices[:5]}{' и др.' if len(invalid_indices) > 5 else ''}")
                    return None
                numeric_series_dict[col] = numeric_series
            
            # Проверяем, что числовые значения находятся в разумных пределах
            # Проверяем, что цены и объемы положительные, используя векторизованные операции
            positive_cols = ['open', 'high', 'low', 'close', 'Volume']
            for col in positive_cols:
                if (df[col] <= 0).any():
                    invalid_indices = df[df[col] <= 0].index.tolist()
                    st.error(f"Файл {file} содержит недопустимые значения (≤ 0) в столбце {col} на строках: {invalid_indices[:5]}{' и др.' if len(invalid_indices) > 5 else ''}")
                    return None
            
            # Переименовываем столбцы в соответствии с ожиданиями остальной части кода
            df = df.rename(columns={'Volume': 'volume'})
            
            # Убедимся, что все необходимые столбцы присутствуют в DataFrame
            required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    st.error(f"Файл {file} не содержит обязательный столбец: {col}")
                    return None
            
            # Если в остальном коде ожидается больше столбцов, добавляем их с пустыми значениями
            additional_columns = ['close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            for col in additional_columns:
                if col not in df.columns:
                    df[col] = 0  # или другое подходящее значение по умолчанию
            
            # Если в остальном коде ожидается столбец 'datetime', создаем его из 'time'
            if 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
            
            # Упорядочиваем столбцы в ожидаемом порядке
            expected_order = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                             'taker_buy_quote_asset_volume', 'ignore', 'Symbol', 'HLdir', 'long_prints', 'short_prints', 'datetime']
            df = df.reindex(columns=[col for col in expected_order if col in df.columns])
            
            pass
            
            # Если все проверки пройдены, возвращаем DataFrame
            return df
            
        except pd.errors.EmptyDataError:
            st.error(f"Файл {file} пустой")
        except pd.errors.ParserError:
            st.error(f"Файл {file} содержит ошибки формата")
        except Exception as e:
            st.error(f"Ошибка при загрузке файла {file}: {str(e)}")
    
    return None

# Функция для загрузки и валидации CSV-файлов
def load_and_validate_csv_files(selected_files, module=None):
    """Загрузка и валидация CSV-файлов"""
    dataframes = []
    for file in selected_files:
        df = process_single_file(file, module)
        if df is not None:
            dataframes.append(df)
    return dataframes
# Универсальная функция для генерации параметров (как для анализа, так и для оптимизации)
def get_common_parameters(module, is_optimization=False):
    """
    Универсальная функция для получения параметров стратегии или диапазонов оптимизации
    
    Args:
        module: строка модуля ('analysis' или 'optimization')
        is_optimization: флаг, указывающий, используется ли функция для оптимизации (по умолчанию False)
    """
    # Определяем суффиксы для ключей в зависимости от режима
    if is_optimization:
        min_suffix = "_min"
        max_suffix = "_max"
        min_label = " (min)"
        max_label = " (max)"
    else:
        min_suffix = ""
        max_suffix = ""
        min_label = ""
        max_label = ""

    # Компактное отображение параметров в сайдбаре - каждая группа в одной строке
    st.markdown("**📊 Фильтр объёма**")
    col1, col2 = st.columns(2)
    with col1:
        if is_optimization:
            vol_pctl_min = st.number_input(f"vol_pctl (%) (min)", value=float(st.session_state.get("vol_pctl_min_optimization", 0.5)), min_value=0.01, step=0.01, key="vol_pctl_min_optimization")
            vol_pctl = vol_pctl_min
        else:
            vol_pctl = st.number_input("vol_pctl (%)", value=float(st.session_state.get(f"vol_pctl_{module}", 1.0)), min_value=0.01, step=0.01, key=f"vol_pctl_{module}")
    with col2:
        if is_optimization:
            vol_period_min = st.number_input("vol_period (min)", value=int(st.session_state.get("vol_period_min_optimization", 10)), min_value=1, step=1, key="vol_period_min_optimization")
            vol_period = vol_period_min
        else:
            vol_period = st.number_input("vol_period", value=int(st.session_state.get(f"vol_period_{module}", 20)), min_value=1, step=1, key=f"vol_period_{module}")

    # Если оптимизация, добавляем max значения
    if is_optimization:
        col1, col2 = st.columns(2)
        with col1:
            vol_pctl_max = st.number_input(f"vol_pctl (%) (max)", value=float(st.session_state.get("vol_pctl_max_optimization", 2.0)), min_value=0.01, step=0.01, key="vol_pctl_max_optimization")
        with col2:
            vol_period_max = st.number_input("vol_period (max)", value=int(st.session_state.get("vol_period_max_optimization", 30)), min_value=1, step=1, key="vol_period_max_optimization")

    st.markdown("**📏 Фильтр диапазона**")
    col1, col2 = st.columns(2)
    with col1:
        if is_optimization:
            rng_pctl_min = st.number_input(f"rng_pctl (%) (min)", value=float(st.session_state.get("rng_pctl_min_optimization", 0.5)), min_value=0.01, step=0.01, key="rng_pctl_min_optimization")
            rng_pctl = rng_pctl_min
        else:
            rng_pctl = st.number_input("rng_pctl (%)", value=float(st.session_state.get(f"rng_pctl_{module}", 1.0)), min_value=0.01, step=0.01, key=f"rng_pctl_{module}")
    with col2:
        if is_optimization:
            range_period_min = st.number_input("range_period (min)", value=int(st.session_state.get("range_period_min_optimization", 10)), min_value=1, step=1, key="range_period_min_optimization")
            range_period = range_period_min
        else:
            range_period = st.number_input("range_period", value=int(st.session_state.get(f"range_period_{module}", 20)), min_value=1, step=1, key=f"range_period_{module}")

    if is_optimization:
        col1, col2 = st.columns(2)
        with col1:
            rng_pctl_max = st.number_input(f"rng_pctl (%) (max)", value=float(st.session_state.get("rng_pctl_max_optimization", 2.0)), min_value=0.01, step=0.01, key="rng_pctl_max_optimization")
        with col2:
            range_period_max = st.number_input("range_period (max)", value=int(st.session_state.get("range_period_max_optimization", 30)), min_value=1, step=1, key="range_period_max_optimization")

    st.markdown("**📉 Фильтр NATR**")
    col1, col2 = st.columns(2)
    with col1:
        if is_optimization:
            natr_min_min = st.number_input(f"natr_min (%) (min)", value=float(st.session_state.get("natr_min_min_optimization", 0.2)), min_value=0.01, step=0.01, key="natr_min_min_optimization")
            natr_min = natr_min_min
        else:
            natr_min = st.number_input("natr_min (%)", value=float(st.session_state.get(f"natr_min_{module}", 0.35)), min_value=0.01, step=0.01, key=f"natr_min_{module}")
    with col2:
        if is_optimization:
            natr_period_min = st.number_input("natr_period (min)", value=int(st.session_state.get("natr_period_min_optimization", 5)), min_value=1, step=1, key="natr_period_min_optimization")
            natr_period = natr_period_min
        else:
            natr_period = st.number_input("natr_period", value=int(st.session_state.get(f"natr_period_{module}", 10)), min_value=1, step=1, key=f"natr_period_{module}")

    if is_optimization:
        col1, col2 = st.columns(2)
        with col1:
            natr_min_max = st.number_input(f"natr_min (%) (max)", value=float(st.session_state.get("natr_min_max_optimization", 0.8)), min_value=0.01, step=0.01, key="natr_min_max_optimization")
        with col2:
            natr_period_max = st.number_input("natr_period (max)", value=int(st.session_state.get("natr_period_max_optimization", 20)), min_value=1, step=1, key="natr_period_max_optimization")

    st.markdown("**📈 Фильтр роста**")
    col1, col2 = st.columns(2)
    with col1:
        if is_optimization:
            min_growth_pct_min = st.number_input(f"min_growth_pct (%) (min)", value=st.session_state.get("min_growth_pct_min_optimization", 0.5), min_value=-100.0, max_value=100.0, step=0.01, key="min_growth_pct_min_optimization")
            min_growth_pct = min_growth_pct_min
        else:
            min_growth_pct = st.number_input("min_growth_pct (%)", value=st.session_state.get(f"min_growth_pct_{module}", 1.0), min_value=-100.0, max_value=100.0, step=0.01, key=f"min_growth_pct_{module}")
    with col2:
        if is_optimization:
            lookback_period_min = st.number_input("lookback_period (min)", value=int(st.session_state.get("lookback_period_min_optimization", 10)), min_value=1, step=1, key="lookback_period_min_optimization")
            lookback_period = lookback_period_min
        else:
            lookback_period = st.number_input("lookback_period", value=int(st.session_state.get(f"lookback_period_{module}", 20)), min_value=1, step=1, key=f"lookback_period_{module}")

    if is_optimization:
        col1, col2 = st.columns(2)
        with col1:
            min_growth_pct_max = st.number_input(f"min_growth_pct (%) (max)", value=st.session_state.get("min_growth_pct_max_optimization", 2.0), min_value=-100.0, max_value=100.0, step=0.01, key="min_growth_pct_max_optimization")
        with col2:
            lookback_period_max = st.number_input("lookback_period (max)", value=st.session_state.get("lookback_period_max_optimization", 30), min_value=1, step=1, key="lookback_period_max_optimization")

    st.markdown("**🖨️ Фильтр принтов**")
    col1, col2 = st.columns(2)
    with col1:
        if is_optimization:
            # Убираем выбор режима анализа принтов, оставляем только режим 1 (фиксированное соотношение)
            prints_analysis_mode_min = 1  # Фиксируем режим 1
            prints_analysis_mode = prints_analysis_mode_min
        else:
            # Убираем выбор режима анализа принтов, оставляем только режим 1 (фиксированное соотношение)
            prints_analysis_mode = 1  # Фиксируем режим 1
    with col2:
        if is_optimization:
            # Убираем max значение для режима анализа принтов, фиксируем режим 1
            prints_analysis_mode_max = 1  # Фиксируем режим 1
        else:
            prints_threshold_ratio = st.number_input("prints_threshold_ratio", value=st.session_state.get(f"prints_threshold_ratio_{module}", 1.0), min_value=0.01, step=0.01, key=f"prints_threshold_ratio_{module}_analysis")

    # Инициализируем переменные для оптимизации, чтобы избежать ошибки UnboundLocalError
    if is_optimization:
        # Инициализируем все возможные переменные
        prints_threshold_ratio_min = st.session_state.get("prints_threshold_ratio_min_optimization", 0.5)
        prints_threshold_ratio_max = st.session_state.get("prints_threshold_ratio_max_optimization", 2.0)
        prints_analysis_period_min = int(st.session_state.get("prints_analysis_period_min_optimization", 1))
        prints_analysis_period_max = int(st.session_state.get("prints_analysis_period_max_optimization", 5))
        
        # Обновляем значения для режима 1 (фиксированное соотношение)
        col1, col2 = st.columns(2)
        with col1:
            prints_threshold_ratio_min = st.number_input("prints_threshold_ratio (min)", value=float(st.session_state.get("prints_threshold_ratio_min_optimization", 0.5)), min_value=0.01, step=0.01, key="prints_threshold_ratio_min_optimization")
            prints_analysis_period_min = st.number_input("prints_analysis_period (min)", value=int(st.session_state.get("prints_analysis_period_min_optimization", 1)), min_value=1, step=1, key="prints_analysis_period_min_optimization")
        with col2:
            prints_threshold_ratio_max = st.number_input("prints_threshold_ratio (max)", value=float(st.session_state.get("prints_threshold_ratio_max_optimization", 2.0)), min_value=0.01, step=0.01, key="prints_threshold_ratio_max_optimization")
            prints_analysis_period_max = st.number_input("prints_analysis_period (max)", value=int(st.session_state.get("prints_analysis_period_max_optimization", 5)), min_value=1, step=1, key="prints_analysis_period_max_optimization")
    else:
        # Для режима анализа
        prints_analysis_period = st.number_input("prints_analysis_period", value=int(st.session_state.get(f"prints_analysis_period_{module}", 2)), min_value=1, step=1, key=f"prints_analysis_period_{module}")

    # Дополнительные параметры
    if is_optimization:
        st.markdown("**🛡️ Управление риском**")
    else:
        st.markdown("**🔧 Дополнительные параметры**")
    
    col1, col2 = st.columns(2)
    with col1:
        if is_optimization:
            stop_loss_pct_min = st.number_input("stop_loss_pct (%) (min)", value=float(st.session_state.get("stop_loss_pct_min_optimization", 1.0)), min_value=0.01, step=0.01, format="%.2f", key="stop_loss_pct_min_optimization")
            stop_loss_pct = stop_loss_pct_min
        else:
            stop_loss_pct = st.number_input("stop_loss_pct (%)", value=float(st.session_state.get(f"stop_loss_pct_{module}", 2.0)), min_value=0.01, step=0.01, format="%.2f", key=f"stop_loss_pct_{module}")
    with col2:
        if is_optimization:
            take_profit_pct_min = st.number_input("take_profit_pct (%) (min)", value=float(st.session_state.get("take_profit_pct_min_optimization", 1.0)), min_value=0.01, step=0.01, format="%.2f", key="take_profit_pct_min_optimization")
            take_profit_pct = take_profit_pct_min
        else:
            take_profit_pct = st.number_input("take_profit_pct (%)", value=float(st.session_state.get(f"take_profit_pct_{module}", 4.0)), min_value=0.01, step=0.01, format="%.2f", key=f"take_profit_pct_{module}")

    if is_optimization:
        col1, col2 = st.columns(2)
        with col1:
            stop_loss_pct_max = st.number_input("stop_loss_pct (%) (max)", value=float(st.session_state.get("stop_loss_pct_max_optimization", 5.0)), min_value=0.01, step=0.01, format="%.2f", key="stop_loss_pct_max_optimization")
        with col2:
            take_profit_pct_max = st.number_input("take_profit_pct (%) (max)", value=float(st.session_state.get("take_profit_pct_max_optimization", 8.0)), min_value=0.01, step=0.01, format="%.2f", key="take_profit_pct_max_optimization")

    # Параметры для дополнительных фильтров
    if is_optimization:
        # Логика для enable_additional_filters теперь полностью определяется диапазонами в get_optimization_parameters
        # Этот параметр будет оптимизироваться по умолчанию в диапазоне [False, True]
        pass
    else:
        enable_additional_filters = st.checkbox(
            "Включить дополнительные фильтры",
            value=st.session_state.get(f"enable_additional_filters_{module}", False),
            key=f"enable_additional_filters_{module}"
        )

    # Формируем словарь параметров в зависимости от режима
    if is_optimization:
        # Для оптимизации возвращаем параметры в формате диапазонов
        params = {
            "vol_period_min": vol_period_min,
            "vol_period_max": vol_period_max,
            "vol_pctl_min": vol_pctl_min,
            "vol_pctl_max": vol_pctl_max,
            "range_period_min": range_period_min,
            "range_period_max": range_period_max,
            "rng_pctl_min": rng_pctl_min,
            "rng_pctl_max": rng_pctl_max,
            "natr_period_min": natr_period_min,
            "natr_period_max": natr_period_max,
            "natr_min_min": natr_min_min,
            "natr_min_max": natr_min_max,
            "lookback_period_min": lookback_period_min,
            "lookback_period_max": lookback_period_max,
            "min_growth_pct_min": min_growth_pct_min,
            "min_growth_pct_max": min_growth_pct_max,
            "prints_analysis_period_min": prints_analysis_period_min,
            "prints_analysis_period_max": prints_analysis_period_max,
            "stop_loss_pct_min": stop_loss_pct_min,
            "stop_loss_pct_max": stop_loss_pct_max,
            "take_profit_pct_min": take_profit_pct_min,
            "take_profit_pct_max": take_profit_pct_max
        }

        # Добавляем параметры hldir_window для оптимизации
        # Убираем из этой функции, так как элементы управления будут в основном интерфейсе
        if is_optimization:
            # Для оптимизации добавляем параметры из session_state
            params["hldir_window_min"] = int(st.session_state.get("hldir_window_min_optimization", 3))
            params["hldir_window_max"] = int(st.session_state.get("hldir_window_max_optimization", 10))
        else:
            # Для анализа добавляем только один параметр hldir_window
            params["hldir_window"] = int(st.session_state.get("hldir_window_analysis", 5))

        # Добавляем параметры для режима 1 (фиксированное соотношение)
        params["prints_threshold_ratio_min"] = prints_threshold_ratio_min
        params["prints_threshold_ratio_max"] = prints_threshold_ratio_max
        params["prints_analysis_period_min"] = prints_analysis_period_min
        params["prints_analysis_period_max"] = prints_analysis_period_max
    else:
        # Для анализа возвращаем параметры в обычном формате
        params = {
            "vol_period": vol_period,
            "vol_pctl": vol_pctl,
            "range_period": range_period,
            "rng_pctl": rng_pctl,
            "natr_period": natr_period,
            "natr_min": natr_min,
            "lookback_period": lookback_period,
            "min_growth_pct": min_growth_pct,
            "prints_analysis_period": prints_analysis_period,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "enable_additional_filters": enable_additional_filters
        }
        
        # Добавляем параметр hldir_window для анализа
        entry_logic_mode = st.session_state.get(f"entry_logic_mode_{module}", "Принты и HLdir")
        if "HLdir" in entry_logic_mode:
            params["hldir_window"] = st.session_state.get(f"hldir_window_{module}", 10)
        
        # Добавляем параметры для режима анализа принтов (только режим 1 - фиксированное соотношение)
        params["prints_threshold_ratio"] = prints_threshold_ratio
        params["prints_analysis_period"] = prints_analysis_period

    # Проверяем, что все значения в допустимом диапазоне
    for param_name, param_value in params.items():
        if param_value is None:
            st.warning(f"Параметр {param_name} не определен")

    # Для оптимизации проверяем, что min не больше max
    if is_optimization:
        param_items = list(params.items())
        for i in range(0, len(param_items), 2):
            if i+1 < len(param_items):
                min_param, min_value = param_items[i]
                max_param, max_value = param_items[i+1]
                
                # Для парных параметров min/max проверяем корректность
                if min_param.endswith('_min') and max_param.endswith('_max'):
                    # Для булевых значений: если min=True и max=False, это ошибка
                    # Но если min=False и max=True, это нормально (означает, что нужно оптимизировать оба варианта)
                    if isinstance(min_value, bool) and isinstance(max_value, bool):
                        if min_value and not max_value:
                            # Это нормальный случай для enable_additional_filters - означает, что нужно оптимизировать оба варианта
                            if min_param != "enable_additional_filters_min":
                                st.warning(f"Диапазон параметра некорректен: {min_param} ({min_value}) > {max_param} ({max_value})")
                # Для числовых значений проверяем стандартное условие
                elif min_value > max_value:
                    st.warning(f"Диапазон параметра некорректен: {min_param} ({min_value}) > {max_param} ({max_value})")

    return params

# Функция для генерации параметров стратегии (теперь использует универсальную функцию)
def get_strategy_parameters(module):
    """Получить параметры стратегии для указанного модуля"""
    return get_common_parameters(module, is_optimization=False)

# Функция для генерации параметров оптимизации (теперь использует универсальную функцию)
def get_optimization_parameters():
    """Получить параметры оптимизации (диапазоны) для модуля оптимизации"""
    return get_common_parameters("optimization", is_optimization=True)

# Функция для получения базовых настроек (для обоих модулей)
def get_basic_settings(module):
    """Получить базовые настройки для модуля"""
    col1, col2 = st.columns(2)
    with col1:
        # Инициализируем значение в session_state, если его нет
        if f"position_size_{module}" not in st.session_state:
            st.session_state[f"position_size_{module}"] = 100.0
        # Создаем виджет с явным указанием значения из session_state
        position_size = st.number_input("Размер позиции", value=round(float(st.session_state[f"position_size_{module}"]), 2), step=10.0, key=f"position_size_{module}")
    with col2:
        # Инициализируем значение в session_state, если его нет
        if f"commission_{module}" not in st.session_state:
            st.session_state[f"commission_{module}"] = 0.1
        # Создаем виджет с явным указанием значения из session_state
        commission = st.number_input("Комиссия (%)", value=round(float(st.session_state[f"commission_{module}"]), 3), step=0.01, key=f"commission_{module}")
    col1, col2 = st.columns(2)
    with col1:
        # Инициализируем значение в session_state, если его нет
        if f"start_date_{module}" not in st.session_state:
            st.session_state[f"start_date_{module}"] = datetime(2025, 9, 1).date()
        # Создаем виджет с явным указанием значения из session_state
        start_date = st.date_input("Дата начала", value=st.session_state[f"start_date_{module}"], key=f"start_date_{module}")
    with col2:
        # Инициализируем значение в session_state, если его нет
        if f"end_date_{module}" not in st.session_state:
            st.session_state[f"end_date_{module}"] = datetime(2025, 9, 26).date()
        # Создаем виджет с явным указанием значения из session_state
        end_date = st.date_input("Дата окончания", value=st.session_state[f"end_date_{module}"], key=f"end_date_{module}")
    
    # Добавляем переключатель для агрессивного режима
    aggressive_mode = st.checkbox(
        "Агрессивный режим",
        value=st.session_state.get(f"aggressive_mode_{module}", False),
        key=f"aggressive_mode_{module}",
        help="Позволяет открывать новую сделку, не дожидаясь закрытия предыдущей, если появляется новый сигнал."
    )
    return position_size, commission, start_date, end_date

# Функция для управления профилями (для обоих модулей)
def manage_profiles(module, params_func):
    """Управление профилями для указанного модуля"""
    st.subheader("Профили")
    col1, col2 = st.columns(2)
    with col1:
        profile_name = st.text_input("Название профиля", key=f"profile_name_{module}")
        if st.button(f"Сохранить {'профиль' if module == 'analysis' else 'диапазоны'}", key=f"save_profile_{module}"):
            if profile_name:
                # Сбор всех параметров стратегии
                profile_data = {
                    "position_size": st.session_state.get(f"position_size_{module}", 1000.0),
                    "commission": st.session_state.get(f"commission_{module}", 0.1),
                    "entry_logic_mode": st.session_state.get(f"entry_logic_mode_{module}", "Принты и HLdir"),
                    "hldir_offset": st.session_state.get(f"hldir_offset_{module}", 0),
                    "aggressive_mode": st.session_state.get(f"aggressive_mode_{module}", False),
                    "start_date": str(st.session_state.get(f"start_date_{module}", datetime(2025, 9, 1))),
                    "end_date": str(st.session_state.get(f"end_date_{module}", datetime(2025, 9, 26)))
                }
                # Добавляем в профиль список выбранных файлов
                selected_files_key = f"selected_files_{module}"
                profile_data["selected_files"] = st.session_state.get(selected_files_key, [])
                
                # Добавляем параметры стратегии в зависимости от модуля
                if module == "analysis":
                    # При сохранении профиля анализа, получаем параметры из session_state напрямую, чтобы избежать дублирования ключей
                    profile_data.update({
                        "vol_period": st.session_state.get("vol_period_analysis", 20),
                        "vol_pctl": round(st.session_state.get("vol_pctl_analysis", 1.0), 2),
                        "range_period": st.session_state.get("range_period_analysis", 20),
                        "rng_pctl": round(st.session_state.get("rng_pctl_analysis", 1.0), 2),
                        "natr_period": st.session_state.get("natr_period_analysis", 10),
                        "natr_min": round(st.session_state.get("natr_min_analysis", 0.35), 2),
                        "lookback_period": st.session_state.get("lookback_period_analysis", 20),
                        "min_growth_pct": round(st.session_state.get("min_growth_pct_analysis", 1.0), 2),
                        "prints_analysis_period": st.session_state.get("prints_analysis_period_analysis", 2),
                        "stop_loss_pct": round(st.session_state.get("stop_loss_pct_analysis", 2.0), 2),
                        "take_profit_pct": round(st.session_state.get("take_profit_pct_analysis", 4.0), 2),
                        "enable_additional_filters": st.session_state.get("enable_additional_filters_analysis", True)
                    })
                    
                    # Добавляем параметры для режима анализа принтов (только режим 1 - фиксированное соотношение)
                    profile_data["prints_threshold_ratio"] = round(st.session_state.get("prints_threshold_ratio_analysis", 1.0), 2)
                    
                    # Добавляем параметр hldir_window, если он используется
                    if "HLdir" in st.session_state.get("entry_logic_mode_analysis", ""):
                        profile_data["hldir_window"] = st.session_state.get("hldir_window_analysis", 10)
                        profile_data["hldir_offset"] = st.session_state.get("hldir_offset_analysis", 0)
                else: # optimization
                    # При сохранении профиля оптимизации, нужно получить параметры без вызова функции, создающей виджеты
                    # Создаем параметры оптимизации напрямую, без вызова Streamlit-элементов
                    # Получаем текущее значение из radio-кнопки для enable_additional_filters
                    enable_additional_filters_option = st.session_state.get("enable_additional_filters_option", "Both")
                    
                    # Устанавливаем значения min и max в зависимости от выбора
                    if enable_additional_filters_option == "False":
                        enable_additional_filters_min = False
                        enable_additional_filters_max = False
                    elif enable_additional_filters_option == "True":
                        enable_additional_filters_min = True
                        enable_additional_filters_max = True
                    else:  # Both
                        enable_additional_filters_min = False
                        enable_additional_filters_max = True
                    
                    profile_data.update({
                        "vol_period_min": st.session_state.get("vol_period_min_optimization", 10),
                        "vol_period_max": st.session_state.get("vol_period_max_optimization", 30),
                        "vol_pctl_min": round(st.session_state.get("vol_pctl_min_optimization", 0.5), 2),
                        "vol_pctl_max": round(st.session_state.get("vol_pctl_max_optimization", 2.0), 2),
                        "range_period_min": st.session_state.get("range_period_min_optimization", 10),
                        "range_period_max": st.session_state.get("range_period_max_optimization", 30),
                        "rng_pctl_min": round(st.session_state.get("rng_pctl_min_optimization", 0.5), 2),
                        "rng_pctl_max": round(st.session_state.get("rng_pctl_max_optimization", 2.0), 2),
                        "natr_period_min": st.session_state.get("natr_period_min_optimization", 5),
                        "natr_period_max": st.session_state.get("natr_period_max_optimization", 20),
                        "natr_min_min": round(st.session_state.get("natr_min_min_optimization", 0.2), 2),
                        "natr_min_max": round(st.session_state.get("natr_min_max_optimization", 0.8), 2),
                        "lookback_period_min": st.session_state.get("lookback_period_min_optimization", 10),
                        "lookback_period_max": st.session_state.get("lookback_period_max_optimization", 30),
                        "min_growth_pct_min": round(st.session_state.get("min_growth_pct_min_optimization", 0.5), 2),
                        "min_growth_pct_max": round(st.session_state.get("min_growth_pct_max_optimization", 2.0), 2),
                        "prints_analysis_period_min": st.session_state.get("prints_analysis_period_min_optimization", 1),
                        "prints_analysis_period_max": st.session_state.get("prints_analysis_period_max_optimization", 5),
                        "stop_loss_pct_min": round(st.session_state.get("stop_loss_pct_min_optimization", 1.0), 2),
                        "stop_loss_pct_max": round(st.session_state.get("stop_loss_pct_max_optimization", 5.0), 2),
                        "take_profit_pct_min": round(st.session_state.get("take_profit_pct_min_optimization", 1.0), 2),
                        "take_profit_pct_max": round(st.session_state.get("take_profit_pct_max_optimization", 8.0), 2),
                        "enable_additional_filters_min": enable_additional_filters_min,
                        "enable_additional_filters_max": enable_additional_filters_max
                    })
                    # Параметры принтов и HLdir также сохраняются через этот механизм

                profile_data["name"] = profile_name
                save_profile(profile_name, profile_data, module)
            else:
                st.warning("Введите название профиля")
    with col2:
        profiles = get_profiles(module)
        selected_profile = st.selectbox(f"Загрузить {'профиль' if module == 'analysis' else 'диапазоны'}", options=profiles, key=f"select_profile_{module}")
        if st.button(f"Загрузить {'профиль' if module == 'analysis' else 'диапазоны'}", key=f"load_profile_{module}") and selected_profile:
            profile_data = load_profile(selected_profile, module)
            if profile_data:
                # Используем специальный флаг для обновления session_state без конфликта с виджетами
                st.session_state["profile_to_load"] = profile_data
                st.session_state["page_to_rerun"] = "Анализ сигналов" if module == "analysis" else "Оптимизация"
                st.rerun()

# Функция для обработки загрузки профиля в session_state
def handle_profile_loading():
    """Обработка загрузки профиля в session_state"""
    # Обработка загрузки профиля
    if "profile_to_load" in st.session_state:
        profile_data = st.session_state["profile_to_load"]
        page_to_rerun = st.session_state.get("page_to_rerun")
        
        # Удаляем флаги из session_state до загрузки данных, чтобы избежать конфликта с виджетами
        del st.session_state["profile_to_load"]
        if "page_to_rerun" in st.session_state:
            del st.session_state["page_to_rerun"]
        
        if page_to_rerun == "Анализ сигналов":
            # Загружаем значения в session_state с правильными ключами
            load_profile_to_session_state(profile_data, "analysis")
            
            # Обработка специфичных для анализа параметров
            if "hldir_window" in profile_data:
                st.session_state["hldir_window_analysis"] = profile_data["hldir_window"]
            
            profile_name = profile_data.get('name', 'неизвестный')
            st.success(f"Профиль '{profile_name}' загружен!")
            
            # Если это временный профиль, созданный из оптимизации, удаляем его
            if profile_name.startswith("temp_analysis_"):
                try:
                    import os
                    profile_path = os.path.join("profiles", "analysis", f"{profile_name}.json")
                    if os.path.exists(profile_path):
                        os.remove(profile_path)
                except Exception as e:
                    st.warning(f"Не удалось удалить временный профиль: {str(e)}")
                
        elif page_to_rerun == "Оптимизация":
            # Загружаем значения в session_state с правильными ключами
            load_profile_to_session_state(profile_data, "optimization")
            
            # Обработка специфичных для оптимизации параметров
            if "hldir_window_min" in profile_data:
                st.session_state["hldir_window_min_optimization"] = profile_data["hldir_window_min"]
            if "hldir_window_max" in profile_data:
                st.session_state["hldir_window_max_optimization"] = profile_data["hldir_window_max"]
            
            st.success(f"Диапазоны '{profile_data.get('name', 'неизвестный')}' загружены!")
        else:
            # Неизвестная страница, выходим
            return
            
        # Перезапускаем страницу для обновления значений виджетов
        st.rerun()

# Функция для получения списка файлов прогонов оптимизации
@st.cache_data(ttl=10)  # Кэшируем на 10 секунд
def get_optimization_run_files():
    try:
        files = [f for f in os.listdir("optimization_runs") if f.endswith('.json')]
        # Сортируем файлы по дате создания в названии файла (от новых к старым)
        # Формат файла: run_YYYYMMDD_HHMMSS.json
        def extract_datetime(f):
            try:
                # Извлекаем дату и время из имени файла
                # Формат файла: run_YYYYMMDD_HHMMSS_... или run_YYYYMMDD_HHMMSS.json
                datetime_str = f.replace('run_', '').replace('.json', '')
                # Извлекаем только начальную часть с датой и временем: YYYYMMDD_HHMMSS
                # Находим первую часть до следующего символа (обычно это _ после времени)
                parts = datetime_str.split('_', 1)  # Разбиваем на 2 части: время и остальное
                time_part = parts[0]  # Берем только первую часть (дату_время)
                
                # Преобразуем строку в datetime объект
                return datetime.strptime(time_part, '%Y%m%d_%H%M%S')
            except ValueError:
                # Если формат не подходит, возвращаем минимальную дату
                return datetime.min
         
        # Сначала сортируем по возрастанию даты, затем разворачиваем список, чтобы новые файлы были сверху
        sorted_files = sorted(files, key=extract_datetime)
        return sorted_files[::-1]  # Новые файлы будут сверху
    except FileNotFoundError:
        return []

# Функция для загрузки данных прогона оптимизации
@st.cache_data(ttl=300)  # Кэшируем на 5 минут
def load_run_data_cached(run_file):
    file_path = os.path.join("optimization_runs", run_file)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Файл прогона '{run_file}' не найден")
        return None
    except json.JSONDecodeError:
        st.error(f"Файл прогона '{run_file}' содержит некорректный JSON")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке прогона '{run_file}': {str(e)}")
        return None

# Основная логика приложения
if current_page == "Анализ сигналов":
    # Проверяем, нужно ли загрузить профиль
    handle_profile_loading()
    
    # Базовые настройки в боковой панели
    with st.sidebar:
        st.header("Настройки анализа")
        # Основные настройки, включая aggressive_mode
        position_size, commission, start_date, end_date = get_basic_settings("analysis")

        # Профили настроек (только для этого модуля)
        manage_profiles("analysis", get_strategy_parameters)
        
        # Параметры стратегии (всего 12)
        st.subheader("Параметры стратегии")
        
        # Переключатель логики входа
        entry_logic_mode_analysis = st.radio(
            "Логика входа",
            options=["Принты и HLdir", "Только по принтам", "Только по HLdir"],
            index=["Принты и HLdir", "Только по принтам", "Только по HLdir"].index(st.session_state.get("entry_logic_mode_analysis", "Принты и HLdir")),
            key="entry_logic_mode_analysis",
            help="Определяет, как будут сочетаться сигналы для входа в сделку."
        )
        
        # Параметры HLdir теперь всегда видны для использования в сравнительном анализе
        st.markdown("**Параметры HLdir** (для режимов с HLdir)")
        col1, col2 = st.columns(2)
        with col1:
            hldir_window = st.number_input(
                "Окно HLdir",
                value=int(st.session_state.get("hldir_window_analysis", 10)),
                min_value=1,
                step=1,
                key="hldir_window_analysis"
            )
        with col2:
            hldir_offset = st.number_input(
                "Смещение окна HLdir",
                value=int(st.session_state.get("hldir_offset_analysis", 0)),
                min_value=0,
                key="hldir_offset_analysis",
                help="0: окно включает сигнальную свечу `i`. 1: окно до `i-1`."
            )
        
        # Получаем параметры для анализа
        params = get_strategy_parameters("analysis")
        vol_period = params["vol_period"]
        vol_pctl = round(params["vol_pctl"], 2)
        range_period = params["range_period"]
        rng_pctl = round(params["rng_pctl"], 2)
        natr_period = params["natr_period"]
        natr_min = round(params["natr_min"], 2)
        lookback_period = params["lookback_period"]
        min_growth_pct = round(params["min_growth_pct"], 2)
        prints_analysis_period = params["prints_analysis_period"]
        prints_threshold_ratio = round(params["prints_threshold_ratio"], 2)
        stop_loss_pct = round(params["stop_loss_pct"], 2)
        take_profit_pct = round(params["take_profit_pct"], 2)
        
        # Добавляем hldir_window в параметры
        params["hldir_window"] = int(hldir_window)
        params["hldir_offset"] = int(hldir_offset)
        
        # Кнопка для обновления графика
        update_chart = st.button("Применить настройки", key="update_chart_sidebar")
        
        # Выбор данных
        csv_files = []
        try:
            for file in os.listdir("dataCSV"):
                if file.endswith(".csv"):
                    csv_files.append(file)
        except FileNotFoundError:
            st.warning("Папка dataCSV не найдена")
        
        if csv_files:
            with st.expander("Выбор данных", expanded=True):
                st.subheader("Выбор данных")
                # Добавляем возможность выбрать все файлы
                select_all = st.checkbox("Выбрать все файлы", key="select_all_csv")
                
                # Создаем чекбоксы для каждого файла
                selected_files = []
                checkbox_key_prefix = f"csv_analysis_"
                for i, file in enumerate(csv_files):
                    # Если "Выбрать все" отмечен, все файлы отмечены по умолчанию
                    # Или если это первый файл и состояние не установлено, выбираем его по умолчанию
                    if f"{checkbox_key_prefix}{file}" not in st.session_state:
                        # При инициализации используем значение select_all
                        st.session_state[f"{checkbox_key_prefix}{file}"] = select_all
                    elif st.session_state.get("select_all_csv_prev", not select_all) != select_all:
                        # Если состояние "Выбрать все" изменилось, обновляем все индивидуальные чекбоксы
                        st.session_state[f"{checkbox_key_prefix}{file}"] = select_all
                    
                    is_selected = st.checkbox(file, key=f"{checkbox_key_prefix}{file}")
                    if is_selected:
                        selected_files.append(file)
                
                # Сохраняем текущее состояние "Выбрать все" для следующей итерации
                st.session_state["select_all_csv_prev"] = select_all
                
                # Сохраняем выбранные файлы в session_state
                st.session_state["selected_files_analysis"] = selected_files
        else:
            st.info("В папке dataCSV нет CSV-файлов")
            st.session_state["selected_files_analysis"] = []
    
    # Выбор данных уже перенесен в боковую панель
    # Отображаем выбранные файлы в строке
    selected_files = st.session_state.get("selected_files_analysis", [])
    if selected_files:
        st.write(f"Выбраны файлы: {', '.join(selected_files)}")
    else:
        st.write("Файлы не выбраны")
    
    # Убираем кнопку, которая вызывает перерисовку, так как это создает конфликты с session_state
    # Вместо этого используем автоматическое обновление при изменении параметров
    # st.rerun() вызывается при изменении параметров в get_strategy_parameters
    
    # Загрузка и объединение данных
    dataframes = load_and_validate_csv_files(selected_files, "analysis")
    
    if dataframes:
        try:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='s')
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
            
            # Сохраняем combined_df в session_state, чтобы иметь к нему доступ из других частей приложения
            st.session_state["combined_df_analysis"] = combined_df
        except Exception as e:
            st.error(f"Ошибка при объединении данных: {str(e)}")
            combined_df = None
            st.session_state["combined_df_analysis"] = None
            
        # Предварительный расчет и отображение количества сигналов
        if combined_df is not None:
            with st.spinner("Расчет количества сигналов..."):
                try:
                    signal_indices = generate_signals(combined_df, params)
                    st.info(f"Найдено сигналов по текущим параметрам: **{len(signal_indices)}**")
                except Exception as e:
                    st.error(f"Ошибка при генерации сигналов: {e}")
                    signal_indices = []
        
        # Добавляем кнопку для запуска симуляции торговли
        if st.button("Запустить симуляцию торговли", key="run_simulation"):
            with st.spinner("Запуск симуляции..."):
                # Подготовим параметры для симуляции
                simulation_params = params.copy()
                simulation_params["position_size"] = round(position_size, 2)
                simulation_params["commission"] = round(commission, 3)  # Используем 3 знака для комиссии из-за малых значений
                simulation_params["stop_loss_pct"] = round(stop_loss_pct, 2)
                simulation_params["aggressive_mode"] = st.session_state.get("aggressive_mode_analysis", False)
                simulation_params["take_profit_pct"] = round(take_profit_pct, 2)
                
                # Добавляем параметры логики входа и HLdir, чтобы они всегда были доступны
                simulation_params["entry_logic_mode"] = entry_logic_mode_analysis
                simulation_params["hldir_window"] = hldir_window
                simulation_params["hldir_offset"] = hldir_offset
                                
                # Запускаем симуляцию
                simulation_results = run_trading_simulation(combined_df, simulation_params)
                
                # Сохраняем результаты в session_state
                st.session_state["simulation_results"] = simulation_results
                st.session_state["simulation_params"] = simulation_params
                
                st.success("Симуляция завершена!")
        
        # Отображаем результаты симуляции, если они есть
        if "simulation_results" in st.session_state:
            results = st.session_state["simulation_results"]
            
            st.subheader("Результаты симуляции")
            
            # Отображаем график с сделками
            if st.button("Показать график сделок", key="show_trades_chart"):
                simulator = TradingSimulator(
                    position_size=position_size,
                    commission=commission,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct
                )
                fig = simulator.plot_trades_on_chart(combined_df, results)
                st.plotly_chart(fig, use_container_width=True)
            
            # Отображаем основные метрики
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Всего сделок", results['total_trades'])
            col2.metric("Прибыльных сделок", results['winning_trades'])
            col3.metric("Процент прибыльных", f"{round(results['win_rate']*100, 2):.2f}%")
            col4.metric("Общий PnL", f"${results['total_pnl']:.2f}")
            col5.metric("Средний PnL", f"${results['avg_pnl']:.2f}")
            col6.metric("Макс. просадка", f"{round(results['max_drawdown']*100, 2):.2f}%")
                 
            # Отображаем таблицу сделок
            if results['trades']:
                st.subheader("Список сделок")
                
                # Используем векторизованные операции для создания DataFrame
                trades_data = {
                    'Индекс входа': [trade['entry_idx'] for trade in results['trades']],
                    'Цена входа': [trade['entry_price'] for trade in results['trades']],
                    'Направление': [trade['direction'] for trade in results['trades']],
                    'Цена выхода': [trade['exit_price'] for trade in results['trades']],
                    'PnL': [trade['pnl'] for trade in results['trades']],
                    'Причина выхода': [trade['exit_reason'] for trade in results['trades']]
                }
                
                trades_df_display = pd.DataFrame(trades_data)
                
                st.dataframe(trades_df_display)
                 
            # Добавляем раздел для сравнения с/без HLdir
            st.subheader("Сравнительный анализ режимов входа")
            if st.button("Сравнить режимы входа", key="compare_hldir"):
                with st.spinner("Проведение сравнительных симуляций..."):
                    # Получаем текущие параметры и результаты
                    # ВАЖНО: Пересобираем параметры из интерфейса, чтобы учесть изменения
                    # в hldir_window/offset, даже если текущий режим их не использует. 
                    # Используем `params`, который уже содержит все значения из UI.
                    current_params = params.copy()
                    current_params["position_size"] = position_size
                    current_params["commission"] = commission
                    current_params["aggressive_mode"] = st.session_state.get("aggressive_mode_analysis", False)

                    current_results = st.session_state["simulation_results"]                    
                    current_mode = current_params.get("entry_logic_mode", "Только по принтам")
                    
                    # Запускаем симуляции для всех трех режимов
                    results_by_mode = {current_mode: current_results}
                    all_modes = ["Принты и HLdir", "Только по принтам", "Только по HLdir"]
                    for mode in all_modes:
                        if mode not in results_by_mode:
                            opposite_params = current_params.copy()
                            opposite_params["entry_logic_mode"] = mode
                            results_by_mode[mode] = run_trading_simulation(combined_df, opposite_params)
                    
                    # Сохраняем результаты в session_state для последующей отрисовки графиков
                    st.session_state['comparison_results_by_mode'] = results_by_mode
                    st.session_state['comparison_current_params'] = current_params
                    st.session_state['comparison_df'] = combined_df
                    
                    # Формируем данные для сравнительной таблицы для всех режимов
                    comparison_data = []
                    # Используем all_modes для сохранения порядка
                    for mode in all_modes: 
                        res = results_by_mode[mode]
                        comparison_data.append({
                            "Режим": mode,
                            "Общий PnL": f"${res['total_pnl']:.2f}",
                            "Win Rate": f"{res['win_rate']*100:.2f}%",
                            "Всего сделок": res['total_trades'],
                            "Макс. просадка": f"{res['max_drawdown']*100:.2f}%",
                            "Profit Factor": f"{res.get('profit_factor', 0):.2f}"
                        })
                    comparison_df = pd.DataFrame(comparison_data).set_index("Режим")
                    
                    st.subheader("Сводная таблица метрик")
                    st.table(comparison_df)
            
            # Кнопка для отображения сравнительных графиков (появляется после расчета)
            if 'comparison_results_by_mode' in st.session_state:
                if st.button("Показать сравнительные графики", key="show_comparison_charts"):
                    with st.spinner("Отрисовка графиков..."):
                        # Загружаем данные из session_state
                        results_by_mode = st.session_state['comparison_results_by_mode']
                        current_params = st.session_state['comparison_current_params']
                        combined_df = st.session_state['comparison_df']

                        st.subheader("Сравнительные графики")
                        col1, col2, col3 = st.columns(3)

                        simulator = TradingSimulator(
                            position_size=current_params.get("position_size"),
                            commission=current_params.get("commission"),
                            stop_loss_pct=current_params.get("stop_loss_pct"),
                            take_profit_pct=current_params.get("take_profit_pct"),
                        )

                        def plot_in_col(column, mode_name):
                            with column:
                                st.markdown(f"**{mode_name}**")
                                params_for_plot = current_params.copy()
                                params_for_plot["entry_logic_mode"] = mode_name
                                fig = simulator.plot_trades_on_chart(
                                    combined_df, results_by_mode[mode_name], show_balance=False, params=params_for_plot
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        plot_in_col(col1, "Принты и HLdir")
                        plot_in_col(col2, "Только по принтам")
                        plot_in_col(col3, "Только по HLdir")

elif current_page == "Оптимизация":
    # Проверяем, нужно ли загрузить профиль
    handle_profile_loading()
    
    # Отображаем выбранные файлы в строке
    selected_files = st.session_state.get("selected_files_optimization", [])
    if selected_files:
        st.write(f"Выбраны файлы: {', '.join(selected_files)}")
    else:
        st.write("Файлы не выбраны")
    
    # Базовые настройки в боковой панели
    with st.sidebar:
        st.header("Настройки оптимизации")
        # Основные настройки, включая aggressive_mode
        position_size, commission, start_date, end_date = get_basic_settings("optimization")

        # Профили настроек (только для этого модуля)
        manage_profiles("optimization", get_optimization_parameters)
    
        # Параметры оптимизации
        st.subheader("Параметры оптимизации")
        
        # Тип оптимизации теперь только один - стандартная Optuna
        st.info("Используется стандартная оптимизация **Optuna**.")
        optimization_type = "optuna"

        # Переключатель логики входа для оптимизации
        entry_logic_mode_optimization = st.selectbox(
            "Логика входа для оптимизации",
            options=["Принты и HLdir", "Только по принтам", "Только по HLdir", "Оптимизировать"],
            index=0,
            key="entry_logic_mode_optimization"
        )
        
        # Показываем диапазон hldir_window, если выбран режим с HLdir
        if "HLdir" in entry_logic_mode_optimization or entry_logic_mode_optimization == "Оптимизировать":
            st.markdown("**Параметры HLdir**")
            col1, col2 = st.columns(2)
            with col1:
                hldir_window_min = st.number_input(
                    "Окно HLdir (min)",
                    value=int(st.session_state.get("hldir_window_min_optimization", 3)),
                    min_value=1,
                    step=1,
                    key="hldir_window_min_optimization"
                )
                hldir_offset_min = st.number_input(
                    "Смещение (min)",
                    value=int(st.session_state.get("hldir_offset_min_optimization", 0)),
                    min_value=0,
                    key="hldir_offset_min_optimization"
                )
            with col2:
                hldir_window_max = st.number_input(
                    "Окно HLdir (max)",
                    value=int(st.session_state.get("hldir_window_max_optimization", 10)),
                    min_value=1,
                    step=1,
                    key="hldir_window_max_optimization"
                )
                hldir_offset_max = st.number_input(
                    "Смещение (max)",
                    value=int(st.session_state.get("hldir_offset_max_optimization", 2)),
                    min_value=0,
                    key="hldir_offset_max_optimization"
                )
        
        # Получаем параметры для оптимизации
        opt_params = get_optimization_parameters()
        
        # Извлекаем все параметры оптимизации и округляем float параметры до 2 знаков после запятой
        vol_period_min = opt_params["vol_period_min"]
        vol_period_max = opt_params["vol_period_max"]
        vol_pctl_min = round(opt_params["vol_pctl_min"], 2)
        vol_pctl_max = round(opt_params["vol_pctl_max"], 2)
        range_period_min = opt_params["range_period_min"]
        range_period_max = opt_params["range_period_max"]
        rng_pctl_min = round(opt_params["rng_pctl_min"], 2)
        rng_pctl_max = round(opt_params["rng_pctl_max"], 2)
        natr_period_min = opt_params["natr_period_min"]
        natr_period_max = opt_params["natr_period_max"]
        natr_min_min = round(opt_params["natr_min_min"], 2)
        natr_min_max = round(opt_params["natr_min_max"], 2)
        lookback_period_min = opt_params["lookback_period_min"]
        lookback_period_max = opt_params["lookback_period_max"]
        min_growth_pct_min = round(opt_params["min_growth_pct_min"], 2)
        min_growth_pct_max = round(opt_params["min_growth_pct_max"], 2)
        prints_analysis_period_min = opt_params["prints_analysis_period_min"]
        prints_analysis_period_max = opt_params["prints_analysis_period_max"]
        stop_loss_pct_min = round(opt_params["stop_loss_pct_min"], 2)
        stop_loss_pct_max = round(opt_params["stop_loss_pct_max"], 2)
        take_profit_pct_min = round(opt_params["take_profit_pct_min"], 2)
        take_profit_pct_max = round(opt_params["take_profit_pct_max"], 2)
        
        # Извлекаем параметры для режима 1 (фиксированное соотношение)
        prints_threshold_ratio_min = round(opt_params["prints_threshold_ratio_min"], 2)
        prints_threshold_ratio_max = round(opt_params["prints_threshold_ratio_max"], 2)
        
        # Извлекаем hldir_window параметры
        hldir_window_min = int(st.session_state.get("hldir_window_min_optimization", 3))
        hldir_window_max = int(st.session_state.get("hldir_window_max_optimization", 10))
        hldir_offset_min = int(st.session_state.get("hldir_offset_min_optimization", 0))
        hldir_offset_max = int(st.session_state.get("hldir_offset_max_optimization", 2))
        
        # Добавляем выбор количества проб для разных типов оптимизации
        col1, col2 = st.columns(2)
        with col1:
            optuna_trials = st.number_input("Количество проб Optuna", value=50, min_value=10, step=10, key="optuna_trials")
        with col2:
            wfo_trials = st.number_input("Количество проб WFO", value=20, min_value=5, step=5, key="wfo_trials")
        
        # Цель оптимизации теперь фиксирована - высокий Win Rate.
        st.info("Цель оптимизации: **Высокий Win Rate** (поиск стратегий с максимальным процентом прибыльных сделок).")

        csv_files = []
        try:
            for file in os.listdir("dataCSV"):
                if file.endswith(".csv"):
                    csv_files.append(file)
        except FileNotFoundError:
            st.warning("Папка dataCSV не найдена")
        
        if csv_files:
            with st.expander("Выбор данных", expanded=True):
                st.subheader("Выбор данных")
                # Добавляем возможность выбрать все файлы
                select_all = st.checkbox("Выбрать все файлы", key="select_all_csv_optimization")
                
                # Создаем чекбоксы для каждого файла
                selected_files = []
                checkbox_key_prefix = f"csv_optimization_"
                for i, file in enumerate(csv_files):
                    # Если "Выбрать все" отмечен, все файлы отмечены по умолчанию
                    # Или если это первый файл и состояние не установлено, выбираем его по умолчанию
                    if f"{checkbox_key_prefix}{file}" not in st.session_state:
                        # При инициализации используем значение select_all
                        st.session_state[f"{checkbox_key_prefix}{file}"] = select_all
                    elif st.session_state.get("select_all_csv_optimization_prev", not select_all) != select_all:
                        # Если состояние "Выбрать все" изменилось, обновляем все индивидуальные чекбоксы
                        st.session_state[f"{checkbox_key_prefix}{file}"] = select_all
                    
                    is_selected = st.checkbox(file, key=f"{checkbox_key_prefix}{file}")
                    if is_selected:
                        selected_files.append(file)
                
                # Сохраняем текущее состояние "Выбрать все" для следующей итерации
                st.session_state["select_all_csv_optimization_prev"] = select_all
                
                # Сохраняем выбранные файлы в session_state
                st.session_state["selected_files_optimization"] = selected_files
        else:
            st.info("В папке dataCSV нет CSV-файлов")
            st.session_state["selected_files_optimization"] = []
        
        # Загрузка и валидация CSV-файлов
        dataframes = load_and_validate_csv_files(selected_files, "optimization")
            
    # Используем единственную целевую функцию, ориентированную на высокий Win Rate.
    objective_func = strategy_objectives.trading_strategy_objective_high_win_rate

    # Запуск оптимизации
    if st.button("Запустить оптимизацию"):
        # Проверяем, что файлы загружены
        if not dataframes:
            st.warning("Пожалуйста, выберите хотя бы один CSV-файл для оптимизации")
        else:
            # Объединяем данные из всех файлов
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['datetime'] = pd.to_datetime(combined_df['time'], unit='s')
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
            
            # Получаем текущее значение из radio-кнопки для enable_additional_filters
            # Теперь это определяется напрямую из диапазонов, заданных в get_optimization_parameters
        
            # Подготавливаем пространство параметров для оптимизации из существующих диапазонов
            param_space = {
                "vol_period": ("int", int(vol_period_min), int(vol_period_max)),
                "vol_pctl": ("float", round(float(vol_pctl_min), 2), round(float(vol_pctl_max), 2)),
                "range_period": ("int", int(range_period_min), int(range_period_max)),
                "rng_pctl": ("float", round(float(rng_pctl_min), 2), round(float(rng_pctl_max), 2)),
                "natr_period": ("int", int(natr_period_min), int(natr_period_max)),
                "natr_min": ("float", round(float(natr_min_min), 2), round(float(natr_min_max), 2)),
                "lookback_period": ("int", int(lookback_period_min), int(lookback_period_max)),
                "min_growth_pct": ("float", round(float(min_growth_pct_min), 2), round(float(min_growth_pct_max), 2)),
                "prints_analysis_period": ("int", int(prints_analysis_period_min), int(prints_analysis_period_max)),
                "prints_threshold_ratio": ("float", round(float(prints_threshold_ratio_min), 2), round(float(prints_threshold_ratio_max), 2)),  # Используется при режиме 1
                "stop_loss_pct": ("float", round(float(stop_loss_pct_min), 2), round(float(stop_loss_pct_max), 2)),
                "take_profit_pct": ("float", round(float(take_profit_pct_min), 2), round(float(take_profit_pct_max), 2)),
                "hldir_window": ("int", int(hldir_window_min), int(hldir_window_max)),  # Параметр для усреднения HLdir
                "hldir_offset": ("int", int(hldir_offset_min), int(hldir_offset_max)),
                "enable_additional_filters": ("categorical", [False, True])  # Оптимизируем по обоим вариантам
            }
            
            # Добавляем логику входа в пространство параметров
            if entry_logic_mode_optimization == "Оптимизировать":
                param_space["entry_logic_mode"] = ("categorical", ["Принты и HLdir", "Только по принтам", "Только по HLdir"])
            else:
                param_space["entry_logic_mode"] = ("categorical", [entry_logic_mode_optimization])

            # Запускаем обычную оптимизацию Optuna
            try:
                
                st.subheader("Результаты оптимизации Optuna")
                
                # Подготовим параметры для обычной оптимизации
                opt_params = {
                    'data': combined_df,
                    'param_space': param_space,
                    'n_trials': optuna_trials,  # Используем количество проб из параметров
                    'direction': 'maximize',
                    'optimization_type': 'optuna',
                    'position_size': position_size,
                    'commission': commission,
                    'stop_loss_pct': stop_loss_pct_min,
                    'aggressive_mode': st.session_state.get("aggressive_mode_optimization", False),
                    'take_profit_pct': take_profit_pct_min,
                    'strategy_func': objective_func # Добавляем целевую функцию в параметры
                }
                
                # Запускаем оптимизацию
                opt_results = wfo_optuna.run_optimization(opt_params)
                
                # Проверяем, что оптимизация вернула результат
                if opt_results and opt_results.get('best_value') is not None:
                    st.success(f"Оптимизация завершена! Лучшее значение: {opt_results['best_value']:.4f}")
                    
                    # Отображаем лучшие параметры
                    st.subheader("Лучшие параметры")
                    best_params_df = pd.DataFrame(list(opt_results['best_params'].items()), columns=['Параметр', 'Значение'])
                    st.dataframe(best_params_df, use_container_width=True)
                    
                    # Отображаем топ-10 результатов
                    if 'top_10_results' in opt_results and opt_results['top_10_results']:
                        st.subheader("Топ-10 результатов оптимизации")
                        top_10_df = pd.DataFrame(opt_results['top_10_results'])
                        st.dataframe(top_10_df, use_container_width=True)
                else:
                    st.error("Оптимизация завершилась, но не удалось найти ни одного подходящего набора параметров. Попробуйте расширить диапазоны оптимизации или проверить данные.")
                
                # Формируем имя файла для результатов обычной оптимизации
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Определяем префикс для файлов данных
                if len(selected_files) == 1:
                    # Если только один файл, используем его имя до первого тире
                    filename = selected_files[0]
                    dash_pos = filename.find('-')
                    data_prefix = filename[:dash_pos] if dash_pos != -1 else filename.rsplit('.', 1)[0]
                elif len(selected_files) > 1:
                    data_prefix = "ALL"
                else:
                    data_prefix = ""
                
                # Извлекаем числовые значения и округляем до целого
                def extract_numeric_value(value_str):
                    """Извлекает числовое значение из строки и возвращает округленное целое"""
                    if value_str is None or (isinstance(value_str, float) and np.isnan(value_str)):
                        return 0
                    
                    # Используем numpy для обработки строковых данных, если это массив
                    if hasattr(value_str, 'dtype') and np.issubdtype(value_str.dtype, np.number):
                        value_str = value_str.item()  # Извлекаем скалярное значение из numpy скаляра
                    
                    # Обрабатываем строку для извлечения числового значения
                    numeric_str = ''.join(filter(lambda x: x.isdigit() or x == '.', str(value_str).replace('$', '').replace('%', '').replace('-', '')))
                    try:
                        return int(float(numeric_str) + 0.5)
                    except ValueError:
                        return 0
                
                best_value = extract_numeric_value(opt_results.get('best_value'))
                
                # Режим анализа принтов всегда 1 (фиксированное соотношение)
                mode_suffix = "_mode1"
                
                # Формируем новое имя файла
                new_run_name = f"run_{timestamp}_{data_prefix}{mode_suffix}_${best_value}_OPTUNA"
                
                # Подготавливаем словарь диапазонов
                ranges_dict = {
                    "vol_period_min": int(vol_period_min),
                    "vol_period_max": int(vol_period_max),
                    "vol_pctl_min": round(float(vol_pctl_min), 2),
                    "vol_pctl_max": round(float(vol_pctl_max), 2),
                    "range_period_min": int(range_period_min),
                    "range_period_max": int(range_period_max),
                    "rng_pctl_min": round(float(rng_pctl_min), 2),
                    "rng_pctl_max": round(float(rng_pctl_max), 2),
                    "natr_period_min": int(natr_period_min),
                    "natr_period_max": int(natr_period_max),
                    "natr_min_min": round(float(natr_min_min), 2),
                    "natr_min_max": round(float(natr_min_max), 2),
                
                    "lookback_period_min": int(lookback_period_min),
                    "lookback_period_max": int(lookback_period_max),
                    "min_growth_pct_min": round(float(min_growth_pct_min), 2),
                    "min_growth_pct_max": round(float(min_growth_pct_max), 2),
                    "prints_analysis_period_min": int(prints_analysis_period_min),
                    "prints_analysis_period_max": int(prints_analysis_period_max),
                    "stop_loss_pct_min": round(float(stop_loss_pct_min), 2),
                    "stop_loss_pct_max": round(float(stop_loss_pct_max), 2),
                    "take_profit_pct_min": round(float(take_profit_pct_min), 2),
                    "take_profit_pct_max": round(float(take_profit_pct_max), 2),
                    # Добавляем диапазоны для hldir_window
                    "hldir_window_min": int(hldir_window_min),
                    "hldir_window_max": int(hldir_window_max),
                    "hldir_offset_min": int(hldir_offset_min),
                    "hldir_offset_max": int(hldir_offset_max),
                    
                    # Добавляем Z-score параметры
                    "prints_threshold_ratio_min": round(float(prints_threshold_ratio_min), 2),
                    "prints_threshold_ratio_max": round(float(prints_threshold_ratio_max), 2)
                }
                
                
                # Подготовим результаты в формате, совместимом с существующими файлами
                optuna_results = []
                if opt_results and opt_results.get('best_params'):
                    result_entry = opt_results['best_params'].copy()
                    result_entry['ID'] = 1
                    result_entry['in_sample_metric'] = opt_results.get('best_value', 0)
                    result_entry['out_sample_metric'] = opt_results.get('best_value', 0)  # Для обычной оптимизации in-sample и out-sample одинаковы
                    result_entry['end_date'] = str(combined_df['datetime'].iloc[-1])
                    
                    # Добавляем метрики
                    result_entry['Total Trades'] = 0 # Будет обновлено после симуляции
                    result_entry['PnL'] = f"${opt_results.get('best_value', 0):.2f}"
                    result_entry['Win Rate'] = "0%"
                    result_entry['Max Drawdown'] = "0%"
                    result_entry['Sharpe Ratio'] = opt_results.get('best_value', 0)  # Используем метрику как приближение
                    result_entry['Profit Factor'] = 1.0  # Временное значение
                    
                    optuna_results.append(result_entry)
                
                run_data = {
                    "run_name": new_run_name,  # Используем новое имя
                    "timestamp": datetime.now().isoformat(),
                    "ranges": ranges_dict,
                    "settings": {
                        "position_size": position_size,
                        "commission": commission,
                        "start_date": str(start_date),
                        "end_date": str(end_date)
                    },
                    "data_files": selected_files, # Добавляем информацию о выбранных файлах данных
                    "results": optuna_results,
                    "best_params": opt_results.get('best_params', {}),
                    "best_result": optuna_results[0] if optuna_results else {},
                    "optimization_type": "optuna", # Указываем тип оптимизации
                    "top_10_results": opt_results.get('top_10_results', [])  # Добавляем топ-10 результатов
                }
                
                # Преобразуем numpy типы в стандартные Python типы для JSON сериализации
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj
                
                # Преобразуем данные перед сохранением
                run_data_converted = convert_numpy_types(run_data)
                
                # Сохраняем результаты в файл
                try:
                    os.makedirs("optimization_runs", exist_ok=True)
                    with open(f"optimization_runs/{new_run_name}.json", 'w', encoding='utf-8') as f:
                        json.dump(run_data_converted, f, ensure_ascii=False, indent=2)
                
                    st.success(f"Результаты Optuna оптимизации сохранены как '{new_run_name}'")
                except Exception as e:
                    st.error(f"Ошибка при сохранении результатов Optuna оптимизации: {str(e)}")
            
            except Exception as e:
                st.error(f"Ошибка при запуске Optuna оптимизации: {str(e)}")
            
            

elif current_page == "Аналитика":
    # Отображение списка сохранённых прогонов оптимизации
    st.header("Аналитика результатов оптимизации")
    st.subheader("Сохранённые прогоны оптимизации")
    
    run_files = get_optimization_run_files()
    
    if run_files:
        # Отображаем все прогоны в компактном виде с кнопками в ряд
        for run_file in run_files:
            run_name = run_file.replace('.json', '')
            
            # Загружаем данные прогона для отображения метрик
            run_data = load_run_data_cached(run_file)
            if run_data is None:
                continue
                
            # Извлекаем метрики из top_10_results[0], если они есть, или из best_params с помощью симуляции
            top_results = run_data.get("top_10_results", [])
            if top_results:
                # Используем первый результат из top_10_results, который должен содержать полные метрики
                best_result = top_results[0]
                # Форматируем значения как в остальной части приложения
                total_pnl = best_result.get('PnL', 0) or best_result.get('total_pnl', 0)
                if total_pnl is None or (isinstance(total_pnl, float) and np.isnan(total_pnl)):
                    pnl = 'N/A'
                else:
                    # Если значение уже строка с форматом доллара, оставляем как есть
                    if isinstance(total_pnl, str):
                        pnl = total_pnl
                    else:
                        pnl = f"${total_pnl:.2f}"
                
                win_rate = best_result.get('Win Rate', 0) or best_result.get('win_rate', 0)
                if win_rate is None or (isinstance(win_rate, float) and np.isnan(win_rate)):
                    win_rate_formatted = 'N/A'
                else:
                    # Проверяем, может быть win_rate уже в формате процента
                    if isinstance(win_rate, str) and win_rate.endswith('%'):
                        win_rate_formatted = win_rate
                    elif isinstance(win_rate, (int, float)):
                        # win_rate из симуляции уже в десятичном формате (например, 0.65),
                        # но может быть уже умноженным на 100 в другом месте, поэтому проверяем значение
                        if win_rate > 1:
                            # Если значение больше 1, то оно уже в формате процента
                            win_rate_formatted = f"{win_rate:.2f}%"
                        else:
                            # Если значение <= 1, то это десятичная доля, нужно умножить на 100
                            win_rate_formatted = f"{round(win_rate * 100, 2):.2f}%"
                    else:
                        win_rate_formatted = 'N/A'
            else:
                # Если top_10_results нет, пытаемся получить метрики из best_params или других полей
                best_result = run_data.get("best_result", {})
                if best_result:
                    # Проверяем, есть ли полные метрики в best_result
                    total_pnl = best_result.get('total_pnl', 0)
                    if total_pnl is None or (isinstance(total_pnl, float) and np.isnan(total_pnl)):
                        pnl = 'N/A'
                    else:
                        pnl = f"${total_pnl:.2f}"
                    
                    win_rate = best_result.get('win_rate', 0)
                    if win_rate is None or (isinstance(win_rate, float) and np.isnan(win_rate)):
                        win_rate_formatted = 'N/A'
                    else:
                        # Проверяем, может быть win_rate уже в формате процента
                        if isinstance(win_rate, str) and win_rate.endswith('%'):
                            win_rate_formatted = win_rate
                        elif isinstance(win_rate, (int, float)):
                            # win_rate из симуляции уже в десятичном формате (например, 0.65),
                            # но может быть уже умноженным на 100 в другом месте, поэтому проверяем значение
                            if win_rate > 1:
                                # Если значение больше 1, то оно уже в формате процента
                                win_rate_formatted = f"{win_rate:.2f}%"
                            else:
                                # Если значение <= 1, то это десятичная доля, нужно умножить на 10
                                win_rate_formatted = f"{round(win_rate * 100, 2):.2f}%"
                        else:
                            win_rate_formatted = 'N/A'
                else:
                    # Пытаемся получить метрики из других возможных полей
                    total_pnl = run_data.get('PnL', 0) or run_data.get('total_pnl', 0)
                    if total_pnl is None or (isinstance(total_pnl, float) and np.isnan(total_pnl)):
                        pnl = 'N/A'
                    else:
                        # Если значение уже строка с форматом доллара, оставляем как есть
                        if isinstance(total_pnl, str):
                            pnl = total_pnl
                        else:
                            pnl = f"${total_pnl:.2f}"
                    
                    win_rate = run_data.get('Win Rate', 0) or run_data.get('win_rate', 0)
                    if win_rate is None or (isinstance(win_rate, float) and np.isnan(win_rate)):
                        win_rate_formatted = 'N/A'
                    else:
                        # Проверяем, может быть win_rate уже в формате процента
                        if isinstance(win_rate, str) and win_rate.endswith('%'):
                            win_rate_formatted = win_rate
                        elif isinstance(win_rate, (int, float)):
                            # win_rate из симуляции уже в десятичном формате (например, 0.65),
                            # но может быть уже умноженным на 100 в другом месте, поэтому проверяем значение
                            if win_rate > 1:
                                # Если значение больше 1, то оно уже в формате процента
                                win_rate_formatted = f"{win_rate:.2f}%"
                            else:
                                # Если значение <= 1, то это десятичная доля, нужно умножить на 100
                                win_rate_formatted = f"{round(win_rate * 100, 2):.2f}%"
                        else:
                            win_rate_formatted = 'N/A'
            
            # Создаем контейнер для компактного отображения
            with st.container():
                # Используем columns для компактного размещения информации и кнопок
                cols = st.columns([1, 3, 2, 2, 2])  # Распределяем ширину: кнопка раскрытия, название, 2 кнопки, метрики
                
                with cols[0]:
                    # Кнопка для отображения/скрытия таблицы результатов
                    show_results_key = f"show_results_{run_name}"
                    # Проверяем, существует ли уже состояние для этой кнопки, если нет - устанавливаем в False
                    if show_results_key not in st.session_state:
                        st.session_state[show_results_key] = False
                    
                    # Создаем кнопку и проверяем, была ли она нажата
                    button_label = "▼" if st.session_state[show_results_key] else "▶"
                    if st.button(button_label, key=show_results_key + "_button"):
                        # Изменяем состояние при нажатии кнопки
                        st.session_state[show_results_key] = not st.session_state[show_results_key]
                    
                with cols[1]:
                    st.markdown(f"**{run_name}**")
                    
                with cols[2]:
                    if st.button(f"→ Оптимизация", key=f"optimizer_{run_name}"):
                        # Загружаем диапазоны и настройки в модуль 2
                        if run_data is not None:
                            # Загрузка в session_state с правильными ключами
                            optimization_data = {**run_data.get("ranges", {}), **run_data.get("settings", {})}
                            load_profile_to_session_state(optimization_data, "optimization")
                            
                            st.session_state["page"] = "Оптимизация"
                            st.rerun()
                            
                with cols[3]:
                    # Если таблица результатов открыта, кнопка "В анализ" не нужна, так как переход будет через кнопки в таблице
                    if run_data is not None and "results" in run_data and st.session_state.get(f"show_results_{run_name}", False):
                        # Показываем только подсказку, что переход в анализ осуществляется через кнопки в таблице
                        st.caption("→ Анализ (из табл.)")
                    else:
                        # Если таблица результатов не открыта или нет данных, используем лучший результат
                        if st.button(f"→ Анализ", key=f"analysis_from_run_{run_name}"):
                            # Загружаем лучший набор параметров в модуль 1
                            if run_data is not None:
                                # Загрузка параметров в session_state с правильными ключами
                                best_params = run_data.get("best_params", {})
                                analysis_data = {k: v for k, v in best_params.items() if k != 'ID'}  # пропускаем ID, так как это не параметр стратегии
                                analysis_data.update(run_data.get("settings", {}))
                                
                                # Приведение значений к правильному типу для избежания ошибок Streamlit
                                for key, value in analysis_data.items():
                                    if isinstance(value, float) and key not in ['position_size', 'commission', 'stop_loss_pct', 'take_profit_pct', 'vol_pctl', 'rng_pctl', 'natr_min', 'min_growth_pct', 'prints_threshold_ratio']:
                                        # Если значение - float, но ключ не из списка float-параметров, преобразуем в int
                                        if not key.endswith('_analysis'):
                                            try:
                                                analysis_data[key] = int(value)
                                            except (ValueError, TypeError):
                                                pass  # Если не удается преобразовать, оставляем как есть
                                    elif isinstance(value, int) and key in ['position_size', 'commission', 'stop_loss_pct', 'take_profit_pct', 'vol_pctl', 'rng_pctl', 'natr_min', 'min_growth_pct', 'prints_threshold_ratio']:
                                        # Если значение - int, но ключ из списка float-параметров, преобразуем в float
                                        analysis_data[key] = float(value)
                                        
                                load_profile_to_session_state(analysis_data, "analysis")
                                
                                st.session_state["page"] = "Анализ сигналов"
                                st.rerun()
                
                with cols[4]:
                    st.caption(f"PnL: {pnl}, WR: {win_rate_formatted}")
            
            # Отображаем таблицу результатов, если она запрошена
            if st.session_state[show_results_key]:
                # Проверяем, есть ли данные в top_10_results (для Optuna) или в results (для WFO)
                if run_data is not None:
                    # Определяем, какие данные использовать для таблицы
                    if "top_10_results" in run_data and run_data["top_10_results"]:
                        # Для Optuna результатов используем top_10_results
                        results_data = run_data["top_10_results"]
                        # Добавляем ID к каждому результату, если его нет
                        for i, result in enumerate(results_data):
                            if 'ID' not in result:
                                result['ID'] = i + 1
                        results_df = pd.DataFrame(results_data)
                    elif "results" in run_data and run_data["results"]:
                        # Для WFO результатов используем results
                        results_df = pd.DataFrame(run_data["results"])
                    else:
                        st.warning("Нет данных для отображения")
                    
                    # Переупорядочиваем столбцы в соответствии с порядком в результатах оптимизации
                    desired_order = [
                        "ID",
                        "Total Trades", "PnL", "Win Rate", "Max Drawdown", "Sharpe Ratio", "Profit Factor",
                        "vol_pctl", "vol_period", "rng_pctl", "range_period", "natr_min", "natr_period",
                        "min_growth_pct", "lookback_period", "prints_analysis_period", "prints_threshold_ratio",
                        "stop_loss_pct", "take_profit_pct"
                    ]
                    
                    # Убедимся, что все столбцы из desired_order присутствуют в DataFrame
                    available_columns = [col for col in desired_order if col in results_df.columns]
                    # Добавим любые столбцы, которые могут отсутствовать в desired_order, в конец
                    additional_columns = [col for col in results_df.columns if col not in desired_order]
                    final_order = available_columns + additional_columns
                    
                    results_df_display = results_df[final_order]
                    
                    # Отображаем параметры (все результаты теперь используют режим 1)
                    display_df = results_df_display.copy()
                    
                    # Добавляем колонку с кнопками для выбора результата
                    selected_result_key = f"selected_result_{run_name}"
                    cols = st.columns([1, 8])  # Колонка для кнопок и основная таблица
                    with cols[0]:
                        st.write("**Действия**")
                        for i in range(len(results_df)):
                            # Получаем информацию о результате для отображения на кнопке
                            result_row = results_df.iloc[i]
                            result_id = result_row.get('ID', i+1)
                            pnl = result_row.get('PnL', 'N/A')
                            win_rate = result_row.get('Win Rate', 'N/A')
                            
                            # Кнопка с информацией о результате (только значения), которая сразу переходит в анализ
                            if st.button(f"{result_id}\n{pnl}\n{win_rate}", key=f"select_{run_name}_result_{i}"):
                                # Загружаем параметры выбранного результата в session_state с правильными ключами
                                selected_params = {k: v for k, v in result_row.items() if k != 'ID'}  # пропускаем ID, так как это не параметр стратегии
                                selected_params.update(run_data.get("settings", {}))
                                load_profile_to_session_state(selected_params, "analysis")
                                
                                st.session_state["page"] = "Анализ сигналов"
                                st.rerun()
                                
                    with cols[1]:
                        # Отображаем таблицу с корректным отображением параметров в зависимости от режима анализа принтов
                        st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("Нет данных для отображения")
            
            st.markdown("---")
    else:
        st.info("Нет сохранённых прогонов оптимизации")