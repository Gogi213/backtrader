import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from signal_generator import generate_signals


class TradingSimulator:
    """
    Модуль симуляции сделок
    """
    
    def __init__(self, position_size=100.0, commission=0.1, stop_loss_pct=2.0, take_profit_pct=4.0):
       """
       Инициализация симулятора
        
       Args:
           position_size: размер позиции
           commission: комиссия в процентах
           stop_loss_pct: стоп-лосс в процентах
           take_profit_pct: тейк-профит в процентах
           use_hldir: использовать ли HLdir для определения направления (если доступен)
       """
       self.position_size = position_size
       self.commission = commission / 100  # преобразуем в доли
       self.stop_loss_pct = stop_loss_pct
       self.take_profit_pct = take_profit_pct
       self.trades = []
       self.pnl_history = []
       self.balance_history = []
       self.current_position = None
       self.initial_balance = position_size
       # Всегда используем HLdir, если он присутствует в данных
        
    def calculate_stop_loss_take_profit(self, entry_price, direction):
        """
        Расчет стоп-лосса и тейк-профита
        
        Args:
            entry_price: цена входа
            direction: направление позиции ('long' или 'short')
            
        Returns:
            кортеж (stop_loss, take_profit)
        """
        if direction == 'long':
            stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
            take_profit = entry_price * (1 + self.take_profit_pct / 100)
        else:  # short
            stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
            take_profit = entry_price * (1 - self.take_profit_pct / 100)
            
        return stop_loss, take_profit
    
    def open_position(self, idx, entry_price, direction, signal_data):
        """
        Открытие позиции
        
        Args:
            idx: индекс свечи
            entry_price: цена входа (цена открытия следующей свечи)
            direction: направление ('long' или 'short')
            signal_data: данные сигнала
        """
        if self.current_position is not None:
            # Если позиция уже открыта, не открываем новую
            return False
            
        stop_loss, take_profit = self.calculate_stop_loss_take_profit(entry_price, direction)
        
        # Конвертируем значения в стандартные типы Python, чтобы избежать проблем с сериализацией
        self.current_position = {
            'entry_idx': int(idx) if hasattr(idx, 'item') else idx,
            'entry_price': float(entry_price) if hasattr(entry_price, 'item') else entry_price,
            'direction': direction,
            'stop_loss': float(stop_loss) if hasattr(stop_loss, 'item') else stop_loss,
            'take_profit': float(take_profit) if hasattr(take_profit, 'item') else take_profit,
            'exit_idx': None,
            'exit_price': None,
            'pnl': None,
            'signal_data': signal_data
        }
        
        return True
    
    def close_position(self, idx, exit_price, reason):
        """
        Закрытие позиции
        
        Args:
            idx: индекс свечи
            exit_price: цена выхода
            reason: причина закрытия ('stop_loss', 'take_profit', 'signal_reverse', 'end_of_data')
        """
        if self.current_position is None:
            return 0  # Нет позиции для закрытия
            
        # Рассчитываем PNL
        entry_price = self.current_position['entry_price']
        direction = self.current_position['direction']
        
        if direction == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) / entry_price
            
        # Учитываем размер позиции
        pnl_amount = self.position_size * pnl
        
        # Учитываем комиссии (вход + выход)
        commission_amount = (self.position_size * self.commission) * 2
            
        pnl_amount -= commission_amount
        
        # Проверяем, что сделка приносит минимальную прибыль после вычета комиссий
        # Если прибыль слишком мала или убыток, считаем, что сделка не состоялась
        if abs(pnl_amount) < (commission_amount * 0.5):  # Порог в половину комиссии
            return 0 # Не учитываем сделку как успешную
        
        # Обновляем позицию
        # Конвертируем значения в стандартные типы Python, чтобы избежать проблем с сериализацией
        self.current_position['exit_idx'] = int(idx)
        self.current_position['exit_price'] = float(exit_price)
        self.current_position['pnl'] = float(pnl_amount)
        self.current_position['exit_reason'] = reason
        
        # Добавляем в историю сделок
        self.trades.append(self.current_position.copy())
        
        # Сохраняем PNL в историю
        self.pnl_history.append(pnl_amount)
        
        # Обновляем историю баланса
        balance_value = self.initial_balance + pnl_amount if len(self.balance_history) == 0 else self.balance_history[-1] + pnl_amount
            
        self.balance_history.append(balance_value)
        
        # Сбрасываем текущую позицию
        self.current_position = None
        
        return pnl_amount
    
    def simulate_trades(self, df, params, hldir_window=10, aggressive_mode=False):
        """
        Симуляция торговли

        Args:
            df: DataFrame с рыночными данными
            params: параметры стратегии
            hldir_window: размер окна для усреднения HLdir значений (не используется в текущей реализации)
            aggressive_mode: если True, позволяет открывать новые сделки, не дожидаясь закрытия старых

        Returns:
            словарь с результатами симуляции
        """
        # Сброс состояния
        self.trades = []
        self.pnl_history = []
        self.balance_history = []
        self.current_position = None

        # Извлекаем параметры для определения направления
        prints_threshold_ratio = params.get("prints_threshold_ratio", 1.0)
        prints_analysis_period = params.get("prints_analysis_period", 2)
        entry_logic_mode = params.get("entry_logic_mode", "Принты и HLdir")

        # Генерируем сигналы
        signal_indices = set(generate_signals(df, params)) # используем set для быстрой проверки
        
              
        # Используем только CPU (numpy)
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        long_prints_values = df['long_prints'].values if 'long_prints' in df.columns else None
        short_prints_values = df['short_prints'].values if 'short_prints' in df.columns else None
        hldir_values = df['HLdir'].values if 'HLdir' in entry_logic_mode and 'HLdir' in df.columns else None

        # --- Начало векторизованной логики ---
        xp = np

        # 1. Предварительно рассчитываем индикаторы для определения направления
        # Эти расчеты производятся для всех свечей, чтобы потом быстро извлекать значения
        
        # Расчет направления по принтам
        prints_long_signals = xp.zeros(len(df), dtype=bool)
        prints_short_signals = xp.zeros(len(df), dtype=bool)
        if long_prints_values is not None and short_prints_values is not None:
            long_sum = pd.Series(long_prints_values).rolling(window=prints_analysis_period, min_periods=1).sum().values
            short_sum = pd.Series(short_prints_values).rolling(window=prints_analysis_period, min_periods=1).sum().values
            long_sum = xp.asarray(long_sum)
            short_sum = xp.asarray(short_sum)

            ratios = xp.full_like(long_sum, float('inf'))
            non_zero_short = short_sum > 0
            ratios[non_zero_short] = long_sum[non_zero_short] / short_sum[non_zero_short]

            prints_long_signals = ratios > prints_threshold_ratio
            prints_short_signals = ratios < (1 / prints_threshold_ratio)

        # Расчет направления по HLdir
        hldir_long_signals = xp.zeros(len(df), dtype=bool)
        hldir_short_signals = xp.zeros(len(df), dtype=bool)
        if "HLdir" in entry_logic_mode:
            # Убеждаемся, что hldir_window передано, иначе используем значение по умолчанию
            hldir_window = params.get("hldir_window", 10)
            hldir_offset = params.get("hldir_offset", 0)
            hldir_rolling_mean = pd.Series(hldir_values).rolling(window=hldir_window, min_periods=1).mean().shift(hldir_offset).values
            hldir_rolling_mean = xp.asarray(hldir_rolling_mean)
            
            # Условие для восходящего тренда HLdir
            hldir_long_signals = hldir_rolling_mean > 0.5

            # Условие для нисходящего тренда HLdir
            hldir_short_signals = hldir_rolling_mean <= 0.5

        # 2. Итерируемся по сигналам, а не по всем свечам
        entry_indices = sorted(list(signal_indices))
        if not entry_indices: # Если нет сигналов, выходим
            return self.calculate_metrics()

        current_idx = 0
        signal_queue = sorted(list(signal_indices))

        while signal_queue:
            # Берем следующий сигнал, который идет после закрытия последней сделки
            next_signal_idx = -1
            for s_idx in signal_queue:
                if s_idx > current_idx:
                    next_signal_idx = s_idx
                    break
            
            if next_signal_idx == -1:
                break # Больше нет подходящих сигналов

            # Удаляем обработанный сигнал из очереди
            signal_queue.pop(0)

            # --- Логика с подтверждением на i и входом на i+1 ---
            analysis_idx = next_signal_idx
            if analysis_idx >= len(df) - 1:
                continue # Не можем войти, так как это предпоследняя свеча

            # Определяем направления по разным источникам на сигнальной свече (i)
            prints_dir = 'long' if prints_long_signals[analysis_idx] else ('short' if prints_short_signals[analysis_idx] else None)
            hldir_dir = 'long' if hldir_long_signals[analysis_idx] else ('short' if hldir_short_signals[analysis_idx] else None)

            # Финальное решение о направлении входа в зависимости от выбранного режима
            direction = None
            if entry_logic_mode == "Принты и HLdir":
                if prints_dir is not None and prints_dir == hldir_dir:
                    direction = prints_dir
            elif entry_logic_mode == "Только по принтам":
                direction = prints_dir
            elif entry_logic_mode == "Только по HLdir":
                direction = hldir_dir

            if direction is None:
                current_idx = analysis_idx
                continue # Сигнал игнорируется, если нет четкого направления

            # Вход происходит на следующей свече (i+1)
            entry_idx = analysis_idx + 1
            entry_price = open_prices[entry_idx]

            # --- Общая логика открытия и закрытия позиции ---
            if direction:
                stop_loss, take_profit = self.calculate_stop_loss_take_profit(entry_price, direction)

                # Векторизованный поиск точки выхода
                future_indices = xp.arange(entry_idx, len(df))
                if len(future_indices) == 0:
                    break

                future_highs = high_prices[future_indices]
                future_lows = low_prices[future_indices]

                if direction == 'long':
                    sl_triggers = future_lows <= stop_loss
                    tp_triggers = future_highs >= take_profit
                else: # short
                    sl_triggers = future_highs >= stop_loss
                    tp_triggers = future_lows <= take_profit

                sl_hit_idx = xp.argmax(sl_triggers) if xp.any(sl_triggers) else -1
                tp_hit_idx = xp.argmax(tp_triggers) if xp.any(tp_triggers) else -1

                if sl_hit_idx != -1 and (tp_hit_idx == -1 or sl_hit_idx <= tp_hit_idx):
                    exit_idx = future_indices[sl_hit_idx]
                    exit_reason = 'stop_loss'
                    # В агрессивном режиме ищем новый сигнал до срабатывания стопа
                    if aggressive_mode:
                        next_signal_in_trade = next((s for s in signal_queue if entry_idx < s < exit_idx), None)
                        if next_signal_in_trade:
                            exit_idx = next_signal_in_trade -1 # Выходим на свече перед новым сигналом
                            exit_reason = 'new_signal'
                            exit_price = close_prices[exit_idx]
                        else:
                            exit_price = stop_loss # This was the source of the error
                    else:
                        exit_price = stop_loss
                elif tp_hit_idx != -1 and (sl_hit_idx == -1 or tp_hit_idx < sl_hit_idx):
                    exit_idx = future_indices[tp_hit_idx]
                    exit_reason = 'take_profit'
                    exit_price = take_profit
                else: # Если не сработал ни SL, ни TP
                    exit_idx = -1 # Сигнал, что сделка не закрылась до конца данных
                    exit_reason = 'end_of_data'
                    exit_price = close_prices[-1]

                if exit_idx != -1:
                    # Записываем сделку
                    trade = {
                        'entry_idx': entry_idx,
                        'entry_price': entry_price,
                        'direction': direction,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'exit_idx': int(exit_idx),
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'signal_data': {} # Упрощено для скорости
                    }
                    self.trades.append(trade)
                    current_idx = trade['exit_idx']
        # --- Конец векторизованной логики ---

        # Закрываем оставшуюся позицию в конце данных
        if self.current_position is not None:
            final_price = df['close'].iloc[-1]  # используем цену закрытия последней свечи
            self.close_position(len(df) - 1, final_price, 'end_of_data')
        
        # Рассчитываем PnL для всех сделок после цикла
        for trade in self.trades:
            if trade['direction'] == 'long':
                pnl = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            else: # short
                pnl = (trade['entry_price'] - trade['exit_price']) / trade['entry_price']
            
            pnl_amount = self.position_size * pnl - (self.position_size * self.commission) * 2
            trade['pnl'] = pnl_amount
            self.pnl_history.append(pnl_amount)
            
            balance_value = self.initial_balance + pnl_amount if len(self.balance_history) == 0 else self.balance_history[-1] + pnl_amount
            self.balance_history.append(balance_value)

        return self.calculate_metrics()

    def calculate_metrics(self):
        """Вычисляет итоговые метрики после симуляции."""
        # Рассчитываем метрики с использованием векторизованных операций
        if self.pnl_history:            
            pnl_array = np.array(self.pnl_history)
            total_trades = len(pnl_array)
            winning_trades = np.sum(pnl_array > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = np.sum(pnl_array)
            
            # Рассчитываем дополнительные метрики
            avg_pnl = np.mean(pnl_array)
            profits = pnl_array[pnl_array > 0]
            losses = np.abs(pnl_array[pnl_array < 0])
            total_profit = np.sum(profits) if len(profits) > 0 else 0
            total_loss = np.sum(losses) if len(losses) > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Рассчитываем максимальную просадку
            combined_balance = [self.initial_balance] + self.balance_history
            balance_array = np.array(combined_balance)
            running_max = np.maximum.accumulate(balance_array)
            drawdowns = (running_max - balance_array) / running_max
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        else:
            total_trades = 0
            winning_trades = 0
            win_rate = 0
            total_pnl = 0
            avg_pnl = 0
            profit_factor = 0
            max_drawdown = 0
        
        # Обеспечиваем, что все значения в результатах являются обычными Python или NumPy значениями
        results = {
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'win_rate': float(win_rate),
            'total_pnl': float(total_pnl),
            'avg_pnl': float(avg_pnl),
            'max_drawdown': float(max_drawdown),
            'profit_factor': float(profit_factor),
            'trades': self.trades,
            'pnl_history': self.pnl_history,
            'balance_history': self.balance_history,
            'final_balance': self.balance_history[-1] if self.balance_history else self.initial_balance
        }
        
        return results

    def plot_trades_on_chart(self, df, results, show_trades=True, show_balance=True, params=None):
       """
       Отображение сделок на графике
       
       Args:
           df: DataFrame с рыночными данными
           results: результаты симуляции
           show_trades: показывать ли сделки на графике
           show_balance: показывать ли баланс на отдельной панели
           params: параметры стратегии для отображения сигналов
           
       Returns:
           объект plotly figure
       """
       # Создаем subplot с двумя панелями: одна для свечей, одна для баланса
       if show_balance:
           fig = make_subplots(
               rows=2, cols=1,
               shared_xaxes=True,
               vertical_spacing=0.05,
               row_heights=[0.7, 0.3]  # Увеличиваем высоту графика цены
           )
       else:
           fig = make_subplots(
               rows=1, cols=1,
               shared_xaxes=True
           )
       
       # Векторизованная подготовка hover текста для свечей, включая объем
       hover_texts = (
           "O: " + df['open'].astype(str) +
           "<br>H: " + df['high'].astype(str) +
           "<br>L: " + df['low'].astype(str) +
           "<br>C: " + df['close'].astype(str) +
           "<br>V: " + df['volume'].round(2).astype(str)
       ).tolist()

       # Добавляем свечной график с пользовательскими подсказками
       fig.add_trace(go.Candlestick(
           x=df['datetime'],
           open=df['open'],
           high=df['high'],
           low=df['low'],
           close=df['close'],
           name='Цена',
           increasing_line_color='green',
           decreasing_line_color='red',
           line_width=0.5,  # Уменьшаем толщину свечей
           hovertext=hover_texts,
           hoverinfo="x+text" # 'x' для даты, 'text' для нашего текста
       ), row=1, col=1)
       
       # Добавляем линию баланса, если нужно
       if show_balance and results['balance_history']:
           combined_balance = [self.initial_balance] + results['balance_history']
           balance_array = np.array(combined_balance)
           
           # Создаем массив дат для баланса - начальная дата плюс даты для каждой сделки
           # Используем векторизованные операции для подготовки дат
           trade_exit_indices = [int(trade['exit_idx']) for trade in results['trades']]
           balance_dates = [df['datetime'].iloc[0]]  # начальный баланс на первую дату
           balance_dates.extend(df['datetime'].iloc[trade_exit_indices].values)
           
           # Если количество дат не совпадает с количеством значений баланса,
           # дополним массив дат последней датой или расширим как нужно
           if len(balance_dates) < len(balance_array):
               # Если у нас меньше дат, чем значений баланса, дополним последней датой
               last_date = df['datetime'].iloc[-1] if len(df) > 0 else balance_dates[-1]
               balance_dates.extend([last_date] * (len(balance_array) - len(balance_dates)))
           elif len(balance_dates) > len(balance_array):
               # Если больше дат, чем значений баланса, усечем
               balance_dates = balance_dates[:len(balance_array)]
           
           fig.add_trace(go.Scatter(
               x=balance_dates,
               y=balance_array,
               mode='lines',
               name='Баланс',
               line=dict(color='blue', width=2),
               hovertemplate='Дата: %{x}<br>Баланс: %{y:.2f}<extra></extra>'
           ), row=2, col=1)
       
       # Добавляем все сгенерированные сигналы на график
       if params:
           all_signal_indices = generate_signals(df, params)
           if all_signal_indices:
               signal_datetimes = df['datetime'].iloc[all_signal_indices]
               signal_prices = df['low'].iloc[all_signal_indices] * 0.998 # Чуть ниже минимума свечи
               fig.add_trace(go.Scatter(
                   x=signal_datetimes,
                   y=signal_prices,
                   mode='markers',
                   marker=dict(
                       symbol='square',
                       size=5,
                       color='white',
                       line=dict(width=1, color='gray')
                   ),
                   name='Все сигналы',
                   hoverinfo='none' # Отключаем hover для этих маркеров, чтобы не загромождать
               ), row=1, col=1)


       # Отображаем на графике только те сигналы, на которых были открыты сделки
       # Используем информацию из истории сделок
       
       # Добавляем сделки на график, если нужно
       if show_trades and results['trades']:
           trades_df = pd.DataFrame(results['trades'])

           # --- Векторизованная отрисовка входов ---
           long_entries = trades_df[trades_df['direction'] == 'long']
           short_entries = trades_df[trades_df['direction'] == 'short']

           # --- Векторизованная отрисовка выходов ---
           long_exits = trades_df[trades_df['direction'] == 'long']
           short_exits = trades_df[trades_df['direction'] == 'short']

           long_tp_exits = long_exits[long_exits['exit_reason'] == 'take_profit']
           long_sl_exits = long_exits[long_exits['exit_reason'] == 'stop_loss']
           short_tp_exits = short_exits[short_exits['exit_reason'] == 'take_profit']
           short_sl_exits = short_exits[short_exits['exit_reason'] == 'stop_loss']
           
           # Добавляем точки входа (теперь раздельно для long и short)
           if not long_entries.empty:
               fig.add_trace(go.Scatter(
                   x=df['datetime'].iloc[long_entries['entry_idx']],
                   y=long_entries['entry_price'],
                   mode='markers',
                   marker=dict(
                       symbol='triangle-up',  # Треугольник вверх для long
                       size=12,
                       color='white',  # Белый цвет для long
                       line=dict(width=2, color='lime')  # Зеленая граница для long
                   ),
                   name='Вход в LONG',
                   hovertemplate='Вход в LONG<br>Цена: %{y:.5f}<extra></extra>'
               ), row=1, col=1)
               
           if not short_entries.empty:
               fig.add_trace(go.Scatter(
                   x=df['datetime'].iloc[short_entries['entry_idx']],
                   y=short_entries['entry_price'],
                   mode='markers',
                   marker=dict(
                       symbol='triangle-down',  # Треугольник вниз для short
                       size=12,
                       color='white',  # Белый цвет для short
                       line=dict(width=2, color='red')  # Красная граница для short
                   ),
                   name='Вход в SHORT',
                   hovertemplate='Вход в SHORT<br>Цена: %{y:.5f}<extra></extra>'
               ), row=1, col=1)
           
           # Добавляем точки выхода по тейк-профиту
           if not long_tp_exits.empty:
               fig.add_trace(go.Scatter(
                   x=df['datetime'].iloc[long_tp_exits['exit_idx']],
                   y=long_tp_exits['exit_price'],
                   mode='markers',
                   marker=dict(
                       symbol='x',
                       size=10,
                       color='green',  # Зеленый для тейк-профита
                       line=dict(width=2, color='darkgreen')
                   ),
                   name='Выход из LONG (Тейк-профит)',
                   hovertemplate='Выход из LONG (Тейк-профит)<br>Цена: %{y:.5f}<extra></extra>'
               ), row=1, col=1)
               
           if not short_tp_exits.empty:
               fig.add_trace(go.Scatter(
                   x=df['datetime'].iloc[short_tp_exits['exit_idx']],
                   y=short_tp_exits['exit_price'],
                   mode='markers',
                   marker=dict(
                       symbol='x',
                       size=10,
                       color='green',  # Зеленый для тейк-профита
                       line=dict(width=2, color='darkgreen')
                   ),
                   name='Выход из SHORT (Тейк-профит)',
                   hovertemplate='Выход из SHORT (Тейк-профит)<br>Цена: %{y:.5f}<extra></extra>'
               ), row=1, col=1)
           
           # Добавляем точки выхода по стоп-лосс
           if not long_sl_exits.empty:
               fig.add_trace(go.Scatter(
                   x=df['datetime'].iloc[long_sl_exits['exit_idx']],
                   y=long_sl_exits['exit_price'],
                   mode='markers',
                   marker=dict(
                       symbol='x',
                       size=10,
                       color='blue',  # Синий для стоп-лосса
                       line=dict(width=2, color='darkblue')
                   ),
                   name='Выход из LONG (Стоп-лосс)',
                   hovertemplate='Выход из LONG (Стоп-лосс)<br>Цена: %{y:.5f}<extra></extra>'
               ), row=1, col=1)
               
           if not short_sl_exits.empty:
               fig.add_trace(go.Scatter(
                   x=df['datetime'].iloc[short_sl_exits['exit_idx']],
                   y=short_sl_exits['exit_price'],
                   mode='markers',
                   marker=dict(
                       symbol='x',
                       size=10,
                       color='blue',  # Синий для стоп-лосса
                       line=dict(width=2, color='darkblue')
                   ),
                   name='Выход из SHORT (Стоп-лосс)',
                   hovertemplate='Выход из SHORT (Стоп-лосс)<br>Цена: %{y:.5f}<extra></extra>'
               ), row=1, col=1)
       
       # Настраиваем макет
       fig.update_layout(
           title='График цен с отображением сделок и баланса',
           height=1200 if show_balance else 1000,
           dragmode='zoom',  # Изменяем режим перетаскивания на zoom для увеличения по обеим осям
           showlegend=True,
           xaxis_rangeslider_visible=False,  # Отключаем встроенный range slider, чтобы избежать конфликта с балансом
           yaxis=dict(fixedrange=False) # Разрешаем изменение масштаба по оси Y
       )
       
       # Настройка осей
       if show_balance:
           fig.update_xaxes(title_text='Дата', row=2, col=1)
           fig.update_yaxes(title_text='Цена', row=1, col=1)
           fig.update_yaxes(title_text='Баланс', row=2, col=1)
       else:
           fig.update_xaxes(title_text='Дата')
           fig.update_yaxes(title_text='Цена', row=1, col=1)
       
       return fig


def run_trading_simulation(df, params):
    """
    Запуск симуляции торговли
     
    Args:
        df: DataFrame с рыночными данными
        params: параметры стратегии
        
    Returns:
        результаты симуляции
    """
    
    # Извлекаем параметры для симулятора и округляем float параметры до 2 знаков после запятой
    position_size = round(params.get("position_size", 100.0), 2)
    commission = round(params.get("commission", 0.1), 3)  # Используем 3 знака для комиссии из-за малых значений
    stop_loss_pct = round(params.get("stop_loss_pct", 2.0), 2)
    take_profit_pct = round(params.get("take_profit_pct", 4.0), 2)
    hldir_window = params.get("hldir_window", 3) # Размер окна для усреднения HLdir (не используется в текущей реализации)
    aggressive_mode = params.get("aggressive_mode", False) # Получаем режим симуляции
    
    # Создаем симулятор
    simulator = TradingSimulator(
        position_size=position_size,
        commission=commission,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
        # Всегда используем HLdir, если он присутствует в данных
    )
    
    # Запускаем симуляцию
    results = simulator.simulate_trades(df, params, hldir_window, aggressive_mode)
    
    return results