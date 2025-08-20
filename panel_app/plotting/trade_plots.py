import panel as pn
import plotly.graph_objs as go
import pandas as pd

def plot_cumulative_profit(trades):
    y_data = None
    y_label = ''
    if 'pnl_net' in trades.columns:
        trades['cum_pnl_net'] = trades['pnl_net'].cumsum()
        y_data = trades['cum_pnl_net']
        y_label = 'Cumulative Profit ($, net)'
    elif 'pnl' in trades.columns:
        trades['cum_pnl'] = trades['pnl'].cumsum()
        y_data = trades['cum_pnl']
        y_label = 'Cumulative Profit ($, gross)'
    if y_data is not None:
        area_trades_fig = go.Figure()
        # Используем время входа, если доступно, иначе индекс сделок
        x_time = trades['entry_time'] if 'entry_time' in trades.columns else (trades.index + 1)
        # Начинаем кумулятивную кривую с нуля: добавим стартовую точку (x0, 0)
        try:
            x0 = x_time.iloc[0] if hasattr(x_time, 'iloc') else (x_time[0] if len(x_time) > 0 else None)
        except Exception:
            x0 = None
        if x0 is not None and len(y_data) > 0:
            xs = [x0] + list(x_time)
            ys = [0] + list(y_data)
        else:
            xs = x_time
            ys = y_data
        area_trades_fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            fill='tozeroy',
            mode='lines',
            name=y_label,
            line=dict()
        ))
        area_trades_fig.update_layout(
            title='Cumulative Profit (Area)',
            height=600,
            xaxis_title='Время входа',
            yaxis_title=y_label,
        )
        return pn.pane.Plotly(area_trades_fig, config={'responsive': True}, sizing_mode='stretch_width')
    return pn.pane.Markdown('⚠️ Нет данных для графика cumulative profit.')

def plot_trade_profits(trades):
    try:
        if 'pnl_net' in trades.columns:
            y_profit = trades['pnl_net']
        elif 'pnl' in trades.columns:
            y_profit = trades['pnl']
        else:
            y_profit = None
        x_time = trades['entry_time'] if 'entry_time' in trades.columns else trades.index
        if y_profit is not None and x_time is not None:
            profit_fig = go.Figure()
            # Цвета по знаку прибыли: зелёный / красный / серый для нулевой
            vals = y_profit.to_numpy()
            colors = [
                ('#2ca02c' if v > 0 else ('#d62728' if v < 0 else '#8c8c8c'))
                for v in vals
            ]
            profit_fig.add_trace(go.Bar(
                x=x_time,
                y=y_profit,
                name='Trade Profit',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(0,0,0,0.35)', width=0.4)
                ),
                opacity=0.9,
                hovertemplate='%{x}<br>PNL: %{y:.4f}<extra></extra>'
            ))
            profit_fig.update_layout(
                title='Прибыль по сделкам',
                height=500,
                xaxis_title='Время входа',
                yaxis_title='Прибыль по сделке',
                bargap=0.1,
                showlegend=False,
                plot_bgcolor='white'
            )
            profit_fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
            profit_fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.08)')
            return pn.pane.Plotly(profit_fig, config={'responsive': True}, sizing_mode='stretch_width')
        else:
            return pn.pane.Markdown('⚠️ Нет данных по прибыли сделок.')
    except Exception as eqerr:
        return pn.pane.Markdown(f'❌ Ошибка при построении графика прибыли по сделкам: {eqerr}')

def plot_candlestick_with_trades(df, trades, disable_trades_chart):
    if disable_trades_chart:
        return None
    try:
        candle_fig = go.Figure()
        candle_fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            showlegend=False
        ))
        if 'entry_idx' in trades.columns and 'exit_idx' in trades.columns:
            show_trades = trades.head(100)
            for _, trade in show_trades.iterrows():
                entry_time = trade.get('entry_time')
                entry_price = trade.get('entry_price')
                exit_time = trade.get('exit_time')
                exit_price = trade.get('exit_price')
                side = trade.get('side', None)
                if side is None and all(x in trade for x in ['entry_price','exit_price','pnl']):
                    if trade['entry_price'] == trade['exit_price'] or trade['pnl'] == 0:
                        side = 'flat'
                    elif (trade['exit_price'] > trade['entry_price'] and trade['pnl'] > 0) or (trade['exit_price'] < trade['entry_price'] and trade['pnl'] < 0):
                        side = 'long'
                    elif (trade['exit_price'] < trade['entry_price'] and trade['pnl'] > 0) or (trade['exit_price'] > trade['entry_price'] and trade['pnl'] < 0):
                        side = 'short'
                    else:
                        side = 'unknown'
                if side == 'short':
                    entry_marker = dict(symbol='triangle-down', size=14, line=dict(width=2, color='black'))
                    exit_marker = dict(symbol='triangle-up', size=14, line=dict(width=2, color='black'))
                else:
                    entry_marker = dict(symbol='triangle-up', size=14, line=dict(width=2, color='black'))
                    exit_marker = dict(symbol='triangle-down', size=14, line=dict(width=2, color='black'))
                candle_fig.add_trace(go.Scatter(
                    x=[entry_time],
                    y=[entry_price],
                    mode='markers',
                    marker=entry_marker,
                    name='Entry',
                    showlegend=False
                ))
                candle_fig.add_trace(go.Scatter(
                    x=[exit_time],
                    y=[exit_price],
                    mode='markers',
                    marker=exit_marker,
                    name='Exit',
                    showlegend=False
                ))
                candle_fig.add_trace(go.Scatter(
                    x=[entry_time, exit_time],
                    y=[entry_price, exit_price],
                    mode='lines',
                    line=dict(width=2, dash='dot'),
                    name='Trade',
                    showlegend=False
                ))
        candle_fig.update_layout(title='Candlestick Chart with Trades', height=900, xaxis_title='Time', yaxis_title='Price', xaxis_rangeslider_visible=False)
        return pn.pane.Plotly(candle_fig, config={'responsive': True}, sizing_mode='stretch_width')
    except Exception as candle_err:
        return pn.pane.Markdown(f'❌ Ошибка при построении свечного графика: {candle_err}')
