import plotly.graph_objs as go
import pandas as pd


def plot_trades_with_plotly(ohlcv_df: pd.DataFrame, trades_df: pd.DataFrame, title: str = "Backtest Chart with Trades"):
    """
    ohlcv_df: DataFrame с колонками ['open', 'high', 'low', 'close', 'volume'], индекс — datetime
    trades_df: DataFrame с колонками ['entry_time', 'entry_price', 'exit_time', 'exit_price', ...]
    """
    fig = go.Figure()
    # Свечи
    fig.add_trace(go.Candlestick(
        x=ohlcv_df.index,
        open=ohlcv_df['open'],
        high=ohlcv_df['high'],
        low=ohlcv_df['low'],
        close=ohlcv_df['close'],
        name='OHLC',
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    ))
    # Сделки
    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            entry_time = trade.get('entry_time')
            entry_price = trade.get('entry_price')
            exit_time = trade.get('exit_time')
            exit_price = trade.get('exit_price')
            # Вход
            fig.add_trace(go.Scatter(
                x=[entry_time],
                y=[entry_price],
                mode='markers',
                marker=dict(symbol='triangle-up', color='lime', size=14, line=dict(width=2, color='black')),
                name='Entry',
                showlegend=False
            ))
            # Выход
            fig.add_trace(go.Scatter(
                x=[exit_time],
                y=[exit_price],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=14, line=dict(width=2, color='black')),
                name='Exit',
                showlegend=False
            ))
            # Линия между входом и выходом (по времени и цене)
            fig.add_trace(go.Scatter(
                x=[entry_time, exit_time],
                y=[entry_price, exit_price],
                mode='lines',
                line=dict(color='royalblue', width=2, dash='dot'),
                name='Trade',
                showlegend=False
            ))
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=700
    )
    return fig
