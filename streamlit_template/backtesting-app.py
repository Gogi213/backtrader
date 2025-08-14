# app.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import backtrader as bt
import plotly_trade_chart
import asyncio
import importlib.util
import os
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'fetch_binance_data_fast.py'))
spec = importlib.util.spec_from_file_location('fetch_binance_data_fast', data_path)
fetch_binance_data_fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_binance_data_fast)
import altair as alt

# Function to run backtest and generate HTML report
def run_backtest(strategy_class, symbol, start_date, end_date, timeframe, **params):
    # ...existing code...
    cerebro = bt.Cerebro()
    # Извлекаем параметры для брокера
    market_commission = params.pop('market_commission', 0.05)
    limit_commission = params.pop('limit_commission', 0.02)
    deposit = params.pop('deposit', 10000)
    order_type = params.pop('order_type', 'Market')
    leverage = params.pop('leverage', 1)
    # Устанавливаем депозит
    cerebro.broker.setcash(deposit)
    # Устанавливаем комиссию и плечо (по типу ордера)
    if order_type == 'Market':
        cerebro.broker.setcommission(commission=market_commission / 100, leverage=leverage)
    else:
        cerebro.broker.setcommission(commission=limit_commission / 100, leverage=leverage)
    # Загрузка с Binance через быстрый асинхронный загрузчик
    start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    save_path = f'data/{symbol}_{timeframe}_{start_str}_{end_str}_fast.csv'
    df = asyncio.run(fetch_binance_data_fast.fetch_binance_ohlcv_fast(symbol, timeframe, start_str, end_str, save_path))
    if df.empty:
        raise ValueError(f"No candles returned for {symbol} {timeframe} in selected date range. Check symbol, timeframe, or date range.")
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    # Pass only strategy params
    # Удаляем дублирующийся ключ 'leverage' из params, если есть
    if 'leverage' in params:
        del params['leverage']
    # Логируем параметры стратегии
    print(f"[DEBUG] addstrategy params: {dict(leverage=leverage, **params)}")
    cerebro.addstrategy(strategy_class, leverage=leverage, **params)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.broker.setcash(100.)
    # Run the backtest
    results = cerebro.run()
    strat = results[0]
    # DEBUG: print executed_trades
    if hasattr(strat, 'executed_trades'):
        print(f"[DEBUG] executed_trades: {strat.executed_trades}")
    # Метрики
    trades = strat.analyzers.trades.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    final_value = cerebro.broker.getvalue()
    def extract_total(val):
        if isinstance(val, dict):
            return val.get('total', 0)
        return val
    # Net Profit = финал - стартовый депозит
    pnl_net = final_value - deposit
    # Корректный расчет total commission по всем сделкам
    total_comm = 0
    if hasattr(strat, 'completed_trades') and strat.completed_trades:
        for trade in strat.completed_trades:
            # Для каждой сделки считаем комиссию на вход и выход
            entry_comm = abs(trade['size']) * trade['entry_price'] * (market_commission / 100 if order_type == 'Market' else limit_commission / 100)
            exit_comm = abs(trade['size']) * trade['exit_price'] * (market_commission / 100 if order_type == 'Market' else limit_commission / 100)
            total_comm += entry_comm + exit_comm
    won_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
    lost_pnl = trades.get('lost', {}).get('pnl', {}).get('total', 0)
    profit_factor = won_pnl / abs(lost_pnl) if lost_pnl else float('inf')
    won_total = trades.get('won', {}).get('total', 0)
    total_trades = trades.get('total', {}).get('total', 0)
    winrate = (won_total / total_trades) * 100 if total_trades else 0
    max_win_streak = trades.get('streak', {}).get('won', {}).get('max', 0)
    max_loss_streak = trades.get('streak', {}).get('lost', {}).get('max', 0)
    max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
    # Возвращаем метрики и график
    returns = strat.analyzers.returns.get_analysis()
    returns_df = pd.DataFrame(list(returns.items()), columns=['Date', 'Return'])
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
    returns_df.set_index('Date', inplace=True)
    # Гарантируем старт с initial deposit
    if not returns_df.empty:
        first_date = returns_df.index.min()
        # Добавляем только если первой точки с Return=0 нет
        if returns_df.iloc[0]['Return'] != 0:
            returns_df = pd.concat([
                pd.DataFrame({'Return': [0]}, index=[first_date]),
                returns_df
            ]).sort_index()
        # Удаляем возможные дубликаты по дате, оставляя первую
        returns_df = returns_df[~returns_df.index.duplicated(keep='first')]
    # Equity curve по завершённым сделкам
    trades_df = pd.DataFrame()
    if hasattr(strat, 'get_trades_df'):
        trades_df = strat.get_trades_df()
    if not trades_df.empty and 'pnl' in trades_df.columns:
        trades_df = trades_df.reset_index(drop=True)
        trades_df['CumulativePnL'] = trades_df['pnl'].cumsum()
        equity_df = pd.DataFrame({'Trade': trades_df.index + 1, 'Equity': trades_df['CumulativePnL']})
        equity_chart = alt.Chart(equity_df).mark_area(
            color='royalblue',
            opacity=0.4
        ).encode(
            x=alt.X('Trade:O', title='Trade Number'),
            y=alt.Y('Equity:Q', title='Cumulative PnL')
        ).properties(title='Equity Curve (Cumulative PnL by Trade Number)', height=700)
    else:
        # fallback: пустой график
        equity_df = pd.DataFrame({'Trade': [0], 'Equity': [0]})
        equity_chart = alt.Chart(equity_df).mark_area(
            color='royalblue',
            opacity=0.4
        ).encode(
            x=alt.X('Trade:O', title='Trade Number'),
            y=alt.Y('Equity:Q', title='Cumulative PnL')
        ).properties(title='Equity Curve (Cumulative PnL by Trade Number)', height=700)

    # Для Plotly: сохраняем OHLCV и сделки
    ohlcv_df = df.copy()
    trades_df = pd.DataFrame()
    if hasattr(strat, 'get_trades_df'):
        trades_df = strat.get_trades_df()
    ohlc_plot = plotly_trade_chart.plot_trades_with_plotly(ohlcv_df, trades_df)

    metrics = {
        'Net Profit': pnl_net,
        'Total Commission': total_comm,
        'Profit Factor': profit_factor,
        'Max Drawdown (%)': max_drawdown,
        'Winrate (%)': winrate,
        'Max Win Streak': max_win_streak,
        'Max Loss Streak': max_loss_streak,
        'Final Portfolio Value': final_value,
        'Leverage': leverage
    }
    return (equity_chart, ohlc_plot), metrics

# Streamlit app
def main():

    st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"], .main, .block-container {
            max-width: 100vw !important;
            width: 100vw !important;
            min-width: 100vw !important;
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
            overflow-x: hidden !important;
        }
        [data-testid="stSidebar"], .stSidebar {
            max-width: 100vw !important;
            width: 100vw !important;
        }
        .element-container, .stColumn, .stDataFrameContainer, .stTable, .stAltairChart, .stEchartsChart {
            width: 100% !important;
            max-width: 100vw !important;
        }
        .stTable, .stDataFrameContainer {
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('Backtest Trading Strategies')
    col1, col2 = st.columns([0.5, 2.5], gap="large")
    with st.container():
        with col1:
            import strategies
            strategy_names = [name for name, obj in vars(strategies).items() if name.endswith('Strategy') and isinstance(obj, type)]
            if not strategy_names:
                st.error('Нет доступных стратегий. Проверьте импорты в strategies/__init__.py')
                return
            default_strategy = 'MomentumBreakoutStrategy' if 'MomentumBreakoutStrategy' in strategy_names else strategy_names[0]
            selected_strategy = st.selectbox('Strategy', strategy_names, index=strategy_names.index(default_strategy))
            selected_strategy_class = getattr(strategies, selected_strategy)
            def to_number(s):
                n = float(s)
                return int(n) if n.is_integer() else n
            strategy_params = {}
            for param_name in dir(selected_strategy_class.params):
                if not param_name.startswith("_") and param_name not in ['isdefault', 'notdefault', 'leverage']:
                    param_value = getattr(selected_strategy_class.params, param_name)
                    strategy_params[param_name] = st.text_input(f'{param_name}', value=param_value)
            strategy_params = {param_name: to_number(strategy_params[param_name]) for param_name in strategy_params}
            symbol = st.text_input('Symbol', 'MYXUSDT')
            start_date = st.date_input('Start date', pd.to_datetime('2025-01-01'))
            end_date = st.date_input('End date', pd.to_datetime('2025-08-31'))
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            timeframe = st.selectbox('Timeframe', timeframes, index=0)
            deposit = st.number_input('Initial Deposit', value=100)
            market_commission = st.number_input('Market Commission (%)', value=0.05, step=0.01, format='%.4f')
            limit_commission = st.number_input('Limit Commission (%)', value=0.02, step=0.01, format='%.4f')
            order_type = st.selectbox('Order Type', ['Market', 'Limit'], index=1)
            leverage = st.number_input('Leverage', value=100, min_value=1, step=1)
            run = st.button('Run Backtest', use_container_width=True)

        figs = None
        metrics_df = None
        with col2:
            if 'run_clicked' not in st.session_state:
                st.session_state['run_clicked'] = False
            if run:
                st.session_state['run_clicked'] = True
            if st.session_state['run_clicked']:
                with col2:
                    if run:
                        with st.spinner('Running backtest...'):
                            try:
                                figs, metrics = run_backtest(
                                    selected_strategy_class,
                                    symbol,
                                    start_date,
                                    end_date,
                                    timeframe,
                                    deposit=deposit,
                                    market_commission=market_commission,
                                    limit_commission=limit_commission,
                                    order_type=order_type,
                                    leverage=leverage,
                                    **strategy_params
                                )
                                equity_chart, ohlc_plot = figs
                                # Метрики в виде таблицы над графиками
                                # Горизонтальная таблица: метрики — в колонках, одна строка
                                metrics_row = pd.DataFrame([metrics])
                                st.dataframe(metrics_row, use_container_width=True)
                                st.write('---')
                                st.altair_chart(equity_chart, use_container_width=True)
                                if ohlc_plot:
                                    st.plotly_chart(ohlc_plot, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error: {e}")

if __name__ == '__main__':
    main()

