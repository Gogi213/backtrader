"""
Chart signals tab for Professional GUI Application
High-performance Plotly implementation for desktop (NOT browser)
Extracted from gui_visualizer following HFT principles: high performance, no duplication, YAGNI compliance
"""
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
from plotly.offline import plot
import tempfile
import os


class ChartSignalsTab:
    """High-performance chart display with Plotly (desktop, not browser)"""

    def __init__(self):
        self.widget = None
        self.web_view = None
        self._init_ui()

    def _init_ui(self):
        """Initialize chart UI with Plotly web view"""
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)

        # Create web view for Plotly chart
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)

    def get_widget(self):
        """Get the widget for tab integration"""
        return self.widget

    def update_chart(self, results_data):
        """Create high-performance chart with Plotly (desktop rendering)"""
        if not results_data:
            return

        try:
            # Get data from results
            bb_data = results_data.get('bb_data')
            trades = results_data.get('trades', [])

            if not bb_data or 'times' not in bb_data:
                return

            # Convert timestamps to datetime
            times = pd.to_datetime(bb_data['times'], unit='ms')
            prices = bb_data['prices']

            # Create Plotly figure
            fig = go.Figure()

            # Add price line
            fig.add_trace(go.Scatter(
                x=times,
                y=prices,
                mode='lines',
                name='Price',
                line=dict(color='blue', width=1.5),
                hovertemplate='<b>Price</b>: $%{y:.4f}<br><b>Time</b>: %{x}<extra></extra>'
            ))

            # Add Bollinger Bands if available
            if 'bb_upper' in bb_data and 'bb_lower' in bb_data:
                bb_upper = bb_data['bb_upper']
                bb_middle = bb_data['bb_middle']
                bb_lower = bb_data['bb_lower']
                bb_period = bb_data.get('bb_period', 20)
                bb_std = bb_data.get('bb_std', 2.0)

                # Upper band
                fig.add_trace(go.Scatter(
                    x=times,
                    y=bb_upper,
                    mode='lines',
                    name=f'BB Upper ({bb_period}, {bb_std})',
                    line=dict(color='red', width=1, dash='dash'),
                    hovertemplate='<b>BB Upper</b>: $%{y:.4f}<extra></extra>'
                ))

                # Middle band (SMA)
                fig.add_trace(go.Scatter(
                    x=times,
                    y=bb_middle,
                    mode='lines',
                    name=f'BB Middle (SMA {bb_period})',
                    line=dict(color='orange', width=1),
                    hovertemplate='<b>SMA</b>: $%{y:.4f}<extra></extra>'
                ))

                # Lower band
                fig.add_trace(go.Scatter(
                    x=times,
                    y=bb_lower,
                    mode='lines',
                    name=f'BB Lower ({bb_period}, {bb_std})',
                    line=dict(color='green', width=1, dash='dash'),
                    hovertemplate='<b>BB Lower</b>: $%{y:.4f}<extra></extra>'
                ))

                # Fill between bands
                fig.add_trace(go.Scatter(
                    x=times.tolist() + times.tolist()[::-1],
                    y=bb_upper.tolist() + bb_lower.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(173, 216, 230, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='BB Channel',
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Add trade signals
            if trades:
                buy_times = []
                buy_prices = []
                sell_times = []
                sell_prices = []

                for trade in trades:
                    entry_time = trade.get('timestamp')
                    entry_price = trade.get('entry_price', 0)
                    side = trade.get('side', 'unknown')

                    if entry_time and entry_price:
                        trade_datetime = pd.to_datetime(entry_time, unit='ms')

                        if side == 'long':
                            buy_times.append(trade_datetime)
                            buy_prices.append(entry_price)
                        else:
                            sell_times.append(trade_datetime)
                            sell_prices.append(entry_price)

                # Add buy signals
                if buy_times:
                    fig.add_trace(go.Scatter(
                        x=buy_times,
                        y=buy_prices,
                        mode='markers',
                        name=f'Long Entry ({len(buy_times)})',
                        marker=dict(
                            symbol='triangle-up',
                            size=8,
                            color='green',
                            line=dict(width=1, color='darkgreen')
                        ),
                        hovertemplate='<b>Long Entry</b><br>Price: $%{y:.4f}<br>Time: %{x}<extra></extra>'
                    ))

                # Add sell signals
                if sell_times:
                    fig.add_trace(go.Scatter(
                        x=sell_times,
                        y=sell_prices,
                        mode='markers',
                        name=f'Short Entry ({len(sell_times)})',
                        marker=dict(
                            symbol='triangle-down',
                            size=8,
                            color='red',
                            line=dict(width=1, color='darkred')
                        ),
                        hovertemplate='<b>Short Entry</b><br>Price: $%{y:.4f}<br>Time: %{x}<extra></extra>'
                    ))

            # Configure layout for professional trading look
            fig.update_layout(
                title=dict(
                    text='HFT Price Chart - Bollinger Bands Strategy',
                    x=0.5,
                    font=dict(size=16, color='black')
                ),
                xaxis=dict(
                    title='Time',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                yaxis=dict(
                    title='Price (USDT)',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                ),
                hovermode='x unified',
                height=600
            )

            # Configure high-performance rendering
            config = {
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'displaylogo': False,
                'responsive': True,
                'scrollZoom': True
            }

            # Generate HTML for desktop rendering (not browser)
            html_str = plot(fig, output_type='div', include_plotlyjs=True, config=config)

            # Create full HTML page
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>HFT Chart</title>
                <style>
                    body {{ margin: 0; padding: 0; }}
                    .chart-container {{ width: 100%; height: 100vh; }}
                </style>
            </head>
            <body>
                <div class="chart-container">
                    {html_str}
                </div>
            </body>
            </html>
            """

            # Save to temp file and load in QWebEngineView (desktop rendering)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                f.write(full_html)
                temp_path = f.name

            # Load in web view for desktop display
            self.web_view.load(f"file:///{temp_path.replace(os.sep, '/')}")

        except Exception as e:
            print(f"Chart rendering error: {str(e)}")

    def clear(self):
        """Clear chart display"""
        if self.web_view:
            self.web_view.setHtml("")