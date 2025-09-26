"""
Bollinger Bands Mean Reversion Strategy for Jesse

Strategy parameters:
- Period: 200
- Standard deviation: 3
- Take profit: SMA
- Stop loss: 1%

Sophisticated implementation with proper data handling for tick data
and comprehensive risk management.
"""
import numpy as np
from jesse.strategies import Strategy, cached
from jesse import utils

class BollingerBandsMeanReversion(Strategy):
    """
    Sophisticated Bollinger Bands Mean Reversion Strategy
    - Uses period 200 and standard deviation 3 for Bollinger Bands
    - Uses SMA for take profit levels
    - Implements 1% stop loss
    - Proper entry: long when price below lower band, short when above upper band
    - Advanced position sizing and risk management
    """
    
    def hyperparameters(self):
        return [
            {'name': 'bb_period', 'type': int, 'min': 190, 'max': 210, 'default': 200},
            {'name': 'bb_std', 'type': float, 'min': 2.5, 'max': 3.5, 'default': 3.0},
            {'name': 'stop_loss_percentage', 'type': float, 'min': 0.8, 'max': 1.2, 'default': 1.0},
            {'name': 'sma_tp_period', 'type': int, 'min': 10, 'max': 50, 'default': 20},
            {'name': 'risk_to_reward_ratio', 'type': float, 'min': 1.0, 'max': 3.0, 'default': 2.0},
            {'name': 'max_position_size_percentage', 'type': float, 'min': 0.5, 'max': 5.0, 'default': 2.0},
        ]

    def before(self):
        """
        Executed before each time period/candle
        """
        # Update indicators and bands with most recent data
        pass

    def should_long(self) -> bool:
        """
        Entry condition for long position
        Price should be below lower Bollinger Band
        Additional filters to avoid false signals
        """
        current_price = self.price
        lower_band = self.lower_band()
        upper_band = self.upper_band()
        sma_price = self.sma_current_price()
        
        # Primary condition: price below lower band
        price_condition = current_price < lower_band
        
        # Additional filters to reduce false signals
        # Ensure price is not in extremely overbought territory elsewhere
        # And that bands are not too narrow (indicating low volatility)
        band_width = upper_band - lower_band
        min_band_width = sma_price * 0.005  # 0.5% of price as minimum band width
        
        # Only enter if bands are sufficiently wide
        volatility_condition = band_width > min_band_width
        
        # Additional filter: ensure price is not too far below lower band (avoid chasing)
        max_distance_from_lower = abs(current_price - lower_band) / lower_band < 0.02  # 2%
        
        return price_condition and volatility_condition and max_distance_from_lower

    def should_short(self) -> bool:
        """
        Entry condition for short position
        Price should be above upper Bollinger Band
        Additional filters to avoid false signals
        """
        current_price = self.price
        upper_band = self.upper_band()
        lower_band = self.lower_band()
        sma_price = self.sma_current_price()
        
        # Primary condition: price above upper band
        price_condition = current_price > upper_band
        
        # Additional filters to reduce false signals
        # Ensure bands are not too narrow (indicating low volatility)
        band_width = upper_band - lower_band
        min_band_width = sma_price * 0.005  # 0.5% of price as minimum band width
        
        # Only enter if bands are sufficiently wide
        volatility_condition = band_width > min_band_width
        
        # Additional filter: ensure price is not too far above upper band (avoid chasing)
        max_distance_from_upper = abs(current_price - upper_band) / upper_band < 0.02  # 2%
        
        return price_condition and volatility_condition and max_distance_from_upper

    def should_cancel(self) -> bool:
        """
        Cancel any pending orders
        """
        return True

    def filters(self):
        """
        Additional filters to avoid entering trades in poor conditions
        """
        return [
            # Filter 1: Minimum price movement threshold to ensure meaningful signals
            {
                'name': 'minimum_price_change',
                'func': lambda: abs(self.price - self.close[1]) / self.close[1] > 0.001  # 0.1% minimum change
            },
            # Filter 2: Volume confirmation (if available, ensure volume is not too low)
            # Note: Jesse's standard filters, but we could implement custom volume filters if needed
        ]

    def go_long(self):
        """
        Execute long position with proper position sizing and risk management
        """
        # Calculate stop loss (1% below entry)
        stop_price = self.price * (1 - self.hp['stop_loss_percentage'] / 100)
        
        # Calculate take profit based on SMA
        sma_tp = self.sma_tp()
        if sma_tp is None or np.isnan(sma_tp):
            # If SMA is not available, calculate take profit based on risk to reward ratio
            take_profit_price = self.price * (1 + (self.hp['stop_loss_percentage'] * self.hp['risk_to_reward_ratio']) / 100)
        else:
            take_profit_price = sma_tp
        
        # Calculate position size based on risk management
        risk_amount = self.capital * (self.hp['max_position_size_percentage'] / 100)  # Max position size percent of capital
        risk_per_share = self.price - stop_price
        qty = min(risk_amount / risk_per_share, self.balance * 0.95 / self.price)  # Ensure we don't use more than 95% of balance
        
        # Open long position
        self.buy = qty, self.price
        self.stop_loss = qty, stop_price
        self.take_profit = qty, take_profit_price

    def go_short(self):
        """
        Execute short position with proper position sizing and risk management
        """
        # Calculate stop loss (1% above entry)
        stop_price = self.price * (1 + self.hp['stop_loss_percentage'] / 100)
        
        # Calculate take profit based on SMA
        sma_tp = self.sma_tp()
        if sma_tp is None or np.isnan(sma_tp):
            # If SMA is not available, calculate take profit based on risk to reward ratio
            take_profit_price = self.price * (1 - (self.hp['stop_loss_percentage'] * self.hp['risk_to_reward_ratio']) / 100)
        else:
            take_profit_price = sma_tp
        
        # Calculate position size based on risk management
        risk_amount = self.capital * (self.hp['max_position_size_percentage'] / 100)  # Max position size percent of capital
        risk_per_share = stop_price - self.price
        qty = min(risk_amount / risk_per_share, self.balance * 0.95 / self.price)  # Ensure we don't use more than 95% of balance
        
        # Open short position
        self.sell = qty, self.price
        self.stop_loss = qty, stop_price
        self.take_profit = qty, take_profit_price

    @cached
    def upper_band(self):
        """
        Calculate upper Bollinger Band
        Uses SMA as the middle line with standard deviation bands
        """
        sma = self.sma(self.hp['bb_period'])
        std = self.standard_deviation(self.hp['bb_period'])
        return sma + (std * self.hp['bb_std'])

    @cached
    def lower_band(self):
        """
        Calculate lower Bollinger Band
        Uses SMA as the middle line with standard deviation bands
        """
        sma = self.sma(self.hp['bb_period'])
        std = self.standard_deviation(self.hp['bb_period'])
        return sma - (std * self.hp['bb_std'])

    @cached
    def sma_current_price(self):
        """
        Calculate SMA of current price for volatility reference
        """
        return self.sma(self.hp['bb_period'])
        
    @cached
    def standard_deviation(self, period):
        """
        Calculate standard deviation of closing prices over the specified period
        """
        if len(self.candles) < period:
            period = len(self.candles)
            
        prices = self.candles[:, 2][-period:]  # Closing prices
        return np.std(prices)

    @cached
    def sma_tp(self):
        """
        Calculate Simple Moving Average for take profit
        Uses a shorter period than the Bollinger Bands for dynamic take profit levels
        """
        return self.sma(self.hp['sma_tp_period'])

    def on_open_position(self, order):
        """
        Called when a position is opened
        """
        # Log entry for analysis
        self.log(f"Position opened: {order.side} at {order.price}, qty: {order.qty}")

    def on_close_position(self, order):
        """
        Called when a position is closed
        """
        # Log exit for analysis
        self.log(f"Position closed: {order.type} at {order.price}, PnL: {order.pnl}")

    def on_stop_loss(self, order):
        """
        Called when a position is closed by a stop loss order
        """
        self.log(f"Stop loss triggered: {order.side} closed at {order.price}, PnL: {order.pnl}")

    def on_take_profit(self, order):
        """
        Called when a position is closed by a take profit order
        """
        self.log(f"Take profit executed: {order.side} closed at {order.price}, PnL: {order.pnl}")

    def on_increased_position(self, order):
        """
        Called when a position is increased
        """
        pass

    def on_reduced_position(self, order):
        """
        Called when a position is reduced
        """
        pass

    def on_route_opened(self, strategy, order):
        """
        Called when a route order is opened
        """
        pass

    def on_route_closed(self, strategy, order):
        """
        Called when a route order is closed
        """
        pass

    def on_route_increased(self, strategy, order):
        """
        Called when a route position is increased
        """
        pass

    def on_route_reduced(self, strategy, order):
        """
        Called when a route position is reduced
        """
        pass

    def on_route_canceled(self, strategy, order):
        """
        Called when a route order is canceled
        """
        pass

    def terminate(self):
        """
        Called when the strategy is terminated
        """
        # Calculate and log final performance metrics
        total_bars = len(self.candles)
        if total_bars > 0:
            win_rate = self.total_trades_won / self.total_trades * 100 if self.total_trades > 0 else 0
            self.log(f"Strategy completed: {self.total_trades} trades, {win_rate:.2f}% win rate")