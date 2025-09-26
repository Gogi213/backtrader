import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    entry_price: float
    size: float
    side: str  # 'long' or 'short'
    entry_time: pd.Timestamp
    stop_loss: float  # 1% stop loss
    take_profit: Optional[float] = None


class BollingerBandsMeanReversionStrategy:
    """
    Bollinger Bands Mean Reversion Strategy (DEL)
    This file is marked for deletion - use vectorized_bollinger_strategy.py instead
    Legacy implementation with candlestick aggregation
    - Period: 200
    - Standard Deviation: 3
    - SMA for take profit
    - 1% stop loss
    """
    
    def __init__(self, 
                 symbol: str,
                 period: int = 200,
                 std_dev: float = 3.0,
                 take_profit_sma_period: int = 20,
                 stop_loss_pct: float = 0.01,
                 initial_capital: float = 10000.0):
        
        self.symbol = symbol
        self.period = period  # Bollinger Bands period
        self.std_dev = std_dev  # Standard deviation multiplier
        self.take_profit_sma_period = take_profit_sma_period
        self.stop_loss_pct = stop_loss_pct  # 1% stop loss
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position: Optional[Position] = None
        self.trade_history: List[Dict] = []
        self.price_history: List[Tuple[pd.Timestamp, float]] = []
        
    def calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands: middle, upper, lower
        """
        if len(prices) < self.period:
            # Use the available data if less than period
            actual_period = min(len(prices), self.period)
            sma = np.convolve(prices, np.ones(actual_period) / actual_period, mode='valid')
            # Pad the beginning with NaN to match length
            sma_padded = np.full(len(prices), np.nan)
            sma_padded[-len(sma):] = sma
            
            # Avoid invalid sqrt values due to floating point precision
            variance = np.convolve(prices**2, np.ones(actual_period) / actual_period, mode='valid') - sma**2
            rolling_std = np.sqrt(np.maximum(variance, 0.0))
            rolling_std_padded = np.full(len(prices), np.nan)
            rolling_std_padded[-len(rolling_std):] = rolling_std
        else:
            sma = np.convolve(prices, np.ones(self.period) / self.period, mode='valid')
            # Pad the beginning with NaN to match length
            sma_padded = np.full(len(prices), np.nan)
            sma_padded[-len(sma):] = sma
            
            rolling_std = np.sqrt(
                np.convolve(prices**2, np.ones(self.period) / self.period, mode='valid') - 
                sma**2
            )
            rolling_std_padded = np.full(len(prices), np.nan)
            rolling_std_padded[-len(rolling_std):] = rolling_std
        
        upper_band = sma_padded + (rolling_std_padded * self.std_dev)
        lower_band = sma_padded - (rolling_std_padded * self.std_dev)
        
        return sma_padded, upper_band, lower_band
    
    def calculate_sma(self, prices: np.ndarray) -> np.ndarray:
        """Calculate Simple Moving Average for take profit"""
        if len(prices) < self.take_profit_sma_period:
            actual_period = min(len(prices), self.take_profit_sma_period)
            sma = np.convolve(prices, np.ones(actual_period) / actual_period, mode='valid')
            sma_padded = np.full(len(prices), np.nan)
            sma_padded[-len(sma):] = sma
        else:
            sma = np.convolve(prices, np.ones(self.take_profit_sma_period) / self.take_profit_sma_period, mode='valid')
            sma_padded = np.full(len(prices), np.nan)
            sma_padded[-len(sma):] = sma
            
        return sma_padded
    
    def should_enter_long(self, current_price: float, lower_band: float) -> bool:
        """Determine if we should enter a long position"""
        return current_price <= lower_band
        
    def should_enter_short(self, current_price: float, upper_band: float) -> bool:
        """Determine if we should enter a short position"""
        return current_price >= upper_band
        
    def should_exit_long(self, current_price: float, position: Position, take_profit_level: float) -> bool:
        """Determine if we should exit a long position"""
        return (current_price >= take_profit_level or 
                current_price <= position.stop_loss)
                
    def should_exit_short(self, current_price: float, position: Position, take_profit_level: float) -> bool:
        """Determine if we should exit a short position"""
        return (current_price <= take_profit_level or 
                current_price >= position.stop_loss)
    
    def process_tick(self, timestamp: pd.Timestamp, price: float) -> List[Dict]:
        """
        Process a single tick and return any trades executed
        """
        # Add current price to history
        self.price_history.append((timestamp, price))
        
        # Extract prices for calculations
        prices = np.array([p[1] for p in self.price_history])
        
        # Calculate Bollinger Bands
        _, upper_band, lower_band = self.calculate_bollinger_bands(prices)
        current_upper_band = upper_band[-1] if not np.isnan(upper_band[-1]) else None
        current_lower_band = lower_band[-1] if not np.isnan(lower_band[-1]) else None
        
        # Calculate SMA for take profit
        take_profit_sma = self.calculate_sma(prices)
        current_tp_sma = take_profit_sma[-1] if not np.isnan(take_profit_sma[-1]) else None
        
        executed_trades = []
        
        # Check for exit conditions if we have an open position
        if self.position:
            if self.position.side == 'long':
                if current_tp_sma and self.should_exit_long(price, self.position, current_tp_sma):
                    # Exit long position
                    trade = self.exit_position(timestamp, price, "take_profit" if price >= current_tp_sma else "stop_loss")
                    if trade:
                        executed_trades.append(trade)
            elif self.position.side == 'short':
                if current_tp_sma and self.should_exit_short(price, self.position, current_tp_sma):
                    # Exit short position
                    trade = self.exit_position(timestamp, price, "take_profit" if price <= current_tp_sma else "stop_loss")
                    if trade:
                        executed_trades.append(trade)
        
        # If no position, check for entry conditions
        if not self.position and current_lower_band is not None and current_upper_band is not None:
            if self.should_enter_long(price, current_lower_band):
                # Enter long position
                trade = self.enter_position(timestamp, price, 'long')
                if trade:
                    executed_trades.append(trade)
            elif self.should_enter_short(price, current_upper_band):
                # Enter short position
                trade = self.enter_position(timestamp, price, 'short')
                if trade:
                    executed_trades.append(trade)
        
        return executed_trades
    
    def enter_position(self, timestamp: pd.Timestamp, price: float, side: str) -> Optional[Dict]:
        """Enter a new position - only store position info, don't create trade record yet"""
        if self.position is not None:
            return None  # Already in a position

        # Calculate stop loss (1% from entry)
        stop_loss_multiplier = (1 - self.stop_loss_pct) if side == 'long' else (1 + self.stop_loss_pct)
        stop_loss = price * stop_loss_multiplier

        # Calculate position size based on available capital
        position_size = min(self.current_capital * 0.1, self.current_capital) / price  # Risk 10% of capital

        # Create position (store entry info)
        self.position = Position(
            symbol=self.symbol,
            entry_price=price,
            size=position_size,
            side=side,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=None
        )

        # Don't create trade record on entry - wait for exit to create complete trade
        return None
    
    def exit_position(self, timestamp: pd.Timestamp, price: float, exit_reason: str) -> Optional[Dict]:
        """Exit current position and create complete trade record"""
        if not self.position:
            return None

        # Calculate PnL
        if self.position.side == 'long':
            pnl = (price - self.position.entry_price) * self.position.size
        else:  # short
            pnl = (self.position.entry_price - price) * self.position.size

        # Calculate duration in minutes
        duration_timedelta = timestamp - self.position.entry_time
        duration_minutes = int(duration_timedelta.total_seconds() / 60)

        # Calculate PnL percentage
        pnl_percentage = (pnl / (self.position.entry_price * self.position.size)) * 100

        # Update capital
        self.current_capital += pnl

        # Create COMPLETE trade record with entry and exit info
        complete_trade = {
            'timestamp': self.position.entry_time,  # Entry timestamp for sorting
            'exit_timestamp': timestamp,           # Exit timestamp
            'symbol': self.symbol,
            'side': self.position.side,
            'entry_price': self.position.entry_price,
            'exit_price': price,
            'size': self.position.size,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'duration': duration_minutes,
            'exit_reason': exit_reason,
            'capital_before': self.current_capital - pnl,
            'capital_after': self.current_capital
        }

        self.trade_history.append(complete_trade)

        # Clear position
        self.position = None

        return complete_trade
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if len(self.trade_history) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'final_capital': self.initial_capital,
                'return_percentage': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Filter to only exit trades
        exit_trades = [trade for trade in self.trade_history if trade.get('type') == 'EXIT']
        
        total_trades = len(exit_trades)
        winning_trades = sum(1 for t in exit_trades if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in exit_trades if t.get('pnl', 0) < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(t.get('pnl', 0) for t in exit_trades)
        return_percentage = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Calculate max drawdown (simplified)
        # Note: This is a simplified calculation; a full implementation would track running capital
        max_drawdown = 0.0  # Placeholder
        
        # Calculate Sharpe ratio (simplified)
        # Note: This is a simplified calculation
        sharpe_ratio = 0.0  # Placeholder
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_capital': self.current_capital,
            'return_percentage': return_percentage,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }