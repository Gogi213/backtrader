"""
Unified Backtest Results

This module provides a unified results class for backtesting
that can be used by both CLI and GUI interfaces.

Author: HFT System
"""
from typing import Dict, Any, List, Optional, Union
import json
import os
from datetime import datetime


class BacktestResults:
    """
    Unified results container for backtesting
    
    This class provides a consistent interface for backtest results
    that can be used by both CLI and GUI interfaces.
    """
    
    def __init__(self, raw_results: Dict[str, Any]):
        """
        Initialize with raw results from backtest engine
        
        Args:
            raw_results: Raw results dictionary from backtest engine
        """
        self.data = raw_results.copy()
        self._generated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to dictionary
        
        Returns:
            Dictionary representation of results
        """
        result = self.data.copy()
        result['generated_at'] = self._generated_at.isoformat()
        return result
    
    def to_console_output(self) -> str:
        """
        Format results for console output
        
        Returns:
            Formatted string for console display
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BACKTEST RESULTS")
        lines.append("=" * 60)
        
        # Basic metrics
        lines.append(f"Strategy: {self.get('strategy_name', 'N/A')}")
        lines.append(f"Symbol: {self.get('symbol', 'N/A')}")
        lines.append(f"Total Trades: {self.get('total', 0)}")
        lines.append(f"Win Rate: {self.get('win_rate', 0):.1%}")
        lines.append(f"Net P&L: ${self.get('net_pnl', 0):,.2f}")
        lines.append(f"Return: {self.get('net_pnl_percentage', 0):.2f}%")
        lines.append(f"Max Drawdown: {self.get('max_drawdown', 0):.2f}%")
        lines.append(f"Sharpe Ratio: {self.get('sharpe_ratio', 0):.2f}")
        lines.append(f"Profit Factor: {self.get('profit_factor', 0):.2f}")
        
        # Additional metrics
        lines.append(f"Winners: {self.get('total_winning_trades', 0)}")
        lines.append(f"Losers: {self.get('total_losing_trades', 0)}")
        lines.append(f"Avg Win: ${self.get('average_win', 0):.2f}")
        lines.append(f"Avg Loss: ${self.get('average_loss', 0):.2f}")
        lines.append(f"Best Trade: ${self.get('largest_win', 0):.2f}")
        lines.append(f"Worst Trade: ${self.get('largest_loss', 0):.2f}")
        
        # Performance metrics
        if 'klines_processed' in self.data:
            lines.append(f"Klines Processed: {self.get('klines_processed', 0):,}")
        if 'processing_time_seconds' in self.data:
            lines.append(f"Processing Time: {self.get('processing_time_seconds', 0):.2f}s")
            processing_time = self.get('processing_time_seconds', 1)
            if processing_time > 0:
                klines_per_sec = self.get('klines_processed', 0) / processing_time
                lines.append(f"Klines per Second: {klines_per_sec:,.0f}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_gui_format(self) -> Dict[str, Any]:
        """
        Format results for GUI display
        
        Returns:
            Dictionary formatted for GUI components
        """
        return {
            # Basic info
            'strategy': self.get('strategy_name', 'N/A'),
            'symbol': self.get('symbol', 'N/A'),
            'total_trades': self.get('total', 0),
            
            # Performance metrics
            'win_rate': self.get('win_rate', 0.0),
            'net_pnl': self.get('net_pnl', 0.0),
            'return_pct': self.get('net_pnl_percentage', 0.0),
            'sharpe_ratio': self.get('sharpe_ratio', 0.0),
            'profit_factor': self.get('profit_factor', 0.0),
            'max_drawdown': self.get('max_drawdown', 0.0),
            
            # Trade statistics
            'winners': self.get('total_winning_trades', 0),
            'losers': self.get('total_losing_trades', 0),
            'avg_win': self.get('average_win', 0.0),
            'avg_loss': self.get('average_loss', 0.0),
            'best_trade': self.get('largest_win', 0.0),
            'worst_trade': self.get('largest_loss', 0.0),
            
            # Processing info
            'klines_processed': self.get('klines_processed', 0),
            'processing_time': self.get('processing_time_seconds', 0.0),
            
            # Raw data for charts
            'trades': self.get('trades', []),
            'bb_data': self.get('bb_data', {}),
            'indicator_data': self.get('indicator_data', {}),
            
            # Metadata
            'generated_at': self._generated_at.isoformat(),
            'status': 'success' if not self.get('error') else 'error'
        }
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save results to file
        
        Args:
            filepath: Path to save the results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for saving
        save_data = self.to_dict()
        
        # Save based on file extension
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
        else:
            # Default to text format
            with open(filepath, 'w') as f:
                f.write(self.to_console_output())
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from results
        
        Args:
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Value or default
        """
        return self.data.get(key, default)
    
    def has_error(self) -> bool:
        """
        Check if results contain an error
        
        Returns:
            True if error exists, False otherwise
        """
        return 'error' in self.data
    
    def get_error(self) -> Optional[str]:
        """
        Get error message if exists
        
        Returns:
            Error message or None
        """
        return self.data.get('error')
    
    def is_successful(self) -> bool:
        """
        Check if backtest was successful
        
        Returns:
            True if successful, False otherwise
        """
        return not self.has_error() and self.get('total', 0) > 0
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Get list of trades
        
        Returns:
            List of trade dictionaries
        """
        return self.get('trades', [])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Dictionary with key performance metrics
        """
        return {
            'total_trades': self.get('total', 0),
            'win_rate': self.get('win_rate', 0.0),
            'net_pnl': self.get('net_pnl', 0.0),
            'return_pct': self.get('net_pnl_percentage', 0.0),
            'sharpe_ratio': self.get('sharpe_ratio', 0.0),
            'profit_factor': self.get('profit_factor', 0.0),
            'max_drawdown': self.get('max_drawdown', 0.0),
            'avg_win': self.get('average_win', 0.0),
            'avg_loss': self.get('average_loss', 0.0)
        }
    
    def get_chart_data(self) -> Dict[str, Any]:
        """
        Get data for charting
        
        Returns:
            Dictionary with chart data
        """
        return {
            'trades': self.get('trades', []),
            'bb_data': self.get('bb_data', {}),
            'indicator_data': self.get('indicator_data', {}),
            'symbol': self.get('symbol', 'N/A'),
            'strategy': self.get('strategy_name', 'N/A')
        }
    
    @classmethod
    def from_error(cls, error_message: str, strategy_name: str = 'unknown', symbol: str = 'UNKNOWN') -> 'BacktestResults':
        """
        Create results from error
        
        Args:
            error_message: Error message
            strategy_name: Strategy name
            symbol: Trading symbol
            
        Returns:
            BacktestResults with error
        """
        return cls({
            'error': error_message,
            'strategy_name': strategy_name,
            'symbol': symbol,
            'total': 0,
            'net_pnl': 0.0
        })
    
    def __str__(self) -> str:
        """String representation"""
        return self.to_console_output()
    
    def __repr__(self) -> str:
        """Debug representation"""
        return f"BacktestResults(strategy={self.get('strategy_name')}, trades={self.get('total', 0)}, success={self.is_successful()})"