"""
Strategy Parameters Helper
Unified utility for working with strategy parameters across GUI components

Author: HFT System
"""
from typing import Dict, Any, List
from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QFormLayout


class StrategyParamsHelper:
    """Unified helper for strategy parameters management"""
    
    @staticmethod
    def get_strategy_params(strategy_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a strategy

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with default parameters
        """
        from ...strategies.strategy_registry import StrategyRegistry
        strategy_class = StrategyRegistry.get(strategy_name)

        # Get default parameters or return empty dict if strategy not found
        default_params = strategy_class.get_default_params().copy() if strategy_class else {}

        # Remove common parameters to avoid conflicts when passed to StrategyFactory.create()
        default_params.pop('initial_capital', None)
        default_params.pop('commission_pct', None)

        return default_params
    
    @staticmethod
    def create_param_widgets(params: Dict[str, Any], layout: QFormLayout) -> Dict[str, Any]:
        """
        Create parameter widgets for a strategy
        
        Args:
            params: Dictionary with parameter names and default values
            layout: Form layout to add widgets to
            
        Returns:
            Dictionary mapping parameter names to widgets
        """
        widgets = {}
        
        for param_name, param_value in params.items():
            if isinstance(param_value, int):
                widget = QSpinBox()
                widget.setRange(1, 1000)
                widget.setValue(param_value)
            elif isinstance(param_value, float):
                widget = QDoubleSpinBox()
                widget.setRange(0.01, 100.0)
                widget.setSingleStep(0.01)
                widget.setValue(param_value)
            else:
                # Skip unsupported parameter types
                continue
            
            widgets[param_name] = widget
            layout.addRow(f"{param_name.replace('_', ' ').title()}:", widget)
        
        return widgets
    
    @staticmethod
    def update_params_from_widgets(widgets: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update parameters dictionary from widget values
        
        Args:
            widgets: Dictionary mapping parameter names to widgets
            params: Parameters dictionary to update
            
        Returns:
            Updated parameters dictionary
        """
        for param_name, widget in widgets.items():
            if isinstance(widget, QSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[param_name] = widget.value()
        
        return params
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """
        Get list of available strategies

        Returns:
            List of strategy names
        """
        from ...strategies.strategy_registry import StrategyRegistry
        return StrategyRegistry.list_strategies()