"""
Минималистичная визуализация с Plotly
Только самое необходимое для свечных графиков
"""
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Plotly import is disabled as visualization is currently turned off.
PLOTLY_AVAILABLE = False


def plot_candlestick_chart(results: Dict[str, Any], save_path: str) -> Optional[str]:
    """
    Создает свечной график с входами и выходами.
    В настоящее время отключено для рефакторинга.
    """
    print("Визуализация отключена. Графики не будут созданы.")
    return None


def quick_plot(results: Dict[str, Any], output_dir: str = "optimization_plots") -> str:
    """
    Быстрое создание графика
    """
    os.makedirs(output_dir, exist_ok=True)
    
    strategy = results.get("strategy_name", "strategy")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{strategy}_trades_{ts}.html"
    save_path = os.path.join(output_dir, filename)
    
    return plot_candlestick_chart(results, save_path) or save_path


def plot_and_open(results: Dict[str, Any], output_dir: str = "optimization_plots") -> str:
    """
    Создает график и открывает его в браузере.
    В настоящее время отключено для рефакторинга.
    """
    print("Визуализация отключена. Графики не будут созданы или открыты.")
    return quick_plot(results, output_dir)


# Класс для совместимости с TUI
class OptimizationVisualizer:
    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def plot_trades(self, output_dir: str = "optimization_plots",
                   fname: Optional[str] = None) -> Optional[str]:
        if fname:
            save_path = os.path.join(output_dir, fname)
            return plot_candlestick_chart(self.results, save_path)
        else:
            return quick_plot(self.results, output_dir)
    
    def plot_and_open(self, output_dir: str = "optimization_plots",
                     fname: Optional[str] = None) -> Optional[str]:
        """
        Создает график и открывает его в браузере.
        В настоящее время отключено для рефакторинга.
        """
        print("Визуализация отключена. Графики не будут созданы или открыты.")
        if fname:
            save_path = os.path.join(output_dir, fname)
            return plot_candlestick_chart(self.results, save_path)
        else:
            return quick_plot(self.results, output_dir)


# Псевдонимы для обратной совместимости
quick_visualize = quick_plot
quick_plot_trades = quick_plot