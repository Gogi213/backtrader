import pkgutil
import importlib
import warnings

# Import the registry that all strategies will use
from .strategy_registry import StrategyRegistry

# --- Dynamic Strategy Loading ---
# This code automatically finds and imports all strategy modules in this package.
# By importing them, the @StrategyRegistry.register('name') decorator in each
# strategy file is executed, populating the central registry.

# Get the current package name (e.g., 'src.strategies')
__path__ = pkgutil.extend_path(__path__, __name__)

# Iterate over all modules in the current package
for importer, modname, ispkg in pkgutil.iter_modules(path=__path__, prefix=__name__ + '.'):
    try:
        # Import the module
        importlib.import_module(modname)
    except ImportError as e:
        # If a strategy fails to import (e.g., missing dependencies),
        # issue a warning but do not crash the application.
        warnings.warn(
            f"Could not import strategy module '{modname}': {e}\n"
            "Please check for missing dependencies or errors in the file."
        )

# --- Public API ---
# Define what is exposed when someone does 'from src.strategies import *'
__all__ = [
    'StrategyRegistry',
]