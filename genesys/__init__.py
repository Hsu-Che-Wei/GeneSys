# genesys/__init__.py
__version__ = "1.0.0"

from .genesys_cli import main as cli_main
from .genesys_model import ClassifierLSTM  # replace with your actual model module/class

__all__ = ["cli_main", "ClassifierLSTM", "__version__"]

