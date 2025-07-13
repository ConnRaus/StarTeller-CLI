"""
StarTeller-CLI Source Package
=============================

Core source code for the StarTeller-CLI astrophotography planning tool.

Modules:
    starteller_cli: Main application with observation planning algorithms
    catalog_manager: NGC/IC catalog loading and management
"""

__version__ = "1.0.0"
__author__ = "ConnRaus"

from .starteller_cli import StarTellerCLI, get_user_location, create_custom_starteller_cli
from .catalog_manager import load_ngc_catalog

__all__ = [
    'StarTellerCLI',
    'get_user_location', 
    'create_custom_starteller_cli',
    'load_ngc_catalog'
] 