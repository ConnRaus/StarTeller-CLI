"""
StarTeller Source Package
========================

Core source code for the StarTeller astrophotography planning tool.

Modules:
    star_teller: Main application with observation planning algorithms
    catalog_manager: NGC/IC catalog loading and management
"""

__version__ = "1.0.0"
__author__ = "Connor Rauscher"

from .star_teller import StarTeller, get_user_location, create_custom_starteller
from .catalog_manager import load_ngc_catalog

__all__ = [
    'StarTeller',
    'get_user_location', 
    'create_custom_starteller',
    'load_ngc_catalog'
] 