"""
StarTeller-CLI Source Package
=============================

Core source code for the StarTeller-CLI astrophotography planning tool.

Modules:
    starteller: Core observation math and StarTellerCLI (no CLI I/O)
    starteller_cli: Interactive prompts, paths, and main()
    catalog_manager: NGC/IC catalog loading and management
"""

__version__ = "2.0.2"
__author__ = "ConnRaus"

from .starteller_cli import StarTellerCLI, get_user_location, main
from .catalog_manager import load_ngc_catalog

__all__ = [
    'StarTellerCLI',
    'get_user_location',
    'main',
    'load_ngc_catalog'
] 