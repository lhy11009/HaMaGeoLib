# HaMaGeo/__init__.py

"""
HaMaGeo: A Geodynamic Research Toolkit

Authors: Haoyuan Li, Magali Billen
Description: HaMaGeo is a Python package designed to provide tools and models for geodynamic research, 
             focusing on processes like mantle convection, plate tectonics, and related simulations.

Usage:
    import hamageo
    # Access placeholder submodules directly, e.g., hamageo.core_foo
"""

# Placeholder imports for future modules
from . import core as hamageolib_core  # Placeholder for the core functionality
from . import utils as hamageolib_utils  # Placeholder for utility functions
from . import visualization as hamageolib_viz  # Placeholder for visualization functions

# Define package metadata
__author__ = "Haoyuan Li, Magali Billen"
__version__ = "0.1.0"
__license__ = "MIT"
__all__ = [
    "hamageolib_core",
    "hamageolib_utils",
    "hamageolib_viz"
]
