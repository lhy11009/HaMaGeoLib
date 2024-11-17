# __init__.py

"""
haoyuan_2d_subduction
----------------------
This module contains configurations, data, and functions for 2D subduction modeling cases.

Modules:
- input: Will contain input configurations and data for initializing 2D subduction cases.
- output: Will contain functions or classes to handle the output of simulation results.
- scripts: Will contain specific scripts to run or analyze 2D subduction cases.

Example usage:
    from hamageolib.research.haoyuan_2d_subduction import default_settings

Author: Haoyuan Li
Affiliation: EPS Department, UC Davis
"""

# Define any shared or default settings that might be useful across the subduction cases
default_settings = {
    "case_name": "2D Subduction",
    "time_step": None,
    "resolution": None,
    # Additional settings can be added as needed
}

__all__ = [
    "default_settings",
]
