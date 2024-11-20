# MIT License
# 
# Copyright (c) 2024 Haoyuan Li
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
File: plot_helpers.py
Author: Haoyuan Li

Description:
This module provides utility functions and classes for generating plots 
related to geodynamic modeling. It is intended to support plot customization 
and visualization tasks, with a focus on ease of use and consistency.

Classes:
    (To be defined)
        - Descriptions will be added once the classes are implemented.

Functions:
    scale_matplotlib_params
        - Scales Matplotlib parameters proportionally based on a given scaling factor. 
        This function adjusts font sizes, line widths, marker sizes, and other visual 
        elements to ensure consistent scaling in plots. It allows fine-tuning of specific 
        parameters such as font size, line width, and legend font size through additional multipliers.
"""

import matplotlib.pyplot as plt

def scale_matplotlib_params(scaling_factor=1.0, **kwargs):
    """
    Scale Matplotlib parameters proportionally by a given scaling factor.
    
    Parameters:
        scaling_factor (float): The factor by which to scale font sizes, linewidths, and markers.
        **kwargs:
            font_scaling_multiplier (float): extra scaling multiplier for font
    """
    font_scaling_multiplier = kwargs.get("font_scaling_multiplier", 1.0)
    line_width_scaling_multiplier = kwargs.get("line_width_scaling_multiplier", 1.0)
    legend_font_scaling_multiplier = kwargs.get("legend_font_scaling_multiplier", 1.0)

    # Base font sizes and line widths
    base_fontsize = 10 * font_scaling_multiplier
    base_linewidth = 1.0
    base_markersize = 6.0
    base_tickwidth = 0.5
    base_ticklength = 3.5
    base_spinewidth = 0.5
    base_gridlinewidth = 0.5
    
    # Update rcParams
    plt.rcParams.update({
        "figure.figsize": (6 * scaling_factor, 4 * scaling_factor),  # Scale figure size
        "font.size": base_fontsize * scaling_factor,                # Scale font size
        "axes.labelsize": base_fontsize * scaling_factor,           # Scale axis label size
        "axes.titlesize": base_fontsize * scaling_factor,           # Scale title size
        "xtick.labelsize": base_fontsize * scaling_factor,          # Scale x-tick label size
        "ytick.labelsize": base_fontsize * scaling_factor,          # Scale y-tick label size
        "legend.fontsize": base_fontsize * scaling_factor * legend_font_scaling_multiplier,          # Scale legend font size
        "lines.linewidth": base_linewidth * scaling_factor * line_width_scaling_multiplier,         # Scale line width
        "lines.markersize": base_markersize * scaling_factor * line_width_scaling_multiplier,       # Scale marker size
        "xtick.major.width": base_tickwidth * scaling_factor * line_width_scaling_multiplier,       # Scale major tick width
        "ytick.major.width": base_tickwidth * scaling_factor * line_width_scaling_multiplier,       # Scale major tick width
        "xtick.minor.width": base_tickwidth * scaling_factor * line_width_scaling_multiplier,       # Scale minor tick width
        "ytick.minor.width": base_tickwidth * scaling_factor * line_width_scaling_multiplier,       # Scale minor tick width
        "xtick.major.size": base_ticklength * scaling_factor,       # Scale major tick length
        "ytick.major.size": base_ticklength * scaling_factor,       # Scale major tick length
        "grid.linewidth": base_gridlinewidth * scaling_factor      # Scale grid line width
    })
