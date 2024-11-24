# MIT License
# 
# Copyright (c) YYYY Haoyuan Li
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
utility_functions.py

Author: Haoyuan Li

This file provides two utility functions:
    - func_name: Retrieves the name of the calling function.
    - check_float: Compares a float value to an expected value within a specified tolerance.
"""
import os
import sys
import warnings


def func_name():
    """
    Returns the name of the calling function.

    Returns:
        str: The name of the function that called this function.
    """
    # Uses the inspect module to get the name of the calling function
    return inspect.currentframe().f_back.f_code.co_name


def check_float(val, val_expected, tolerance=1e-8):
    """
    Checks if a float value is within a specified tolerance of an expected value.

    Parameters:
        val (float): The value to check.
        val_expected (float): The expected value to compare against.
        tolerance (float, optional): The allowable relative difference. Default is 1e-8.

    Returns:
        bool: True if val is within the specified tolerance of val_expected, False otherwise.
    """
    # Compares the relative difference between val and val_expected to the tolerance
    return (abs((val - val_expected) / val_expected) < tolerance)


class Mute:
    """Context manager to suppress stdout, stderr, and warnings."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.simplefilter("ignore")
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._warnings.__exit__(exc_type, exc_val, exc_tb)
