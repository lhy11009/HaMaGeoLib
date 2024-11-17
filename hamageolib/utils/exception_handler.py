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
exception_handler.py

Author: Haoyuan Li

This file defines a utility function to assert conditions at runtime, raising specified errors with 
custom messages if conditions are not met.

Functions:
    my_assert(condition, error_type, message):
        Asserts a condition and raises an error with a specified message if the condition is not met.
"""

def my_assert(condition, error_type, message):
    '''
    Asserts a condition at runtime and raises an error with a specified message if the condition is not met.

    Parameters:
        condition (bool): The condition to assert. If False, an error is raised.
        error_type (Type[Exception]): The type of error to raise if the condition is False.
        message (str): The message to include with the raised error.
    '''
    # Raises the specified error type with the provided message if the condition is False.
    if not condition:
        raise error_type(message)
