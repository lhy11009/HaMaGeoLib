# world_builder_param_parser.py
# MIT License
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
world_builder_param_parser.py

Author: Haoyuan Li
Description: A utility for parsing and handling deal.ii parameter files 
             in the hamageolib package.
"""


from .handy_shortcuts_haoyuan import func_name

def find_wb_feature(Inputs_wb, key):
    '''
    find index of feature in a world builder inputs by its key
    Inputs:
        Inputs_wb (dict): World buider dictionary
        key (str): name of the feature
    '''
    assert(type(Inputs_wb) == dict)
    Features = Inputs_wb['features']
    i = 0
    for feature in Features:
        if feature['name'] == key:
            break
        i += 1
        if i == len(Features):  # not found
            raise ValueError("%s: There is no feature named %s" % (func_name(), key))
    return i