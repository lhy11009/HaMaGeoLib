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
wb_file_parser.py
-----------------
A utility module for parsing and manipulating Geodynamic WorldBuilder configuration files.

This module provides functions and classes to:
- Read WorldBuilder configuration files.
- Modify configuration parameters.
- Write updated configurations back to file.

Author: Haoyuan Li
License: MIT
"""

import json

def find_feature_by_name(wb_dict, feature_name):
    """
    Searches the 'features' list in the provided dictionary for a feature with a matching 'name'.
    
    Args:
        wb_dict (dict): The dictionary containing the 'features' list.
        feature_name (str): The name of the feature to search for.
    
    Returns:
        dict: The matching feature dictionary if found, else None.
    """
    # Ensure 'features' exists as a key and its value is a list.
    assert "features" in wb_dict and isinstance(wb_dict["features"], list), \
        "'features' must exist as a key in the dictionary and be a list."
    
    # Extract the 'features' list from the dictionary.
    features = wb_dict["features"]
    # Iterate through the features to find a feature with a matching 'name'.
    for feature in features:
        if feature.get("name") == feature_name:
            return feature  # Return the feature dictionary if a match is found.
    return None  # Return None if no matching feature is found.


def update_or_add_feature(wb_dict, feature_name, feature_obj):
    """
    Updates an existing feature with the given name in the 'features' list, or adds it if not found.
    
    Args:
        wb_dict (dict): The dictionary containing the 'features' list.
        feature_name (str): The name of the feature to search for and update.
        feature_obj (dict): The new feature object to substitute or add.
        
    Returns:
        dict: The updated dictionary with the feature added or updated.
    
    Raises:
        AssertionError: 
            - If wb_dict does not contain the 'features' key or the value is not a list.
            - If feature_obj['name'] does not match feature_name.
    """
    # Assert wb_dict has the 'features' key and the value is a list
    assert "features" in wb_dict, "wb_dict must contain a 'features' key."
    assert isinstance(wb_dict["features"], list), "'features' must be a list."

    # Assert feature_obj's 'name' matches feature_name
    assert feature_obj.get("name") == feature_name, \
        f"feature_obj's 'name' ({feature_obj.get('name')}) does not match feature_name ({feature_name})"
    
    # Retrieve the list of features
    features = wb_dict["features"]
    
    # Search for the feature by name
    for idx, feature in enumerate(features):
        if feature.get("name") == feature_name:
            # Substitute the feature with the new one
            features[idx] = feature_obj
            return wb_dict
    
    # If not found, add the new feature
    features.append(feature_obj)
    
    return wb_dict