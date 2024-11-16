# dealii_param_parser.py
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
dealii_param_parser.py

Author: Haoyuan Li
Description: A utility for parsing and handling deal.ii parameter files 
             in the HaMaGeoLib package.

Functions:
    - parse_parameters: Parses a deal.ii parameter file and returns a dictionary 
                        of parameters.
    - save_parameters: Saves a dictionary of parameters to a deal.ii formatted file.
"""

import re

def parse_parameters_to_dict(file_input):
    """
    Parses a deal.ii parameter file and returns a dictionary of parameters.

    Args:
        file_input (TextIO): A file object opened for reading, which contains the parameter data.

    Returns:
        dict: A dictionary with parameter names as keys and their corresponding values, 
              including support for nested subsections as dictionaries.
    """
    # Dictionary to store parsed parameters
    parameters = {}
    current_line = file_input.readline()
    while current_line != "":
        # Inputs formats:
        # - Comments: Lines starting with '#'
        # - Section markers: "subsection name" to start and "end" to close
        # - Key-value pairs: 'set key = value'
        if re.match(r'^(\t| )*#', current_line):
            # Skip lines that are comments (indicated by #)
            pass
        elif re.match(r'^(\t| )*set', current_line):
            # Parse key-value pairs in 'set key = value' format
            # Clean up 'set' keyword and split by '='
            line_cleaned = re.sub(r'^(\t| )*set ', '', current_line, count=1)
            key_value_pair = line_cleaned.split('=', maxsplit=1)
            key = key_value_pair[0]
            key = re.sub(r'(\t| )*$', '', key)
            value = key_value_pair[1]
            value = re.sub(r'^ *', '', value)
            value = re.sub(r' *(#.*)?\n$', '', value)
            while value[-1] == '\\':
                # Handle multi-line values where lines end with '\'
                current_line = file_input.readline()
                current_line = re.sub(r' *(#.*)?\n$', '', current_line)
                value = value + '\n' + current_line
            parameters[key] = value
        elif re.match(r'^.*subsection', current_line):
            # Start of a subsection; create a nested dictionary
            subsection_name = re.sub(r'^.*subsection ', '', current_line)
            subsection_name = re.sub(r' *(#.*)?\n$', '', subsection_name)
            try:
                # Handle repeated subsections by updating existing entries
                parameters[subsection_name]
                print('%s is already presented, going to update.' % subsection_name)
            except KeyError:
                # Recursively parse the subsection if not already in dictionary
                parameters[subsection_name] = parse_parameters_to_dict(file_input)
            else:
                # Update the existing subsection with new entries if present
                new_entries = parse_parameters_to_dict(file_input)
                parameters[subsection_name].update(new_entries.items())
        elif re.match(r'^.*end', current_line):
            # End of a subsection; return the current dictionary
            return parameters
        current_line = file_input.readline()
    return parameters


def save_parameters_from_dict(fout, parameters_dict, indent_level=0):
    """
    Saves a dictionary of parameters to a deal.ii formatted file, preserving 
    nested structure using subsections.

    Args:
        fout (TextIO): An open file object where the parameters will be written.
        parameters_dict (dict): Dictionary of parameters to save, where each key 
                                is a parameter or subsection name and each value is either 
                                a string (for parameter values) or a nested dictionary (for subsections).
        indent_level (int): Current indentation level to format nested sections (default is 0).

    Raises:
        ValueError: If a dictionary value is not of type str or dict, raises an error indicating 
                    an unsupported value type.
    """
    indent = ' ' * 4 * indent_level  # Set indentation based on the current level for nested sections
    for key, value in parameters_dict.items():
        # Determine the output format: either set a parameter or define a subsection
        if isinstance(value, str):
            fout.write(indent + 'set %s = %s\n' % (key, value))
        elif isinstance(value, dict):
            if indent_level == 0:
                fout.write('\n')
            fout.write(indent + 'subsection %s\n' % key)
            next_indent_level = indent_level + 1
            save_parameters_from_dict(fout, value, next_indent_level)
            fout.write(indent + 'end\n')
            if indent_level == 0:
                fout.write('\n')
        else:
            # Raise an error if the value type is unsupported
            raise ValueError('Value in dictionary must be str or dict, received:\n key: '\
                             + key + "\n type of value: " + str(type(value)) + "\n value: " + str(value))

