"""
Module for ASPECT prm and GWB wb utilities
"""
import re
from copy import deepcopy
from .io import parse_entry_as_list, format_list_as_entry,\
        parse_composition_entry, format_composition_entry,\
        parse_isosurfaces_entry, format_isosurfaces_entry

def delete_composition_from_prm(prm_dict, composition_name):
    """
    Remove a specific composition entry from an ASPECT prm file text.

    This function searches for lines that define compositional fields and
    removes the specified composition name from those definitions.

    Parameters:
        prm_text (str): Full text content of an ASPECT prm file.
        composition_name (str): Name of the composition to remove.

    Returns:
        str: Modified prm file text with the composition removed.
    """

    names_of_fields = parse_entry_as_list(prm_dict["Compositional fields"]["Names of fields"])
    number_of_compositional_fields = len(names_of_fields)

    # Get the index of the composition to remove
    remove_idx = names_of_fields.index(composition_name)

    # Remove it from the name of field
    names_of_fields.pop(remove_idx)
    prm_dict["Compositional fields"]["Names of fields"] = format_list_as_entry(names_of_fields)

    prm_dict["Compositional fields"]["Number of fields"] = str(number_of_compositional_fields-1)

    # Remove it from the compositional field methods
    try:
        compositional_field_methods = parse_entry_as_list(prm_dict["Compositional fields"]["Compositional field methods"])
    except KeyError:
        pass
    else:
        compositional_field_methods.pop(remove_idx)
        prm_dict["Compositional fields"]["Compositional field methods"] = format_list_as_entry(compositional_field_methods)

    # Remove it from the list of relevant compositions
    try:
        relevant_compositions_in_worldbuilder = parse_entry_as_list(prm_dict["Initial composition model"]["World builder"]["List of relevant compositions"])
    except KeyError:
        pass
    else:
        relevant_compositions_in_worldbuilder.pop(remove_idx)
        prm_dict["Initial composition model"]["World builder"]["List of relevant compositions"] = format_list_as_entry(relevant_compositions_in_worldbuilder)

    # Remove it from compositional field thresholds
    try:
        compositional_field_thresholds = parse_entry_as_list(prm_dict["Mesh refinement"]["Composition threshold"]["Compositional field thresholds"])
    except KeyError:
        pass
    else:
        compositional_field_thresholds.pop(remove_idx)
        prm_dict["Mesh refinement"]["Composition threshold"]["Compositional field thresholds"] = format_list_as_entry(compositional_field_thresholds)

    # Remove it from the isosurfaces
    try:
        isosurfaces = parse_isosurfaces_entry(prm_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"])
    except KeyError:
        pass
    else:
        for i, isosurface in enumerate(isosurfaces):
            if isosurface["composition"] == composition_name:
                isosurfaces.pop(i)
                break
        prm_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"] = format_isosurfaces_entry(isosurfaces)

    # Further remove composition in entry with complex strings in the pattern of "composition name: foo"
    remove_composition_from_prm_recursive(prm_dict, remove_compositions=[composition_name])

    return remove_idx


def remove_composition_from_prm_recursive(prm_dict, *, remove_compositions=[]):
    """
    Remove entries of composition from the prm file
    prm_dict : dict
        Deal.II/ASPECT-style nested parameter dictionary to be modified
        in-place by the rule.
    remove_compositions: list
        names of compositions to remove from prm file
    """
    for key, value in prm_dict.items():
        if isinstance(value, dict):
            # call function recursively in case value is dict
            remove_composition_from_prm_recursive(value, remove_compositions=remove_compositions)
        else:
            if not isinstance(value, str):
                raise TypeError("value must be dict or str, get %s" % str(value))

            try:
                # look for entries with composition options
                comp_dict = parse_composition_entry(value)
            except ValueError:
                pass
            else:
                # delete certain compositions from them and parsed back
                for composition in remove_compositions:
                    if composition in comp_dict:
                        comp_dict.pop(composition)
                prm_dict[key] = format_composition_entry(comp_dict)


def find_WB_feature_by_name(wb_dict, feature_name):
    """
    Locate a feature in a Geodynamic World Builder dictionary by name.

    This function searches through the list of features in a GWB configuration
    dictionary and returns both the index of the feature and the feature
    dictionary itself. If no matching feature is found, the function returns
    two None values.

    Parameters:
        wb_dict (dict): Parsed Geodynamic World Builder configuration dictionary.
        feature_name (str): Name of the feature to locate.

    Returns:
        tuple[int or None, dict or None]:
            Index of the matching feature and the feature dictionary if found.
            Returns (None, None) if no matching feature exists.
    """

    features = wb_dict.get("features", [])

    for i, feature in enumerate(features):
        if feature.get("name") == feature_name:
            return i, feature

    return None, None



def duplicate_composition_from_prm(prm_dict, composition_from_name, composition_to_name):
    """
    Duplicate a specific composition entry to another composition from an ASPECT prm file text.

    This function searches for lines that define compositional fields and
    duplicates the specified composition in those definitions and assign it to
    another composition

    Parameters:
        prm_text (str): Full text content of an ASPECT prm file.
        composition_from_name (str): Name of the composition to duplicate from.
        composition_to_name (str): Name of the composition to assign to.

    Returns:
        str: Modified prm file text with the composition removed.
    """

    names_of_fields = parse_entry_as_list(prm_dict["Compositional fields"]["Names of fields"])
    number_of_compositional_fields = len(names_of_fields)

    # Get the index of the composition to remove
    from_idx = names_of_fields.index(composition_from_name)
    to_idx = number_of_compositional_fields

    # Add new entry to the name of field
    names_of_fields.append(composition_to_name)
    prm_dict["Compositional fields"]["Names of fields"] = format_list_as_entry(names_of_fields)

    prm_dict["Compositional fields"]["Number of fields"] = str(number_of_compositional_fields+1)

    # Add new entry to the compositional field methods
    try:
        compositional_field_methods = parse_entry_as_list(prm_dict["Compositional fields"]["Compositional field methods"])
    except KeyError:
        pass
    else:
        new_value = copy_or_substitute_string_entry(composition_from_name, composition_to_name, compositional_field_methods[from_idx])
        compositional_field_methods.append(new_value)
        prm_dict["Compositional fields"]["Compositional field methods"] = format_list_as_entry(compositional_field_methods)

    # Add new entry to the list of relevant compositions
    try:
        relevant_compositions_in_worldbuilder = parse_entry_as_list(prm_dict["Initial composition model"]["World builder"]["List of relevant compositions"])
    except KeyError:
        pass
    else:
        new_value = copy_or_substitute_string_entry(composition_from_name, composition_to_name, relevant_compositions_in_worldbuilder[from_idx])
        relevant_compositions_in_worldbuilder.append(new_value)
        prm_dict["Initial composition model"]["World builder"]["List of relevant compositions"] = format_list_as_entry(relevant_compositions_in_worldbuilder)

    # Add new entry to the compositional field thresholds
    try:
        compositional_field_thresholds = parse_entry_as_list(prm_dict["Mesh refinement"]["Composition threshold"]["Compositional field thresholds"])
    except KeyError:
        pass
    else:
        new_value = copy_or_substitute_string_entry(composition_from_name, composition_to_name, compositional_field_thresholds[from_idx])
        compositional_field_thresholds.append(new_value)
        prm_dict["Mesh refinement"]["Composition threshold"]["Compositional field thresholds"] = format_list_as_entry(compositional_field_thresholds)

    # Add new entry to the isosurfaces
    try:
        isosurfaces = parse_isosurfaces_entry(prm_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"])
    except KeyError:
        pass
    else:
        for isosurface in isosurfaces:
            if isosurface["composition"] == composition_from_name:
                new_isosurface = deepcopy(isosurface)
                new_isosurface["composition"] = composition_to_name
                isosurfaces.append(new_isosurface)
                break
        prm_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"] = format_isosurfaces_entry(isosurfaces)

    # Further duplicate composition in entry with complex strings in the pattern of "composition name: foo"
    duplicate_composition_in_prm_recursive(prm_dict, composition_from_name, composition_to_name)

    return to_idx


def duplicate_composition_in_prm_recursive(prm_dict, composition_from_name, composition_to_name):
    """
    Duplicate entries of composition from the prm file
    prm_dict : dict
        Deal.II/ASPECT-style nested parameter dictionary to be modified
        in-place by the rule.
    composition_from_name: str:
        name of compositions to duplicate from
    composition_to_name: str:
        name of compositions to duplicate to
    """
    for key, value in prm_dict.items():
        if isinstance(value, dict):
            # call function recursively in case value is dict
            duplicate_composition_in_prm_recursive(value, composition_from_name, composition_to_name)
        else:
            if not isinstance(value, str):
                raise TypeError("value must be dict or str, get %s" % str(value))

            try:
                # look for entries with composition options
                comp_dict = parse_composition_entry(value)
            except ValueError:
                pass
            else:
                # delete certain compositions from them and parsed back
                if composition_from_name in comp_dict:
                    value = copy_or_substitute_string_entry(composition_from_name, composition_to_name, comp_dict[composition_from_name])
                    comp_dict[composition_to_name] = value

                prm_dict[key] = format_composition_entry(comp_dict)


def copy_or_substitute_string_entry(str_from, str_to, entry):
    """
    Substitute a substring in a string entry using regular expressions.

    If the provided entry is a string, this function replaces all occurrences
    of the pattern str_from with str_to using re.sub. If the entry is not a
    string, it is returned unchanged.

    Parameters:
        str_from (str): Regular expression pattern to search for.
        str_to (str): Replacement string.
        entry (any): Input value that may or may not be a string.

    Returns:
        any: Modified string if entry is a string, otherwise the original entry.
    """

    if isinstance(entry, str):
        return re.sub(str_from, str_to, entry)
    else:
        return entry
