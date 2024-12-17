import os
import pytest
from hamageolib.utils.world_builder_file_parser import find_feature_by_name, update_or_add_feature

def test_find_feature_by_name_valid_feature():
    """
    Test that the function correctly finds a feature by its name in a valid dictionary.
    """
    wb_dict = {
        "features": [
            {"name": "feature1", "value": 10},
            {"name": "feature2", "value": 20},
        ]
    }
    result = find_feature_by_name(wb_dict, "feature2")
    assert result == {"name": "feature2", "value": 20}

def test_find_feature_by_name_missing_feature():
    """
    Test that the function returns None when the feature name is not found.
    """
    wb_dict = {
        "features": [
            {"name": "feature1", "value": 10},
        ]
    }
    result = find_feature_by_name(wb_dict, "nonexistent")
    assert result is None

def test_find_feature_by_name_empty_features():
    """
    Test that the function returns None when the 'features' list is empty.
    """
    wb_dict = {
        "features": []
    }
    result = find_feature_by_name(wb_dict, "feature1")
    assert result is None

def test_find_feature_by_name_no_features_key():
    """
    Test that the function raises an AssertionError when 'features' is missing.
    """
    wb_dict = {}
    with pytest.raises(AssertionError, match="'features' must exist as a key.*"):
        find_feature_by_name(wb_dict, "feature1")

def test_find_feature_by_name_features_not_a_list():
    """
    Test that the function raises an AssertionError when 'features' is not a list.
    """
    wb_dict = {
        "features": "not_a_list"
    }
    with pytest.raises(AssertionError, match="'features' must exist as a key.*"):
        find_feature_by_name(wb_dict, "feature1")

def test_find_feature_by_name_invalid_dict():
    """
    Test that the function raises an AssertionError when input is not a dictionary.
    """
    wb_dict = "not_a_dict"
    with pytest.raises(AssertionError):
        find_feature_by_name(wb_dict, "feature1")

def test_update_existing_feature():
    """
    Test that the function correctly updates an existing feature.
    """
    wb_dict = {
        "features": [
            {"name": "feature1", "value": 10},
            {"name": "feature2", "value": 20},
        ]
    }
    feature_obj = {"name": "feature2", "value": 99}
    result = update_or_add_feature(wb_dict, "feature2", feature_obj)
    assert result["features"][1] == {"name": "feature2", "value": 99}

def test_add_new_feature():
    """
    Test that the function adds a new feature if it does not exist.
    """
    wb_dict = {
        "features": [{"name": "feature1", "value": 10}]
    }
    feature_obj = {"name": "feature2", "value": 20}
    result = update_or_add_feature(wb_dict, "feature2", feature_obj)
    assert result["features"][-1] == {"name": "feature2", "value": 20}
    assert len(result["features"]) == 2

def test_missing_features_key():
    """
    Test that the function raises an AssertionError when 'features' is missing.
    """
    wb_dict = {}
    feature_obj = {"name": "feature1", "value": 10}
    with pytest.raises(AssertionError, match="wb_dict must contain a 'features' key."):
        update_or_add_feature(wb_dict, "feature1", feature_obj)

def test_features_not_a_list():
    """
    Test that the function raises an AssertionError when 'features' is not a list.
    """
    wb_dict = {"features": "not_a_list"}
    feature_obj = {"name": "feature1", "value": 10}
    with pytest.raises(AssertionError, match="'features' must be a list."):
        update_or_add_feature(wb_dict, "feature1", feature_obj)

def test_name_mismatch():
    """
    Test that the function raises an AssertionError when feature_obj's 'name' does not match feature_name.
    """
    wb_dict = {"features": [{"name": "feature1", "value": 10}]}
    feature_obj = {"name": "wrong_name", "value": 20}
    with pytest.raises(AssertionError, match="feature_obj's 'name' .* does not match feature_name"):
        update_or_add_feature(wb_dict, "feature1", feature_obj)

def test_update_or_add_with_empty_features():
    """
    Test that the function adds a feature when the 'features' list is empty.
    """
    wb_dict = {"features": []}
    feature_obj = {"name": "feature1", "value": 10}
    result = update_or_add_feature(wb_dict, "feature1", feature_obj)
    assert result["features"] == [{"name": "feature1", "value": 10}]