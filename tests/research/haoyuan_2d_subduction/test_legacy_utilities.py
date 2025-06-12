import pytest
from hamageolib.research.haoyuan_2d_subduction.legacy_utilities import *

########################################
# Tests for JSON_OPT
########################################
@pytest.fixture
def sample_json_opt():
    opt = JSON_OPT()
    opt.add_key(description="Test float parameter",
                _type=float,
                keys=["test_param"],
                default_value=3.14,
                nick="tp")
    return opt


def test_add_key_initialization(sample_json_opt):
    assert sample_json_opt.number_of_keys() == 1
    assert sample_json_opt.values[0] == 3.14


def test_import_options_updates_values(sample_json_opt):
    sample_json_opt.import_options({"test_param": 2.71})
    assert sample_json_opt.values[0] == 2.71


def test_document_output(sample_json_opt):
    doc = sample_json_opt.document_str()
    assert "test_param" in doc
    assert "Default value: 3.14" in doc


def test_read_json_dict(sample_json_opt):
    sample_json_opt.read_json({"test_param": 1.23})
    assert sample_json_opt.values[0] == 1.23


def test_add_key_wrong_type_raises():
    opt = JSON_OPT()
    with pytest.raises(TypeError):
        opt.add_key(description="Wrong type", _type=int, keys=["bad"], default_value="not an int")


@pytest.mark.parametrize("key_path, value", [
    (["test_param"], 1.0),
    (["test_param"], 5.0),
])
def test_multiple_value_import(key_path, value):
    opt = JSON_OPT()
    opt.add_key(description="Test key", _type=float, keys=key_path, default_value=0.0)
    opt.import_options({key_path[0]: value})
    assert opt.values[0] == value
