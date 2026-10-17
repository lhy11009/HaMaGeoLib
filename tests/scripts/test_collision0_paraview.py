import re
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "paraview_scripts"
    / "Collision0"
    / "collision0.py"
)
BASE_PATH = SCRIPT_PATH.parents[1] / "base.py"
CASE_OPTIONS_PATH = (
    SCRIPT_PATH.parents[3] / "hamageolib" / "utils" / "case_options.py"
)


def test_fastscape_uses_custom_signed_color_scales():
    source = SCRIPT_PATH.read_text()

    assert "TPOGRAPHY_COLOR_POINTS" in source
    assert "EROSION_DEPOSITION_COLOR_POINTS" in source
    assert "rescale_transfer_function_combined(field, -11000.0, 4000.0)" in source
    assert "rescale_transfer_function_combined(field, -1e-2, 1e-2)" in source
    assert "rescale_transfer_function_combined(field, -1e4, 1e4)" in source
    assert "(0.00, 127 / 255.0, 0 / 255.0, 0 / 255.0)" in source
    assert "(1.00, 37 / 255.0, 52 / 255.0, 148 / 255.0)" in source
    assert "Deposition (-) / erosion (+) rate" in source
    assert "Cumulative deposition (-) / erosion (+)" in source

    erosion_rate_function = source.split(
        "def set_fastscape_erosion_rate_plot", 1
    )[1].split("def set_fastscape_erosion_rate_colorbar", 1)[0]
    total_erosion_function = source.split(
        "def set_fastscape_total_erosion_plot", 1
    )[1].split("def set_fastscape_total_erosion_colorbar", 1)[0]
    assert "UseLogScale = 1" not in erosion_rate_function
    assert "UseLogScale = 1" not in total_erosion_function
    assert "SCM_batlow" not in erosion_rate_function
    assert "SCM_batlow" not in total_erosion_function


def test_fastscape_creates_only_cumulative_signed_sedimentation():
    source = SCRIPT_PATH.read_text()

    assert "fastscape_programmable1" in source
    assert "Sedimentation = Basement - Topography" in source
    assert '"sedimentation_rate"' not in source
    assert '"sedimentation"' in source
    assert "PLOT_TYPE_sedimentation_rate" not in source
    assert "PLOT_TYPE_total_sedimentation" in source
    assert "set_fastscape_total_sedimentation_plot" in source


def test_fastscape_adds_requested_dashed_contours():
    base_source = BASE_PATH.read_text()
    case_source = SCRIPT_PATH.read_text()

    assert "def add_dashed_contour" in base_source
    assert "Contour(" in base_source
    assert "ProgrammableFilter(" in base_source
    assert "dash_length" in base_source
    assert "gap_length" in base_source

    expected_contours = (
        ('_sourceF', 'erosion_rate', '1e-3'),
        ('_sourceF', 'total_erosion', '1e3'),
        ('_sourceFPG', 'sedimentation', '-1e3'),
    )
    for source, field, value in expected_contours:
        assert re.search(
            rf'add_optional_fastscape_contour\(\s*{source}, "{field}", {value}',
            case_source,
        )


def test_fastscape_contours_are_optional_and_off_by_default():
    case_source = SCRIPT_PATH.read_text()
    options_source = CASE_OPTIONS_PATH.read_text()

    assert 'self.options["PLOT_FASTSCAPE_CONTOURS"] = "False"' in options_source
    assert "def add_optional_fastscape_contour" in case_source
    optional_helper = case_source.split(
        "def add_optional_fastscape_contour", 1
    )[1].split("def remove_optional_fastscape_contour", 1)[0]
    assert "if not PLOT_FASTSCAPE_CONTOURS:" in optional_helper
