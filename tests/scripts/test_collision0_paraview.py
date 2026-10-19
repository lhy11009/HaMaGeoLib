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
    assert "minimum, maximum = -1e-2, 1e-2" in source
    assert "rescale_transfer_function_combined(field, -5e3, 5e3)" in source
    assert "(0.00, 127 / 255.0, 0 / 255.0, 0 / 255.0)" in source
    assert "(1.00, 37 / 255.0, 52 / 255.0, 148 / 255.0)" in source
    assert "Erosion (-) / sedimentation (+) rate" in source
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


def test_sedimentation_positive_rate_with_both_ranges():
    import ast
    from types import SimpleNamespace

    tree = ast.parse(SCRIPT_PATH.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ("set_fastscape_erosion_rate_plot", "apply_custom_color_points") or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
             t.id == "EROSION_DEPOSITION_COLOR_POINTS" for t in n.targets)]
    for legacy in (False, True):
        lut = SimpleNamespace()
        ns = dict(USE_LEGACY_EROSION_RATE_RANGE=legacy,
                  ColorBy=lambda *a: None,
                  rescale_transfer_function_combined=lambda *a: None,
                  GetColorTransferFunction=lambda field: lut)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "plot", "exec"), ns)
        field, _ = ns["set_fastscape_erosion_rate_plot"](None)
        assert field == "signed_sedimentation_rate"
        assert lut.RGBPoints[0] == (-1e-2 if legacy else -3e-4)
        assert lut.RGBPoints[-4] == (1e-2 if legacy else 3e-4)
        assert lut.RGBPoints[-3] > lut.RGBPoints[-1]
        assert lut.RGBPoints[1] < lut.RGBPoints[3]
    collision_options = SCRIPT_PATH.parents[3] / "hamageolib/research/haoyuan_collision0/case_options.py"
    assert 'self.options["USE_LEGACY_EROSION_RATE_RANGE"] = "False"' in collision_options.read_text()
    assert "USE_LEGACY_EROSION_RATE" not in CASE_OPTIONS_PATH.read_text()
    assert 'Function = "-erosion_rate"' in SCRIPT_PATH.read_text()


def test_rate_palette_is_defined_directly_and_cumulative_palette_is_preserved():
    import ast

    source = SCRIPT_PATH.read_text()
    palette_names = {"EROSION_DEPOSITION_COLOR_POINTS", "CUMULATIVE_EROSION_DEPOSITION_COLOR_POINTS"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id in palette_names
                 for target in node.targets)]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "palettes", "exec"), ns)
    rate = ns["EROSION_DEPOSITION_COLOR_POINTS"]
    cumulative = ns["CUMULATIVE_EROSION_DEPOSITION_COLOR_POINTS"]
    assert rate[0] == (0.0, 37 / 255.0, 52 / 255.0, 148 / 255.0)
    assert rate[-1] == (1.0, 127 / 255.0, 0.0, 0.0)
    assert cumulative[0] == (0.0, 127 / 255.0, 0.0, 0.0)
    assert cumulative[-1] == (1.0, 37 / 255.0, 52 / 255.0, 148 / 255.0)
    assert "reversed(EROSION_DEPOSITION_COLOR_POINTS)" not in source
    assert source.count("fieldLUT, -5e3, 5e3, CUMULATIVE_EROSION_DEPOSITION_COLOR_POINTS") == 2
