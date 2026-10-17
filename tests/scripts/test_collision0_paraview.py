from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "paraview_scripts"
    / "Collision0"
    / "collision0.py"
)


def test_fastscape_uses_custom_signed_color_scales():
    source = SCRIPT_PATH.read_text()

    assert "TOPOGRAPHY_COLOR_POINTS" in source
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


def test_fastscape_does_not_create_separate_sedimentation_plots():
    source = SCRIPT_PATH.read_text()

    assert "fastscape_programmable1" not in source
    assert '"sedimentation_rate"' not in source
    assert '"sedimentation"' not in source
    assert "PLOT_TYPE_sedimentation_rate" not in source
    assert "PLOT_TYPE_total_sedimentation" not in source
