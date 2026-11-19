"""Keep erosion sediment distinct in processed and raw composition plots."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import pytest

from hamageolib.research.haoyuan_collision0.post_process import (
    PYVISTA_PROCESS_COLLISION, save_composition_indicator_legend,
)

SCRIPT = Path(__file__).parents[3] / "scripts/paraview_scripts/Collision0/collision0.py"


def grid_for_compositions(split, sediment1):
    grid = pv.PolyData(np.column_stack((np.arange(9), np.zeros(9), np.zeros(9))))
    for name in ["gabbro", "MORB", "sediment", "crust_upper", "crust_lower"]:
        grid[name] = np.zeros(9)
    grid["crust_upper"][1] = 1.
    grid["crust_lower"][2] = 1.
    grid["gabbro"][3] = 1.
    grid["sediment"][4] = 1.
    if split:
        for name, index in [("sp_crust_upper", 1), ("sp_crust_lower", 2),
                            ("ov_crust_upper", 6), ("ov_crust_lower", 7)]:
            grid[name] = np.eye(9)[index]
        grid["crust_upper"][6] = 1.
        grid["crust_lower"][7] = 1.
    grid["T"] = np.full(9, 1800.)
    grid["T"][5] = 1000.
    if sediment1:
        grid["sediment1"] = np.eye(9)[8]
    return grid


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("sediment1", [False, True])
@pytest.mark.parametrize("vtk_namespace", [False, True])
def test_composition_paths_agree(split, sediment1, vtk_namespace):
    grid = grid_for_compositions(split, sediment1)
    processor = PYVISTA_PROCESS_COLLISION.__new__(PYVISTA_PROCESS_COLLISION)
    processor.grid = grid
    processor.composition_names = ["gabbro", "MORB", "sediment", "crust_upper", "crust_lower"]
    processor.lithospheric_T = 1573.15
    processor.add_background()
    processor.add_composition_indicator(upper_lower_plate=split)
    expected = [0, 1, 2, 3, 4, 5, 6 if split else 0, 7 if split else 0, 8 if sediment1 else 0]
    np.testing.assert_array_equal(grid["composition_indicator"], expected)
    if sediment1:
        assert grid["background"][8] == 0.

    # Execute the actual ParaView programmable-filter body using local VTK.
    node = next(n for n in ast.parse(SCRIPT.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "add_composition_indicator")
    filter_object = SimpleNamespace()
    namespace = dict(FindSource=lambda name: grid,
                     ProgrammableFilter=lambda **kw: filter_object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    namespace["add_composition_indicator"]("solution")
    code = filter_object.Script.replace("CONTINENT_COMPOSITIONS", repr(["crust_upper", "crust_lower"]))
    code = code.replace("OCEANIC_CRUST_COMPOSITIONS", repr(["gabbro", "MORB"]))
    code = code.replace("LITHOSPHERE_TEMPERATURE", "1573.15")
    output = pv.PolyData()
    filter_namespace = dict(self=SimpleNamespace(GetInputDataObject=lambda *a: grid), output=output)
    if vtk_namespace:
        # ParaView injects VTK array algorithms into programmable-filter globals.
        from vtk.numpy_interface import algorithms
        filter_namespace["all"] = algorithms.all
    exec(code, filter_namespace)
    np.testing.assert_array_equal(output["composition_indicator"], expected)


def test_animation_legend_matches_paraview_palette(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    tree = ast.parse(SCRIPT.read_text())
    palette = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute)
                   and t.attr == "IndexedColors" for t in n.targets))
    annotations = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute)
                       and t.attr == "Annotations" for t in n.targets))
    figures = []
    original = Figure.savefig
    def save(fig, *args, **kwargs):
        figures.append(fig)
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", save)
    filename = tmp_path / "legend.png"
    assert save_composition_indicator_legend(filename) == str(filename)
    assert filename.stat().st_size > 0
    legend = figures[0].legends[0]
    assert [text.get_text() for text in legend.get_texts()] == annotations[1::2]
    colors = [handle.get_facecolor()[:3] for handle in legend.legend_handles]
    np.testing.assert_allclose(colors, np.array(palette).reshape(-1, 3))
    np.testing.assert_allclose(colors[8], [0.557, 0.267, 0.678])
