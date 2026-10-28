"""Validate workflow inputs before opening a ParaView pipeline."""
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / 'scripts/paraview_scripts/PrescribedVelocity/prescribed_velocity.py'


def test_missing_solution_rejected(tmp_path):
    workflow = runpy.run_path(str(SCRIPT))
    with pytest.raises(FileNotFoundError):
        workflow['plot_prescribed_velocity'](tmp_path / 'missing.pvd', tmp_path / 'plot.png')


@pytest.mark.parametrize('scale', [0, -1, float('nan')])
def test_invalid_glyph_scale_rejected(tmp_path, scale):
    pvd = tmp_path / 'solution.pvd'
    pvd.touch()
    workflow = runpy.run_path(str(SCRIPT))
    with pytest.raises(ValueError):
        workflow['plot_prescribed_velocity'](pvd, tmp_path / 'plot.png', glyph_scale=scale)
