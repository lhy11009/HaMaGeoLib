import ast
import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).parents[2]
    / 'notebooks/research/prescribe_velocity/PreProcess.ipynb'
)
SLAB_CODES = {
    'ALU', 'CAL', 'CAM', 'CAR', 'CAS', 'COT', 'HAL', 'HEL', 'HIM',
    'HIN', 'IZU', 'KER', 'KUR', 'MAK', 'MAN', 'MUE', 'PAM', 'PHI',
    'PNG', 'PUY', 'RYU', 'SAM', 'SCO', 'SOL', 'SUL', 'SUM', 'VAN',
}


def notebook_cells():
    notebook = json.loads(NOTEBOOK.read_text())
    return {cell['id']: cell for cell in notebook['cells']}


def literal_assignments(source):
    assignments = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return assignments


def test_regional_configuration_covers_every_slab_code():
    cells = notebook_cells()
    source = ''.join(cells['regional-configuration']['source'])
    assignments = literal_assignments(source)

    regions = assignments['regional_map_regions']
    assert set(regions) == SLAB_CODES
    assert assignments['regional_slab_code'] in regions
    assert all(
        len(bounds) == 4
        and bounds[0] < bounds[1]
        and bounds[2] < bounds[3]
        for bounds in regions.values()
    )


def test_regional_plot_cells_are_valid_python():
    cells = notebook_cells()
    for cell_id in ('regional-configuration', 'plot-regional-observations'):
        compile(''.join(cells[cell_id]['source']), f'<{cell_id}>', 'exec')
