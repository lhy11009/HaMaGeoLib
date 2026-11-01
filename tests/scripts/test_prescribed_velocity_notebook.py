import ast
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[2] / 'notebooks/research/prescribe_velocity/PrescribeVelocity.ipynb'


def parameters(minimum, model='chunk'):
    return f'''set Dimension = 3
subsection Geometry model
  set Model name = {model}
  subsection Chunk
    set Chunk inner radius = 5e6
    set Chunk outer radius = 6e6
    set Chunk minimum longitude = {minimum}
    set Chunk maximum longitude = 42
    set Chunk minimum latitude = -20
    set Chunk maximum latitude = 55
  end
end
'''


@pytest.mark.parametrize('recorded', [False, True])
def test_frame_reads_recorded_or_input_domain(tmp_path, monkeypatch, recorded):
    monkeypatch.chdir(NOTEBOOK.parents[3])
    monkeypatch.setenv('PVPYTHON', '/bin/true')
    (tmp_path/'output').mkdir()
    (tmp_path/'output/solution.pvd').touch()
    (tmp_path/'case.prm').write_text(parameters(-2))
    if recorded:
        (tmp_path/'output/original.prm').write_text(parameters(-8))
    notebook=json.loads(NOTEBOOK.read_text())
    source=''.join(notebook['cells'][10]['source'])
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            name=node.targets[0].id
            if name=='visualization_case_dir':
                node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(tmp_path))],keywords=[])
            elif name=='show_velocity_frame':
                node.value=ast.Constant(True)
    exec(compile(ast.fix_missing_locations(tree), '<notebook>', 'exec'), {})
    runner=(tmp_path/'paraview_scripts/run_prescribed_velocity.py').read_text()
    call=ast.parse(runner).body[-1].value
    options=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='frame_options'))
    assert options['radii']==(5e6,6e6)
    assert options['longitude']==(-8 if recorded else -2,42)
    assert options['latitude']==(-20,55)
