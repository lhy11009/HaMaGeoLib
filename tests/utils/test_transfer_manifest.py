"""Interval transfer selection and complete parallel dataset coverage."""
import xml.etree.ElementTree as ET

import pytest

from hamageolib.utils.transfer_manifest import build_manifest, write_manifest


def make_case(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    for family in ('solution', 'particles'):
        folder = output / family
        folder.mkdir()
        root = ET.Element('VTKFile')
        collection = ET.SubElement(root, 'Collection')
        for i, time in enumerate((0, 240000, 510000, 1000000)):
            name = f'{family}/{family}-{i:05d}.pvtu'
            ET.SubElement(collection, 'DataSet', timestep=str(time), file=name)
            (output / name).write_text('<VTKFile><PUnstructuredGrid><Piece Source="piece%d.vtu"/></PUnstructuredGrid></VTKFile>' % i)
            (folder / f'piece{i}.vtu').write_text('data')
        ET.ElementTree(root).write(output / f'{family}.pvd')
    return tmp_path


def test_select_independent_families_and_write(tmp_path):
    case = make_case(tmp_path)
    result = build_manifest(case, 500000, families=('solution', 'particles'))
    assert len(result['selections']) == 6
    assert 'output/solution/piece2.vtu' in result['files']
    assert 'output/particles/piece2.vtu' in result['files']
    assert 'output/solution/piece1.vtu' not in result['files']
    dest = tmp_path.parent / (tmp_path.name + '_manifest')
    write_manifest(result, dest)
    assert (dest / 'transfer_files.txt').is_file()
    tree = ET.parse(dest / 'output/solution.pvd')
    assert len(tree.findall('.//DataSet')) == 3


def test_missing_piece_is_error(tmp_path):
    make_case(tmp_path)
    (tmp_path / 'output/solution/piece2.vtu').unlink()
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path, 500000)


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf')])
def test_invalid_interval(tmp_path, interval):
    with pytest.raises(ValueError):
        build_manifest(tmp_path, interval)


def test_topography_and_fastscape_use_simulation_steps(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'statistics').write_text(
        '# 1: Time step number\n# 2: Time (years)\n'
        '0 0\n17 240000\n131 510000\n260 1000000\n')
    for family in ('topography', 'fastscape'):
        (output / family).mkdir()
    for step in (0, 17, 131, 260):
        (output / f'topography/topography.{step:05d}').write_text('data')
        (output / f'fastscape/Topography{step:07d}.vtk').write_text('data')
    result = build_manifest(tmp_path, 500000, families=('topography', 'fastscape'))
    assert 'output/topography/topography.00131' in result['files']
    assert 'output/fastscape/Topography0000131.vtk' in result['files']
    assert 'output/topography/topography.00017' not in result['files']
    assert len(result['selections']) == 6


def test_particle_clock_is_independent(tmp_path):
    make_case(tmp_path)
    path = tmp_path / 'output/particles.pvd'
    tree = ET.parse(path)
    for node, time in zip(tree.findall('.//DataSet'), (0, 500000, 750000, 1000000)):
        node.set('timestep', str(time))
    tree.write(path)
    result = build_manifest(tmp_path, 500000, families=('solution', 'particles'))
    assert 'output/solution/piece2.vtu' in result['files']
    assert 'output/particles/piece1.vtu' in result['files']
    assert 'output/particles/piece2.vtu' not in result['files']


def test_bounds_ties_and_deduplication(tmp_path):
    make_case(tmp_path)
    result = build_manifest(tmp_path, 10000, start=240000, end=510000)
    assert len(result['selections']) == 2
    assert {r['actual_time'] for r in result['selections']} == {240000, 510000}
    result = build_manifest(tmp_path, 375000, start=0, end=510000)
    assert result['selections'][-1]['actual_time'] == 510000


def test_external_reference_rejected(tmp_path):
    make_case(tmp_path)
    pvd = tmp_path / 'output/solution.pvd'
    tree = ET.parse(pvd)
    tree.find('.//DataSet').set('file', '../../outside.vtu')
    tree.write(pvd)
    with pytest.raises(ValueError, match='escapes case'):
        build_manifest(tmp_path, 500000)


def test_missing_family_and_bad_bounds(tmp_path):
    make_case(tmp_path)
    with pytest.raises(FileNotFoundError, match='topography'):
        build_manifest(tmp_path, 500000, families=('topography',))
    with pytest.raises(ValueError):
        build_manifest(tmp_path, 500000, start=10, end=0)


def test_fastscape_collection_and_partitions(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    folder = output / 'fastscape'
    folder.mkdir()
    (folder / 'a.vtk').write_text('data')
    (folder / 'b.vtk').write_text('data')
    (folder / 'fastscape.pvd').write_text(
        '<VTKFile><Collection>'
        '<DataSet timestep="0" part="0" file="a.vtk"/>'
        '<DataSet timestep="0" part="1" file="b.vtk"/>'
        '</Collection></VTKFile>')
    result = build_manifest(tmp_path, 500000, families=('fastscape',))
    assert result['files'] == ['output/fastscape/a.vtk', 'output/fastscape/b.vtk']
    assert len(result['selections']) == 2


def test_no_outputs_in_bounds(tmp_path):
    make_case(tmp_path)
    with pytest.raises(ValueError, match='No snapshots'):
        build_manifest(tmp_path, 500000, start=2000000)


def test_commands_quote_paths():
    import shlex
    from hamageolib.utils.transfer_manifest import transfer_commands
    lines = transfer_commands('ucd:/case with spaces', '/data/case name', './manifest').splitlines()
    args = shlex.split(lines[1])
    assert args[-2:] == ['ucd:/case with spaces/', '/data/case name/']
    assert '--files-from=manifest/transfer_files.txt' in args
