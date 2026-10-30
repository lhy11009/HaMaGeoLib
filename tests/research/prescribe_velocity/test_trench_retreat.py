import json
from pathlib import Path

import numpy as np
import pytest

from hamageolib.research.prescribe_vecloity.trench_retreat import generate_retreat_case


@pytest.fixture
def source_case(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    world = {
        'version': '1.2',
        'coordinate system': {'model': 'spherical', 'depth method': 'begin segment'},
        'cross section': [[0, 0], [10, 0]],
        'features': [{
            'model': 'subducting plate', 'name': 'slab',
            'coordinates': [[15, 41], [15, 25], [5, 5], [5, -1]],
            'dip point': [80, 0],
            'segments': [
                {'length': 200e3, 'thickness': [95e3], 'angle': [0, 45]},
                {'length': 400e3, 'thickness': [95e3], 'angle': [45]},
                {'length': 200e3, 'thickness': [95e3], 'angle': [45, 0]},
                {'length': 100e3, 'thickness': [95e3], 'angle': [0]},
            ],
            'composition models': [{'model': 'uniform', 'compositions': [0],
                                    'max distance slab top': 100e3}],
            'velocity models': [{'model': 'along surface', 'velocity magnitude': .05}],
            'indicator models': [{'model': 'uniform', 'indicators': ['velocity'],
                                  'max distance slab top': 100e3}],
        }],
    }
    (source / 'case.wb').write_text(json.dumps(world))
    (source / 'case.prm').write_text('''set Dimension = 3
set End time = 0
subsection Geometry model
  set Model name = chunk
  subsection Chunk
    set Chunk outer radius = 6371e3
    set Chunk inner radius = 5451e3
    set Chunk minimum longitude = 0
    set Chunk maximum longitude = 25
    set Chunk minimum latitude = -10
    set Chunk maximum latitude = 50
  end
end
''')
    return source


@pytest.mark.parametrize('retreat', [50e3, 100e3])
def test_retreat_and_terminal_anchor(source_case, tmp_path, retreat):
    target = tmp_path / 'result'
    report = generate_retreat_case(source_case, target, trench_retreat_m=retreat)
    assert (target / 'case.prm').read_bytes() == (source_case / 'case.prm').read_bytes()
    assert report['trench_retreat_m'] == retreat
    assert report['connector_endpoint_cross_track_error_max_m'] < 1e-3
    geometry = np.load(target / 'geometry.npz')
    old, new = geometry['old_trench'], geometry['new_trench']
    distance = 6371e3 * np.arctan2(np.linalg.norm(np.cross(old, new), axis=1),
                                  (old * new).sum(axis=1))
    np.testing.assert_allclose(distance, retreat, atol=1e-5)
    world = json.loads((target / 'case.wb').read_text())
    original = json.loads((source_case / 'case.wb').read_text())['features'][0]
    fixed = world['features'][1]
    assert fixed['coordinates'] == original['coordinates']
    assert fixed['segments'][-1] == original['segments'][-1]
    for feature in world['features']:
        assert feature['velocity models'] == original['velocity models']
        assert feature['indicator models'] == original['indicator models']


@pytest.mark.parametrize('retreat', [0, -1, float('nan'), float('inf')])
def test_invalid_retreat_does_not_create_case(source_case, tmp_path, retreat):
    target = tmp_path / 'result'
    with pytest.raises(ValueError):
        generate_retreat_case(source_case, target, trench_retreat_m=retreat)
    assert not target.exists()


def test_existing_destination_preserved(source_case, tmp_path):
    target = tmp_path / 'existing'
    target.mkdir()
    (target / 'marker').write_text('keep')
    with pytest.raises(FileExistsError):
        generate_retreat_case(source_case, target)
    assert (target / 'marker').read_text() == 'keep'


def test_unsupported_geometry_rejected(source_case, tmp_path):
    p = source_case / 'case.wb'
    world = json.loads(p.read_text())
    world['coordinate system']['model'] = 'cartesian'
    p.write_text(json.dumps(world))
    with pytest.raises(ValueError, match='spherical'):
        generate_retreat_case(source_case, tmp_path / 'result')
    assert not (tmp_path / 'result').exists()


def test_python_sampler_matches_worldbuilder_reference():
    from hamageolib.research.prescribe_vecloity._trench_bezier import sample_trench

    path = Path(__file__).parents[2] / 'fixtures/research/prescribe_velocity/trench_bezier_reference.json'
    data = json.loads(path.read_text())
    np.testing.assert_allclose(sample_trench(data['coordinates']), data['rows'], atol=2e-14, rtol=0)


def _profile_vertices(segments, radius):
    """Independent midpoint-heading integration of constant-curvature arcs."""
    p = np.array([0., radius])
    vertices = [p.copy()]
    for segment in segments:
        a, b = np.radians([segment['angle'][0], segment['angle'][-1]])
        offset = np.arctan2(p[0], p[1])
        mean = (a + b) / 2 + offset
        half_turn = (b - a) / 2
        distance = segment['length'] * np.sinc(half_turn / np.pi)
        p = p + distance * np.array([np.cos(mean), -np.sin(mean)])
        vertices.append(p.copy())
    return np.asarray(vertices)


def test_split_transition_preserves_both_terminal_endpoints(source_case, tmp_path):
    target = tmp_path / 'result'
    generate_retreat_case(source_case, target)
    original = json.loads((source_case / 'case.wb').read_text())['features'][0]
    fixed = json.loads((target / 'case.wb').read_text())['features'][1]
    before = _profile_vertices(original['segments'], 6371e3)
    after = _profile_vertices(fixed['segments'], 6371e3)
    np.testing.assert_allclose(after[-2:], before[-2:], atol=1e-5, rtol=0)


def test_failed_fit_leaves_no_destination(source_case, tmp_path, monkeypatch):
    import hamageolib.research.prescribe_vecloity.trench_retreat as module
    from types import SimpleNamespace

    monkeypatch.setattr(module, 'root', lambda *a, **k: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match='fit'):
        generate_retreat_case(source_case, tmp_path / 'result')
    assert not (tmp_path / 'result').exists()


def test_cli_generates_case(source_case, tmp_path, capsys):
    from hamageolib.research.prescribe_vecloity.trench_retreat import main

    main([str(source_case), str(tmp_path / 'cli'), '--trench-retreat-m', '50000'])
    assert json.loads(capsys.readouterr().out)['trench_retreat_m'] == 50000
    assert (tmp_path / 'cli/case.wb').is_file()
