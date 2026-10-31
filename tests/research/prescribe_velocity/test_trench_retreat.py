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


REGRESSION = Path(__file__).parents[2] / 'fixtures/research/prescribe_velocity/trench_retreat_regression'


def _assert_nested_close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_nested_close(a, b)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-7, rel=1e-10)
    else:
        assert actual == expected


@pytest.mark.parametrize('retreat', [100000, 300000])
def test_current_case_outputs_preserved(tmp_path, retreat):
    """Frozen from unmodified main using the actual notebook's _2 source."""
    target = tmp_path / 'baseline'
    source = REGRESSION / 'source'
    report = generate_retreat_case(source, target, trench_retreat_m=retreat)
    assert (target / 'case.prm').read_bytes() == (source / 'case.prm').read_bytes()
    expected = REGRESSION / str(retreat)
    _assert_nested_close(json.loads((target / 'case.wb').read_text()),
                         json.loads((expected / 'case.wb').read_text()))
    report.pop('source_case_dir')
    report.pop('destination_case_dir')
    _assert_nested_close(report, json.loads((expected / 'report.json').read_text()))
    with np.load(target / 'geometry.npz') as actual, np.load(expected / 'geometry.npz') as golden:
        assert set(actual.files) == set(golden.files)
        for key in golden.files:
            np.testing.assert_allclose(actual[key], golden[key], atol=1e-7, rtol=1e-10)


def test_explicit_default_options_preserve_outputs(source_case, tmp_path):
    default, explicit = tmp_path / 'default', tmp_path / 'explicit'
    generate_retreat_case(source_case, default)
    generate_retreat_case(source_case, explicit, chunk_min_longitude=None,
                          chunk_max_longitude=None, longitude_shift_deg=0.0)
    for name in ('case.prm', 'case.wb'):
        assert (default / name).read_bytes() == (explicit / name).read_bytes()
    with np.load(default / 'geometry.npz') as a, np.load(explicit / 'geometry.npz') as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_enlarge_chunk_without_moving_slab(source_case, tmp_path):
    original, expanded = tmp_path / 'original', tmp_path / 'expanded'
    generate_retreat_case(source_case, original)
    report = generate_retreat_case(source_case, expanded, chunk_min_longitude=-5,
                                   chunk_max_longitude=35)
    assert (expanded / 'case.wb').read_bytes() == (original / 'case.wb').read_bytes()
    from gdmate.aspect_tools import parse_parameters_to_dict
    with (source_case / 'case.prm').open() as stream:
        expected = parse_parameters_to_dict(stream)
    expected['Geometry model']['Chunk'].update({
        'Chunk minimum longitude': '-5', 'Chunk maximum longitude': '35'})
    with (expanded / 'case.prm').open() as stream:
        assert parse_parameters_to_dict(stream) == expected
    assert report['chunk_longitude_bounds_deg'] == [-5, 35]


@pytest.mark.parametrize('shift', [-5., 5.])
def test_longitude_shift_is_rigid_rotation(source_case, tmp_path, shift):
    original, moved = tmp_path / 'original', tmp_path / 'moved'
    baseline = generate_retreat_case(source_case, original)
    report = generate_retreat_case(source_case, moved, chunk_min_longitude=-10,
                                   chunk_max_longitude=40, longitude_shift_deg=shift)
    before = json.loads((original / 'case.wb').read_text())
    after = json.loads((moved / 'case.wb').read_text())
    for a, b in zip(before['features'], after['features']):
        for field in ('coordinates',):
            expected = np.array(a[field], dtype=float); expected[:, 0] += shift
            np.testing.assert_allclose(b[field], expected, atol=1e-12)
        np.testing.assert_allclose(b['dip point'], [a['dip point'][0] + shift, a['dip point'][1]])
        for key in ('segments', 'sections', 'velocity models', 'indicator models', 'composition models'):
            assert a.get(key) == b.get(key)
    expected = np.array(before['cross section'], dtype=float); expected[:, 0] += shift
    np.testing.assert_allclose(after['cross section'], expected)
    angle = np.radians(shift)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    vectors = {'old_trench', 'new_trench', 'old_direction', 'new_direction',
               'anchor_global', 'upper_profiles', 'tail_samples'}
    with np.load(original / 'geometry.npz') as a, np.load(moved / 'geometry.npz') as b:
        for key in vectors:
            np.testing.assert_allclose(b[key], a[key] @ rotation.T, atol=1e-8)
        np.testing.assert_array_equal(a['old_profile_points'], b['old_profile_points'])
        for key in ('old_trench_lonlat', 'new_trench_lonlat'):
            expected = a[key].copy(); expected[:, 0] += shift
            np.testing.assert_allclose(b[key], expected)
    for key in ('trench_retreat_m', 'old_anchor_depth_m', 'terminal_segment_length_m'):
        assert report[key] == baseline[key]
    assert report['longitude_shift_deg'] == shift


@pytest.mark.parametrize('options', [
    {'chunk_min_longitude': 30},
    {'chunk_max_longitude': 0},
    {'chunk_min_longitude': float('nan')},
    {'chunk_max_longitude': float('inf')},
    {'longitude_shift_deg': float('nan')},
    {'longitude_shift_deg': float('inf')},
])
def test_invalid_longitude_options_leave_no_case(source_case, tmp_path, options):
    with pytest.raises(ValueError):
        generate_retreat_case(source_case, tmp_path / 'invalid', **options)
    assert not (tmp_path / 'invalid').exists()


def test_shift_requires_domain_to_fit_deep_slab(source_case, tmp_path):
    # The shifted trench itself fits 0..25, but the deep terminal segment does not.
    with pytest.raises(ValueError, match='chunk domain'):
        generate_retreat_case(source_case, tmp_path / 'outside', longitude_shift_deg=5)
    assert not (tmp_path / 'outside').exists()


def test_longitude_override_standard_writer_preserves_parameters(source_case, tmp_path):
    path = source_case / 'case.prm'
    text = '# spherical model: longitude–latitude\n' + path.read_text()
    text = text.replace('Chunk maximum longitude = 25',
                        'Chunk maximum longitude = 25   # eastern boundary')
    text = text.replace('\n', '\r\n')
    path.write_bytes(text.encode('utf-8'))
    target = tmp_path / 'comments'
    generate_retreat_case(source_case, target, chunk_max_longitude=35)
    from gdmate.aspect_tools import parse_parameters_to_dict
    with path.open(encoding='utf-8') as stream:
        expected = parse_parameters_to_dict(stream)
    expected['Geometry model']['Chunk']['Chunk maximum longitude'] = '35'
    with (target / 'case.prm').open(encoding='utf-8') as stream:
        assert parse_parameters_to_dict(stream) == expected
    output = (target / 'case.prm').read_bytes()
    assert b'# eastern boundary' not in output
    assert b'\r\n' not in output


def test_longitude_cli_options(source_case, tmp_path, capsys):
    from hamageolib.research.prescribe_vecloity.trench_retreat import main

    target = tmp_path / 'cli-longitude'
    main([str(source_case), str(target), '--chunk-min-longitude', '-5',
          '--chunk-max-longitude', '35', '--longitude-shift-deg', '5'])
    report = json.loads(capsys.readouterr().out)
    assert report['longitude_shift_deg'] == 5
    assert report['chunk_longitude_bounds_deg'] == [-5, 35]


def test_standard_gdmate_io_interface():
    import gdmate.aspect_tools as io
    import hamageolib.research.prescribe_vecloity.trench_retreat as module

    assert module.parse_parameters_to_dict is io.parse_parameters_to_dict
    assert module.save_parameters_from_dict is io.save_parameters_from_dict


def test_modified_parameters_use_gdmate_writer(source_case, tmp_path, monkeypatch):
    import gdmate.aspect_tools as io
    import hamageolib.research.prescribe_vecloity.trench_retreat as module
    from unittest.mock import Mock

    writer = Mock(wraps=io.save_parameters_from_dict)
    monkeypatch.setattr(module, 'save_parameters_from_dict', writer, raising=False)
    generate_retreat_case(source_case, tmp_path / 'modified', chunk_max_longitude=35)
    writer.assert_called_once()
    assert writer.call_args.args[1]['Geometry model']['Chunk']['Chunk maximum longitude'] == '35'


def test_none_retreat_preserves_original_case(source_case, tmp_path, monkeypatch):
    import hamageolib.research.prescribe_vecloity.trench_retreat as module

    def no_fit(*args, **kwargs):
        pytest.fail('None must not fit a connector')
    monkeypatch.setattr(module, 'root', no_fit)
    target = tmp_path / 'unchanged'
    report = generate_retreat_case(source_case, target, trench_retreat_m=None)
    for name in ('case.prm', 'case.wb'):
        assert (target / name).read_bytes() == (source_case / name).read_bytes()
    world = json.loads((target / 'case.wb').read_text())
    assert len(world['features']) == 1
    assert 'sections' not in world['features'][0]
    assert report['trench_retreat_m'] is None
    assert report['actual_retreat_m'] == 0
    assert report['connector_fit_performed'] is False
    assert report['connector_overlap_length_m'] == 0
    with np.load(target / 'geometry.npz') as geometry:
        np.testing.assert_array_equal(geometry['old_trench'], geometry['new_trench'])
        np.testing.assert_array_equal(geometry['old_direction'], geometry['new_direction'])


def test_none_retreat_with_shift_and_bounds(source_case, tmp_path):
    target = tmp_path / 'shifted-original'
    report = generate_retreat_case(source_case, target, trench_retreat_m=None,
                                   chunk_min_longitude=-5, chunk_max_longitude=35,
                                   longitude_shift_deg=5)
    before = json.loads((source_case / 'case.wb').read_text())
    after = json.loads((target / 'case.wb').read_text())
    assert len(after['features']) == 1
    slab_before, slab_after = before['features'][0], after['features'][0]
    expected = np.array(slab_before['coordinates'], dtype=float); expected[:, 0] += 5
    np.testing.assert_allclose(slab_after['coordinates'], expected)
    assert slab_after['segments'] == slab_before['segments']
    assert slab_after['dip point'] == [slab_before['dip point'][0] + 5, slab_before['dip point'][1]]
    assert report['trench_retreat_m'] is None
    assert report['longitude_shift_deg'] == 5
    with (target / 'case.prm').open() as stream:
        from gdmate.aspect_tools import parse_parameters_to_dict
        chunk = parse_parameters_to_dict(stream)['Geometry model']['Chunk']
    assert float(chunk['Chunk minimum longitude']) == -5
    assert float(chunk['Chunk maximum longitude']) == 35
    with np.load(target / 'geometry.npz') as geometry:
        np.testing.assert_array_equal(geometry['old_trench'], geometry['new_trench'])


def test_none_cli(source_case, tmp_path, capsys):
    from hamageolib.research.prescribe_vecloity.trench_retreat import main

    target = tmp_path / 'cli-none'
    main([str(source_case), str(target), '--trench-retreat-m', 'none'])
    assert json.loads(capsys.readouterr().out)['trench_retreat_m'] is None
    assert (target / 'case.wb').read_bytes() == (source_case / 'case.wb').read_bytes()
