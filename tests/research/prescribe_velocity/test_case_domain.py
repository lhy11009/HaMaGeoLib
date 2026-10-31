"""Independent domain modification and domain-then-retreat workflows."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from gdmate.aspect_tools import parse_parameters_to_dict

from hamageolib.research.prescribe_vecloity.case_domain import modify_case_domain
from hamageolib.research.prescribe_vecloity.trench_retreat import generate_retreat_case


@pytest.fixture
def original(tmp_path):
    fixture = Path(__file__).parents[2] / 'fixtures/research/prescribe_velocity/trench_retreat_regression/source'
    source = tmp_path / 'original'
    shutil.copytree(fixture, source)
    return source


def parameters(case):
    with (case / 'case.prm').open() as stream:
        return parse_parameters_to_dict(stream)


def test_original_domain_clone_does_not_refit(original, tmp_path, monkeypatch):
    import hamageolib.research.prescribe_vecloity.trench_retreat as retreat
    monkeypatch.setattr(retreat, 'root', lambda *a, **k: pytest.fail('No fit allowed'))
    target = tmp_path / 'clone'
    report = modify_case_domain(original, target)
    for name in ('case.prm', 'case.wb'):
        assert (target / name).read_bytes() == (original / name).read_bytes()
    assert report['slab_refit_performed'] is False
    assert report['operation'] == 'modify_case_domain'
    assert not (target / 'output').exists()


def test_modify_original_bounds_and_position(original, tmp_path):
    target = tmp_path / 'domain'
    modify_case_domain(original, target, chunk_min_longitude=-5,
                       chunk_max_longitude=35, longitude_shift_deg=5)
    before, after = parameters(original), parameters(target)
    before['Geometry model']['Chunk'].update({'Chunk minimum longitude': '-5',
                                               'Chunk maximum longitude': '35'})
    assert after == before
    a = json.loads((original / 'case.wb').read_text())
    b = json.loads((target / 'case.wb').read_text())
    assert len(b['features']) == 1
    for key in ('segments', 'temperature models', 'composition models', 'velocity models', 'indicator models'):
        assert a['features'][0][key] == b['features'][0][key]
    expected = np.array(a['features'][0]['coordinates'], dtype=float); expected[:, 0] += 5
    np.testing.assert_allclose(b['features'][0]['coordinates'], expected)


def test_modify_retreated_case_without_new_fit(original, tmp_path, monkeypatch):
    import hamageolib.research.prescribe_vecloity.trench_retreat as retreat
    source, target = tmp_path / 'retreated', tmp_path / 'moved'
    generate_retreat_case(original, source, trench_retreat_m=100e3)
    monkeypatch.setattr(retreat, 'root', lambda *a, **k: pytest.fail('Must not refit a retreated case'))
    modify_case_domain(source, target, chunk_max_longitude=35, longitude_shift_deg=5)
    a = json.loads((source / 'case.wb').read_text())
    b = json.loads((target / 'case.wb').read_text())
    assert len(b['features']) == 2
    for old, new in zip(a['features'], b['features']):
        assert old['segments'] == new['segments']
        assert old.get('sections') == new.get('sections')
        assert old['velocity models'] == new['velocity models']
    theta = np.radians(5)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    with np.load(source / 'geometry.npz') as old, np.load(target / 'geometry.npz') as new:
        np.testing.assert_allclose(new['anchor_global'], old['anchor_global'] @ rotation.T, atol=1e-8)
        np.testing.assert_array_equal(new['old_profile_points'], old['old_profile_points'])
    report = json.loads((target / 'geometry_report.json').read_text())
    assert report['trench_retreat_m'] == 100e3
    assert report['longitude_shift_deg'] == 5


def test_domain_then_retreat_matches_combined_api(original, tmp_path):
    domain, sequential, combined = (tmp_path / name for name in ('domain', 'sequential', 'combined'))
    modify_case_domain(original, domain, chunk_min_longitude=-5,
                       chunk_max_longitude=35, longitude_shift_deg=5)
    generate_retreat_case(domain, sequential, trench_retreat_m=300e3)
    generate_retreat_case(original, combined, trench_retreat_m=300e3,
                          chunk_min_longitude=-5, chunk_max_longitude=35, longitude_shift_deg=5)
    assert (sequential / 'case.prm').read_bytes() == (domain / 'case.prm').read_bytes()
    assert parameters(sequential) == parameters(combined)
    with np.load(sequential / 'geometry.npz') as a, np.load(combined / 'geometry.npz') as b:
        for key in ('new_trench', 'anchor_global', 'upper_profiles'):
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=.02)


@pytest.mark.parametrize('options', [{'chunk_min_longitude': 40},
                                    {'chunk_max_longitude': float('nan')},
                                    {'longitude_shift_deg': float('inf')},
                                    {'longitude_shift_deg': 5}])
def test_invalid_domain_leaves_no_case(original, tmp_path, options):
    with pytest.raises(ValueError):
        modify_case_domain(original, tmp_path / 'invalid', **options)
    assert not (tmp_path / 'invalid').exists()


def test_existing_destination_preserved(original, tmp_path):
    target = tmp_path / 'existing'; target.mkdir()
    (target / 'keep').write_text('keep')
    with pytest.raises(FileExistsError):
        modify_case_domain(original, target)
    assert (target / 'keep').read_text() == 'keep'


def test_retreated_source_requires_diagnostics(original, tmp_path):
    source = tmp_path / 'retreated'
    generate_retreat_case(original, source)
    (source / 'geometry.npz').unlink()
    with pytest.raises(ValueError, match='geometry.npz'):
        modify_case_domain(source, tmp_path / 'unsupported', chunk_max_longitude=35)
    assert not (tmp_path / 'unsupported').exists()


def test_repeated_domain_shifts_accumulate_without_changing_shape(original, tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    modify_case_domain(original, first, chunk_max_longitude=35, longitude_shift_deg=5)
    report = modify_case_domain(first, second, longitude_shift_deg=3)
    assert report['longitude_shift_deg'] == 3  # Relative to this operation's source.
    geometry_report = json.loads((second / 'geometry_report.json').read_text())
    assert geometry_report['longitude_shift_deg'] == 8
    a = json.loads((original / 'case.wb').read_text())['features'][0]
    b = json.loads((second / 'case.wb').read_text())['features'][0]
    assert a['segments'] == b['segments']
    expected = np.array(a['coordinates'], dtype=float); expected[:, 0] += 8
    np.testing.assert_allclose(b['coordinates'], expected)


def test_domain_cli(original, tmp_path, capsys):
    from hamageolib.research.prescribe_vecloity.case_domain import main

    target = tmp_path / 'cli'
    main([str(original), str(target), '--chunk-max-longitude', '35',
          '--longitude-shift-deg', '5'])
    report = json.loads(capsys.readouterr().out)
    assert report['slab_refit_performed'] is False
    assert report['chunk_longitude_bounds_deg'] == [0, 35]
    assert (target / 'case.wb').is_file()
