"""Modify the domain/longitude of an existing prescribed-velocity case.

This operation copies the existing slab definitions without fitting them.
Original four-segment cases can be sampled directly. Retreated or sectioned
cases require their corresponding geometry.npz diagnostics for domain checks.
"""
import copy
from io import BytesIO, StringIO
import json
import math
from pathlib import Path

import numpy as np
from gdmate.aspect_tools import parse_parameters_to_dict, save_parameters_from_dict

from .trench_retreat import _check_slab_domain, _original_geometry, _shift_longitude


_GEOMETRY_KEYS = {
    'old_trench', 'new_trench', 'old_direction', 'new_direction',
    'old_profile_points', 'anchor_global', 'upper_profiles', 'tail_samples',
    'old_trench_lonlat', 'new_trench_lonlat',
}


def modify_case_domain(source_case_dir, destination_case_dir, *,
                       chunk_min_longitude=None, chunk_max_longitude=None,
                       longitude_shift_deg=0.0, samples_per_segment=20):
    """Copy a case with optional longitude bounds/rotation and no slab refit.

    None bounds inherit the source. A shift in degrees rotates all slab
    features, dip points, cross section and global geometry diagnostics; it
    does not change segment lengths, angles, sections, materials or velocity.
    Radii, latitude bounds and mesh repetitions are inherited. Changed bounds
    are saved through gdmate's standard parameter writer (comments/formatting
    are not preserved); otherwise case.prm bytes are copied exactly. With no
    shift, case.wb bytes are also copied exactly. Longitudes are not wrapped.

    Source geometry.npz must correspond to source case.wb. If absent, only
    an original four-segment single-feature case is supported. Domain checks
    use sampled geometry, not full volume containment. Creates inputs,
    geometry diagnostics, geometry_report.json and domain_report.json only.
    Refuses an existing destination; never runs ASPECT or copies its outputs.
    Returns the domain-operation report, with shift measured from the source.
    """
    source = Path(source_case_dir).resolve()
    destination = Path(destination_case_dir).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    if isinstance(samples_per_segment, bool) or not isinstance(samples_per_segment, int) or samples_per_segment < 4:
        raise ValueError('samples_per_segment must be an integer >= 4')
    prm_bytes = (source / 'case.prm').read_bytes()
    parameters = parse_parameters_to_dict(StringIO(prm_bytes.decode('utf-8'), newline=None))
    if parameters.get('Dimension') != '3' or parameters.get('Geometry model', {}).get('Model name') != 'chunk':
        raise ValueError('Source must use a 3-D chunk geometry')
    chunk = parameters['Geometry model']['Chunk']
    inherited_min = float(chunk['Chunk minimum longitude'])
    inherited_max = float(chunk['Chunk maximum longitude'])
    minimum = inherited_min if chunk_min_longitude is None else float(chunk_min_longitude)
    maximum = inherited_max if chunk_max_longitude is None else float(chunk_max_longitude)
    shift = float(longitude_shift_deg)
    radius = float(chunk['Chunk outer radius'])
    if not all(math.isfinite(v) for v in (minimum, maximum, shift, radius)) or minimum >= maximum or radius <= 0:
        raise ValueError('Domain bounds/shift must be finite and ordered, with a positive radius')
    wb_bytes = (source / 'case.wb').read_bytes()
    world = json.loads(wb_bytes)
    if (world.get('coordinate system', {}).get('model') != 'spherical'
            or world['coordinate system'].get('depth method') != 'begin segment'):
        raise ValueError('Source must use spherical begin segment coordinates')
    features = world.get('features', [])
    if not features or any(f.get('model') != 'subducting plate' for f in features):
        raise ValueError('Supported cases contain subducting plate features only')
    for feature in features:
        coordinates = np.asarray(feature.get('coordinates', []), dtype=float)
        dip = np.asarray(feature.get('dip point', []), dtype=float)
        if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3
                or not np.isfinite(coordinates).all() or dip.shape != (2,) or not np.isfinite(dip).all()):
            raise ValueError('Each slab requires finite trench coordinates and a dip point')

    geometry_path = source / 'geometry.npz'
    if geometry_path.is_file():
        with np.load(geometry_path, allow_pickle=False) as data:
            if not _GEOMETRY_KEYS.issubset(data.files):
                raise ValueError('Source geometry.npz is missing required diagnostic arrays')
            geometry = {key: data[key].copy() for key in data.files}
        report_path = source / 'geometry_report.json'
        geometry_report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    else:
        if len(features) != 1 or features[0].get('sections') or len(features[0].get('segments', [])) != 4:
            raise ValueError('Retreated or sectioned cases require matching source geometry.npz')
        _, geometry, geometry_report = _original_geometry(world, radius, samples_per_segment)
    world = copy.deepcopy(world)
    changes = {}
    if minimum != inherited_min:
        changes['Chunk minimum longitude'] = minimum
    if maximum != inherited_max:
        changes['Chunk maximum longitude'] = maximum
    if shift:
        _shift_longitude(world, geometry, shift)
        wb_bytes = (json.dumps(world, indent=2) + '\n').encode('utf-8')
    if changes or shift:
        _check_slab_domain(geometry, radius, minimum, maximum,
                           float(chunk['Chunk minimum latitude']), float(chunk['Chunk maximum latitude']))
    if changes:
        chunk.update({key: format(value, '.15g') for key, value in changes.items()})
        stream = StringIO()
        save_parameters_from_dict(stream, parameters)
        prm_bytes = stream.getvalue().encode('utf-8')
    report = dict(operation='modify_case_domain', slab_refit_performed=False,
                  source_case_dir=str(source), destination_case_dir=str(destination),
                  longitude_shift_deg=shift, chunk_longitude_bounds_deg=[minimum, maximum])
    geometry_report.setdefault('source_case_dir', str(source))
    geometry_report.update(
        destination_case_dir=str(destination), domain_source_case_dir=str(source),
        longitude_shift_deg=float(geometry_report.get('longitude_shift_deg', 0.0)) + shift,
        chunk_longitude_bounds_deg=[minimum, maximum])
    archive = BytesIO()
    np.savez(archive, **geometry)
    geometry_text = json.dumps(geometry_report, indent=2) + '\n'
    report_text = json.dumps(report, indent=2) + '\n'
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'case.prm').write_bytes(prm_bytes)
    (destination / 'case.wb').write_bytes(wb_bytes)
    (destination / 'geometry.npz').write_bytes(archive.getvalue())
    (destination / 'geometry_report.json').write_text(geometry_text)
    (destination / 'domain_report.json').write_text(report_text)
    return report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_case_dir')
    parser.add_argument('destination_case_dir')
    parser.add_argument('--chunk-min-longitude', type=float)
    parser.add_argument('--chunk-max-longitude', type=float)
    parser.add_argument('--longitude-shift-deg', type=float, default=0.0)
    args = parser.parse_args(argv)
    print(json.dumps(modify_case_domain(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
