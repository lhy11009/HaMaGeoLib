"""Generate spherical trench-retreat cases with an anchored terminal segment.

This implements the two-feature construction used for the prescribed-velocity
_3 case. It supports the east-dipping four-segment _2 profile, spherical
``begin segment`` depths, a constant thickness and equal transition arc lengths.
Fits and probe coordinates are geometric diagnostics; they do not replace
WorldBuilder field queries or an ASPECT solve for a new retreat distance.
"""
import copy
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import root, brentq
from scipy.interpolate import CubicSpline
from hamageolib.utils.aspect.io import parse_parameters_to_dict
from ._trench_bezier import sample_trench


def _construct(world, radius, retreat, samples_per_segment, overlap_m):
    """Fit the connecting slab and return WorldBuilder data, samples and diagnostics."""
    world = copy.deepcopy(world)
    reference = world['features'][0]
    rows = sample_trench(reference['coordinates'], samples_per_segment)
    R = radius
    delta = retreat / R
    transition_length = reference['segments'][0]['length']
    thickness = reference['segments'][0]['thickness'][0]
    lon, lat = rows[:, :2].T
    radial = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    def directions(lon, lat, derivative):
        """Compute the local east-dipping normal convention used by this WorldBuilder profile."""
        e = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)])
        n = np.column_stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
        v = -derivative[:, 1, None] * np.cos(lat)[:, None] * e + derivative[:, 0, None] * n
        v *= np.where((v * e).sum(axis=1) > 0, 1, -1)[:, None]
        return v / np.linalg.norm(v, axis=1)[:, None]
    direction = directions(lon, lat, rows[:, 2:4])

    def offset_curve(rotations):
        """Move trench samples along spherical arcs, adjusting end bearings smoothly."""
        weights0 = np.exp(-np.arange(len(rows)) / 12.0)
        weights1 = weights0[::-1]
        angles = rotations[0] * weights0 + rotations[1] * weights1
        side = np.cross(radial, direction)
        offsets = np.cos(angles)[:, None] * direction + np.sin(angles)[:, None] * side
        rr = np.cos(delta) * radial - np.sin(delta) * offsets
        ll = np.unwrap(np.arctan2(rr[:, 1], rr[:, 0]))
        pp = np.arcsin(rr[:, 2])
        xy = np.column_stack([ll, pp])
        dif = np.diff(xy, axis=0)
        dif /= np.linalg.norm(dif, axis=1)[:, None]
        deriv = np.vstack([dif[0], dif[:-1] + dif[1:], dif[-1]])
        dd = directions(ll, pp, deriv)
        return (rr, ll, pp, dd)

    def advance(p, length, a, b):
        """Integrate a circular arc with linearly varying absolute dip."""
        if abs(b - a) < 1e-10:
            return p + length * np.array([np.cos(a), -np.sin(a)])
        return p + length * np.array([(np.sin(b) - np.sin(a)) / (b - a), (np.cos(b) - np.cos(a)) / (b - a)])
    points = [np.array([0.0, R])]
    abs_angles = []
    for segment in reference['segments']:
        p = points[-1]
        phi = np.arctan2(p[0], p[1])
        angle = np.radians(segment['angle'])
        a, b = (angle[0] + phi, angle[-1] + phi)
        points.append(advance(p, segment['length'], a, b))
        abs_angles.append((a, b))
    anchor = points[3]
    tail_angle = abs_angles[3][0]
    anchor_global = anchor[0] * direction + anchor[1] * radial
    v_tail = np.cos(tail_angle) * direction - np.sin(tail_angle) * radial

    def end_plane_errors(v):
        """Require the retreated end planes to contain the original terminal end caps."""
        rr, ll, pp, dd = offset_curve(v)
        return [np.dot(np.cross(rr[i], dd[i]), anchor_global[i]) / 100000.0 for i in [0, -1]]
    end_fit = root(end_plane_errors, [0.0, 0.0])
    if not end_fit.success or np.linalg.norm(end_plane_errors(end_fit.x)) > 1e-06:
        raise ValueError('Could not fit trench end bearings to the fixed anchor')
    new_radial, new_lon, new_lat, new_direction = offset_curve(end_fit.x)
    anchor_interp = CubicSpline(np.arange(len(rows)), anchor_global, axis=0)
    heading_interp = CubicSpline(np.arange(len(rows)), v_tail, axis=0)
    anchor_targets = []
    heading_targets = []
    for i in range(len(rows)):
        plane = np.cross(new_radial[i], new_direction[i])
        f = lambda t: float(np.dot(plane, anchor_interp(t)))
        values = anchor_global @ plane
        brackets = [(j, j + 1) for j in range(len(rows) - 1) if values[j] * values[j + 1] <= 0]
        if i == 0:
            t = 0.0
        elif i == len(rows) - 1:
            t = float(len(rows) - 1)
        else:
            if not brackets:
                raise ValueError(f'No anchor intersection at trench section {i}')
            bracket = min(brackets, key=lambda v: abs((v[0] + v[1]) / 2 - i))
            t = brentq(f, *bracket, xtol=1e-12)
        anchor_targets.append(anchor_interp(t))
        heading_targets.append(heading_interp(t))
    anchor_targets = np.array(anchor_targets)
    heading_targets = np.array(heading_targets)
    sections = []
    endpoint_errors = []
    profiles = []
    metrics = []
    for i in range(len(rows)):
        target = np.array([np.dot(anchor_targets[i], new_direction[i]), np.dot(anchor_targets[i], new_radial[i])])
        end_angle = np.arctan2(-np.dot(heading_targets[i], new_radial[i]), np.dot(heading_targets[i], new_direction[i]))

        def build(params):
            """Evaluate the two bends and intervening straight segment for a trial fit."""
            bend, length = params
            a0 = np.array([0.0, R])
            a1 = advance(a0, transition_length, 0, bend)
            a2 = advance(a1, length, bend, bend)
            a3 = advance(a2, transition_length, bend, end_angle)
            return [a0, a1, a2, a3]
        fit = root(lambda v: (build(v)[-1] - target) / 100000.0, [np.radians(40), 500000.0])
        if not fit.success:
            raise ValueError(f'Connector fit failed at section {i}: {fit.message}')
        bend, length = fit.x
        if not (0 < bend < np.pi / 2 and length > 0) or np.linalg.norm(build(fit.x)[-1] - target) > 0.001:
            raise ValueError(f'Nonphysical connector at section {i}')
        pts = build(fit.x)
        angles = [(0, bend), (bend, bend), (bend, end_angle)]
        segs = []
        sample = []
        for j, (a, b) in enumerate(angles):
            p = pts[j]
            phi = np.arctan2(p[0], p[1])
            L = [transition_length, length, transition_length][j]
            segs.append({'length': float(L), 'thickness': [thickness], 'angle': np.degrees([a - phi, b - phi]).tolist()})
            for fraction in np.linspace(0, 1, 41):
                q = advance(p, L * fraction, a, a + (b - a) * fraction)
                sample.append(q[0] * new_direction[i] + q[1] * new_radial[i])
        phi = np.arctan2(pts[-1][0], pts[-1][1])
        segs.append({'length': overlap_m, 'thickness': [thickness], 'angle': [float(np.degrees(end_angle - phi))]})
        sections.append({'coordinate': i, 'segments': segs})
        profiles.append(sample)
        end_global = pts[-1][0] * new_direction[i] + pts[-1][1] * new_radial[i]
        endpoint_errors.append(float(np.linalg.norm(end_global - anchor_targets[i])))
        metrics.append([float(np.degrees(bend)), float(length)])
    upper = copy.deepcopy(reference)
    upper['name'] = 'Retreated upper slab'
    upper['coordinates'] = np.degrees(np.column_stack([new_lon, new_lat])).tolist()
    upper['segments'] = copy.deepcopy(sections[0]['segments'])
    upper['sections'] = sections
    # Evaluate the original deep geometry last; suppress only its upper thickness.
    fixed = copy.deepcopy(reference)
    fixed['name'] = 'Anchored original terminal segment'
    for segment in fixed['segments'][:3]:
        segment['thickness'] = [0]
    # Compensate spherical dip angles when splitting the original transition.
    # This keeps both endpoints of the final horizontal segment fixed.
    p2 = points[2]
    phi2 = np.arctan2(p2[0], p2[1])
    aa, bb = abs_angles[2]
    cut_angle = aa + (bb - aa) * (1 - overlap_m / transition_length)
    cut = advance(p2, transition_length - overlap_m, aa, cut_angle)
    cut_phi = np.arctan2(cut[0], cut[1])
    fixed['segments'][2] = {'length': transition_length - overlap_m, 'thickness': [0], 'angle': np.degrees([aa - phi2, cut_angle - phi2]).tolist()}
    fixed['segments'].insert(3, {'length': overlap_m, 'thickness': [thickness], 'angle': np.degrees([cut_angle - cut_phi, bb - cut_phi]).tolist()})
    world['features'] = [upper, fixed]
    tail_samples = []
    for i in range(5, len(rows) - 5, 5):
        for s in np.array([0.05, 0.25, 0.5, 0.75, 0.95]) * reference['segments'][-1]['length']:
            q = advance(anchor, s, tail_angle, tail_angle)
            for depth in np.array([5, 25, 50, 75, 90]) / 95 * thickness:
                offset = depth * np.array([-np.sin(tail_angle), -np.cos(tail_angle)])
                qq = q + offset
                tail_samples.append(qq[0] * direction[i] + qq[1] * radial[i])
    geometry = dict(old_trench=R * radial, new_trench=R * new_radial, old_direction=direction, new_direction=new_direction, old_profile_points=np.array(points), anchor_global=anchor_global, upper_profiles=np.array(profiles), tail_samples=tail_samples, old_trench_lonlat=np.degrees(rows[:, :2]), new_trench_lonlat=upper['coordinates'])
    report = {'trench_retreat_m': retreat, 'end_bearing_adjustments_deg': np.degrees(end_fit.x).tolist(), 'number_trench_samples': len(rows), 'old_anchor_depth_m': float(R - np.linalg.norm(anchor)), 'old_distal_endpoint_depth_m': float(R - np.linalg.norm(points[4])), 'terminal_segment_length_m': reference['segments'][-1]['length'], 'connector_overlap_length_m': overlap_m, 'anchored_transition_overlap_length_m': overlap_m, 'connector_endpoint_cross_track_error_max_m': max(endpoint_errors), 'worst_connector_index': int(np.argmax(endpoint_errors)), 'endpoint_errors_ends_m': [endpoint_errors[0], endpoint_errors[-1]], 'connector_endpoint_cross_track_error_median_m': float(np.median(endpoint_errors)), 'middle_segment_length_range_m': [float(np.min(np.array(metrics)[:, 1])), float(np.max(np.array(metrics)[:, 1]))], 'bend_absolute_angle_range_deg': [float(np.min(np.array(metrics)[:, 0])), float(np.max(np.array(metrics)[:, 0]))]}
    return (world, geometry, report)


def generate_retreat_case(source_case_dir, destination_case_dir, trench_retreat_m=100000.0, *, samples_per_segment=20, overlap_m=20000.0):
    """Create a new case; never overwrite a destination or run ASPECT.

    ``trench_retreat_m`` is a finite positive distance in metres. The source
    must be an unmodified-style _2 slab, not a previously retreated two-feature
    case. The source case.prm is copied byte-for-byte. Writes case.wb,
    geometry.npz (including sampled profiles/probes), and geometry_report.json.
    The final horizontal segment is retained through the anchored feature;
    both features overlap by ``overlap_m``. Arbitrary retreat distances may
    have no valid fit or leave the chunk and raise ValueError before writing.
    The returned dict contains fit diagnostics and the output directory.
    """
    source = Path(source_case_dir).resolve()
    destination = Path(destination_case_dir).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    retreat = float(trench_retreat_m)
    overlap = float(overlap_m)
    if not math.isfinite(retreat) or retreat <= 0:
        raise ValueError('trench_retreat_m must be finite and positive')
    if not math.isfinite(overlap) or overlap <= 0:
        raise ValueError('overlap_m must be finite and positive')
    if isinstance(samples_per_segment, bool) or not isinstance(samples_per_segment, int) or samples_per_segment < 4:
        raise ValueError('samples_per_segment must be an integer >= 4')
    prm_bytes = (source / 'case.prm').read_bytes()
    parameters = parse_parameters_to_dict(StringIO(prm_bytes.decode('utf-8')))
    world = json.loads((source / 'case.wb').read_text())
    coordinates = world.get('coordinate system', {})
    if coordinates.get('model') != 'spherical' or coordinates.get('depth method') != 'begin segment':
        raise ValueError('Only spherical coordinates with begin segment depths are supported')
    if parameters.get('Dimension') != '3' or parameters.get('Geometry model', {}).get('Model name') != 'chunk':
        raise ValueError('Source must use a 3-D chunk geometry')
    chunk = parameters['Geometry model']['Chunk']
    radius = float(chunk['Chunk outer radius'])
    if not math.isfinite(radius) or radius <= 0 or retreat >= math.pi * radius / 2:
        raise ValueError('Retreat must be less than a quarter circumference of a positive-radius chunk')
    features = world.get('features', [])
    if len(features) != 1 or features[0].get('model') != 'subducting plate' or features[0].get('sections'):
        raise ValueError('Source must contain one subducting plate without section overrides')
    slab = features[0]
    segments = slab.get('segments', [])
    if len(segments) != 4:
        raise ValueError('Source must have four slab segments')
    angles = [np.asarray(s.get('angle', []), dtype=float) for s in segments]
    if any((len(a) not in (1, 2) or not np.isfinite(a).all() for a in angles)) or not np.isclose(angles[0][0], 0) or (not np.isclose(angles[0][-1], angles[1][0])) or (not np.isclose(angles[1][0], angles[1][-1])) or (not np.isclose(angles[2][0], angles[1][0])) or (not np.isclose(angles[2][-1], 0)) or (not np.allclose(angles[3], 0)):
        raise ValueError('Expected bend, constant-dip, unbend, horizontal terminal profile')
    lengths = [float(s['length']) for s in segments]
    if not all((math.isfinite(v) and v > 0 for v in lengths)):
        raise ValueError('Slab lengths must be finite and positive')
    if not math.isclose(lengths[0], lengths[2]) or overlap >= min(lengths[2:]):
        raise ValueError('Transition lengths must match; overlap must be shorter than transition and tail')
    thickness = segments[0].get('thickness', [])
    if len(thickness) != 1 or not math.isfinite(float(thickness[0])) or thickness[0] <= 0 or any((s.get('thickness') != thickness or s.get('top truncation') for s in segments)):
        raise ValueError('Source must have a constant positive thickness and no top truncation')
    points = np.asarray(slab['coordinates'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or (not np.isfinite(points).all()):
        raise ValueError('Trench must have at least three finite lon/lat coordinates')
    if np.any(np.abs(points[:, 1]) >= 89) or np.ptp(points[:, 0]) >= 180:
        raise ValueError('Polar and dateline-crossing trenches are not supported')
    if slab['dip point'][0] <= points[:, 0].max():
        raise ValueError('This construction supports east-dipping slabs only')
    result, geometry, report = _construct(world, radius, retreat, samples_per_segment, overlap)
    new_points = np.asarray(result['features'][0]['coordinates'])
    lon_min, lon_max = (float(chunk[k]) for k in ('Chunk minimum longitude', 'Chunk maximum longitude'))
    lat_min, lat_max = (float(chunk[k]) for k in ('Chunk minimum latitude', 'Chunk maximum latitude'))
    if np.any(new_points[:, 0] < lon_min) or np.any(new_points[:, 0] > lon_max) or np.any(new_points[:, 1] < lat_min) or np.any(new_points[:, 1] > lat_max):
        raise ValueError('Retreated trench leaves the chunk domain')
    report.update(source_case_dir=str(source), destination_case_dir=str(destination), samples_per_segment=samples_per_segment)
    wb_text = json.dumps(result, indent=2) + '\n'
    geometry_bytes = BytesIO()
    np.savez(geometry_bytes, **geometry)
    report_text = json.dumps(report, indent=2) + '\n'
    # Write only after all geometry checks and serialization have succeeded.
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'case.prm').write_bytes(prm_bytes)
    (destination / 'case.wb').write_text(wb_text)
    (destination / 'geometry.npz').write_bytes(geometry_bytes.getvalue())
    (destination / 'geometry_report.json').write_text(report_text)
    return report


def main(argv=None):
    """Command-line entry point: generate inputs only, without running ASPECT."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_case_dir')
    parser.add_argument('destination_case_dir')
    parser.add_argument('--trench-retreat-m', type=float, default=100000.0)
    parser.add_argument('--samples-per-segment', type=int, default=20)
    parser.add_argument('--overlap-m', type=float, default=20000.0)
    args = parser.parse_args(argv)
    report = generate_retreat_case(**vars(args))
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
