"""Prepare time-sampled ASPECT file lists; never execute transfers.

Times use the case's output units (normally years). Each family is sampled
independently, anchored at start or its earliest available output.
"""
import argparse
import copy
import json
import math
import shlex
from pathlib import Path
import xml.etree.ElementTree as ET

FAMILIES = ('solution', 'particles', 'topography', 'fastscape')


def _relative(case, path):
    path = path.resolve()
    try:
        relative = path.relative_to(case)
    except ValueError as error:
        raise ValueError(f'Output reference escapes case: {path}') from error
    if not path.is_file():
        raise FileNotFoundError(path)
    name = relative.as_posix()
    if '\n' in name or '\r' in name:
        raise ValueError('Manifest paths cannot contain newlines')
    return name


def _datasets(case, path):
    tree = ET.parse(path)
    rows = []
    for node in tree.findall('.//DataSet'):
        time = float(node.attrib['timestep'])
        if not math.isfinite(time):
            raise ValueError(f'Invalid time in {path}')
        source = path.parent / node.attrib['file']
        rows.append((time, source, copy.deepcopy(node)))
    return rows


def build_manifest(case, interval, families=('solution',), start=None, end=None):
    """Return file list, selection audit and filtered collections.

    Particle and FastScape PVD collections supply their own time mapping.
    Without a FastScape collection, FastScape and topography filenames are
    mapped through statistics' simulation timestep, never visualization index.
    Missing requested families or referenced pieces are errors.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('interval must be finite and positive')
    for value in (start, end):
        if value is not None and not math.isfinite(value):
            raise ValueError('Time bounds must be finite')
    if start is not None and end is not None and start > end:
        raise ValueError('start must not exceed end')
    if not families or set(families) - set(FAMILIES):
        raise ValueError(f'Choose families from {FAMILIES}')
    case = Path(case).resolve()
    files, selections, collections = set(), [], {}
    outputs = sorted(p for p in case.glob('output*') if p.is_dir())
    for family in dict.fromkeys(families):
        found = False
        for output in outputs:
            pvd = output / f'{family}.pvd'
            if not pvd.exists():
                candidates = sorted((output / family).glob('*.pvd'))
                if len(candidates) > 1:
                    raise ValueError(f'Ambiguous {family} collections: {candidates}')
                pvd = candidates[0] if candidates else pvd
            rows = []
            if pvd.exists():
                rows = _datasets(case, pvd)
            elif family in ('topography', 'fastscape'):
                from .file_reader import read_aspect_header_file
                statistics = output / 'statistics'
                if not statistics.exists():
                    continue
                frame = read_aspect_header_file(str(statistics))
                # Latest occurrence wins after a restart.
                frame = frame.drop_duplicates('Time step number', keep='last')
                for _, row in frame.iterrows():
                    step = int(row['Time step number'])
                    name = (f'topography/topography.{step:05d}' if family == 'topography'
                            else f'fastscape/Topography{step:07d}.vtk')
                    source = output / name
                    if source.is_file():
                        rows.append((float(row['Time']), source, None))
            if not rows:
                continue
            found = True
            low = start if start is not None else min(r[0] for r in rows)
            high = end if end is not None else max(r[0] for r in rows)
            rows = [r for r in rows if low <= r[0] <= high]
            if not rows:
                continue
            selected, nodes = set(), []
            for index in range(math.floor((high - low) / interval) + 1):
                target = low + index * interval
                nearest_time = min((r[0] for r in rows), key=lambda t: (abs(t-target), -t))
                # Include every partition recorded at the chosen physical time.
                for time, source, node in rows:
                    if time != nearest_time or source in selected:
                        continue
                    selected.add(source)
                    relative = _relative(case, source)
                    files.add(relative)
                    if source.suffix in ('.pvtu', '.pvtp'):
                        for piece in ET.parse(source).findall('.//Piece'):
                            files.add(_relative(case, source.parent / piece.attrib['Source']))
                    selections.append(dict(family=family, target_time=target,
                                           actual_time=time, offset=time-target, file=relative))
                    if node is not None:
                        nodes.append(node)
            if nodes:
                root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
                collection = ET.SubElement(root, 'Collection')
                collection.extend(nodes)
                collections[pvd.relative_to(case).as_posix()] = ET.tostring(root, encoding='unicode')
        if not found:
            raise FileNotFoundError(f'No timed outputs found for requested family {family}')
    if not selections:
        raise ValueError('No snapshots within requested time bounds')
    for output in outputs:
        for name in ('statistics', 'log.txt', 'original.prm', 'parameters.prm'):
            path = output / name
            if path.is_file():
                files.add(_relative(case, path))
    for path in case.glob('*.prm'):
        files.add(_relative(case, path))
    return dict(files=sorted(files), selections=selections, collections=collections,
                interval=interval, start=start, end=end, case=str(case))


def write_manifest(manifest, destination):
    """Write rsync list, JSON audit and collection overlay outside the case."""
    destination = Path(destination).resolve()
    case = Path(manifest['case'])
    if destination == case or case in destination.parents:
        raise ValueError('Write manifest outside the source case')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'transfer_files.txt').write_text(''.join(p + '\n' for p in manifest['files']))
    (destination / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for relative, xml in manifest['collections'].items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml)



def transfer_commands(source, target, manifest_directory):
    """Render commands for the receiving machine, using its local manifest copy."""
    manifest = Path(manifest_directory)
    quote = shlex.quote
    return (
        f"mkdir -p -- {quote(target)}\n"
        f"rsync -av --files-from={quote(str(manifest / 'transfer_files.txt'))} "
        f"-- {quote(source.rstrip('/') + '/')} {quote(target.rstrip('/') + '/')}\n"
        f"rsync -av --include='*/' --include='*.pvd' --exclude='*' -- "
        f"{quote(str(manifest) + '/')} {quote(target.rstrip('/') + '/')}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--interval', required=True, type=float)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=['solution'])
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    parser.add_argument('--destination', required=True, help='New manifest directory outside case')
    parser.add_argument('--source', help='rsync source, e.g. SSH_ALIAS:/case')
    parser.add_argument('--target', help='Destination case directory on receiver')
    parser.add_argument('--receiver-manifest', help='Manifest copy directory on receiver')
    args = parser.parse_args()
    if any((args.source, args.target, args.receiver_manifest)) and not all((args.source, args.target, args.receiver_manifest)):
        parser.error('--source, --target and --receiver-manifest must be supplied together')
    result = build_manifest(args.case, args.interval, args.families, args.start, args.end)
    write_manifest(result, args.destination)
    if args.source:
        commands = transfer_commands(args.source, args.target, args.receiver_manifest)
        (Path(args.destination) / 'transfer.sh').write_text(commands)
        print(commands)
    print(f"Selected {len(result['files'])} files; manifest: {args.destination}")


if __name__ == '__main__':
    main()
