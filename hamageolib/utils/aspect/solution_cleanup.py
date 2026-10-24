"""Plan time-based ASPECT snapshot removal without deleting solution files."""

from bisect import bisect_left
from fractions import Fraction
import math
from pathlib import Path
import re
import shlex
import xml.etree.ElementTree as ET

from hamageolib.utils.case_aspect import CaseASPECT


def write_solution_cleanup_script(case: CaseASPECT, time_interval_years, script_path):
    """Write explicit ``rm --`` commands for unselected, complete snapshots.

    Match log records to files currently in ``case.output_dir / 'solution'``.
    Keep the nearest complete, recorded snapshot to every target from zero
    through the last available time, spaced by ``time_interval_years``.
    Ties favor the earlier time, then the smaller snapshot number. A snapshot
    selected by several targets is kept once; the final snapshot is not
    automatically retained. Seconds are converted using 365.25 days/year.

    Incomplete, unrecorded, ambiguously timed, and orphan files are left alone.
    Shared VTU pieces referenced by preserved snapshots are also left alone.
    Only files inside the solution directory are eligible. VTU contents are
    not validated. The script uses absolute, shell-quoted filenames, contains
    no wildcards, and is never executed here. An existing script is not
    overwritten. The script's parent directory must already exist.

    Return a dict with ``kept_snapshots``, ``removed_snapshots`` (sorted
    numbers), ``files_to_remove`` (sorted Paths), and ``script_path`` (Path).
    ``graphical_outputs`` and the CaseASPECT snapshot attributes are unchanged.
    """
    interval = float(time_interval_years)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("time_interval_years must be finite and positive")
    interval = Fraction(str(interval))
    times_by_number = {}
    for record in case.graphical_outputs:
        if record.timestep is None:
            continue
        value = float(record.timestep.time)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Snapshot times must be finite and nonnegative")
        time = Fraction(str(value))
        unit = record.timestep.time_unit.lower()
        if unit in ("s", "second", "seconds"):
            time /= 31557600
        elif unit not in ("yr", "yrs", "year", "years"):
            raise ValueError(f"Unsupported log time unit: {unit!r}")
        times_by_number.setdefault(record.solution_number, set()).add(time)

    solution = (case.output_dir / "solution").resolve()
    complete = {}
    references = {}
    for pvtu in solution.glob("solution-*.pvtu"):
        match = re.fullmatch(r"solution-(\d+)\.pvtu", pvtu.name)
        if match is None or pvtu.is_symlink():
            continue
        try:
            root = ET.parse(pvtu).getroot()
        except (OSError, ET.ParseError):
            references[pvtu] = set(solution.glob(f"{pvtu.stem}.*.vtu"))
            continue
        sources = [piece.get("Source", "")
                   for piece in root.findall("./PUnstructuredGrid/Piece")]
        paths = [(pvtu.parent / source).resolve() for source in sources if source]
        references[pvtu] = set(paths)
        if (root.tag == "VTKFile" and root.get("type") == "PUnstructuredGrid"
                and sources and len(paths) == len(sources)
                and all(path.is_relative_to(solution) and path.suffix == ".vtu"
                        and path.is_file() for path in paths)):
            complete[pvtu] = int(match.group(1))

    candidates = sorted({(next(iter(times_by_number[number])), number)
                         for number in complete.values()
                         if len(times_by_number.get(number, ())) == 1})
    kept = set()
    if candidates:
        times = [time for time, _ in candidates]
        last_target = times[-1] // interval
        # Only grid points adjacent to a snapshot time can introduce a new
        # nearest snapshot. This avoids iterating billions of tiny intervals.
        target_indices = {0}
        for time in times:
            index = time // interval
            target_indices.update(k for k in (index, index + 1)
                                  if 0 <= k <= last_target)
        for index in target_indices:
            target = index * interval
            position = bisect_left(times, target)
            indices = {max(0, position - 1), min(position, len(times) - 1)}
            neighbors = [candidates[bisect_left(times, times[i])] for i in indices]
            _, number = min(neighbors, key=lambda item: (
                abs(item[0] - target), item[0], item[1]))
            kept.add(number)

    eligible = {number for _, number in candidates}
    removed = eligible - kept
    discarded_pvtu = {pvtu for pvtu, number in complete.items() if number in removed}
    protected = set().union(*(paths for pvtu, paths in references.items()
                              if pvtu not in discarded_pvtu))
    files = set(discarded_pvtu)
    for pvtu in discarded_pvtu:
        files.update(references[pvtu] - protected)
    files = sorted(files)
    script_path = Path(script_path).absolute()
    # Exclusive creation also prevents accidentally overwriting an input file.
    script_path.unlink(missing_ok=True)
    with script_path.open("x", encoding="utf-8") as stream:
        stream.write("#!/bin/sh\nset -eu\n")
        stream.write("# Review before running; generated snapshot removal plan.\n")
        stream.write(f"# Kept snapshots: {sorted(kept)}\n")
        stream.write(f"# Removed snapshots: {sorted(removed)}\n")
        for path in files:
            stream.write(f"rm -- {shlex.quote(str(path))}\n")
    print("Write script %s" % script_path)
    return {"kept_snapshots": sorted(kept), "removed_snapshots": sorted(removed),
            "files_to_remove": files, "script_path": script_path}
