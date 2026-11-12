"""Select a preserved checkpoint from a log and prepare a portable copy script.

No checkpoint payloads or metadata files are read locally. Times have the
precision printed in the log, rather than the higher precision of metadata.
"""

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
import os
import shlex
from typing import Optional

from gdmate.aspect_tools import AspectLogParser
from gdmate.aspect_tools.log import AspectTimestep

from .io import parse_parameters_to_dict


YEAR_IN_SECONDS = 31556952.0  # ASPECT: 365.2425 days/year.
_SNAPSHOT = re.compile(r"^\s*\*\*\* Snapshot (?P<path>.+/restart/(?P<slot>\d+)/?) created!\s*$")
_VERSION = re.compile(r"^\s*--\s+\.\s+version\s+")


@dataclass(frozen=True)
class CheckpointSnapshot:
    """One reported checkpoint; timestep is None if its saved time is unknown."""

    path: str
    slot: int
    line_number: int
    timestep: Optional[AspectTimestep] = None


class CheckpointLogParser(AspectLogParser):
    """Extend gdmate's log parser with checkpoint creation records.

    ASPECT advances time before checkpointing. The next timestep header
    describes the saved state; the preceding header's dt must not be added
    to estimate it. A snapshot without a following header in the same run
    is retained with timestep=None. Two streaming readers avoid loading the
    entire log into memory, and reuse gdmate's timestep parsing.
    """

    def iter_checkpoint_outputs(self):
        headers = iter(self.iter_timesteps())
        following = next(headers, None)
        pending = []
        with self.path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if following is not None and line_number == following.line_number:
                    for snapshot in pending:
                        yield replace(snapshot, timestep=following)
                    pending = []
                    following = next(headers, None)
                elif _VERSION.match(line):
                    yield from pending
                    pending = []
                if line.lstrip().startswith("*** Snapshot"):
                    match = _SNAPSHOT.fullmatch(line)
                    if match is None:
                        raise ValueError(f"{self.path}:{line_number}: malformed checkpoint record")
                    pending.append(CheckpointSnapshot(
                        path=match["path"], slot=int(match["slot"]), line_number=line_number,
                    ))
        yield from pending

    def parse_checkpoint_outputs(self):
        """Collect reported snapshots, including unresolved ones."""
        return list(self.iter_checkpoint_outputs())


def _seconds(value, unit):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Checkpoint times must be finite and nonnegative")
    if unit.lower() in ("yr", "yrs", "year", "years"):
        return value * YEAR_IN_SECONDS
    if unit.lower() in ("s", "second", "seconds"):
        return value
    raise ValueError(f"Unsupported checkpoint time unit: {unit!r}")


def select_checkpoint_from_log(log_path, target_time, *, time_unit="years"):
    """Select the nearest reported checkpoint, assuming none were removed.

    Reused slot IDs or unresolved times are rejected: the preserved-history
    assumption cannot identify all candidates reliably in those cases.
    Ties follow ASPECT: prefer the last reported slot, then the smaller ID.
    Selection is limited to the precision of printed timestep times.
    """
    target = _seconds(target_time, time_unit)
    snapshots = CheckpointLogParser(log_path).parse_checkpoint_outputs()
    if not snapshots:
        raise ValueError("No checkpoint creation records found in the first-stage log")
    slots = [snapshot.slot for snapshot in snapshots]
    if len(set(slots)) != len(slots):
        raise ValueError("Checkpoint slot reused; cannot assume all logged checkpoints are preserved")
    unresolved = [snapshot.path for snapshot in snapshots if snapshot.timestep is None]
    if unresolved:
        raise ValueError(f"Checkpoint saved times are unresolved (no following timestep header): {unresolved}")
    last_slot = snapshots[-1].slot
    return min(snapshots, key=lambda snapshot: (
        abs(_seconds(snapshot.timestep.time, snapshot.timestep.time_unit) - target),
        snapshot.slot != last_slot, snapshot.slot,
    ))


def _relative_directory(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError("Checkpoint copy scripts require case-relative output directories")
    return path


def write_restart_copy_script(case_dir, second_stage_parameters, tag):
    """Write copy_restart_{tag}.sh using only case.prm and its output/log.txt.

    The second-stage dictionary supplies Output directory and Checkpointing /
    Resume time. All logged checkpoint slots are assumed preserved on the
    server. Read the first-stage checkpoint interval and retention settings;
    actual log messages (including forced checkpoints) determine candidates.

    The script is placed beside case.prm and uses paths relative to itself,
    so the case folder can be moved to the server. It rsyncs only the selected
    slot and writes an independent last_good_checkpoint.txt after success.
    It is never executed here and an existing script is not overwritten.
    """
    if not isinstance(tag, str) or not tag or "/" in tag or "\\" in tag:
        raise ValueError("tag must be a nonempty filename tag")
    case_dir = Path(case_dir)
    with (case_dir / "case.prm").open(encoding="utf-8") as stream:
        first_stage = parse_parameters_to_dict(stream)
    settings = first_stage.get("Checkpointing", {})
    interval = int(settings.get("Steps between checkpoint", 0))
    retention = int(settings.get("Number of checkpoints to keep", 3))
    if interval < 0 or retention < 1:
        raise ValueError("Invalid first-stage checkpoint interval or retention setting")
    additional = len([value for value in settings.get("Additional checkpoint times", "").split(",") if value.strip()])
    source_output = _relative_directory(first_stage.get("Output directory", "output"))
    target_output = _relative_directory(second_stage_parameters["Output directory"])
    if source_output == target_output:
        raise ValueError("Second-stage output directory must differ from the first stage")
    checkpointing = second_stage_parameters["Checkpointing"]
    if int(checkpointing.get("Resume checkpoint", 0)) != 0:
        raise ValueError("Use Resume time, not an explicit Resume checkpoint, for log-based selection")
    target_time = float(checkpointing["Resume time"])
    # ASPECT defaults to years when neither the current parameter nor its
    # deprecated alias is explicitly present in the input file.
    use_years = str(second_stage_parameters.get(
        "Use years instead of seconds",
        second_stage_parameters.get("Use years in output instead of seconds", "true"),
    )).strip().lower()
    if use_years not in ("true", "false"):
        raise ValueError("Use years instead of seconds must be true or false")
    unit = "years" if use_years == "true" else "s"
    snapshot = select_checkpoint_from_log(case_dir / source_output / "log.txt", target_time, time_unit=unit)
    if snapshot.slot < 1 or snapshot.slot > retention + additional:
        raise ValueError("Selected checkpoint slot exceeds the first-stage retention configuration")
    target_capacity = int(checkpointing.get("Number of checkpoints to keep", 3)) + len([
        value for value in checkpointing.get("Additional checkpoint times", "").split(",") if value.strip()
    ])
    if snapshot.slot > target_capacity:
        raise ValueError("Second-stage checkpoint retention must include the selected slot ID")
    # Preserve the exact slot spelling reported by ASPECT (including zero padding).
    slot_name = Path(snapshot.path).name
    source = source_output / "restart" / slot_name
    target_restart = target_output / "restart"
    target_slot_name = f"{snapshot.slot:0{max(2, len(str(target_capacity)))}d}"
    script_path = case_dir / f"copy_restart_{tag}.sh"
    delta = _seconds(snapshot.timestep.time, snapshot.timestep.time_unit) - _seconds(target_time, unit)
    lines = [
        "#!/bin/bash", "set -euo pipefail",
        "# Run this on the server before starting the second stage.",
        "# Checkpoints were selected from log.txt; no local payload files were checked.",
        f"# First-stage interval: {interval} steps; retained regular slots: {retention}.",
        f"# Requested time: {target_time:g} {unit}.",
        f"# Selected slot {snapshot.slot}: t={snapshot.timestep.time:g} {snapshot.timestep.time_unit}, timestep {snapshot.timestep.number}.",
        f"# Signed time difference: {delta:g} seconds (using printed log precision).",
        'script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)',
        'cd -- "$script_dir"',
        f"source_slot={shlex.quote(source.as_posix())}",
        f"destination_restart={shlex.quote(target_restart.as_posix())}",
        f'destination_slot="$destination_restart/{target_slot_name}"',
        'mkdir -p -- "$destination_slot"',
        'rsync -a --partial --info=progress2 -- "$source_slot/" "$destination_slot/"',
        '# Atomically replace the pointer with an independent file after successful transfer.',
        'pointer=$(mktemp -- "$destination_restart/.last_good_checkpoint.XXXXXX")',
        f"printf '%s\\n' {snapshot.slot} > \"$pointer\"",
        'mv -f -- "$pointer" "$destination_restart/last_good_checkpoint.txt"',
    ]

    if script_path.is_file():
        print("Going to remove the old file of %s" % str(script_path))
        os.remove(str(script_path))

    with script_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return {"checkpoint": snapshot, "script_path": script_path,
            "steps_between_checkpoint": interval, "source": source,
            "destination": target_restart / target_slot_name,
            "time_difference_seconds": delta}
