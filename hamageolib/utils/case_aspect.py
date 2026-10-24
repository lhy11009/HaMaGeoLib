"""Parse an ASPECT case using gdmate's aspect_tools parsers."""

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


class CaseASPECT:
    """Load ASPECT inputs and log records without writing files.

    ``output_dir`` defaults to ``output`` beneath ``case_dir``. It contains
    ``original.prm``, ``log.txt`` and optionally ``original.wb``. Parameters
    and World Builder JSON are exposed as ``parameters`` and ``world_builder``
    dicts; ``world_builder`` is None when ``original.wb`` is absent.
    Log records retain their gdmate dataclass types.

    ``existing_snapshots`` lists sorted snapshot numbers found in the
    solution directory with a readable PVTU and all its referenced VTU
    pieces present. ``n_existing_snapshots`` counts these snapshots,
    independently of the log's ``graphical_outputs`` records. This checks
    file presence, not VTU contents, at construction time.

    Requires gdmate's ``aspect_tools`` API (currently on its dev branch).
    The log parser requires ASPECT major version 3. Missing or malformed
    inputs raise the underlying reader's exception.
    """

    def __init__(self, case_dir, output_dir="output"):
        from gdmate.aspect_tools import AspectLogParser, parse_parameters_to_dict

        self.case_dir = Path(case_dir)
        self.output_dir = self.case_dir / output_dir
        with (self.output_dir / "original.prm").open(encoding="utf-8") as stream:
            self.parameters = parse_parameters_to_dict(stream)
        self.world_builder = None
        try:
            with (self.output_dir / "original.wb").open(encoding="utf-8") as stream:
                self.world_builder = json.load(stream)
        except FileNotFoundError:
            pass
        parser = AspectLogParser(self.output_dir / "log.txt")
        self.timesteps = parser.parse()
        self.graphical_outputs = parser.parse_graphical_outputs()
        self.timing_blocks = parser.parse_timing_blocks()
        self.existing_snapshots = self._find_existing_snapshots()
        self.n_existing_snapshots = len(self.existing_snapshots)

    def _find_existing_snapshots(self):
        snapshots = set()
        for pvtu in (self.output_dir / "solution").glob("solution-*.pvtu"):
            match = re.fullmatch(r"solution-(\d+)\.pvtu", pvtu.name)
            if match is None:
                continue
            try:
                root = ET.parse(pvtu).getroot()
            except (OSError, ET.ParseError):
                # A snapshot may still be in the process of being written.
                continue
            if root.tag != "VTKFile" or root.get("type") != "PUnstructuredGrid":
                continue
            pieces = root.findall("./PUnstructuredGrid/Piece")
            sources = [piece.get("Source", "") for piece in pieces]
            if sources and all(
                source.endswith(".vtu") and (pvtu.parent / source).is_file()
                for source in sources
            ):
                snapshots.add(int(match.group(1)))
        return sorted(snapshots)
