"""Parse an ASPECT case using gdmate's aspect_tools parsers."""

import json
from pathlib import Path


class CaseASPECT:
    """Load ASPECT inputs and log records without writing files.

    ``output_dir`` defaults to ``output`` beneath ``case_dir``. It contains
    ``original.prm``, ``original.wb`` and ``log.txt``. Parameters and World
    Builder JSON are exposed as ``parameters`` and ``world_builder`` dicts.
    Log records retain their gdmate dataclass types.

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
        with (self.output_dir / "original.wb").open(encoding="utf-8") as stream:
            self.world_builder = json.load(stream)
        parser = AspectLogParser(self.output_dir / "log.txt")
        self.timesteps = parser.parse()
        self.graphical_outputs = parser.parse_graphical_outputs()
        self.timing_blocks = parser.parse_timing_blocks()
