"""Parse an ASPECT case using gdmate's aspect_tools parsers."""

from dataclasses import asdict
import json
from pathlib import Path


class CaseASPECT:
    """Read a case and write parsed records to ``output_dir/case.json``.

    ``case_dir`` must contain ``case.prm`` and ``output/log.txt``. The
    destination is a directory for parsed results, separate from ASPECT's
    simulation output. Construction parses inputs immediately; missing or
    malformed inputs raise the underlying parser exception before exporting.

    Requires gdmate with ``gdmate.aspect_tools`` (currently on its dev branch).
    Log parsing retains gdmate's default ASPECT major-version-3 validation.
    Parameters, timesteps, graphical_outputs and timing_blocks remain
    available as attributes; log records retain their gdmate dataclass types.
    """

    def __init__(self, case_dir, output_dir):
        from gdmate.aspect_tools import AspectLogParser, parse_parameters_to_dict

        self.case_dir = Path(case_dir)
        self.output_dir = Path(output_dir)
        with (self.case_dir / "case.prm").open(encoding="utf-8") as stream:
            self.parameters = parse_parameters_to_dict(stream)
        parser = AspectLogParser(self.case_dir / "output" / "log.txt")
        self.timesteps = parser.parse()
        self.graphical_outputs = parser.parse_graphical_outputs()
        self.timing_blocks = parser.parse_timing_blocks()
        result = {
            "parameters": self.parameters,
            "timesteps": [asdict(record) for record in self.timesteps],
            "graphical_outputs": [asdict(record) for record in self.graphical_outputs],
            "timing_blocks": [asdict(record) for record in self.timing_blocks],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "case.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
