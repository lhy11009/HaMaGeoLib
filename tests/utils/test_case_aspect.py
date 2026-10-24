import json

import pytest

from hamageolib.utils.case_aspect import CaseASPECT


@pytest.fixture
def case_dir(tmp_path):
    case = tmp_path / "case"
    (case / "output").mkdir(parents=True)
    (case / "case.prm").write_text("set Dimension = 2\n")
    (case / "output" / "log.txt").write_text(
        "--     . version 3.0.0\n"
        "*** Timestep 0: t=0 years, dt=0 years\n"
        "Writing graphical output: output/solution/solution-00000\n"
    )
    return case


def test_parse_and_export(case_dir, tmp_path):
    destination = tmp_path / "parsed"
    case = CaseASPECT(case_dir, destination)
    assert case.parameters == {"Dimension": "2"}
    assert case.timesteps[0].number == 0
    assert case.graphical_outputs[0].solution_number == 0
    assert case.timing_blocks == []
    result = json.loads((destination / "case.json").read_text())
    assert result["timesteps"][0]["time"] == 0
    assert result["parameters"] == case.parameters


@pytest.mark.parametrize("missing", ["case.prm", "output/log.txt"])
def test_missing_input(case_dir, tmp_path, missing):
    (case_dir / missing).unlink()
    with pytest.raises(FileNotFoundError):
        CaseASPECT(case_dir, tmp_path / "parsed")
    assert not (tmp_path / "parsed").exists()


def test_malformed_log(case_dir, tmp_path):
    (case_dir / "output/log.txt").write_text(
        "--     . version 3.0.0\n*** Timestep invalid\n"
    )
    with pytest.raises(ValueError):
        CaseASPECT(case_dir, tmp_path / "parsed")
    assert not (tmp_path / "parsed").exists()


def test_timing_export(case_dir, tmp_path):
    with (case_dir / "output/log.txt").open("a") as stream:
        stream.write(
            "| Total wallclock time elapsed since start | 12s |\n"
            "| Section | no. calls | wall time | % of total |\n"
            "+-------------+----------+-----------+------------+\n"
            "| Solve | 2 | 6s | 50% |\n"
            "+-------------+----------+-----------+------------+\n"
        )
    case = CaseASPECT(case_dir, tmp_path / "parsed")
    assert case.timing_blocks[0].total_wallclock_seconds == 12
    data = json.loads((case.output_dir / "case.json").read_text())
    assert data["timing_blocks"][0]["sections"][0]["name"] == "Solve"
