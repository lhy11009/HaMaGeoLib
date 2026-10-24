import json

import pytest

from hamageolib.utils.case_aspect import CaseASPECT


@pytest.fixture
def case_dir(tmp_path):
    case = tmp_path / "case"
    (case / "output").mkdir(parents=True)
    (case / "output/original.prm").write_text("set Dimension = 2\n")
    (case / "output" / "log.txt").write_text(
        "--     . version 3.0.0\n"
        "*** Timestep 0: t=0 years, dt=0 years\n"
        "Writing graphical output: output/solution/solution-00000\n"
    )
    (case / "output/original.wb").write_text('{"features": []}')
    return case


def test_parse_default_output(case_dir):
    before = sorted(case_dir.rglob("*"))
    case = CaseASPECT(case_dir)
    assert case.output_dir == case_dir / "output"
    assert case.parameters == {"Dimension": "2"}
    assert case.world_builder == {"features": []}
    assert case.timesteps[0].number == 0
    assert case.graphical_outputs[0].solution_number == 0
    assert case.timing_blocks == []
    assert sorted(case_dir.rglob("*")) == before


def test_custom_output_subdirectory(case_dir):
    (case_dir / "output").rename(case_dir / "custom")
    case = CaseASPECT(case_dir, "custom")
    assert case.output_dir == case_dir / "custom"
    assert case.parameters == {"Dimension": "2"}
    assert case.world_builder == {"features": []}
    assert len(case.timesteps) == 1


@pytest.mark.parametrize("missing", ["output/original.prm", "output/original.wb", "output/log.txt"])
def test_missing_input(case_dir, tmp_path, missing):
    (case_dir / missing).unlink()
    with pytest.raises(FileNotFoundError):
        CaseASPECT(case_dir)
    assert not (tmp_path / "parsed").exists()


def test_malformed_log(case_dir, tmp_path):
    (case_dir / "output/log.txt").write_text(
        "--     . version 3.0.0\n*** Timestep invalid\n"
    )
    with pytest.raises(ValueError):
        CaseASPECT(case_dir)
    assert not (tmp_path / "parsed").exists()


def test_timing_records(case_dir, tmp_path):
    with (case_dir / "output/log.txt").open("a") as stream:
        stream.write(
            "| Total wallclock time elapsed since start | 12s |\n"
            "| Section | no. calls | wall time | % of total |\n"
            "+-------------+----------+-----------+------------+\n"
            "| Solve | 2 | 6s | 50% |\n"
            "+-------------+----------+-----------+------------+\n"
        )
    case = CaseASPECT(case_dir)
    assert case.timing_blocks[0].total_wallclock_seconds == 12
    assert case.timing_blocks[0].sections[0].name == "Solve"


def test_malformed_world_builder(case_dir):
    (case_dir / "output/original.wb").write_text("invalid json")
    with pytest.raises(json.JSONDecodeError):
        CaseASPECT(case_dir)
