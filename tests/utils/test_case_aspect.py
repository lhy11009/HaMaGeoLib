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


@pytest.mark.parametrize("missing", ["output/original.prm", "output/log.txt"])
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


def test_missing_world_builder(case_dir):
    (case_dir / "output/original.wb").unlink()
    case = CaseASPECT(case_dir)
    assert case.world_builder is None
    assert case.parameters == {"Dimension": "2"}
    assert len(case.timesteps) == 1


def write_snapshot(solution, number, pieces):
    pvtu = solution / f"solution-{number:05d}.pvtu"
    sources = "".join(f'<Piece Source="{piece}"/>' for piece in pieces)
    pvtu.write_text(
        '<VTKFile type="PUnstructuredGrid"><PUnstructuredGrid>'
        + sources + '</PUnstructuredGrid></VTKFile>'
    )


def test_existing_snapshots_are_complete_and_independent_of_log(case_dir):
    solution = case_dir / "output/solution"
    solution.mkdir()
    for number in (10, 2):
        pieces = [f"solution-{number:05d}.{rank:04d}.vtu" for rank in (0, 1)]
        write_snapshot(solution, number, pieces)
        for piece in pieces:
            (solution / piece).touch()
    # Snapshot 0 is reported in the log but has a missing second piece.
    write_snapshot(solution, 0, ["solution-00000.0000.vtu", "solution-00000.0001.vtu"])
    (solution / "solution-00000.0000.vtu").touch()
    # Orphan VTU files do not constitute a snapshot.
    (solution / "solution-00003.0000.vtu").touch()
    case = CaseASPECT(case_dir)
    assert case.existing_snapshots == [2, 10]
    assert case.n_existing_snapshots == 2
    assert [record.solution_number for record in case.graphical_outputs] == [0]


@pytest.mark.parametrize("contents", ["<broken", '<VTKFile/>',
    '<VTKFile><PUnstructuredGrid><Piece/></PUnstructuredGrid></VTKFile>'])
def test_incomplete_pvtu_not_counted(case_dir, contents):
    solution = case_dir / "output/solution"
    solution.mkdir()
    (solution / "solution-00000.pvtu").write_text(contents)
    case = CaseASPECT(case_dir)
    assert case.existing_snapshots == []
    assert case.n_existing_snapshots == 0


def test_missing_solution_directory(case_dir):
    case = CaseASPECT(case_dir)
    assert case.existing_snapshots == []
    assert case.n_existing_snapshots == 0


def test_snapshots_use_custom_output_directory(case_dir):
    (case_dir / "output").rename(case_dir / "custom")
    solution = case_dir / "custom/solution"
    solution.mkdir()
    write_snapshot(solution, 4, ["solution-00004.0000.vtu"])
    (solution / "solution-00004.0000.vtu").touch()
    case = CaseASPECT(case_dir, "custom")
    assert case.existing_snapshots == [4]
    assert case.n_existing_snapshots == 1
