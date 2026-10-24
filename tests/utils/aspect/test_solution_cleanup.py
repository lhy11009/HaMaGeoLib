import shlex
import subprocess

import pytest

from hamageolib.utils.case_aspect import CaseASPECT
from hamageolib.utils.aspect.solution_cleanup import write_solution_cleanup_script


def make_case(tmp_path, times, unit="years", output="output"):
    case_dir = tmp_path / "case with ' quotes; $literal"
    destination = case_dir / output
    solution = destination / "solution"
    solution.mkdir(parents=True)
    (destination / "original.prm").write_text("set Dimension = 2\n")
    log = ["--     . version 3.0.0\n"]
    for number, time in enumerate(times):
        log.append(f"*** Timestep {number}: t={time} {unit}, dt=1 {unit}\n")
        log.append(f"Writing graphical output: {output}/solution/solution-{number:05d}\n")
        add_snapshot(solution, number)
    (destination / "log.txt").write_text("".join(log))
    return CaseASPECT(case_dir, output)


def add_snapshot(solution, number):
    prefix = f"solution-{number:05d}"
    pieces = [f"{prefix}.{rank:04d}.vtu" for rank in range(2)]
    (solution / f"{prefix}.pvtu").write_text(
        '<VTKFile type="PUnstructuredGrid"><PUnstructuredGrid>'
        + "".join(f'<Piece Source="{piece}"/>' for piece in pieces)
        + '</PUnstructuredGrid></VTKFile>'
    )
    for piece in pieces:
        (solution / piece).touch()


def test_nearest_selection_and_quoted_script(tmp_path):
    case = make_case(tmp_path, [0, 0.4e6, 0.9e6, 1.4e6, 2.1e6])
    before = sorted(case.case_dir.rglob("*"))
    script = tmp_path / "cleanup.sh"
    plan = write_solution_cleanup_script(case, 1e6, script)
    assert plan["kept_snapshots"] == [0, 2, 4]
    assert plan["removed_snapshots"] == [1, 3]
    assert len(plan["files_to_remove"]) == 6
    assert sorted(case.case_dir.rglob("*")) == before
    commands = [shlex.split(line) for line in script.read_text().splitlines()
                if line.startswith("rm -- ")]
    assert [command[2] for command in commands] == [str(p) for p in plan["files_to_remove"]]
    assert all(command[:2] == ["rm", "--"] for command in commands)
    subprocess.run(["sh", "-n", str(script)], check=True)


def test_tie_chooses_earlier_time(tmp_path):
    case = make_case(tmp_path, [0, 0.75e6, 1.25e6, 2e6])
    plan = write_solution_cleanup_script(case, 1e6, tmp_path / "cleanup.sh")
    assert plan["kept_snapshots"] == [0, 1, 3]


def test_incomplete_and_unrecorded_files_are_preserved(tmp_path):
    case = make_case(tmp_path, [0, 0.4e6, 1e6])
    solution = case.output_dir / "solution"
    # Remove a piece after CaseASPECT construction: the writer must rescan.
    (solution / "solution-00001.0001.vtu").unlink()
    add_snapshot(solution, 9)
    (solution / "solution-00008.0000.vtu").touch()
    plan = write_solution_cleanup_script(case, 1e6, tmp_path / "cleanup.sh")
    assert plan["kept_snapshots"] == [0, 2]
    assert plan["files_to_remove"] == []


def test_seconds_and_custom_output(tmp_path):
    year = 365.25 * 24 * 3600
    case = make_case(tmp_path, [0, 0.4e6 * year, 1e6 * year], "seconds", "custom")
    plan = write_solution_cleanup_script(case, 1e6, tmp_path / "cleanup.sh")
    assert plan["kept_snapshots"] == [0, 2]
    assert plan["removed_snapshots"] == [1]


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_interval(tmp_path, interval):
    case = make_case(tmp_path, [0])
    script = tmp_path / "cleanup.sh"
    with pytest.raises(ValueError):
        write_solution_cleanup_script(case, interval, script)
    assert not script.exists()


def test_missing_solution_directory(tmp_path):
    case = make_case(tmp_path, [])
    (case.output_dir / "solution").rmdir()
    plan = write_solution_cleanup_script(case, 1e6, tmp_path / "cleanup.sh")
    assert plan["files_to_remove"] == []


def test_existing_script_not_overwritten(tmp_path):
    case = make_case(tmp_path, [0])
    script = tmp_path / "cleanup.sh"
    script.write_text("keep me")
    with pytest.raises(FileExistsError):
        write_solution_cleanup_script(case, 1e6, script)
    assert script.read_text() == "keep me"


def test_duplicate_times_choose_smaller_number(tmp_path):
    case = make_case(tmp_path, [0, 0, 0.2e6, 2e6])
    plan = write_solution_cleanup_script(case, 1e6, tmp_path / "cleanup.sh")
    assert plan["kept_snapshots"] == [0, 2, 3]
    assert plan["removed_snapshots"] == [1]


def test_very_small_interval_does_not_enumerate_all_targets(tmp_path):
    case = make_case(tmp_path, [0, 1e6, 2e6])
    plan = write_solution_cleanup_script(case, 1e-20, tmp_path / "cleanup.sh")
    assert plan["kept_snapshots"] == [0, 1, 2]


def test_last_snapshot_is_not_automatically_kept(tmp_path):
    case = make_case(tmp_path, [0, 1e6, 1.2e6])
    plan = write_solution_cleanup_script(case, 1e6, tmp_path / "cleanup.sh")
    assert plan["removed_snapshots"] == [2]


def test_shared_piece_referenced_by_unrecorded_snapshot_is_preserved(tmp_path):
    case = make_case(tmp_path, [0, 0.4e6, 1e6])
    solution = case.output_dir / "solution"
    add_snapshot(solution, 9)
    pvtu = solution / "solution-00009.pvtu"
    pvtu.write_text(pvtu.read_text().replace("solution-00009.0000.vtu",
                                            "solution-00001.0000.vtu"))
    plan = write_solution_cleanup_script(case, 1e6, tmp_path / "cleanup.sh")
    assert plan["removed_snapshots"] == [1]
    assert solution / "solution-00001.0000.vtu" not in plan["files_to_remove"]
    assert solution / "solution-00001.0001.vtu" in plan["files_to_remove"]


def test_conflicting_log_times_are_preserved(tmp_path):
    case = make_case(tmp_path, [0, 0.4e6, 1e6])
    log = case.output_dir / "log.txt"
    with log.open("a") as stream:
        stream.write("*** Timestep 3: t=600000 years, dt=1 years\n"
                     "Writing graphical output: output/solution/solution-00001\n")
    case = CaseASPECT(case.case_dir)
    plan = write_solution_cleanup_script(case, 1e6, tmp_path / "cleanup.sh")
    assert plan["files_to_remove"] == []


def test_unknown_time_unit_does_not_write_script(tmp_path):
    case = make_case(tmp_path, [0], "fortnights")
    script = tmp_path / "cleanup.sh"
    with pytest.raises(ValueError, match="Unsupported log time unit"):
        write_solution_cleanup_script(case, 1e6, script)
    assert not script.exists()
