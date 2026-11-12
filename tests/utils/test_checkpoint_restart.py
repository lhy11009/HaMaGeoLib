"""Select checkpoints from logs without needing their large files locally."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from hamageolib.utils.aspect.checkpoint_restart import (
    CheckpointLogParser, select_checkpoint_from_log, write_restart_copy_script,
)


LOG = """-- . version 3.2.0-pre
*** Timestep 49:  t=9 years, dt=0.2 years
*** Snapshot output/restart/00001/ created!
*** Timestep 50:  t=10 years, dt=1 years
*** Timestep 99:  t=19 years, dt=0.2 years
*** Snapshot output/restart/00002/ created!
*** Timestep 100:  t=20 years, dt=1 years
*** Timestep 125:  t=29 years, dt=0.2 years
*** Snapshot output/restart/00003/ created!
*** Timestep 126:  t=30 years, dt=1 years
"""


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "case"
    (root / "output").mkdir(parents=True)
    (root / "output/log.txt").write_text(LOG)
    (root / "case.prm").write_text("""set Output directory = output
set Use years instead of seconds = true
subsection Checkpointing
  set Steps between checkpoint = 50
  set Number of checkpoints to keep = 10000
end
""")
    return root


def second_stage():
    return {"Output directory": "output_test", "Use years instead of seconds": "true",
            "Checkpointing": {"Resume time": "21", "Number of checkpoints to keep": "10000"}}


def test_checkpoint_times_are_following_headers(case):
    records = CheckpointLogParser(case / "output/log.txt").parse_checkpoint_outputs()
    assert [r.slot for r in records] == [1, 2, 3]
    assert [r.timestep.time for r in records] == [10., 20., 30.]
    assert [r.timestep.number for r in records] == [50, 100, 126]
    # A forced, off-interval checkpoint is an actual candidate as well.
    assert select_checkpoint_from_log(case / "output/log.txt", 29).slot == 3


def test_checkpoint_selection_units_and_ties(case):
    log = case / "output/log.txt"
    assert select_checkpoint_from_log(log, 21).slot == 2
    assert select_checkpoint_from_log(log, 21 * 31556952, time_unit="s").slot == 2
    # ASPECT prefers the last reported checkpoint on a tie; otherwise the lower slot.
    assert select_checkpoint_from_log(log, 25).slot == 3
    assert select_checkpoint_from_log(log, 15).slot == 1
    log.write_text(LOG.replace("years", "s"))
    assert select_checkpoint_from_log(log, 21, time_unit="s").slot == 2


@pytest.mark.parametrize("suffix, message", [
    ("*** Snapshot output/restart/00004/ created!\n", "unresolved"),
    ("*** Snapshot output/restart/00001/ created!\n*** Timestep 150:  t=40 years, dt=1 years\n", "reused"),
])
def test_incomplete_or_rotated_history_is_not_silently_used(case, suffix, message):
    log = case / "output/log.txt"
    log.write_text(LOG + suffix)
    with pytest.raises(ValueError, match=message):
        select_checkpoint_from_log(log, 21)


def test_snapshot_does_not_take_time_from_another_run(case):
    log = case / "output/log.txt"
    log.write_text(LOG + "*** Snapshot output/restart/00004/ created!\n"
                   "-- . version 3.2.0-pre\n*** Timestep 0:  t=0 years, dt=0 years\n")
    assert CheckpointLogParser(log).parse_checkpoint_outputs()[-1].timestep is None


@pytest.mark.parametrize("text, message", [
    ("-- . version 3.2.0-pre\n", "No checkpoint"),
    ("-- . version 3.2.0-pre\n*** Snapshot broken created!\n", "malformed checkpoint"),
])
def test_missing_or_malformed_checkpoint_records(case, text, message):
    log = case / "output/log.txt"
    log.write_text(text)
    with pytest.raises(ValueError, match=message):
        select_checkpoint_from_log(log, 20)


def test_script_rejects_same_source_and_destination(case):
    parameters = second_stage()
    parameters["Output directory"] = "output"
    with pytest.raises(ValueError, match="must differ"):
        write_restart_copy_script(case, parameters, "test")


def test_script_generation_without_local_checkpoints(case):
    plan = write_restart_copy_script(case, second_stage(), "test")
    assert not (case / "output/restart").exists()
    assert not (case / "output_test").exists()
    assert plan["checkpoint"].slot == 2
    assert plan["steps_between_checkpoint"] == 50
    assert plan["script_path"] == case / "copy_restart_test.sh"
    script = plan["script_path"].read_text()
    assert "output/restart/00002" in script
    assert "00001" not in script and "00003" not in script
    assert "rsync" in script
    subprocess.run(["bash", "-n", str(plan["script_path"])], check=True)
    with pytest.raises(FileExistsError):
        write_restart_copy_script(case, second_stage(), "test")


def test_script_respects_second_stage_slot_width(case):
    parameters = second_stage()
    parameters["Checkpointing"]["Number of checkpoints to keep"] = "3"
    plan = write_restart_copy_script(case, parameters, "test")
    assert plan["destination"] == Path("output_test/restart/02")


def test_second_stage_target_in_seconds(case):
    parameters = second_stage()
    parameters["Use years instead of seconds"] = "false"
    parameters["Checkpointing"]["Resume time"] = str(21 * 31556952)
    assert write_restart_copy_script(case, parameters, "test")["checkpoint"].slot == 2


@pytest.mark.parametrize("setting", [None, ("Use years in output instead of seconds", "true")])
def test_missing_unit_setting_uses_aspect_years_default(case, setting):
    parameters = second_stage()
    parameters.pop("Use years instead of seconds")
    if setting is not None:
        parameters[setting[0]] = setting[1]
    plan = write_restart_copy_script(case, parameters, "test")
    assert plan["checkpoint"].slot == 2
    assert plan["time_difference_seconds"] == -31556952


def test_legacy_seconds_setting(case):
    parameters = second_stage()
    parameters.pop("Use years instead of seconds")
    parameters["Use years in output instead of seconds"] = "false"
    parameters["Checkpointing"]["Resume time"] = str(21 * 31556952)
    assert write_restart_copy_script(case, parameters, "test")["checkpoint"].slot == 2


def test_script_copies_only_selected_slot_after_moving_to_server(case, tmp_path):
    plan = write_restart_copy_script(case, second_stage(), "test")
    remote = tmp_path / "server case ' with spaces"
    shutil.move(case, remote)
    for slot in ("00001", "00002", "00003"):
        directory = remote / "output/restart" / slot
        directory.mkdir(parents=True)
        for filename in ("mesh", "mesh.info", "mesh_fixed.data", "resume.z", "metadata.txt"):
            (directory / filename).write_text(slot)
    source_pointer = remote / "output/restart/last_good_checkpoint.txt"
    source_pointer.write_text("3\n")
    # Stand in for rsync so tests do not require it to be installed.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rsync = bin_dir / "rsync"
    rsync.write_text('#!/bin/bash\nset -eu\ncp -a -- "${@: -2:1}/." "${@: -1}"\n')
    rsync.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    subprocess.run(["bash", str(remote / plan["script_path"].name)], cwd=tmp_path, env=env, check=True)
    target = remote / "output_test/restart"
    assert sorted(p.name for p in target.iterdir()) == ["00002", "last_good_checkpoint.txt"]
    assert (target / "00002/mesh_fixed.data").read_text() == "00002"
    assert (target / "last_good_checkpoint.txt").read_text() == "2\n"
    assert source_pointer.read_text() == "3\n"
    # A repeated run is resumable and does not create a nested 00002/00002.
    subprocess.run(["bash", str(remote / plan["script_path"].name)], env=env, check=True)
    assert not (target / "00002/00002").exists()


def test_failed_transfer_does_not_mark_checkpoint_ready(case, tmp_path):
    plan = write_restart_copy_script(case, second_stage(), "test")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rsync = bin_dir / "rsync"
    rsync.write_text("#!/bin/sh\nexit 23\n")
    rsync.chmod(0o755)
    result = subprocess.run(["bash", str(plan["script_path"])], env=dict(
        os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"]
    ))
    assert result.returncode == 23
    assert not (case / "output_test/restart/last_good_checkpoint.txt").exists()
