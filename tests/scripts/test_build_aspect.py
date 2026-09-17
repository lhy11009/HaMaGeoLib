import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "bash_scripts" / "build_aspect.sh"


def run_sourced_function(command: str, *, env: dict[str, str] | None = None):
    shell_env = os.environ.copy()
    if env:
        shell_env.update(env)
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {command}'],
        check=False,
        capture_output=True,
        text=True,
        env=shell_env,
    )


def test_compute_build_directory_without_tag():
    result = run_sourced_function(
        'compute_build_directory "feature/branch" ""',
        env={"ASPECT_SOURCE_DIR": "/tmp/aspect source"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/tmp/aspect source/build_feature_branch"


def test_compute_build_directory_with_manual_tag():
    result = run_sourced_function(
        'compute_build_directory "main" "gcc-12"',
        env={"ASPECT_SOURCE_DIR": "/tmp/aspect"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/tmp/aspect/build_main_gcc-12"


def test_parse_arguments_defaults_to_debugrelease():
    result = run_sourced_function(
        'parse_arguments feature; printf "%s\n%s\n%s\n%s\n" '
        '"$branch" "$mode" "$tag" "$jobs"'
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["feature", "debugrelease", "", "1"]


def test_parse_arguments_accepts_mode_and_manual_tag():
    result = run_sourced_function(
        'parse_arguments main --mode release --tag optimized; '
        'printf "%s\n%s\n%s\n" "$branch" "$mode" "$tag"'
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["main", "release", "optimized"]


def test_parse_arguments_uses_date_tag():
    result = run_sourced_function(
        'date() { printf "260916\n"; }; parse_arguments main --date-tag; '
        'printf "%s\n" "$tag"'
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "260916"


def test_parse_arguments_rejects_two_tag_options():
    result = run_sourced_function(
        "parse_arguments main --tag custom --date-tag"
    )

    assert result.returncode != 0
    assert "cannot be used together" in result.stderr


def test_parse_arguments_rejects_plugin_paths():
    result = run_sourced_function("parse_arguments main --plugin ../outside")

    assert result.returncode != 0
    assert "plugin names" in result.stderr


def test_discover_plugins_lists_only_directories(tmp_path):
    plugins_directory = tmp_path / "plugins"
    (plugins_directory / "plugin_b").mkdir(parents=True)
    (plugins_directory / "plugin_a").mkdir()
    (plugins_directory / "README.md").write_text("not a plugin")

    result = run_sourced_function(
        "discover_plugins", env={"ASPECT_SOURCE_DIR": str(tmp_path)}
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["plugin_a", "plugin_b"]


def test_build_plugin_copies_to_build_directory_and_uses_aspect_dir(tmp_path):
    aspect_directory = tmp_path / "aspect source"
    plugin_source = aspect_directory / "plugins" / "example"
    build_directory = aspect_directory / "build_main_custom"
    fake_bin = tmp_path / "bin"
    cmake_log = tmp_path / "cmake.log"
    plugin_source.mkdir(parents=True)
    build_directory.mkdir()
    fake_bin.mkdir()
    (plugin_source / "CMakeLists.txt").write_text("project(example)")
    fake_cmake = fake_bin / "cmake"
    fake_cmake.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CMAKE_LOG"\n'
    )
    fake_cmake.chmod(0o755)

    result = run_sourced_function(
        f'jobs=3; build_plugin "{build_directory}" example',
        env={
            "ASPECT_SOURCE_DIR": str(aspect_directory),
            "CMAKE_LOG": str(cmake_log),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    assert result.returncode == 0, result.stderr
    assert (build_directory / "example" / "CMakeLists.txt").is_file()
    assert cmake_log.read_text().splitlines() == [
        f"-S {build_directory}/example -B {build_directory}/example "
        f"-DAspect_DIR={build_directory}",
        f"--build {build_directory}/example --parallel 3",
    ]


def test_prepare_build_directory_starts_clean_when_configuring(tmp_path):
    aspect_directory = tmp_path / "aspect"
    build_directory = aspect_directory / "build_main"
    build_directory.mkdir(parents=True)
    stale_file = build_directory / "stale-file"
    stale_file.write_text("old build output")

    result = run_sourced_function(
        f'configure=1; prepare_build_directory "{build_directory}"',
        env={"ASPECT_SOURCE_DIR": str(aspect_directory)},
    )

    assert result.returncode == 0, result.stderr
    assert build_directory.is_dir()
    assert not stale_file.exists()
