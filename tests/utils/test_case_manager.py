"""Regression coverage for the CaseManager API migrated from gdmate."""

import pandas as pd
import pytest

from hamageolib.utils.case_manager import CaseManager


def test_case_lifecycle_and_csv_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_file = tmp_path / "cases.csv"
    manager = CaseManager(csv_file)
    assert manager.list_cases().empty
    assert manager.next_case_id() == 0
    manager.add_column("status", "pending")
    assert manager.have_column("status")
    assert list(manager.df.columns) == ["case_id", "status", "case_path"]

    idx, row = manager.add_case("case_a")
    assert idx == 0
    assert row["case_id"] == 0
    assert row["case_path"] == str(tmp_path / "case_a")
    manager.set_case_row({"case_id": 0, "status": "done", "case_path": "case_a"})
    manager.add_case("case_b", case_id=7)
    assert manager.next_case_id() == 8
    manager.save()

    restored = CaseManager(csv_file)
    assert restored.get_case_by_path("case_a")[1]["status"] == "done"
    assert restored.get_case_by_id(7)[1]["case_path"] == str(tmp_path / "case_b")
    restored.remove_case(0)
    assert list(restored.df.index) == [0]
    assert list(restored.df["case_id"]) == [7]


def test_load_normalizes_paths_and_preserves_extra_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv_file = tmp_path / "cases.csv"
    pd.DataFrame({"case_id": [3], "label": ["test"], "case_path": ["case_a"]}).to_csv(
        csv_file, index=False
    )
    manager = CaseManager(csv_file)
    assert manager.get_case_by_id(3)[1]["label"] == "test"
    assert manager.get_case_by_path(tmp_path / "case_a")[0] == 0


def test_duplicate_and_missing_entries(tmp_path):
    manager = CaseManager(tmp_path / "cases.csv")
    manager.add_case(tmp_path / "case_a", case_id=2)
    with pytest.raises(ValueError, match="Case path already exists"):
        manager.add_case(tmp_path / "case_a")
    with pytest.raises(ValueError, match="case_id already exists"):
        manager.add_case(tmp_path / "case_b", case_id=2)
    with pytest.raises(ValueError, match="already exists"):
        manager.add_column("case_id")
    with pytest.raises(CaseManager.TabelRowError):
        manager.get_case_by_id(99)
    with pytest.raises(CaseManager.TabelRowError):
        manager.get_case_by_path(tmp_path / "missing")
    with pytest.raises(CaseManager.TabelRowError):
        manager.set_case_row({"case_id": 99})
    with pytest.raises(ValueError, match="Missing columns"):
        manager.set_case_row({"case_id": 2})


def test_series_update(tmp_path):
    manager = CaseManager(tmp_path / "cases.csv")
    manager.add_case(tmp_path / "case_a")
    manager.add_column("status", "pending")
    _, row = manager.get_case_by_id(0)
    updated = row.copy()
    updated["status"] = "done"
    manager.set_case_row(updated)
    assert manager.get_case_by_id(0)[1]["status"] == "done"


def test_duplicate_ids_in_csv(tmp_path):
    csv_file = tmp_path / "cases.csv"
    pd.DataFrame({"case_id": [1, 1], "case_path": ["a", "b"]}).to_csv(csv_file, index=False)
    with pytest.raises(AssertionError, match="case_id must be unique"):
        CaseManager(csv_file)
