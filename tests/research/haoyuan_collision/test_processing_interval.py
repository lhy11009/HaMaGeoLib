"""Select processing snapshots by model time, retaining full summary metadata."""
import pandas as pd
import pytest

from hamageolib.research.haoyuan_collision0 import post_process
from hamageolib.research.haoyuan_collision0.case_options import CASE_OPTIONS_TWOD


@pytest.mark.parametrize("interval, lower, upper, single, expected", [
    (None, None, None, None, [0, 1, 2, 3]),
    (2e5, None, None, None, [0, 2]),
    (1e4, None, None, None, [0, 1, 2, 3]),
    (1e5, 0, 3, None, [1, 2]),
    (2e5, None, None, 1, [1]),
    (2e5, 4, None, None, []),
])
def test_processing_interval(tmp_path, monkeypatch, interval, lower, upper, single, expected):
    options = CASE_OPTIONS_TWOD.__new__(CASE_OPTIONS_TWOD)
    options.summary_df = pd.DataFrame({
        "Vtu step": [0, 1, 2, 3], "Time": [0., 1.01e5, 2.02e5, 3.03e5],
        "slab_depth": [0.] * 4,
    })
    options.summary_file = str(tmp_path / "summary.csv")
    options.get_pvtu_step = lambda step: step + 5
    processed = []
    def process(case, snapshot, opts, **kwargs):
        processed.append(snapshot - 5)
        return {"slab_depth": 99.}
    monkeypatch.setattr(post_process, "ProcessVtuFileTwoDStep", process)
    post_process.process_all_vtu_steps(str(tmp_path), options, time_interval=interval,
                                     graphical_step_min=lower, graphical_step_max=upper,
                                     one_vtu_step=single)
    assert processed == expected
    result = pd.read_csv(options.summary_file)
    assert result["slab_depth"].tolist() == [99. if s in expected else 0. for s in range(4)]


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_processing_interval(tmp_path, interval):
    with pytest.raises(ValueError, match="time_interval"):
        post_process.process_all_vtu_steps(str(tmp_path), None, time_interval=interval)
