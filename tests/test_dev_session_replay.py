"""Replay of the user's first live (dev) session from saved keypoint traces (local only, never committed).

The traces come from replaying the overlay recording, so they approximate the live input. Reference (manual count
from contact sheets): ~13 +- 1 squats + 1 partial knee bend in 0-51.9 s, then walking toward the camera (51.9-61 s),
an empty wall and a face close-up -- none of which may add reps.
"""
import numpy as np
import pytest
from conftest import DEV_FIXTURE

from pipeline import Pipeline

pytestmark = pytest.mark.skipif(not DEV_FIXTURE.exists(), reason="local dev-session fixture not present")


@pytest.fixture(scope="module")
def replay():
    d = np.load(DEV_FIXTURE)
    p = Pipeline(str(d["view"]))
    recs = [r for k in d["kpts"] for r in p.push(k)] + p.flush()
    return [r["rep"] for r in recs if r["rep"]], p


def test_walk_toward_camera_adds_no_reps(replay):
    reps, _ = replay
    counted = [e for e in reps if not e.get("rejected") and e["start"] / 15 >= 51.9]
    rejected = [e for e in reps if e.get("rejected") and e["start"] / 15 >= 51.9]
    assert counted == [] and len(rejected) >= 1          # F1 regression: 8 spurious reps before the gate


def test_squat_part_count_matches_manual_reference(replay):
    reps, _ = replay
    n = sum(1 for e in reps if not e.get("rejected") and e["start"] / 15 < 51.9)
    assert 12 <= n <= 15                                  # 13 +- 1 squats (+1 partial bend counted)


def test_no_reps_after_the_person_leaves(replay):
    _, p = replay
    assert p.counter.count == sum(1 for e in replay[0] if not e.get("rejected"))
