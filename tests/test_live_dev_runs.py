"""Live dev runs after the Phase 7 fixes: exact replay of logged keypoints (local fixture, never committed) and the
output-file overwrite guard (F14)."""
import subprocess
import sys

import numpy as np
import pytest
from conftest import ROOT, RUN2_FIXTURE

from pipeline import Pipeline


@pytest.fixture(scope="module")
def run2():
    if not RUN2_FIXTURE.exists():
        pytest.skip("local live-run fixture not present")
    p = Pipeline("side")
    recs = [r for k in np.load(RUN2_FIXTURE)["kpts"] for r in p.push(k)] + p.flush()
    return [r["rep"] for r in recs if r["rep"]], p


def test_run2_replay_matches_live_counts(run2):
    reps, p = run2
    counted = [(e["start"], e["end"]) for e in reps if not e.get("rejected")]
    assert counted == [(69, 94), (107, 133), (166, 193), (234, 264), (282, 319)] and p.counter.count == 5


def test_run2_walk_up_to_camera_is_rejected(run2):
    reps, _ = run2
    late = [e for e in reps if e["start"] >= 23 * 15]          # walking up to the camera from ~23 s
    assert late and all(e.get("rejected") for e in late)       # F1 gate, first live evidence


def test_run2_full_depth_reps_not_flagged_shallow(run2):
    reps, _ = run2                                              # F12: knee 18-24 deg, hip below knee -> depth OK
    assert all(e["form"]["shallow_depth"]["flag"] is False for e in reps if not e.get("rejected"))


def test_outputs_are_never_overwritten(tmp_path):
    out = tmp_path / "session.npz"
    out.write_bytes(b"keep")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_demo.py"), "--view", "side", "--video", "none.mp4",
                        "--headless", "--log-session", str(out)], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "refusing to overwrite" in r.stderr and out.read_bytes() == b"keep"   # F14


def test_session_log_saves_displayed_warnings(tmp_path):
    import cv2
    vid, out = tmp_path / "noise.mp4", tmp_path / "s.npz"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(30):
        w.write(rng.integers(0, 60, (240, 320, 3), dtype=np.uint8))
    w.release()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_demo.py"), "--view", "side", "--video", str(vid),
                        "--headless", "--log-session", str(out)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    d = np.load(out)
    assert len(d["view_warn"]) == len(d["upper_missing_warn"]) == len(d["t"]) > 0
