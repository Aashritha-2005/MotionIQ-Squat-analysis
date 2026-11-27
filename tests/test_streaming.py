"""Streaming pipeline: offline/streaming feature parity, empty frames, framing gate, display checks."""
import numpy as np
import pytest
from conftest import POSE_DIR

from extract_features import FEATURE_NAMES, StreamingFeatures, body_valid, fill_keypoints, geometry, torso_length
from pipeline import Pipeline


def offline(kpts):
    xy, _, _ = fill_keypoints(kpts)
    valid = body_valid(kpts)
    tl = torso_length(xy)
    scale = float(np.median(tl[valid.astype(bool)] if valid.any() else tl))
    return geometry(xy, scale), valid, scale


def stream(kpts, **kw):
    sf = StreamingFeatures(**kw)
    out = [r for k in kpts for r in sf.push(k)] + sf.flush()
    assert [r[0] for r in out] == list(range(len(kpts)))
    return np.stack([r[1] for r in out]), np.array([r[2] for r in out])


def synthetic_squats(T=300, seed=0):
    rng = np.random.default_rng(seed)
    k = np.zeros((T, 17, 3), np.float32)
    phase = np.sin(np.linspace(0, 6 * np.pi, T)) ** 2            # 3 squats
    for t in range(T):
        hip_y = 300 + 120 * phase[t]
        pts = {5: (200, hip_y - 160), 6: (210, hip_y - 160), 11: (200, hip_y), 12: (210, hip_y),
               13: (260, 420 - 0 * phase[t]), 14: (270, 420), 15: (220, 520), 16: (230, 520)}
        for j, (x, y) in pts.items():
            k[t, j] = (x + rng.normal(0, 1), y + rng.normal(0, 1), 0.95)
    k[40:44, [13, 15]] = np.nan                                     # short gap (interpolated offline)
    k[100:120, 5] = np.nan                                          # long gap (held)
    return k


def test_parity_synthetic_exact_given_offline_scale():
    k = synthetic_squats()
    G, V, scale = offline(k)
    F, Vs = stream(k, lookahead=7, fixed_scale=scale)
    assert (Vs == V).all()
    assert np.abs(F - G).max() <= 1e-4


@pytest.mark.skipif(not (POSE_DIR / "subject_002_squat_bad_side.npz").exists(), reason="local pose cache not present")
def test_parity_on_validation_clip():
    k = np.load(POSE_DIR / "subject_002_squat_bad_side.npz")["kpts"]
    G, V, scale = offline(k)
    F, _ = stream(k, lookahead=7, fixed_scale=scale)
    assert np.abs(F - G).max() <= 1e-4


def test_empty_frames_give_no_person_state_and_zero_reps():
    p = Pipeline("side")
    nan = np.full((17, 3), np.nan, np.float32)
    recs = [r for _ in range(90) for r in p.push(nan)] + p.flush()
    assert len(recs) == 90
    assert not any(p.stream.feat.valid)
    assert p.counter.count == 0 and all(r["rep"] is None for r in recs)
    assert np.isfinite(np.stack(p.F)).all()


def _drive(pipe, phases, shoulders_seen, valid=1):
    """Feed the counter/gate controlled phases (bypassing pose + model) to test the framing gate in isolation."""
    feat = pipe.stream.feat
    out = []
    for f, ph in enumerate(phases):
        o = np.ones(17, bool); o[[5, 6]] = shoulders_seen
        feat.obs.append(o)
        prob = np.eye(4, dtype=np.float32)[ph]
        out += pipe._consume([(f, np.zeros(len(FEATURE_NAMES), np.float32), valid, prob)])
    return out


PHASES = [0] * 10 + [1] * 8 + [2] * 5 + [3] * 8 + [0] * 10


def test_framing_gate_counts_framed_rep():
    p = Pipeline("side")
    ev = [r["rep"] for r in _drive(p, PHASES, True) if r["rep"]]
    assert p.counter.count == 1 and len(ev) == 1 and not ev[0].get("rejected")


def test_framing_gate_rejects_rep_without_upper_body():
    p = Pipeline("side")
    ev = [r["rep"] for r in _drive(p, PHASES, False) if r["rep"]]
    assert p.counter.count == 0 and len(ev) == 1 and "not fully in frame" in ev[0]["rejected"]


def test_framing_gate_rejects_rep_without_lower_body():
    p = Pipeline("side")
    _drive(p, PHASES, True, valid=0)
    assert p.counter.count == 0


def test_view_check_and_upper_body_warning():
    import run_demo as R
    side = [(t / 15, 0.08) for t in range(30)]
    front = [(t / 15, 0.9) for t in range(30)]
    assert R.view_warning(side, 2.0, "side") == "" and R.view_warning(front, 2.0, "side") != ""
    assert R.view_warning(side, 2.0, "front") != "" and R.view_warning(front, 2.0, "front") == ""
    assert R.view_warning(front[:5], 2.0, "side") == ""            # too few recent readings: no warning
    assert R.view_warning(front, 60.0, "side") == ""               # stale readings are ignored
    o = np.ones(17, bool); o[[5, 6]] = False
    run = 0
    for _ in range(R.UPPER_MISSING_WARN):
        run = R.upper_missing_update(run, True, o)
    assert run >= R.UPPER_MISSING_WARN and R.upper_missing_update(run, False, o) == 0
