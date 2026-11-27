"""Geometry features and form rules on hand-built poses with known answers."""
import numpy as np
import pytest

from extract_features import FEATURE_NAMES, geometry
from form_rules import evaluate_rep, load_config, rep_stat

FI = {n: i for i, n in enumerate(FEATURE_NAMES)}


def pose(hip=(0, 100), knee=(0, 150), ankle=(0, 200), sho=(0, 50), width=0.0):
    """Side-view stick figure (image y grows downward). Left/right points offset by `width` in x."""
    xy = np.zeros((17, 2), np.float32)
    for (l, r), p in {(5, 6): sho, (11, 12): hip, (13, 14): knee, (15, 16): ankle}.items():
        xy[l] = (p[0] - width / 2, p[1]); xy[r] = (p[0] + width / 2, p[1])
    return xy


def feats(*poses, scale=50.0):
    return geometry(np.stack(poses), scale)


def test_standing_straight():
    f = feats(pose(), pose())[0]
    assert f[FI["knee_min"]] == pytest.approx(1.0, abs=1e-3)      # 180 deg / 180
    assert f[FI["hip_min"]] == pytest.approx(1.0, abs=1e-3)
    assert f[FI["trunk_lean"]] == pytest.approx(0.0, abs=1e-3)
    assert f[FI["depth"]] == pytest.approx(2.0, abs=1e-3)         # hip 100 px above ankle / torso 50 px
    assert f[FI["hip_below_knee"]] == pytest.approx(-1.0, abs=1e-3)


def test_right_angle_knee():
    p = pose(hip=(-50, 150), knee=(0, 150), ankle=(0, 200), sho=(-50, 100))
    assert feats(p, p)[0][FI["knee_min"]] == pytest.approx(0.5, abs=1e-3)   # 90 deg


def test_45_degree_trunk_lean():
    p = pose(hip=(0, 100), sho=(35.355, 64.645))
    assert feats(p, p)[0][FI["trunk_lean"]] == pytest.approx(0.5, abs=1e-3)


def test_lean_is_clipped_for_upside_down_glitch():
    p = pose(hip=(0, 100), sho=(0, 150))
    assert feats(p, p)[0][FI["trunk_lean"]] == pytest.approx(1.0, abs=1e-6)


def test_hip_below_knee_sign():
    deep = pose(hip=(-40, 160), knee=(0, 150))
    assert feats(deep, deep)[0][FI["hip_below_knee"]] > 0


def test_velocity_central_difference():
    p0, p1, p2 = pose(hip=(0, 100)), pose(hip=(0, 105)), pose(hip=(0, 110))
    f = feats(p0, p1, p2)
    assert f[1, FI["d_depth"]] == pytest.approx(-0.1 * 15, abs=1e-4)   # depth drops 0.1 torso/frame at 15 fps


def test_knee_ankle_ratio_valgus_cue():
    p = pose(width=40); p[[13, 14], 0] = (-5, 5)               # knees 10 px apart, ankles 40 px apart
    assert feats(p, p)[0][FI["knee_ankle_ratio"]] == pytest.approx(0.25, abs=1e-3)


def _rep_features(hbk_max, lean, n=20):
    F = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    F[:, FI["hip_below_knee"]] = np.linspace(-1.0, hbk_max, n)
    F[:, FI["trunk_lean"]] = lean
    F[:, FI["depth"]] = np.linspace(2.0, 1.0, n)
    F[:, FI["knee_ankle_ratio"]] = 1.0
    return F


def test_rules_side_view():
    cfg = load_config()
    thr = cfg["form_rules"]["shallow_depth"]["threshold_per_view"]["side"]
    shallow = evaluate_rep(_rep_features(thr - 0.1, 0.1), np.ones(20), "side", cfg)
    deep = evaluate_rep(_rep_features(thr + 0.1, 0.1), np.ones(20), "side", cfg)
    assert shallow["shallow_depth"]["flag"] is True and deep["shallow_depth"]["flag"] is False
    assert deep["knee_valgus"]["flag"] is None                    # not observable from the side
    lean = evaluate_rep(_rep_features(0.0, 0.9), np.ones(20), "side", cfg)
    assert lean["forward_lean"]["flag"] is True


def test_rules_front_view_only_valgus():
    out = evaluate_rep(_rep_features(-0.9, 0.9), np.ones(20), "front", load_config())
    assert out["shallow_depth"]["flag"] is None and out["forward_lean"]["flag"] is None
    assert out["knee_valgus"]["flag"] is False


def test_rule_abstains_without_valid_frames():
    assert rep_stat(_rep_features(0, 0), np.zeros(20), "trunk_lean", "max") is None
    out = evaluate_rep(_rep_features(0, 0), np.array([1, 1] + [0] * 18), "side", load_config())
    assert out["shallow_depth"]["flag"] is None
