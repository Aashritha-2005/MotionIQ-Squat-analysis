"""Scripted live test runs 1 and 2 (Phase 7), replayed from logged keypoints (local fixtures, never committed).

Replay is exact: the same keypoints through the same pipeline give the same rep segments as the live session.
Confirmed failures that are NOT fixed (known limitations) are marked xfail(strict=True): they document the correct
behaviour, and the suite fails loudly if a later change makes them pass, so the limitation entry can be updated.
Frame indices are at 15 fps from app start.
"""
import numpy as np
import pytest
from conftest import TEST_RUN1, TEST_RUN2

from pipeline import Pipeline


def replay(path):
    if not path.exists():
        pytest.skip("local live-run fixture not present")
    p = Pipeline("side")
    recs = [r for k in np.load(path)["kpts"] for r in p.push(k)] + p.flush()
    return [r["rep"] for r in recs if r["rep"]], p


@pytest.fixture(scope="module")
def run1():
    return replay(TEST_RUN1)


@pytest.fixture(scope="module")
def run2():
    return replay(TEST_RUN2)


def counted(reps, a=0, b=10**9):
    return [e for e in reps if not e.get("rejected") and a <= e["start"] < b]


# ---------------------------------------------------------------- run 1: B 4 good, C 3 shallow, J 1 good, F walk-up
def test_run1_counts_8_of_8(run1):
    reps, p = run1
    assert p.counter.count == 8 and len(counted(reps)) == 8


def test_run1_shallow_flags_exactly_block_C(run1):
    flags = [e["form"]["shallow_depth"]["flag"] for e in counted(run1[0])]
    assert flags == [False] * 4 + [True] * 3 + [False]


def test_run1_no_lean_flag_on_good_reps(run1):
    assert not any(e["form"]["forward_lean"]["flag"] for e in counted(run1[0]))


def test_run1_walk_up_rejected(run1):
    late = [e for e in run1[0] if e["start"] >= 540]
    assert late and all(e.get("rejected") for e in late)


# ---------------------------------------------------------------- run 2: A, D 5 deep, E dips, H pick-up, I still, walk
def test_run2_block_D_five_deep_squats_counted_and_depth_ok(run2):
    d = counted(run2[0], 0, 390)                               # 0-26 s
    assert len(d) == 5 and not any(e["form"]["shallow_depth"]["flag"] for e in d)


def test_run2_stand_still_and_walk_up_add_nothing(run2):
    reps, p = run2
    assert counted(reps, 870) == [] and p.counter.count == 17  # nothing after 58 s (I: 61-81 s, walk-up 82-91 s)
    assert all(e.get("rejected") for e in reps if e["start"] >= 870)


@pytest.mark.xfail(strict=True, reason="known limitation: a held forward bend (pick-up) is counted as 8 reps; the "
                                       "descent/ascent flicker closes a rep at each bounce (k_bounce=2, require 'da')")
def test_run2_pick_up_from_floor_adds_no_reps(run2):
    assert counted(run2[0], 690, 870) == []                    # 46-58 s


@pytest.mark.xfail(strict=True, reason="known limitation: knee dips (knee 92-109 deg) look like shallow squats in the "
                                       "training data; counted and flagged shallow")
def test_run2_knee_dips_add_no_reps(run2):
    assert counted(run2[0], 390, 690) == []                    # 26-46 s


@pytest.mark.xfail(strict=True, reason="known limitation: forward_lean measures the bottom of the rep (27-33 deg here); "
                                       "the intended lean (46-52 deg) happens mid-rep")
def test_run2_intended_lean_reps_flagged(run2):
    d = counted(run2[0], 0, 390)
    assert sum(bool(e["form"]["forward_lean"]["flag"]) for e in d) >= 4


@pytest.mark.xfail(strict=True, reason="known limitation (F13): the 2 s view-check window carries shoulder rotation from "
                                       "the bottom of deep reps into the following standing frames")
def test_run1_no_side_on_warning_while_standing_side_on():
    import run_demo as R
    from extract_features import FEATURE_NAMES
    if not TEST_RUN1.exists():
        pytest.skip("local live-run fixture not present")
    sw_i, p, hist, warn_standing = FEATURE_NAMES.index("shoulder_width"), Pipeline("side"), [], []
    for i, k in enumerate(np.load(TEST_RUN1)["kpts"][:340]):              # blocks B (deep, side-on) up to rep 5
        for r in p.push(k):
            if p.stream.feat.obs[r["frame"]][[5, 6]].all():
                hist = (hist + [(i / 15, float(p.F[r["frame"]][sw_i]))])[-30:]
        if not p.counter.moving:
            warn_standing.append(bool(R.view_warning(hist, i / 15, "side")))
    assert not any(warn_standing)
