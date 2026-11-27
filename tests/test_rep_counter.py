"""Rep counter on synthetic phase sequences (0 standing, 1 descent, 2 bottom, 3 ascent), with the deployed config."""
import json

import pytest

from config import ARTIFACTS
from rep_counter import RepCounter

S, D, B, A = 0, 1, 2, 3
CFG = json.load(open(ARTIFACTS / "pipeline_config.json"))["rep_counter"]


def count(seq, **cfg):
    rc = RepCounter(**(cfg or CFG))
    events = [e for t, p in enumerate(seq) if (e := rc.update(t, p)) is not None]
    return rc.count, events


def rep(d=8, b=5, a=8):
    return [D] * d + [B] * b + [A] * a


def test_single_clean_rep():
    n, ev = count([S] * 10 + rep() + [S] * 10)
    assert n == 1 and ev[0]["start"] == 10 and ev[0]["end"] == 10 + 21 - 1


def test_three_reps_with_standing_pauses():
    assert count([S] * 10 + (rep() + [S] * 15) * 3)[0] == 3


def test_flicker_below_hysteresis_is_ignored():
    seq = [S] * 20
    for i in range(0, 200, 10):
        seq += [D] * (CFG["k_in"] - 1) + [S] * 5   # short non-standing blips never start a movement
    assert count(seq)[0] == 0


def test_single_frame_alternating_noise():
    assert count([S, B] * 100)[0] == 0


def test_partial_descent_without_ascent_is_not_a_rep():
    assert count([S] * 10 + [D] * 12 + [S] * 20)[0] == 0


def test_incomplete_rep_stopped_at_bottom_and_stood_up_without_ascent():
    assert count([S] * 10 + [D] * 8 + [B] * 10 + [S] * 20)[0] == 0


def test_continuous_reps_without_standing_bounce():
    seq = [S] * 10 + rep() + rep() + rep() + [S] * 10   # ascent -> descent without standing up
    assert count(seq)[0] == 3


def test_long_pause_at_bottom_is_one_rep():
    assert count([S] * 10 + rep(b=90) + [S] * 10)[0] == 1


def test_long_pause_standing_between_reps():
    assert count([S] * 10 + rep() + [S] * 300 + rep() + [S] * 10)[0] == 2


def test_clip_starting_mid_squat_is_not_counted():
    assert count([B] * 5 + [A] * 8 + [S] * 10 + rep() + [S] * 10)[0] == 1


def test_bottom_not_required_but_order_is():
    assert count([S] * 10 + [D] * 8 + [A] * 8 + [S] * 10)[0] == 1          # deployed "da": bottom optional
    assert count([S] * 10 + [A] * 8 + [D] * 8 + [S] * 10)[0] == 0          # ascent before descent: no rep


def test_reject_last_uncounts():
    rc = RepCounter(**CFG)
    for t, p in enumerate([S] * 10 + rep() + [S] * 5):
        if rc.update(t, p):
            rc.reject_last()
    assert rc.count == 0


@pytest.mark.parametrize("k_in", [1, 3, 6])
def test_counts_are_stable_across_hysteresis(k_in):
    assert count([S] * 10 + (rep() + [S] * 12) * 4, **{**CFG, "k_in": k_in})[0] == 4
