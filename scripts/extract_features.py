"""Task 2.2: per-frame geometry features from the 15 fps YOLO keypoints (also imported by the live demo).

Gap-fill policy (so blurred / missed frames never produce NaN):
  * a keypoint is "observed" in a frame if its confidence >= KP_CONF (0.3)
  * unobserved runs of <= MAX_INTERP frames (0.5 s) that have observed frames on both sides -> linear interp
  * longer runs / runs at the clip edges -> hold the nearest observed value (last valid; first valid at the start)
  * a keypoint never observed in the clip -> copy its left/right mirror; if both missing, the clip-mean of
    the observed body keypoints (should not happen for the dataset; counted in the log)
Per-frame flags saved alongside the features:
  valid  = 1 if a person was detected AND mean hip/knee/ankle confidence >= 0.5 (same "body visible"
           test as the Phase 1 blur audit), else 0. The model gets this as an input channel.
  fill   = 0 observed, 1 interpolated, 2 held  (worst case over the hip/knee/ankle/shoulder keypoints)

Scale normalisation: distances are divided by the clip's median torso length (shoulder-mid to hip-mid over
valid frames). The live demo uses a running median of the same quantity.

Features (FEATURE_NAMES, all float32):
  knee_l, knee_r, knee_min          knee angle hip-knee-ankle / 180 (1 = straight)
  hip_l, hip_r, hip_min             hip angle shoulder-hip-knee / 180
  trunk_lean                        angle of hip->shoulder vector from vertical / 90 (0 = upright), clipped to [0, 1]
  depth                             (ankle_y - hip_y) / torso   hip height above ankles
  hip_below_knee                    (hip_y - knee_y) / torso    > 0 means hip below knee (past parallel)
  knee_over_toe                     |knee_x - ankle_x| / torso (mean of sides)
  knee_width, ankle_width, hip_width, shoulder_width   |x_l - x_r| / torso
  knee_ankle_ratio                  knee_width / max(ankle_width, 0.05)   (< 1 = knees inside ankles: valgus cue)
  d_knee_min, d_hip_min, d_trunk_lean, d_depth        frame-to-frame velocity (per second, central difference)

Usage: .venv/bin/python scripts/extract_features.py
"""
import bisect
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, FPS, KP, LABELS_CSV, POSE_DIR  # noqa: E402

GEOM_DIR = DATA / "geom_features"
KP_CONF, BODY_CONF, MAX_INTERP = 0.3, 0.5, int(0.5 * FPS)
MIRROR = {1: 2, 2: 1, 3: 4, 4: 3, **{i: i + 1 for i in range(5, 17, 2)}, **{i: i - 1 for i in range(6, 17, 2)}}
LOWER = [11, 12, 13, 14, 15, 16]
CORE = [5, 6, 11, 12, 13, 14, 15, 16]
FEATURE_NAMES = ["knee_l", "knee_r", "knee_min", "hip_l", "hip_r", "hip_min", "trunk_lean", "depth",
                 "hip_below_knee", "knee_over_toe", "knee_width", "ankle_width", "hip_width", "shoulder_width",
                 "knee_ankle_ratio", "d_knee_min", "d_hip_min", "d_trunk_lean", "d_depth"]


def fill_keypoints(kpts):
    """(T,17,3) with NaNs -> filled (T,17,2) xy, per-keypoint fill code (T,17): 0 obs, 1 interp, 2 held."""
    T = len(kpts)
    conf = np.nan_to_num(kpts[:, :, 2])
    obs = conf >= KP_CONF
    xy = np.full((T, 17, 2), np.nan, np.float32)
    code = np.zeros((T, 17), np.int8)
    for j in range(17):
        o = np.flatnonzero(obs[:, j])
        if len(o) == 0:
            continue
        xy[o, j] = kpts[o, j, :2]
        miss = np.flatnonzero(~obs[:, j])
        if len(miss) == 0:
            continue
        nxt = np.searchsorted(o, miss)            # index into o of the next observed frame
        has_prev, has_next = nxt > 0, nxt < len(o)
        prev_f = np.where(has_prev, o[np.maximum(nxt - 1, 0)], -1)
        next_f = np.where(has_next, o[np.minimum(nxt, len(o) - 1)], -1)
        gap = next_f - prev_f - 1
        interp = has_prev & has_next & (gap <= MAX_INTERP)
        for c in range(2):
            xy[miss, j, c] = np.where(
                interp, np.interp(miss, o, kpts[o, j, c]),
                np.where(has_prev, kpts[prev_f, j, c], kpts[next_f, j, c]))
        code[miss, j] = np.where(interp, 1, 2)
    never = [j for j in range(17) if np.isnan(xy[:, j, 0]).all()]
    for j in never:
        m = MIRROR.get(j)
        if m is not None and not np.isnan(xy[:, m, 0]).all():
            xy[:, j], code[:, j] = xy[:, m], 2
    for j in [j for j in range(17) if np.isnan(xy[:, j, 0]).all()]:
        xy[:, j], code[:, j] = np.nanmean(xy[:, CORE], axis=1), 2
    xy = np.nan_to_num(xy)  # only if the clip had no person at all
    return xy, code, never


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _vel(x):
    return np.gradient(x) * FPS if len(x) > 1 else np.zeros_like(x)


def body_valid(kpts):
    person = ~np.isnan(kpts[:, 0, 0])
    return (person & (np.nan_to_num(kpts[:, LOWER, 2]).mean(1) >= BODY_CONF)).astype(np.int8)


def torso_length(xy):
    sho, hip = xy[:, [5, 6]].mean(1), xy[:, [11, 12]].mean(1)
    return np.linalg.norm(sho - hip, axis=1)


def geometry(xy, scale):
    """Filled keypoints (T,17,2) + scalar body scale -> (T, len(FEATURE_NAMES)) float32."""
    k = {n: xy[:, i] for n, i in KP.items()}
    s = max(float(scale), 1e-3)
    knee_l, knee_r = _angle(k["l_hip"], k["l_knee"], k["l_ank"]) / 180, _angle(k["r_hip"], k["r_knee"], k["r_ank"]) / 180
    hip_l, hip_r = _angle(k["l_sho"], k["l_hip"], k["l_knee"]) / 180, _angle(k["r_sho"], k["r_hip"], k["r_knee"]) / 180
    sho_m, hip_m = (k["l_sho"] + k["r_sho"]) / 2, (k["l_hip"] + k["r_hip"]) / 2
    knee_m, ank_m = (k["l_knee"] + k["r_knee"]) / 2, (k["l_ank"] + k["r_ank"]) / 2
    up = sho_m - hip_m
    lean = np.clip(np.degrees(np.arctan2(np.abs(up[:, 0]), -up[:, 1] + 1e-6)) / 90, 0, 1)  # >90 deg = pose glitch
    depth = (ank_m[:, 1] - hip_m[:, 1]) / s
    hbk = (hip_m[:, 1] - knee_m[:, 1]) / s
    kot = (np.abs(k["l_knee"][:, 0] - k["l_ank"][:, 0]) + np.abs(k["r_knee"][:, 0] - k["r_ank"][:, 0])) / (2 * s)
    w = {n: np.abs(k[f"l_{n}"][:, 0] - k[f"r_{n}"][:, 0]) / s for n in ("knee", "ank", "hip", "sho")}
    ratio = np.clip(w["knee"] / np.maximum(w["ank"], 0.05), 0, 5)
    knee_min, hip_min = np.minimum(knee_l, knee_r), np.minimum(hip_l, hip_r)
    F = [knee_l, knee_r, knee_min, hip_l, hip_r, hip_min, lean, depth, hbk, kot,
         w["knee"], w["ank"], w["hip"], w["sho"], ratio,
         _vel(knee_min), _vel(hip_min), _vel(lean), _vel(depth)]
    return np.stack(F, 1).astype(np.float32)


class StreamingFeatures:
    """Causal, frame-by-frame version of process() for the live demo / pipeline (same fill rule, same geometry()).

    push(kpts_t) takes one raw YOLO frame (17,3; NaN rows = no person) and returns the frames that became final:
    a list of (frame_index, feats (19,), valid). Frame f is final once frame f + 1 + lookahead has arrived:
      * lookahead frames let a missing-keypoint gap be interpolated exactly like the offline fill when it closes
        within the lookahead (lookahead >= MAX_INTERP reproduces the offline rule exactly except at clip start);
        otherwise the last observed value is held (the offline rule for long gaps)
      * +1 frame so velocities use the same central difference as np.gradient offline
    Scale = running median of torso length over valid frames so far (offline: whole-clip median), capped to the
    last `scale_window` valid frames. flush() finalises the remaining frames at the end of a clip.
    """

    def __init__(self, lookahead=0, scale_window=1800, fixed_scale=None, scale_needs_observed=True):
        self.D, self.cap, self.fixed_scale = int(lookahead), int(scale_window), fixed_scale  # fixed_scale: parity tests only
        self.scale_needs_observed = scale_needs_observed
        self.raw, self.obs, self.valid, self.xy = [], [], [], []   # per frame
        self.last_obs = [None] * 17                                  # (frame, xy) of last observed, per keypoint
        self.torso_sorted, self.torso_fifo, self.torso_all = [], [], []
        self.n_feat = 0                                              # frames with features emitted

    # -- fill one frame f using frames up to self.t
    def _fill(self, f, horizon):
        k = self.raw[f]
        xy = np.full((17, 2), np.nan, np.float32)
        for j in range(17):
            if self.obs[f][j]:
                xy[j] = k[j, :2]
                continue
            nxt = next((g for g in range(f + 1, horizon + 1) if self.obs[g][j]), None)
            prev = self.last_obs[j]
            if prev is not None and nxt is not None and nxt - prev[0] - 1 <= MAX_INTERP:
                a = (f - prev[0]) / (nxt - prev[0])
                xy[j] = prev[1] + a * (self.raw[nxt][j, :2] - prev[1])
            elif prev is not None:
                xy[j] = prev[1]
            elif nxt is not None:
                xy[j] = self.raw[nxt][j, :2]
        for j in range(17):                                          # still missing: mirror, then core mean
            if np.isnan(xy[j, 0]) and MIRROR.get(j) is not None and not np.isnan(xy[MIRROR[j], 0]):
                xy[j] = xy[MIRROR[j]]
        if np.isnan(xy).any():
            core = xy[CORE][~np.isnan(xy[CORE, 0])]
            xy[np.isnan(xy[:, 0])] = core.mean(0) if len(core) else 0.0
        for j in range(17):
            if self.obs[f][j]:
                self.last_obs[j] = (f, k[j, :2].astype(np.float32))
        self.xy.append(xy)
        tl = float(np.linalg.norm(xy[[5, 6]].mean(0) - xy[[11, 12]].mean(0)))
        self.torso_all.append(tl)
        # scale samples only from frames where both shoulders AND hips were really observed (not filled in):
        # with leg-only close-ups / someone too near the camera, held shoulder positions give a bogus torso length
        if self.valid[f] and (not self.scale_needs_observed or all(self.obs[f][[5, 6, 11, 12]])):
            bisect.insort(self.torso_sorted, tl)
            self.torso_fifo.append(tl)
            if len(self.torso_fifo) > self.cap:
                self.torso_sorted.pop(bisect.bisect_left(self.torso_sorted, self.torso_fifo.pop(0)))

    def scale(self):
        if self.fixed_scale is not None:
            return float(self.fixed_scale)
        L = self.torso_sorted
        if not L:
            return float(np.median(self.torso_all))
        n = len(L)
        return L[n // 2] if n % 2 else 0.5 * (L[n // 2 - 1] + L[n // 2])

    def _emit(self, f, last):
        lo, hi = max(f - 1, 0), (f if last else f + 1)
        win = np.stack(self.xy[lo:hi + 1])
        feats = geometry(win, self.scale())[f - lo]
        self.n_feat += 1
        return f, feats, int(self.valid[f])

    def push(self, kpts):
        kpts = np.asarray(kpts, np.float32)
        self.raw.append(kpts)
        self.obs.append(np.nan_to_num(kpts[:, 2]) >= KP_CONF)
        self.valid.append(int(body_valid(kpts[None])[0]))
        t = len(self.raw) - 1
        out = []
        while len(self.xy) <= t - self.D:                          # fill frames whose lookahead is available
            self._fill(len(self.xy), t)
        while self.n_feat + 1 < len(self.xy):                      # features need the next filled frame
            out.append(self._emit(self.n_feat, last=False))
        return out

    def flush(self):
        t = len(self.raw) - 1
        while len(self.xy) <= t:
            self._fill(len(self.xy), t)
        out = []
        while self.n_feat < len(self.xy):
            out.append(self._emit(self.n_feat, last=self.n_feat == len(self.xy) - 1))
        return out


def process(clip_id):
    d = np.load(POSE_DIR / f"{clip_id}.npz")
    kpts = d["kpts"]
    xy, code, never = fill_keypoints(kpts)
    valid = body_valid(kpts)
    tl = torso_length(xy)
    ref = tl[valid.astype(bool)] if valid.any() else tl
    scale = float(np.median(ref))
    feats = geometry(xy, scale)
    fill = code[:, CORE].max(1)
    assert np.isfinite(feats).all(), clip_id
    GEOM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GEOM_DIR / f"{clip_id}.npz", feats=feats, names=np.array(FEATURE_NAMES), valid=valid,
                        fill=fill, kpts_filled=xy.astype(np.float32), scale=scale, frame_idx=d["frame_idx"],
                        t=d["t"])
    return len(feats), valid, fill, never


def main():
    clips = sorted({r["clip_id"] for r in csv.DictReader(open(LABELS_CSV))})
    T = 0
    inv = interp = held = 0
    nev = {}
    for c in clips:
        n, v, f, never = process(c)
        T += n
        inv += int((v == 0).sum())
        interp += int((f == 1).sum())
        held += int((f == 2).sum())
        if never:
            nev[c] = never
    print(f"{len(clips)} clips, {T} frames, {len(FEATURE_NAMES)} features -> {GEOM_DIR}")
    print(f"valid=0 (no person or lower-body conf < {BODY_CONF}): {inv} frames ({inv / T:.2%})")
    print(f"core keypoints filled: interpolated {interp} ({interp / T:.2%}), held {held} ({held / T:.2%}), "
          f"any {interp + held} ({(interp + held) / T:.2%})")
    print(f"keypoints never observed in a clip: {nev or 'none'}")


if __name__ == "__main__":
    main()
