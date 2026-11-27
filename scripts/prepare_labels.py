"""Task 1.3 automatic passes: filename labels, auto phase boundaries, subject-wise split.

Usage:
    .venv/bin/python scripts/prepare_labels.py            # pose (cached) + phases + split + labels.csv
    .venv/bin/python scripts/prepare_labels.py --spotcheck 8

Writes data/labels.csv with ONE ROW PER REP (clips with no detected rep get one row with rep_idx=-1).
Fault tags already entered by the user (tag_source=user) are preserved on re-runs.
"""
import argparse
import csv
import random
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import (DATA, FAULTS, FPS, KP, LABELS_CSV, OBSERVABLE, POSE_DIR, POSE_MODEL,  # noqa: E402
                    RAW_DIR, SPOTCHECK_DIR)

NAME_RE = re.compile(r"^subject_(\d{3})_squat_(good|bad)_(front|side|diagonal)\.mp4$", re.I)
COLUMNS = ["clip_id", "filename", "subject_id", "view", "quality", "split", "n_frames", "duration_s",
           "rep_idx", "descent_start_s", "bottom_start_s", "bottom_end_s", "ascent_end_s", "phase_source",
           *FAULTS, "other", "unclear", "body_unclear_frac", "form_tags", "tag_source", "notes"]
HELDOUT_FRAC = 0.2
BLUR_DIR = DATA / "blur"
BLUR_CONF_THR = 0.5   # frame is body-unclear if mean hip/knee/ankle conf < this (see blur_audit.py)
BLUR_REP_THR = 0.5    # rep is mostly blurred if > this fraction of its frames are body-unclear
SPLIT_SEED = 42


# ---------------------------------------------------------------- pass a: filenames
def parse_clips():
    clips = []
    for p in sorted(RAW_DIR.glob("*.mp4")):
        m = NAME_RE.match(p.name)
        if not m:
            print(f"  ! skipping unparseable filename {p.name}")
            continue
        clips.append(dict(clip_id=p.stem, filename=p.name, path=p, subject_id=f"subject_{m[1]}",
                          quality=m[2].lower(), view=m[3].lower()))
    return clips


# ---------------------------------------------------------------- pose at 15 fps (cached)
def pick_person(result):
    """Index of the main subject: largest box area x confidence (gyms have bystanders)."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    b = result.boxes
    wh = (b.xyxy[:, 2:] - b.xyxy[:, :2]).cpu().numpy()
    score = wh[:, 0] * wh[:, 1] * b.conf.cpu().numpy()
    return int(score.argmax())


def extract_pose(clip, model, device):
    out = POSE_DIR / f"{clip['clip_id']}.npz"
    if out.exists():
        return np.load(out)
    cap = cv2.VideoCapture(str(clip["path"]))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_src / src_fps
    targets = [int(round(k / FPS * src_fps)) for k in range(int(duration * FPS))]
    targets = [t for t in targets if t < n_src]
    want, frames, src_idx = set(targets), [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            frames.append(fr)
            src_idx.append(i)
        i += 1
    cap.release()
    kpts = np.full((len(frames), 17, 3), np.nan, np.float32)
    for s in range(0, len(frames), 16):
        for j, r in enumerate(model(frames[s:s + 16], verbose=False, device=device)):
            k = pick_person(r)
            if k is not None:
                xy = r.keypoints.xy[k].cpu().numpy()
                cf = r.keypoints.conf[k].cpu().numpy()
                kpts[s + j, :, :2], kpts[s + j, :, 2] = xy, cf
    POSE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, kpts=kpts, frame_idx=np.arange(len(frames)), src_frame_idx=np.array(src_idx),
                        t=np.arange(len(frames)) / FPS, src_fps=src_fps, width=w, height=h)
    return np.load(out)


# ---------------------------------------------------------------- auto phase boundaries
def _side_mean(k, a, b):
    """Confidence-weighted mean of the left/right keypoint y-coordinate."""
    ya, yb, ca, cb = k[:, a, 1], k[:, b, 1], np.nan_to_num(k[:, a, 2]), np.nan_to_num(k[:, b, 2])
    wsum = ca + cb
    y = (np.nan_to_num(ya) * ca + np.nan_to_num(yb) * cb) / np.where(wsum > 0, wsum, np.nan)
    y[wsum < 0.3] = np.nan
    return y


def _interp(x):
    x = x.copy()
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return None
    x[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), x[ok])
    return x


def _smooth(x, med=5, avg=3):
    pad = med // 2
    xp = np.pad(x, pad, mode="edge")
    x = np.array([np.median(xp[i:i + med]) for i in range(len(x))])
    return np.convolve(np.pad(x, avg // 2, mode="edge"), np.ones(avg) / avg, mode="valid")


def depth_signal(kpts):
    """0 = standing tall, larger = hips lower. Hip height above ankles, normalised by standing height.
    View-independent (works front/side/diagonal), unlike a 2D knee angle which foreshortens in front view."""
    hip = _side_mean(kpts, KP["l_hip"], KP["r_hip"])
    ank = _side_mean(kpts, KP["l_ank"], KP["r_ank"])
    sho = _side_mean(kpts, KP["l_sho"], KP["r_sho"])
    hip, ank, sho = _interp(hip), _interp(ank), _interp(sho)
    if hip is None or ank is None or sho is None:
        return None
    body = np.nanpercentile(ank - sho, 90)
    h = (ank - hip) / max(body, 1e-6)
    return _smooth(1.0 - h / np.nanpercentile(h, 90))


# 1.3c fix: on these side-view clips the hip-height signal splits/garbles reps (pose glitch spikes on
# 001; quarter-depth reps + camera drift on 012). The 2D knee angle is reliable on side view, so these
# clips use it instead. Applied only to clips the user flagged with 'x' and verified visually.
KNEE_SIGNAL_CLIPS = {"subject_001_squat_bad_side", "subject_012_squat_bad_side"}


def knee_signal(kpts):
    """0 = straight legs, larger = more knee flexion: (180 - confidence-weighted L/R knee angle) / 180."""
    num = den = 0.0
    for h, k, a in [(KP["l_hip"], KP["l_knee"], KP["l_ank"]), (KP["r_hip"], KP["r_knee"], KP["r_ank"])]:
        v1, v2 = kpts[:, h, :2] - kpts[:, k, :2], kpts[:, a, :2] - kpts[:, k, :2]
        cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        w = np.nan_to_num(np.minimum.reduce([kpts[:, h, 2], kpts[:, k, 2], kpts[:, a, 2]]))
        num, den = num + np.nan_to_num(ang, nan=180.0) * w, den + w
    return _smooth((180 - num / (den + 1e-6)) / 180)


def detect_reps(s, min_gap_s=0.6, win_s=2.0, rel_prom=0.35, abs_prom=0.05):
    """Local-prominence rep detector on the depth signal (robust to baseline drift and glitch spikes,
    unlike a global threshold). Each rep = one depth peak that stands out from the minima on both
    sides within +-win_s; rep boundaries sit at the minimum between neighbouring peaks.
    Returns list of (descent_start, bottom_start, bottom_end, ascent_end) frame indices."""
    n, W, G = len(s), int(win_s * FPS), int(min_gap_s * FPS)
    cand = [i for i in range(1, n - 1) if s[i] >= s[max(0, i - G // 2):i + G // 2 + 1].max()]
    prom = {i: s[i] - max(s[max(0, i - W):i + 1].min(), s[i:i + W + 1].min()) for i in cand}
    strong = [p for p in prom.values() if p >= abs_prom]
    if not strong:
        return []
    thr = max(abs_prom, rel_prom * float(np.median(strong)))
    peaks = []
    for i in sorted(c for c in cand if prom[c] >= thr):
        if peaks and i - peaks[-1] < G:
            if s[i] > s[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    reps = []
    for k, pk in enumerate(peaks):
        a = max(0, pk - W) if k == 0 else peaks[k - 1]
        b = min(n - 1, pk + W) if k == len(peaks) - 1 else peaks[k + 1]
        a = max(a, pk - W) + int(np.argmin(s[max(a, pk - W):pk + 1]))
        b = pk + int(np.argmin(s[pk:min(b, pk + W) + 1]))
        peak = s[pk]
        amp_l, amp_r = peak - s[a], peak - s[b]
        ds = a
        while ds < pk and s[ds + 1] <= s[a] + 0.1 * amp_l:
            ds += 1
        ae = b
        while ae > pk and s[ae - 1] <= s[b] + 0.1 * amp_r:
            ae -= 1
        bs = pk
        while bs > ds and s[bs - 1] >= peak - 0.15 * amp_l:
            bs -= 1
        be = pk
        while be < ae and s[be + 1] >= peak - 0.15 * amp_r:
            be += 1
        reps.append((ds, bs, be, ae))
    return reps


def phase_per_frame(n, reps):
    ph = np.zeros(n, np.int8)  # 0 standing
    for ds, bs, be, ae in reps:
        ph[ds:bs], ph[bs:be + 1], ph[be + 1:ae + 1] = 1, 2, 3
    return ph


# ---------------------------------------------------------------- split
SPLIT_OVERRIDE = {"subject_001": "train", "subject_010": "heldout"}


def propose_split(subjects):
    subs = sorted(subjects)
    rng = random.Random(SPLIT_SEED)
    held = set(rng.sample(subs, max(1, round(HELDOUT_FRAC * len(subs)))))
    # 1.3c decision (user): swap 001 -> train, 010 -> held-out so forward_lean has >=15 held-out reps
    for s, side in SPLIT_OVERRIDE.items():
        (held.add if side == "heldout" else held.discard)(s)
    return {s: ("heldout" if s in held else "train") for s in subs}


# ---------------------------------------------------------------- spot-check image
PH_COL = [(200, 200, 200), (80, 160, 255), (60, 60, 230), (90, 200, 90)]  # BGR: standing, descent, bottom, ascent


def spotcheck_png(clip, s, reps, out):
    W, H = 1200, 260
    img = np.full((H + 210, W, 3), 255, np.uint8)
    ph = phase_per_frame(len(s), reps)
    xs = np.linspace(0, W - 1, len(s)).astype(int)
    for i in range(len(s) - 1):
        cv2.rectangle(img, (xs[i], 0), (xs[i + 1], H), PH_COL[ph[i]], -1)
    lo, hi = s.min(), s.max() + 1e-6
    ys = (H - 10 - (s - lo) / (hi - lo) * (H - 20)).astype(int)
    cv2.polylines(img, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, (0, 0, 0), 2)
    cv2.putText(img, f"{clip['clip_id']}  reps={len(reps)}  (grey=standing orange=descent red=bottom green=ascent)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    if reps:  # thumbnails of rep 1 at each phase
        ds, bs, be, ae = reps[0]
        cap = cv2.VideoCapture(str(clip["path"]))
        src = np.load(POSE_DIR / f"{clip['clip_id']}.npz")["src_frame_idx"]
        for j, (lab, f) in enumerate([("standing", max(ds - 3, 0)), ("descent", (ds + bs) // 2),
                                      ("bottom", (bs + be) // 2), ("ascent", (be + ae) // 2)]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(src[f]))
            ok, fr = cap.read()
            if ok:
                th = cv2.resize(fr, (int(200 * fr.shape[1] / fr.shape[0]), 200))
                x0 = 10 + j * 300
                img[H + 5:H + 205, x0:x0 + th.shape[1]] = th[:, :min(th.shape[1], W - x0)]
                cv2.putText(img, f"rep1 {lab}", (x0, H + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cap.release()
    cv2.imwrite(str(out), img)


# ---------------------------------------------------------------- main
def fmt(f):
    return f"{f / FPS:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spotcheck", type=int, default=8, help="number of sample clips to render")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    clips = parse_clips()
    print(f"[a] {len(clips)} clips parsed from filenames")
    from ultralytics import YOLO
    model = YOLO(POSE_MODEL)

    prev = {}
    if LABELS_CSV.exists():
        with open(LABELS_CSV) as fh:
            for r in csv.DictReader(fh):
                if r["tag_source"] == "user":
                    prev[(r["clip_id"], r["rep_idx"])] = r

    split = propose_split({c["subject_id"] for c in clips})
    rows, per_clip = [], {}
    for n, c in enumerate(clips, 1):
        d = extract_pose(c, model, args.device)
        kp = d["kpts"]
        knee = c["clip_id"] in KNEE_SIGNAL_CLIPS
        s = knee_signal(kp) if knee else depth_signal(kp)
        reps = detect_reps(s) if s is not None else []
        miss = float(np.isnan(kp[:, 0, 0]).mean())
        per_clip[c["clip_id"]] = (c, s, reps)
        print(f"  [{n:3d}/{len(clips)}] {c['clip_id']:40s} frames={len(kp):4d} reps={len(reps):2d} no-person={miss:.0%}")
        base = dict(clip_id=c["clip_id"], filename=c["filename"], subject_id=c["subject_id"], view=c["view"],
                    quality=c["quality"], split=split[c["subject_id"]], n_frames=len(kp),
                    duration_s=f"{len(kp) / FPS:.2f}", phase_source="auto_knee" if knee else "auto")
        notes = "" if reps else ("no person / pose failed" if s is None else "no rep detected")
        if miss > 0.15:
            notes = ";".join(x for x in (notes, f"low_pose_coverage_{miss:.0%}") if x)
        bpath = BLUR_DIR / f"{c['clip_id']}.npz"  # from scripts/blur_audit.py
        body_unclear = np.load(bpath)["lower_conf"] < BLUR_CONF_THR if bpath.exists() else None
        for ri, rep in enumerate(reps or [None]):
            r = dict(base, rep_idx=ri if rep else -1, notes=notes)
            frac = None
            if rep:
                r.update(descent_start_s=fmt(rep[0]), bottom_start_s=fmt(rep[1]),
                         bottom_end_s=fmt(rep[2]), ascent_end_s=fmt(rep[3] + 1))
                if body_unclear is not None:
                    frac = float(body_unclear[rep[0]:rep[3] + 1].mean())
                    r["body_unclear_frac"] = f"{frac:.2f}"
            if c["quality"] == "good" and frac is not None and frac > BLUR_REP_THR:
                # user only tags bad clips, so mostly-blurred reps in good clips are auto-marked unclear
                for f in FAULTS:
                    r[f] = "excl" if c["view"] in OBSERVABLE[f] else "n/a"
                r.update(other="0", unclear="1", form_tags="unclear", tag_source="auto_blur")
            elif c["quality"] == "good":  # dataset says good: no faults on views where they are observable
                for f in FAULTS:
                    r[f] = "0" if c["view"] in OBSERVABLE[f] else "n/a"
                r.update(other="0", unclear="0", form_tags="good", tag_source="dataset")
            else:
                for f in FAULTS:
                    r[f] = "" if c["view"] in OBSERVABLE[f] else "n/a"
                r.update(other="", unclear="", form_tags="", tag_source="pending")
            old = prev.get((r["clip_id"], str(r["rep_idx"])))
            if old:
                for f in [*FAULTS, "other", "unclear", "form_tags", "tag_source"]:
                    r[f] = old.get(f, "")
            rows.append(r)

    with open(LABELS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[wrote] {LABELS_CSV} ({len(rows)} rows)")

    # spot-check sample: one per view x quality, then random fill
    SPOTCHECK_DIR.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(0)
    pick = []
    for v in ("front", "side", "diagonal"):
        for q in ("good", "bad"):
            cand = [k for k, (c, _, _) in per_clip.items() if c["view"] == v and c["quality"] == q]
            if cand:
                pick.append(rnd.choice(cand))
    rest = [k for k in per_clip if k not in pick]
    pick += rnd.sample(rest, max(0, min(len(rest), args.spotcheck - len(pick))))
    for k in pick:
        c, s, reps = per_clip[k]
        if s is not None:
            spotcheck_png(c, s, reps, SPOTCHECK_DIR / f"{k}.png")
    print(f"[spotcheck] {len(pick)} images in {SPOTCHECK_DIR}")

    held = sorted(s for s, v in split.items() if v == "heldout")
    print(f"[split] train={len(split) - len(held)} subjects, heldout={len(held)}: {held}")


if __name__ == "__main__":
    main()
