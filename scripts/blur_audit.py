"""Data-quality audit: how often does the dataset's privacy blur cover the body (not just the face)?

Per 15 fps frame (same grid as data/pose/), using the cached YOLO keypoints:
  body box   = bounding box of shoulders/hips/knees/ankles with conf > 0.3, padded 10%
  sharp_ratio = Laplacian variance inside body box / Laplacian variance of the whole frame
               (normalises away camera/resolution/compression differences between clips)
  lower_conf  = mean keypoint confidence of hips, knees, ankles
Writes data/blur/<clip>.npz (per-frame arrays) and data/blur_audit.csv (per-clip summary).

Finding (calibrated on subject_001_squat_bad_diagonal): when the blur box covers the body, YOLO loses the
lower body (conf ~0.0-0.05 vs ~1.0 when visible), and the body box disappears so sharp_ratio is NaN. So
'body unclear' = lower_conf < 0.5. sharp_ratio is kept for reference only (did not discriminate).

Usage: .venv/bin/python scripts/blur_audit.py [--conf-thr 0.5] [--rep-thr 0.5]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, FPS, LABELS_CSV, POSE_DIR, RAW_DIR  # noqa: E402

BLUR_DIR = DATA / "blur"
BODY = [5, 6, 11, 12, 13, 14, 15, 16]
LOWER = [11, 12, 13, 14, 15, 16]


def lapvar(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray.size else np.nan


def frame_metrics(clip_id):
    out = BLUR_DIR / f"{clip_id}.npz"
    if out.exists():
        return dict(np.load(out))
    d = np.load(POSE_DIR / f"{clip_id}.npz")
    kp, src = d["kpts"], d["src_frame_idx"]
    want = {int(s): i for i, s in enumerate(src)}
    ratio = np.full(len(kp), np.nan)
    lower = np.nanmean(np.where(np.isnan(kp[:, LOWER, 2]), 0, kp[:, LOWER, 2]), axis=1)
    cap, i = cv2.VideoCapture(str(RAW_DIR / f"{clip_id}.mp4")), 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        j = want.get(i)
        i += 1
        if j is None:
            continue
        k = kp[j, BODY]
        k = k[np.nan_to_num(k[:, 2]) > 0.3]
        if len(k) < 4:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        H, W = g.shape
        x0, y0 = k[:, :2].min(0)
        x1, y1 = k[:, :2].max(0)
        px, py = 0.1 * (x1 - x0) + 5, 0.1 * (y1 - y0) + 5
        x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
        x1, y1 = int(min(W, x1 + px)), int(min(H, y1 + py))
        full = lapvar(g)
        if (x1 - x0) * (y1 - y0) > 400 and full > 0:
            ratio[j] = lapvar(g[y0:y1, x0:x1]) / full
    cap.release()
    BLUR_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, sharp_ratio=ratio, lower_conf=lower)
    return dict(sharp_ratio=ratio, lower_conf=lower)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf-thr", type=float, default=0.5, help="frame is 'body unclear' below this lower-body conf")
    ap.add_argument("--rep-thr", type=float, default=0.5, help="rep is 'mostly unclear' above this fraction")
    a = ap.parse_args()
    labels = list(csv.DictReader(open(LABELS_CSV)))
    reps_by_clip = {}
    for r in labels:
        reps_by_clip.setdefault(r["clip_id"], []).append(r)
    rows = []
    for p in sorted(RAW_DIR.glob("*.mp4")):
        m = frame_metrics(p.stem)
        unclear = m["lower_conf"] < a.conf_thr
        lab = reps_by_clip[p.stem]
        reps = [r for r in lab if r["rep_idx"] != "-1"]
        frac = [unclear[int(round(float(r["descent_start_s"]) * FPS)):int(round(float(r["ascent_end_s"]) * FPS)) + 1].mean()
                for r in reps]
        rows.append(dict(clip_id=p.stem, subject_id=lab[0]["subject_id"], view=lab[0]["view"],
                         quality=lab[0]["quality"], split=lab[0]["split"], frames=len(unclear),
                         pct_frames_body_unclear=round(100 * float(unclear.mean()), 1),
                         reps=len(reps), reps_mostly_unclear=int(sum(f > a.rep_thr for f in frac)),
                         median_sharp_ratio=round(float(np.nanmedian(m["sharp_ratio"])), 3)
                         if np.isfinite(m["sharp_ratio"]).any() else ""))
    with open(DATA / "blur_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {DATA / 'blur_audit.csv'} ({len(rows)} clips)")
    P = np.array([r["pct_frames_body_unclear"] for r in rows])
    print(f"all frames: {np.average(P, weights=[r['frames'] for r in rows]):.1f}% body-unclear; "
          f"clip % percentiles 50/75/90: {np.round(np.percentile(P, [50, 75, 90]), 1)}")
    print("view     qual  clips  mean%  >=20%  >=30%   reps  reps>50%unclear")
    for v in ("front", "side", "diagonal"):
        for q in ("good", "bad"):
            g = [r for r in rows if r["view"] == v and r["quality"] == q]
            print(f"{v:8s} {q:4s} {len(g):6d} {np.mean([r['pct_frames_body_unclear'] for r in g]):6.1f} "
                  f"{sum(r['pct_frames_body_unclear'] >= 20 for r in g):6d} {sum(r['pct_frames_body_unclear'] >= 30 for r in g):6d} "
                  f"{sum(r['reps'] for r in g):6d} {sum(r['reps_mostly_unclear'] for r in g):8d}")


if __name__ == "__main__":
    main()
