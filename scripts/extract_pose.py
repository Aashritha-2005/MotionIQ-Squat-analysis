"""Task 2.1: YOLO11s-pose keypoints on the 15 fps grid for every clip in data/labels.csv.

Output per clip, data/pose/<clip>.npz:
  kpts          (T, 17, 3) float32  COCO-17 x, y (pixels) and confidence; NaN rows = no person detected
  frame_idx     (T,)  index on the 15 fps grid (0..T-1)
  src_frame_idx (T,)  index of the decoded source-video frame used for that grid step
  t             (T,)  timestamp in seconds (frame_idx / FPS)
  src_fps, width, height
The resampling + inference code lives in prepare_labels.extract_pose (written in Phase 1 for the auto phase
boundaries); results are cached, so this script reuses existing files and only computes missing ones, then
validates every cache file against the 2.1 contract.

Usage: .venv/bin/python scripts/extract_pose.py [--device mps]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FPS, LABELS_CSV, POSE_DIR, POSE_MODEL, RAW_DIR  # noqa: E402
from prepare_labels import extract_pose  # noqa: E402


def validate(clip_id):
    d = np.load(POSE_DIR / f"{clip_id}.npz")
    T = len(d["kpts"])
    assert d["kpts"].shape == (T, 17, 3), clip_id
    assert (d["frame_idx"] == np.arange(T)).all(), clip_id
    assert np.allclose(d["t"], np.arange(T) / FPS), clip_id
    n_src = int(cv2.VideoCapture(str(RAW_DIR / f"{clip_id}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    assert len(d["src_frame_idx"]) == T and d["src_frame_idx"].max() < n_src, clip_id
    return T, float(d["src_fps"]), float(np.isnan(d["kpts"][:, 0, 0]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    clips = sorted({r["clip_id"] for r in csv.DictReader(open(LABELS_CSV))})
    todo = [c for c in clips if not (POSE_DIR / f"{c}.npz").exists()]
    if todo:
        from ultralytics import YOLO
        model = YOLO(POSE_MODEL)
        for c in todo:
            extract_pose(dict(clip_id=c, path=RAW_DIR / f"{c}.mp4"), model, a.device)
    stats = [validate(c) for c in clips]
    print(f"{len(clips)} clips ({len(todo)} newly extracted, {len(clips) - len(todo)} from cache), "
          f"{sum(s[0] for s in stats)} frames, source fps {sorted({s[1] for s in stats})}, "
          f"no-person frames {np.mean([s[2] for s in stats]):.2%} (mean per clip)")


if __name__ == "__main__":
    main()
