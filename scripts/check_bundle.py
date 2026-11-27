"""Task 2.5: sanity checks on artifacts/train_data.npz. Exits non-zero on any failed assertion.

Checks: shapes; per-clip frame counts identical across pose / geometry / DINOv2 / ResNet-18 / labels (from the
per-clip source files, not just the bundle); no NaN/inf anywhere in the bundle; label alignment (bundle
phase == phases rebuilt from labels.csv, rep frames inside the clip); subject-disjoint split; phase balance
per split; and frame-alignment spot-checks: for random frames, re-decode frame i of the video, re-run DINOv2,
ResNet-18 and YOLO on it, and compare with stored row i (and with rows i +- 5 as a control).

Usage: .venv/bin/python scripts/check_bundle.py [--spot 6]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS, DATA, LABELS_CSV, POSE_DIR, POSE_MODEL, RAW_DIR  # noqa: E402
from extract_embeddings import load_models, preprocess  # noqa: E402

B = ARTIFACTS / "train_data.npz"


def cos(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", type=int, default=6)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    d = dict(np.load(B))
    N = len(d["phase"])
    print(f"== {B} ({B.stat().st_size / 1e6:.1f} MB), {N} frames, {len(d['clip_id'])} clips, {len(d['rep_clip'])} reps")
    for k in sorted(d):
        print(f"  {k:15s} {str(d[k].shape):16s} {d[k].dtype}")

    # 1. no NaN / inf
    for k, v in d.items():
        if v.dtype.kind == "f":
            assert np.isfinite(v).all(), f"non-finite values in {k}"
    print("OK  no NaN/inf in any float array")

    # 2. per-frame arrays all length N; per-clip counts identical across sources
    for k in ["geom", "valid", "fill", "kpts", "dino", "resnet", "phase", "frame_clip", "frame_idx", "t",
              "frame_rep", "frame_split", "frame_faults"]:
        assert len(d[k]) == N, k
    labels = list(csv.DictReader(open(LABELS_CSV)))
    lab_n = {r["clip_id"]: int(r["n_frames"]) for r in labels}
    for ci, c in enumerate(d["clip_id"]):
        s, T = int(d["clip_start"][ci]), int(d["clip_len"][ci])
        counts = dict(pose=len(np.load(POSE_DIR / f"{c}.npz")["kpts"]),
                      geom=len(np.load(DATA / "geom_features" / f"{c}.npz")["feats"]),
                      dino=len(np.load(DATA / "dino_features" / f"{c}.npz")["emb"]),
                      resnet=len(np.load(DATA / "resnet_features" / f"{c}.npz")["emb"]),
                      labels=lab_n[c], bundle=T)
        assert len(set(counts.values())) == 1, (c, counts)
        assert (d["frame_clip"][s:s + T] == ci).all() and (d["frame_idx"][s:s + T] == np.arange(T)).all(), c
    assert d["clip_start"][-1] + d["clip_len"][-1] == N
    print(f"OK  frame counts identical across pose/geometry/DINOv2/ResNet-18/labels for all {len(d['clip_id'])} clips")

    # 3. label alignment
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_training_bundle import rep_frames
    from prepare_labels import phase_per_frame
    for ci, c in enumerate(d["clip_id"]):
        s, T = int(d["clip_start"][ci]), int(d["clip_len"][ci])
        reps = [rep_frames(r) for r in labels if r["clip_id"] == c and r["rep_idx"] != "-1"]
        assert all(0 <= ds <= bs <= be <= ae < T for ds, bs, be, ae in reps), c
        assert (phase_per_frame(T, reps) == d["phase"][s:s + T]).all(), c
    n_rep_rows = sum(r["rep_idx"] != "-1" for r in labels)
    assert len(d["rep_clip"]) == n_rep_rows
    inside = d["frame_rep"] >= 0
    assert ((d["phase"] > 0) <= inside).all() and (d["frame_faults"][~inside] == -2).all()
    print(f"OK  phases match labels.csv rep boundaries for every clip; {n_rep_rows} reps; non-standing frames all inside a rep")

    # 4. split
    subj = {}
    for ci, c in enumerate(d["clip_id"]):
        subj.setdefault(d["subject_id"][ci], set()).add(d["split"][ci])
        s, T = int(d["clip_start"][ci]), int(d["clip_len"][ci])
        assert (d["frame_split"][s:s + T] == list(d["split_names"]).index(d["split"][ci])).all()
    tr = {s for s, v in subj.items() if v == {"train"}}
    ho = {s for s, v in subj.items() if v == {"heldout"}}
    assert len(tr) + len(ho) == len(subj) and not tr & ho
    print(f"OK  subjects disjoint: train {len(tr)}, held-out {len(ho)} {sorted(ho)}")

    # 5. phase balance + valid share per split
    print("    split      frames   " + "  ".join(f"{p:>15s}" for p in d["phase_names"]) + "   valid=0")
    for si, sn in enumerate(d["split_names"]):
        m = d["frame_split"] == si
        cnt = np.bincount(d["phase"][m], minlength=4)
        print(f"    {sn:8s} {m.sum():8d}   " + "  ".join(f"{c:7d} ({c / m.sum():5.1%})" for c in cnt)
              + f"   {(d['valid'][m] == 0).mean():.2%}")

    # 6. frame-alignment spot checks
    from ultralytics import YOLO
    dino, rn = load_models(a.device)
    yolo = YOLO(POSE_MODEL)
    rng = np.random.default_rng(0)
    print(f"    spot-checks: re-decode frame i, recompute; cos(stored i, recomputed) vs best of stored i+-5")
    for ci in rng.choice(len(d["clip_id"]), a.spot, replace=False):
        c, s, T = d["clip_id"][ci], int(d["clip_start"][ci]), int(d["clip_len"][ci])
        i = int(rng.integers(5, T - 5))
        src = int(np.load(POSE_DIR / f"{c}.npz")["src_frame_idx"][i])
        cap = cv2.VideoCapture(str(RAW_DIR / f"{c}.mp4"))
        for _ in range(src + 1):  # sequential decode, same as extraction (no seek drift)
            ok, fr = cap.read()
        cap.release()
        with torch.no_grad():
            x = torch.from_numpy(preprocess(fr)[None]).to(a.device)
            dv = dino(pixel_values=x).last_hidden_state[0, 0].float().cpu().numpy()
            rv = rn(x)[0].float().cpu().numpy()
        cd, cr = cos(d["dino"][s + i], dv), cos(d["resnet"][s + i], rv)
        nd = max(cos(d["dino"][s + j], dv) for j in (i - 5, i + 5))
        nr = max(cos(d["resnet"][s + j], rv) for j in (i - 5, i + 5))
        r = yolo(fr, verbose=False, device=a.device)[0]
        kp_err = np.nan
        if len(r.boxes):
            st = d["kpts"][s + i].astype(np.float32)
            k = int(np.argmin([np.nanmean(np.abs(r.keypoints.xy[j].cpu().numpy() - st[:, :2])) for j in range(len(r.boxes))]))
            kp_err = float(np.mean(np.abs(r.keypoints.xy[k].cpu().numpy() - st[:, :2])[st[:, 2] > 0.3]))
        print(f"    {c:40s} i={i:4d}  dino {cd:.4f} (i+-5 {nd:.3f})  resnet {cr:.4f} (i+-5 {nr:.3f})  "
              f"kpt |dxy| {kp_err:.2f}px")
        assert cd > 0.99 and cr > 0.99 and cd > nd and cr > nr, c
        assert kp_err < 3.0, c
    print("OK  stored row i matches video frame i (embeddings and keypoints)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
