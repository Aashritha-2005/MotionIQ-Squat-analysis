"""Task 2.4: merge pose, geometry, valid flag, DINOv2, ResNet-18 and labels into artifacts/train_data.npz.

Layout: all per-frame arrays are concatenated across clips in clip order (N frames total); clip k occupies
rows clip_start[k] : clip_start[k] + clip_len[k]. Sequence models should window within a clip only.

Per-frame (N rows):
  geom (N,19) f32, geom_names       valid (N,) i8 (1 = body visible, 0 = filled in)   fill (N,) i8 (0/1/2)
  kpts (N,17,3) f16 (raw YOLO x,y,conf; 0,0,0 where no person was detected)
  dino (N,384) f16                  resnet (N,512) f16
  phase (N,) i8  0 standing 1 descent 2 bottom 3 ascent (frames outside every rep = standing)
  frame_clip (N,) i16 index into the clip table   frame_idx (N,) i32   t (N,) f32
  frame_rep (N,) i16  rep index within the clip, -1 outside reps
  frame_split (N,) i8 0 train 1 heldout
  frame_faults (N,5) i8  [knee_valgus, shallow_depth, forward_lean, other, unclear] of the frame's rep:
                         1 / 0, -1 = n/a (not observable) or excl, -2 = outside any rep
Per clip: clip_id, subject_id, view, quality, split, clip_start, clip_len
Per rep:  rep_clip, rep_idx, rep_start, rep_end (frame, inclusive), rep_faults (R,5) same coding, rep_tag_source
Names: phase_names, fault_names, split_names
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS, DATA, FPS, LABELS_CSV, PHASES, POSE_DIR  # noqa: E402

FAULT_COLS = ["knee_valgus", "shallow_depth", "forward_lean", "other", "unclear"]
SPLITS = ["train", "heldout"]
OUT = ARTIFACTS / "train_data.npz"


def code(v):
    return {"1": 1, "0": 0}.get(v, -1)  # n/a, excl -> -1


def rep_frames(r):
    """labels.csv seconds -> (ds, bs, be, ae) frame indices; inverse of prepare_labels.fmt (ae stored +1)."""
    f = lambda k: int(round(float(r[k]) * FPS))  # noqa: E731
    return f("descent_start_s"), f("bottom_start_s"), f("bottom_end_s"), f("ascent_end_s") - 1


def main():
    rows = list(csv.DictReader(open(LABELS_CSV)))
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    clips = sorted(by_clip)
    P = {k: [] for k in ["geom", "valid", "fill", "kpts", "dino", "resnet", "phase", "frame_clip", "frame_idx",
                         "t", "frame_rep", "frame_split", "frame_faults"]}
    C = {k: [] for k in ["clip_id", "subject_id", "view", "quality", "split", "clip_start", "clip_len"]}
    R = {k: [] for k in ["rep_clip", "rep_idx", "rep_start", "rep_end", "rep_faults", "rep_tag_source"]}
    start = 0
    for ci, c in enumerate(clips):
        pose = np.load(POSE_DIR / f"{c}.npz")
        geo = np.load(DATA / "geom_features" / f"{c}.npz")
        dino = np.load(DATA / "dino_features" / f"{c}.npz")
        rn = np.load(DATA / "resnet_features" / f"{c}.npz")
        T = len(pose["kpts"])
        for name, arr in [("geom", geo["feats"]), ("dino", dino["emb"]), ("resnet", rn["emb"])]:
            assert len(arr) == T, f"{c}: {name} has {len(arr)} frames, pose has {T}"
        assert (dino["src_frame_idx"] == pose["src_frame_idx"]).all() and (rn["src_frame_idx"] == pose["src_frame_idx"]).all()
        r0 = by_clip[c][0]
        split = SPLITS.index(r0["split"])
        phase = np.zeros(T, np.int8)
        frep = np.full(T, -1, np.int16)
        ffault = np.full((T, 5), -2, np.int8)
        for r in sorted(by_clip[c], key=lambda r: int(r["rep_idx"])):
            if r["rep_idx"] == "-1":
                continue
            ds, bs, be, ae = rep_frames(r)
            ae = min(ae, T - 1)
            phase[ds:bs], phase[bs:be + 1], phase[be + 1:ae + 1] = 1, 2, 3
            fv = [code(r[k]) for k in FAULT_COLS]
            frep[ds:ae + 1], ffault[ds:ae + 1] = int(r["rep_idx"]), fv
            for k, v in zip(R, [ci, int(r["rep_idx"]), ds, ae, fv, r["tag_source"]]):
                R[k].append(v)
        P["geom"].append(geo["feats"]); P["valid"].append(geo["valid"]); P["fill"].append(geo["fill"])
        P["kpts"].append(np.nan_to_num(pose["kpts"]).astype(np.float16))
        P["dino"].append(dino["emb"]); P["resnet"].append(rn["emb"])
        P["phase"].append(phase); P["frame_clip"].append(np.full(T, ci, np.int16))
        P["frame_idx"].append(pose["frame_idx"].astype(np.int32)); P["t"].append(pose["t"].astype(np.float32))
        P["frame_rep"].append(frep); P["frame_split"].append(np.full(T, split, np.int8))
        P["frame_faults"].append(ffault)
        for k, v in zip(C, [c, r0["subject_id"], r0["view"], r0["quality"], r0["split"], start, T]):
            C[k].append(v)
        start += T
    out = {k: np.concatenate(v) for k, v in P.items()}
    out.update({k: np.array(v) for k, v in C.items()})
    out.update({k: np.array(v) for k, v in R.items()})
    out.update(geom_names=np.load(DATA / "geom_features" / f"{clips[0]}.npz")["names"],
               phase_names=np.array(PHASES), fault_names=np.array(FAULT_COLS), split_names=np.array(SPLITS),
               fps=np.array(FPS))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)  {len(clips)} clips, {start} frames, {len(R['rep_clip'])} reps")


if __name__ == "__main__":
    main()
