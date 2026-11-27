"""Tasks 2.3 / 2.3b: frozen visual embeddings on exactly the pose frames (data/pose/<clip>.npz src_frame_idx).

  DINOv2-small (HF transformers, facebook/dinov2-small): CLS token of last_hidden_state -> 384-d
  ResNet-18 (torchvision, ImageNet1K_V1 weights): global-average-pooled penultimate layer -> 512-d
Both frozen, eval mode, on MPS. Saved as float16 to data/dino_features/<clip>.npz and
data/resnet_features/<clip>.npz with the matching frame_idx/src_frame_idx. Resumable (skips finished clips).

Preprocessing (shared with the live demo via preprocess()): BGR->RGB, pad to a square with the ImageNet
mean colour (portrait clips would lose head/feet under the usual centre crop), resize to 224x224,
ImageNet mean/std normalisation.

Usage: .venv/bin/python scripts/extract_embeddings.py [--device mps] [--batch 64]
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, LABELS_CSV, POSE_DIR, RAW_DIR  # noqa: E402

DINO_DIR, RESNET_DIR = DATA / "dino_features", DATA / "resnet_features"
DINO_NAME = "facebook/dinov2-small"
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(frame_bgr):
    """BGR uint8 frame -> normalised float32 CHW 224x224 (full frame kept by square padding)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.empty((s, s, 3), np.uint8)
    canvas[:] = (MEAN * 255).astype(np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    x = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def load_models(device):
    import torchvision
    from transformers import AutoModel
    dino = AutoModel.from_pretrained(DINO_NAME).to(device).eval()
    rn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    rn.fc = torch.nn.Identity()
    rn = rn.to(device).eval()
    for p in [*dino.parameters(), *rn.parameters()]:
        p.requires_grad_(False)
    return dino, rn


@torch.no_grad()
def embed(batch, dino, rn, device):
    x = torch.from_numpy(np.stack(batch)).to(device)
    d = dino(pixel_values=x).last_hidden_state[:, 0]
    r = rn(x)
    return d.float().cpu().numpy(), r.float().cpu().numpy()


def clip_frames(clip_id, src_idx):
    want = {int(s): k for k, s in enumerate(src_idx)}
    cap, i = cv2.VideoCapture(str(RAW_DIR / f"{clip_id}.mp4")), 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            yield want[i], fr
        i += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    clips = sorted({r["clip_id"] for r in csv.DictReader(open(LABELS_CSV))})
    DINO_DIR.mkdir(parents=True, exist_ok=True)
    RESNET_DIR.mkdir(parents=True, exist_ok=True)
    dino, rn = load_models(a.device)
    t0, nf = time.time(), 0
    for n, c in enumerate(clips, 1):
        if (DINO_DIR / f"{c}.npz").exists() and (RESNET_DIR / f"{c}.npz").exists():
            continue
        p = np.load(POSE_DIR / f"{c}.npz")
        T = len(p["kpts"])
        D, R = np.full((T, 384), np.nan, np.float32), np.full((T, 512), np.nan, np.float32)
        buf, idx = [], []
        for k, fr in clip_frames(c, p["src_frame_idx"]):
            buf.append(preprocess(fr))
            idx.append(k)
            if len(buf) == a.batch:
                D[idx], R[idx] = embed(buf, dino, rn, a.device)
                buf, idx = [], []
        if buf:
            D[idx], R[idx] = embed(buf, dino, rn, a.device)
        missing = int(np.isnan(D[:, 0]).sum())
        if missing:
            raise RuntimeError(f"{c}: {missing} pose frames could not be decoded")
        meta = dict(frame_idx=p["frame_idx"], src_frame_idx=p["src_frame_idx"])
        np.savez_compressed(DINO_DIR / f"{c}.npz", emb=D.astype(np.float16), model=DINO_NAME, **meta)
        np.savez_compressed(RESNET_DIR / f"{c}.npz", emb=R.astype(np.float16), model="resnet18_imagenet1k_v1", **meta)
        nf += T
        el = time.time() - t0
        print(f"[{n:3d}/{len(clips)}] {c:40s} T={T:4d}  {nf / el:6.1f} fr/s  elapsed {el / 60:5.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
