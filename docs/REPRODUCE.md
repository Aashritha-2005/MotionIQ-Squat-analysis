# Reproduce and debug

[← Back to README](../README.md)

## Pipeline details

- **Pose:** pretrained YOLO11s-pose (CNN). In a crowded gym, the main subject is the box with the largest area × confidence.
- **Geometry features (per frame):**
  - knee and hip angles, trunk lean, hip height, hip-below-knee, knee-over-toe
  - knee/ankle/hip/shoulder widths, the knee-to-ankle width ratio
  - four velocities
  - Distances are divided by a body scale (the median torso length).
  - Short keypoint gaps are interpolated and longer ones held.
  - A per-frame `valid` flag tells the model when the lower body was not actually visible.
- **Phase model:** a unidirectional (causal) 2-layer LSTM over the last 32 frames (2.1 s). It predicts the phase of the newest frame, and normalisation is built into the exported ONNX.
- **Rep counter:** decides only from past frames.
  - Hysteresis applies on the standing ↔ moving boundary.
  - A rep also ends on an ascent → descent "bounce", for continuous reps.
  - A segment counts when it contains descent … ascent in order.
- **Form rules:** pure thresholds on per-rep geometry statistics. Each rule runs only where its fault is visible in 2D:
  - depth and lean: side and diagonal views
  - valgus: front and diagonal views

## Tech stack

YOLO11-pose (Ultralytics), PyTorch, frozen DINOv2-small (Hugging Face `transformers`, self-supervised ViT) and
ResNet-18 (torchvision) for the ablation, ONNX + ONNX Runtime, OpenCV, NumPy. Training ran on Colab with a T4 GPU;
everything else runs on a Mac (MPS/CPU).

## Reproducing the pipeline

| step | command | output |
|---|---|---|
| labels: auto rep/phase boundaries, split, spot-checks | `scripts/prepare_labels.py` | `data/labels.csv` |
| manual per-rep fault tagging (author) | `scripts/tag_faults.py` | fault columns in `labels.csv` |
| privacy-blur audit | `scripts/blur_audit.py` | `data/blur_audit.csv` |
| pose (YOLO11s-pose, 15 fps) | `scripts/extract_pose.py` | `data/pose/` |
| geometry features | `scripts/extract_features.py` | `data/geom_features/` |
| DINOv2 + ResNet-18 embeddings | `scripts/extract_embeddings.py` | `data/dino_features/`, `data/resnet_features/` |
| training bundle + sanity checks | `scripts/build_training_bundle.py`, `scripts/check_bundle.py` | `artifacts/train_data.npz` (106 MB, not committed) |
| **training (Colab GPU)** | upload `train_data.npz`, run `colab_training.ipynb` (Runtime → T4 GPU → Run all) | `model_geom.onnx`, `model_best.onnx`, `metrics.json` |
| parity test, counter/rule tuning, held-out evaluation | `scripts/evaluate_phase4.py` | `pipeline_config.json`, `form_metrics.json` |
| replay of all held-out clips through the demo | `scripts/run_demo.py --video … --log …`, then `scripts/evaluate_replay.py` | `replay_metrics.json` |

The remaining modules are `config.py` (shared constants such as FPS = 15), `pipeline.py` (the streaming pipeline),
`rep_counter.py` and `form_rules.py`.

## Camera self-test and debugging

Camera self-test (10 s each, prints PASS/FAIL checks): once facing an empty scene, once with a person standing
in full view.

```bash
.venv/bin/python scripts/run_demo.py --view front --selftest-camera --expect empty
```

```bash
.venv/bin/python scripts/run_demo.py --view front --selftest-camera --expect person
```

Debug a live session (saves the raw frames and a per-frame log of keypoints, features, phase probabilities,
counter state and stage timings, so the session can be replayed exactly with `--video`):

```bash
.venv/bin/python scripts/run_demo.py --view side --record-raw data/demo/session_raw.mp4 --log-session data/debug/session.npz
```

On macOS, if the camera does not open: System Settings → Privacy & Security → Camera → enable the app you launched
from (Terminal, iTerm, VS Code …), then restart that app.

## Tests

Tests (rep counter, geometry and rules, streaming parity, empty frames, framing gate; tests that need local data are
skipped on a fresh clone):

```bash
.venv/bin/python -m pytest -q tests
```
