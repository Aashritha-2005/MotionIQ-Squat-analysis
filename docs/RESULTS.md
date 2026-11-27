# Results

[← Back to README](../README.md)

All results are on the **5 held-out subjects (30 clips)** unless stated otherwise. Nothing was tuned on them.

## 1. Phase classifier (per frame; claim separate from form rules)

Causal LSTM, W = 32, 3 seeds per feature set. Selection used internal-validation subjects only.

| input features | dim | val macro-F1 | held-out accuracy | held-out macro-F1 |
|---|---|---|---|---|
| **geometry + valid** | 20 | 0.864 ± 0.002 | **0.896 ± 0.002** | **0.891 ± 0.002** |
| ResNet-18 (frozen) | 512 | 0.649 ± 0.019 | 0.753 ± 0.005 | 0.737 ± 0.003 |
| DINOv2-small CLS (frozen) | 384 | 0.732 ± 0.000 | 0.781 ± 0.005 | 0.765 ± 0.005 |
| geometry + ResNet-18 | 532 | 0.778 ± 0.001 | 0.826 ± 0.008 | 0.813 ± 0.009 |
| geometry + DINOv2 | 404 | 0.813 ± 0.006 | 0.837 ± 0.002 | 0.827 ± 0.002 |

Per-phase held-out F1:

| model | standing | descent | bottom | ascent |
|---|---|---|---|---|
| geometry | 0.919 | 0.873 | 0.864 | 0.906 |
| geometry + DINOv2 | 0.886 | 0.797 | 0.797 | 0.828 |

- **Geometry wins,** which is expected because the phase labels were generated from pose (see [limitations](LIMITATIONS.md)).
- **The visual-only models learn phases without any pose input:** DINOv2 reaches 0.77 macro-F1 and beats ResNet-18 by about 3 points.
- **Adding embeddings to geometry hurts,** likely overfitting to how subjects and scenes look, with only 16 training subjects.
- **Deployed model:** the geometry-only model (`model_geom.onnx`). Streaming on the Mac it reaches macro-F1 0.889 on held-out clips, vs 0.891 offline.

## 2. Rep counting

The reference reps were **auto-generated** from the pose-based depth signal, so this measures agreement with that
reference, not with an independent human count.

| view | reps counted / reference | clips with exact count | mean abs. error per clip |
|---|---|---|---|
| front | 133 / 135 | 8 / 10 | 0.2 |
| side | 118 / 120 | 9 / 10 | 0.2 |
| diagonal | 130 / 132 | 8 / 10 | 0.2 |
| **all** | **381 / 387** | **25 / 30** (29/30 within ±1) | **0.2** |

All errors are undercounts. The counter was tuned on the internal-validation subjects (MAE 0.24 reps/clip).

## 3. Form rules (per rep; agreement with my own tags)

Rules were tuned on the 20 training-split subjects. These numbers measure **agreement with my own fault tags**,
not an external ground truth. They're computed on reference rep segments; "end-to-end" uses the model's own
rep segmentation, and counts missed reps as false negatives and spurious reps as false positives.

| rule (view) | rule | TP / FP / FN / TN | precision | recall | end-to-end P / R | training-split F1 | held-out F1 |
|---|---|---|---|---|---|---|---|
| shallow depth (side, diagonal) | max hip-below-knee over the rep < threshold | 82 / 18 / 12 / 129 | 0.82 | 0.87 | 0.81 / 0.84 | 0.64 | 0.85 |
| forward lean (side, diagonal) — **provisional** | trunk angle at the bottom > 34° (diagonal) / 45° (side) | 31 / 11 / 6 / 193 | 0.74 | 0.84 | 0.74 / 0.84 | **0.28** | 0.79 |
| knee valgus (front, diagonal) — **experimental, unvalidated** | knee/ankle width ratio at the bottom < 0.60 | 0 positives | – | – | – | not tuned | **false-alarm rate 1/253 = 0.4%** |

- By view (precision / recall): depth is 0.83 / 0.83 side and 0.81 / 0.92 diagonal; lean is 0.83 / 0.83 side and 0.70 / 0.84 diagonal.
- **Forward lean is provisional.**
  - The rule separates the training-split tags poorly (F1 0.28), but scores 0.79 on held-out.
  - The held-out positives come from only 2 subjects (37 reps) whose lean is pronounced.
  - Treat that gap as a warning, not a success.
- **Knee valgus is a fixed literature threshold** and was never tuned.
  - The threshold is the normalised knee-separation cut-off (< ~60% of ankle separation) from the drop-jump screening test: Noyes et al. 2005, *Am J Sports Med* 33(2):197–207.
  - The dataset has 0 tagged valgus reps, so only its false-alarm rate can be measured.

## 4. Latency

| what | time |
|---|---|
| phase model, ONNX Runtime CPU, one 32-frame window (Colab CPU) | 0.47 ms median (p95 0.56) — geometry model; 0.69 ms for geometry + DINOv2 |
| phase model, PyTorch FP32 on a T4 GPU | 0.45 ms |
| **end-to-end per frame on an Apple Silicon Mac (MPS)**: YOLO11s-pose + features + ONNX + counter + rules | **20.55 ms median of per-clip medians; worst-clip p95 26.3 ms** across the 30 held-out replays (budget 66.7 ms at 15 fps); YOLO is ~20 ms of it |
| TensorRT FP16 | attempted on Colab, blocked by an API change (`BuilderFlag.FP16` removed in the TensorRT 10 that pip installed); not retried because the result is reporting-only |

## 5. Robustness checks (no live person)

- **Same code path:** replaying all 30 held-out clips through `run_demo.py` gives frame-for-frame identical phases and reps to the offline evaluation.
- **Streaming vs offline features:** given the offline body scale, the streaming feature code matches the training features exactly (difference 0.0).
  - In deployment the body scale is a running median, because the offline whole-clip median needs future frames.
  - That leaves a small skew: 98.3% frame-level phase agreement, and identical rep counts on 30/30 clips.
- **External videos:**
  - Clean side-view clip: all 6 real squats were counted at the right times, despite strong backlighting and a small subject.
  - Two kettlebell pick-ups were also counted as reps (see [limitations](LIMITATIONS.md)).
  - Instructional clips with close-ups and cutaways ran without crashes.
  - One bug found this way was fixed: leg-only close-ups had corrupted the body scale.
- **Live camera self-test** (author's Mac camera, 10 s per scenario; no squats performed):

  | scenario | person box | state / visibility | reps | processed | latency median / p95 | result |
  |---|---|---|---|---|---|---|
  | empty scene (wall, nobody in frame) | 0% of 147 frames | "no person / lower body not visible" on 100% | 0 | 14.6 fps | 20.9 / 28.3 ms | PASS |
  | person standing in full view | 100% of 145 frames (median max confidence 0.92, p10 0.89) | lower body visible on 85% | 0 | 14.5 fps | 23.9 / 32.9 ms | PASS |

  This verifies the live capture path, person detection, lower-body keypoints, empty-scene handling and latency.
  It does **not** validate squat counting or form feedback on a live person.
