# Limitations, descoped items and next steps

[← Back to README](../README.md)

## Known limitations

- **Phase labels are auto-generated from pose angles** (a hip-depth signal; the knee angle for two clips), spot-checked but not independently verified. The geometry-only phase model is therefore strong partly by construction, and phase accuracy should be read with that caveat.
- **Fault tags are self-annotated** on top of the dataset's binary good/bad labels, not provided by the dataset authors. Per-rule precision/recall validates consistency with my own labelling, not an external ground truth.
- **Rep-count reference is auto-generated** (pose-derived), not an independent human count.
- **Subject IDs may not map one-to-one to people:** subjects 005/006 and 015/026 shared byte-identical clips, so the subject-wise split could be slightly optimistic.
- **Privacy-blurred training videos vs a real webcam:** the dataset's anonymisation blur sometimes covers the body (2.8% of frames). The DINOv2/ResNet embeddings were extracted from these blurred gym videos and may not transfer to an unblurred webcam. The deployed model uses geometry only, partly for this reason.
- **Live use is tested on one person only** (side view, one room, ~2.3 min of scripted test runs, with hand support at the bottom; see [LIVE_TEST.md](LIVE_TEST.md)). Squat counting and shallow-depth flags behaved as intended there. Nothing is validated on other people, front or diagonal live views, other rooms or cameras.
- **Non-squat movements are counted as reps.** Live: 4 small knee dips → 4 reps; one held forward bend to pick something up → 8 reps. External clip: kettlebell pick-up/put-down hinges. Dips are indistinguishable from quarter-depth squats, which must be counted to be flagged. During a held bend the phase model flickers between descent and ascent, and the counter's bounce rule closes a rep at each flicker. Fixing this needs non-squat training examples or a counter rule validated on the training/validation split.
- **The forward-lean flag is not reliable.** It measures the torso at the bottom of the rep only. Live, 0 of 5 intended leaning squats were flagged: their lean at the bottom was 27–32°, while the visible 46–52° lean is mid-rep. On the validation subjects it reaches only P 0.08 / R 0.20. Measuring lean over the whole rep was evaluated and rejected, because it flags 15.5% (train) / 30.6% (validation) of good reps; large mid-rep lean is normal in good squats.
- **The side-on (view) check can flash for 0.5–1.5 s after deep reps** when the shoulders rotate at the bottom (e.g. hands reaching forward). A standing-only variant was evaluated and rejected: it gave more false warnings on training clips (0.31% → 1.13% of frames).
- **A rejected movement re-shows faded verdicts** together with its "Not counted" message (display only).
- **Rep-counter deviation from the original spec:** a rep needs descent … ascent in order, and does **not** require a separate predicted bottom phase. Requiring it more than doubled the validation count error (0.24 → 0.57 reps per clip), because the model sometimes predicts only 0–2 bottom frames on fast or shallow reps.
- **Forward lean is weak on training subjects** (F1 0.28); its held-out numbers come from 2 subjects. **Knee valgus is unvalidated** (0 positive examples, and not tested live).
- **Body-scale skew:** live features use a causal running-median body scale, while the model was trained on whole-clip medians. The first seconds of a session are less reliable (the overlay says "calibrating"). Retraining on causal-scale features would remove this.
- Small data overall: 25 subjects, one gym-dominated dataset, and three external clips. Exact-hash duplicate checking cannot catch re-encoded near-duplicates.

## Descoped, on purpose

These were designed for but cut to fit a ~36–38 hour budget:
- **SAM2 segmentation:** person masks would help in crowded gyms. The largest-box heuristic was adequate here, and SAM2 would add a second heavy model to the live loop.
- **VideoMAE fine-tuning:** a video transformer on raw clips needs far more GPU time and data than one Colab session and 136 clips. Frozen per-frame embeddings plus a small temporal model answered the same question (do visual features help?) cheaply.
- **VLM weak labeling:** auto-tagging faults with a vision-language model would have scaled the labels, but without a trusted ground truth its errors would have been unmeasurable. I tagged reps by hand instead.

## What I would do next

1. Record or collect a small, consented, **independently labelled live set** (reps, phases and faults, including real valgus) to validate live mode and the valgus rule.
2. Retrain the phase model on **causal running-scale features** to remove the train/live skew, and add **non-squat negatives** (hinges, bends, walking) so the counter can reject them.
3. Replace the pose-derived phase labels with a small set of hand-labelled phase boundaries, so the geometry-vs-visual comparison is fair.
4. Learn the form rules (or a small classifier) from more subjects with a second annotator, and report inter-annotator agreement.
5. Retry the TensorRT FP16 benchmark with the TensorRT 10 API (strongly-typed networks).
