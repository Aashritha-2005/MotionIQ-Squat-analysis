# Live sessions

[← Back to README](../README.md)

## First live session: debugging

A first live webcam session (side view, about 87 s, the author squatting) was used **for diagnosis only**. It
showed clear failures. The session was reconstructed from its overlay recording, and every number below comes from
replaying it through the pipeline.

| | before fixes | after fixes |
|---|---|---|
| reps counted in the session (manual reference: ~13 squats + 1 partial bend, then walking toward the camera, empty wall, face close-up) | 22 in replay (25 live) | **14** in replay |
| "reps" counted while walking toward the camera with only the legs in frame | 8 | **0** (rejected: body not fully in frame) |
| held-out regression gate (phase macro-F1, reps, rule counts) | 0.889, 381/387, unchanged | **identical** |
| validation subjects (phase, reps, rules) | — | **identical**, 0 reps rejected |

Root causes and fixes:
- **Reps were counted with the body out of frame** (walking toward the camera). Each completed rep must now have the upper body observed on ≥ 25% of its frames and the lower body on ≥ 25%. Both bounds are below the minimum over all 1,488 training-split reps (0.375 / 0.294), so no training rep is rejected.
- **The camera was low and close**, with the shoulders out of frame on 60% of squat frames. That made the body scale unstable and kept "calibrating" reappearing. There's now an explicit warning, "upper body out of frame".
- **`--view` didn't match the actual stance** (squats facing the camera or turned diagonal while `--view side`). There's now a view check from shoulder width, with a 0.3% mean false-warning rate on training clips.
- **Per-rep verdicts stayed on screen indefinitely.** They now fade 10 s after the last rep.
- **A phantom person was detected on clothes hanging on the wall.** It's contained (no reps counted) but not fixed, since a confidence threshold would also drop real frames.
- **Still open:** extra counts seen live but not in the replay (reaching for the camera, turning between views) need a raw-frame session to diagnose.

Full-loop latency in replay (pose, features, ONNX, rules, drawing, overlay and raw recording): 24.6 ms median,
p95 32.7 ms. During the live session, per-frame time rose from ~24 to ~33–37 ms after 40 s (load or thermal),
which is still within budget.

The extra live counts from that session could not be reproduced afterwards. In the scripted test runs below (raw
frames logged), every extra count was traced to a cause.

## Live test (scripted, one person, side view)

After the fixes, two scripted test runs were recorded and scored from their raw-frame logs. The pipeline was frozen,
and nothing was tuned on them. Replaying the logged keypoints reproduces the live rep segments exactly.
**Scope:** one person, one room, side view only, about 2.3 minutes in total. In run 1 the hands were on the wall for
every squat; in run 2 they reached toward a box at the bottom. The chair-sit block was not done.

| block | intended | counted | flags shown | verdict |
|---|---|---|---|---|
| deep squats, run 1 (4 + 1 good) | 5 | **5** | 0 shallow, 0 leaning | as intended |
| deep squats, run 2 (the "leaning" block: chest forward) | 5 | **5** | 0 shallow | counted as intended |
| shallow squats (run 1) | 3 | **3** | 3/3 "too shallow" (closest miss: 0.095 torso lengths) | as intended |
| **all squats** | **13** (10 deep + 3 shallow) | **13** | | as intended |
| lean flag on the run-2 "leaning" block (the same 5 squats as above, not additional reps) | 5 flagged "leaning" | 5 (same reps as above) | **0/5 "leaning"**: lean at the bottom 27–32° vs 45° threshold; the visible 46–52° lean happens mid-rep | **fails** |
| small knee dips (run 2) | 0 reps | **4** | all "too shallow" | **fails**: dips look like quarter squats |
| pick up an object from the floor (run 2) | 0 reps | **8** (one held forward bend) | all "too shallow" + "leaning" | **fails**: phase flicker during a held bend |
| stand still 20 s (run 2) | 0 reps; feedback fades | 0 | feedback cleared ~9.7 s after the last rep | as intended |
| walk toward the camera (both runs) | 0 reps | **0** | "Not counted: body not fully in frame" | as intended |

- **Latency** (full live loop including display, on this Mac): 15 fps held, 0 dropped frames.
  - Processing p95 53.5 ms against a 66.7 ms budget.
  - Pose takes ~31 ms live vs ~21 ms in replay.
- **Side-on warning:** correct when facing the camera. It also flashes for 0.5–1.5 s after deep reps where the hands reach forward: the 2 s window still holds the rotated shoulders from the bottom of the rep.
- **Calibration** cleared in ~3.3 s when standing still. It is display-only and never blocks counting.

**Next to the recorded-clip numbers** (held-out: `artifacts/form_metrics.json`, `artifacts/replay_metrics.json`,
`artifacts/regression_gate_heldout.json`; validation subjects: `scripts/regression_check.py --role val`; live: local
session logs, which are not published because they contain the author's own recordings):

| | held-out subjects (5) | validation subjects (4) | live test (1 person) |
|---|---|---|---|
| rep counting | 381/387, 25/30 clips exact | 283/284 | 13/13 squats; but 12 non-squat "reps" (4 dips + 8 from one pick-up) |
| shallow depth P / R | 0.82 / 0.87 | 0.48 / 1.00 | 3/3 shallow flagged, 0/10 deep flagged |
| forward lean P / R | 0.74 / 0.84 (2 subjects) | 0.08 / 0.20 | 0/5 intended lean flagged; 0 false flags on the 5 good run-1 squats |
| knee valgus | false alarms 1/253 | false alarms 2/178 | not tested (side view) |
