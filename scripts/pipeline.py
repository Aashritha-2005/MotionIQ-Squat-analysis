"""Task 4.3: streaming pipeline = YOLO keypoints -> StreamingFeatures -> ONNX phase model -> RepCounter -> form rules.

Everything is causal: each pushed frame can only finalise frames up to `lag` frames in the past
(lag = lookahead + 1, see extract_features.StreamingFeatures). The live demo (Phase 5) feeds YOLO keypoints
from the webcam; the offline evaluation feeds the cached held-out pose frame by frame.

    p = Pipeline(view="side")                # view picked by the user; rules only run where observable
    for kpts in frames: for rec in p.push(kpts): ...      # rec per finalised frame
    for rec in p.flush(): ...                              # end of clip only
rec = {frame, phase (raw argmax), prob, smooth (smoothed phase), count, rep (None or event with form flags)}
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS  # noqa: E402
from extract_features import StreamingFeatures  # noqa: E402
from form_rules import evaluate_rep, load_config  # noqa: E402
from rep_counter import RepCounter  # noqa: E402

MODEL = ARTIFACTS / "model_geom.onnx"
# Framing gate (Phase 7, failure F1): a completed rep is only counted if the upper body (both shoulders observed,
# conf >= 0.3) was seen on >= GATE_UPPER of its frames AND the lower body was valid on >= GATE_LOWER of its frames.
# Bounds lie below the minimum over all 1,488 training-split reps (0.375 / 0.294), so no training rep is rejected;
# not fitted to any live session. Reps made with the body not in frame are outside the model's training data.
GATE_UPPER, GATE_LOWER = 0.25, 0.25
METRICS = ARTIFACTS / "metrics.json"


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class PhaseStream:
    """Keypoints -> finalised (frame, feats, valid, probs). Model input = [19 geometry features, valid] per frame,
    window of the last W frames, left-padded with the first frame (same as training, metrics.json)."""

    def __init__(self, lookahead=0, model=MODEL, fixed_scale=None):
        meta = json.load(open(METRICS))
        exp = meta["exported"][Path(model).name]
        assert exp["blocks"] == ["geom", "valid"], f"pipeline expects the geometry-only model, got {exp['blocks']}"
        self.W = int(meta["W"])
        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.feat = StreamingFeatures(lookahead, fixed_scale=fixed_scale)
        self.win = deque(maxlen=self.W)

    @property
    def lag(self):
        return self.feat.D + 1

    def _step(self, recs):
        out = []
        for f, feats, valid in recs:
            x = np.concatenate([feats, [valid]]).astype(np.float32)
            if not self.win:
                self.win.extend([x] * self.W)
            self.win.append(x)
            logits = self.sess.run(None, {self.inp: np.stack(self.win)[None]})[0][0]
            out.append((f, feats, valid, softmax(logits)))
        return out

    def push(self, kpts):
        t0 = time.perf_counter()
        recs = self.feat.push(kpts)
        t1 = time.perf_counter()
        out = self._step(recs)
        self.t_features_ms, self.t_onnx_ms = (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3   # instrumentation only
        return out

    def flush(self):
        return self._step(self.feat.flush())


class Pipeline:
    def __init__(self, view, cfg=None, model=MODEL):
        self.view, self.cfg = view, cfg or load_config()
        self.stream = PhaseStream(self.cfg.get("lookahead", 0), model)
        self.counter = RepCounter(**self.cfg["rep_counter"])
        self.F, self.V = [], []   # finalised features / valid, for rule evaluation over a rep

    def _consume(self, recs):
        out = []
        for f, feats, valid, prob in recs:
            self.F.append(feats); self.V.append(valid)
            ph = int(prob.argmax())
            ev = self.counter.update(f, ph, float(prob[ph]))
            if ev is not None:
                s, e = ev["start"], ev["end"]
                upper = float(np.mean([all(self.stream.feat.obs[i][[5, 6]]) for i in range(s, e + 1)]))
                lower = float(np.mean(self.V[s:e + 1]))
                ev.update(upper_share=upper, lower_share=lower)
                ev["form"] = evaluate_rep(np.stack(self.F[s:e + 1]), np.array(self.V[s:e + 1]), self.view, self.cfg)
                if upper < GATE_UPPER or lower < GATE_LOWER:
                    self.counter.reject_last()
                    ev["rejected"] = (f"body not fully in frame (upper body seen on {upper:.0%}, "
                                      f"lower body on {lower:.0%} of the movement)")
            out.append({"frame": f, "phase": ph, "prob": prob, "smooth": self.counter.state,
                        "count": self.counter.count, "rep": ev})
        return out

    def push(self, kpts):
        recs = self.stream.push(kpts)
        t0 = time.perf_counter()
        out = self._consume(recs)
        self.t_logic_ms = (time.perf_counter() - t0) * 1e3   # counter + rules (instrumentation only)
        return out

    def flush(self):
        return self._consume(self.stream.flush())


def run_clip(kpts, view, cfg=None):
    p = Pipeline(view, cfg)
    recs = [r for k in kpts for r in p.push(k)] + p.flush()
    return recs, p
