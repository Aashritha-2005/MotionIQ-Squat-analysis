"""Task 4.2: per-rep form rules on the geometry features (pure thresholds, no model).

A rule looks at the frames of one rep (start..end), keeps only frames with valid == 1 (body visible, not
filled in), reduces one geometry feature to a per-rep statistic and compares it with a threshold.
A rule is only applied on views where its fault is observable (config.OBSERVABLE); elsewhere it returns
None ("n/a"). It also returns None ("abstain") if the rep has fewer than MIN_VALID valid frames.

Statistics:  max / min / p90 / p10 over valid rep frames, or "deep" = median over the DEEP_K valid frames
             where the hips are lowest (smallest `depth` feature), i.e. the bottom of the squat.

shallow_depth, forward_lean: feature, statistic and threshold are TUNED on training-split subjects only
(scripts/evaluate_phase4.py; written to artifacts/pipeline_config.json). Per-view thresholds are used when
the training split has >= MIN_POS_PER_VIEW positive reps for that view, otherwise one pooled threshold.

knee_valgus: NOT tuned (0 positive examples). Fixed literature-motivated rule on the knee-to-ankle width
ratio at the bottom of the squat: valgus if knee separation / ankle separation < 0.60. Source: the
normalised knee separation distance of the drop-jump screening test, where knee separation below ~60% of
ankle separation is read as excessive valgus alignment (Noyes, Barber-Westin, Fleckenstein, Walsh & West,
"The drop-jump screening test", Am J Sports Med 33(2):197-207, 2005). Caveats: defined for a drop-jump
landing in the frontal plane, not a squat, and the 2D ratio is foreshortened in diagonal view. Unvalidated
here (0 positive reps in the dataset).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS, FAULTS, OBSERVABLE  # noqa: E402
from extract_features import FEATURE_NAMES  # noqa: E402

FI = {n: i for i, n in enumerate(FEATURE_NAMES)}
MIN_VALID, DEEP_K, MIN_POS_PER_VIEW = 3, 5, 15
CONFIG_PATH = ARTIFACTS / "pipeline_config.json"

KNEE_VALGUS_RULE = {"feature": "knee_ankle_ratio", "stat": "deep", "op": "lt", "threshold": 0.60,
                    "tuned": False, "status": "unvalidated (0 positive examples in the dataset)",
                    "source": "Noyes et al. 2005, Am J Sports Med 33(2):197-207 (drop-jump normalised knee "
                              "separation < ~60% of ankle separation)"}

# candidate (feature, statistic, op) per tunable fault; op "lt": fault if stat < thr, "gt": fault if stat > thr
CANDIDATES = {
    "shallow_depth": [("hip_below_knee", "max", "lt"), ("hip_below_knee", "p90", "lt"), ("hip_below_knee", "deep", "lt"),
                      ("knee_min", "min", "gt"), ("knee_min", "p10", "gt"), ("knee_min", "deep", "gt")],
    "forward_lean": [("trunk_lean", "max", "gt"), ("trunk_lean", "p90", "gt"), ("trunk_lean", "deep", "gt")],
}


def rep_stat(F, valid, feature, stat):
    """F (n,19) features of one rep, valid (n,) -> scalar statistic, or None if too few valid frames."""
    m = np.asarray(valid).astype(bool)
    if m.sum() < MIN_VALID:
        return None
    x = F[m, FI[feature]]
    if stat == "max":
        return float(x.max())
    if stat == "min":
        return float(x.min())
    if stat == "p90":
        return float(np.percentile(x, 90))
    if stat == "p10":
        return float(np.percentile(x, 10))
    if stat == "deep":
        k = np.argsort(F[m, FI["depth"]])[:DEEP_K]
        return float(np.median(x[k]))
    raise ValueError(stat)


def fires(value, op, thr):
    return value < thr if op == "lt" else value > thr


def load_config():
    return json.load(open(CONFIG_PATH))


def rule_for(fault, view, cfg):
    if fault == "knee_valgus":
        return KNEE_VALGUS_RULE
    r = cfg["form_rules"][fault]
    return {**r, "threshold": r["threshold_per_view"].get(view, r["threshold"])}


def evaluate_rep(F, valid, view, cfg):
    """-> {fault: {"flag": True/False/None, "value": float|None, "reason": str}} for one rep."""
    out = {}
    for fault in FAULTS:
        if view not in OBSERVABLE[fault]:
            out[fault] = {"flag": None, "value": None, "reason": "n/a (not observable from this view)"}
            continue
        r = rule_for(fault, view, cfg)
        v = rep_stat(F, valid, r["feature"], r["stat"])
        if v is None:
            out[fault] = {"flag": None, "value": None, "reason": "abstain (too few valid frames)"}
        else:
            out[fault] = {"flag": bool(fires(v, r["op"], r["threshold"])), "value": v,
                          "reason": f"{r['feature']}[{r['stat']}]={v:.3f} {'<' if r['op'] == 'lt' else '>'} {r['threshold']:.3f}?"}
    return out


def best_threshold(values, labels, op):
    """Max-F1 threshold over midpoints of the sorted values. Returns (thr, f1, tp, fp, fn, tn)."""
    v, y = np.asarray(values, float), np.asarray(labels, int)
    u = np.unique(v)
    cands = np.concatenate([[u[0] - 1e-6], (u[1:] + u[:-1]) / 2, [u[-1] + 1e-6]])
    best = None
    for t in cands:
        p = v < t if op == "lt" else v > t
        tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
        fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        key = (f1, tp / max(tp + fp, 1))
        if best is None or key > best[0]:
            best = (key, (float(t), f1, tp, fp, fn, tn))
    return best[1]
