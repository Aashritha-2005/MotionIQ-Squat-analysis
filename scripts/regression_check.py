"""Regression check: run the full streaming pipeline (scripts/pipeline.run_clip) on cached pose for one subject group
and report phase / rep-count / form-rule metrics, so a change can be compared before vs after.

  --role val      the 4 internal-validation subjects (002, 012, 023, 026): use this during development
  --role heldout  the 5 held-out subjects: READ-ONLY regression gate, run once at the end; compared with the stored
                  numbers in artifacts/form_metrics.json. Never use it for tuning or diagnosis.

Usage: .venv/bin/python scripts/regression_check.py --role val --out data/debug/val_before.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FAULTS, OBSERVABLE  # noqa: E402
from evaluate_phase4 import clip_info, kpts_of, role  # noqa: E402
from form_rules import evaluate_rep, load_config  # noqa: E402
from pipeline import run_clip  # noqa: E402


def conf(rows):
    y = np.array([r[0] for r in rows], int); p = np.array([bool(r[1]) for r in rows])
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None, "fpr": fp / (fp + tn) if fp + tn else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["val", "heldout"], required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    want = "test" if a.role == "heldout" else "val"
    cfg = load_config()
    cm = np.zeros((4, 4), int)
    clips, gt_rows, e2e_rows = [], {f: [] for f in FAULTS}, {f: [] for f in FAULTS}
    for ci in range(136):
        if role(ci) != want:
            continue
        info = clip_info(ci)
        recs, p = run_clip(kpts_of(info["clip"]), info["view"], cfg)
        F, V = np.stack(p.F), np.array(p.V)
        ph = np.array([r["phase"] for r in recs])
        cm += np.bincount(info["phase"] * 4 + ph, minlength=16).reshape(4, 4)
        reps = [r["rep"] for r in recs if r["rep"]]
        counted = [e for e in reps if not e.get("rejected")]
        for a_, b_, fv in info["reps"]:
            fr = evaluate_rep(F[a_:b_ + 1], V[a_:b_ + 1], info["view"], cfg)
            for k, f in enumerate(FAULTS):
                if fv[k] in (0, 1):
                    gt_rows[f].append((int(fv[k]), fr[f]["flag"]))
            # end-to-end: best-overlapping counted rep (IoU >= 0.3)
            best = max(counted, key=lambda e: (min(b_, e["end"]) - max(a_, e["start"]) + 1)
                       / (max(b_, e["end"]) - min(a_, e["start"]) + 1), default=None)
            iou = 0 if best is None else (min(b_, best["end"]) - max(a_, best["start"]) + 1) / (max(b_, best["end"]) - min(a_, best["start"]) + 1)
            for k, f in enumerate(FAULTS):
                if fv[k] in (0, 1):
                    e2e_rows[f].append((int(fv[k]), best["form"][f]["flag"] if iou >= 0.3 else False))
        clips.append({"clip": info["clip"], "view": info["view"], "ref": len(info["reps"]), "counted": len(counted),
                      "rejected": len(reps) - len(counted)})
    f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
    err = np.array([c["counted"] - c["ref"] for c in clips])
    out = {"role": a.role, "clips": len(clips),
           "phase": {"acc": float(np.diag(cm).sum() / cm.sum()), "macro_f1": float(f1.mean())},
           "reps": {"ref": int(sum(c["ref"] for c in clips)), "counted": int(sum(c["counted"] for c in clips)),
                    "rejected_by_gates": int(sum(c["rejected"] for c in clips)), "exact_clips": int((err == 0).sum()),
                    "mae_per_clip": float(np.abs(err).mean())},
           "rules_gt_segments": {f: conf(gt_rows[f]) for f in FAULTS},
           "rules_end_to_end": {f: conf(e2e_rows[f]) for f in FAULTS},
           "per_clip": clips}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    s = {k: out[k] for k in ("phase", "reps")}
    s.update({f"{f}_gt": (out["rules_gt_segments"][f]["precision"], out["rules_gt_segments"][f]["recall"], out["rules_gt_segments"][f]["fpr"]) for f in FAULTS})
    s.update({f"{f}_e2e": (out["rules_end_to_end"][f]["precision"], out["rules_end_to_end"][f]["recall"]) for f in FAULTS})
    print(json.dumps(s, default=lambda x: round(x, 4) if isinstance(x, float) else x))


if __name__ == "__main__":
    main()
