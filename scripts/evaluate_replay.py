"""Task 5.2 (revised): score run_demo.py --video replays of all held-out clips (logs in data/demo/heldout_replay/)
against (a) the Phase 4 pipeline on cached pose with the same code, (b) the phase labels, rep reference and the
author's per-rep fault tags. Writes artifacts/replay_metrics.json. No tuning happens here.

Usage: .venv/bin/python scripts/evaluate_replay.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS, DATA, FAULTS, OBSERVABLE  # noqa: E402
from evaluate_phase4 import clip_info, kpts_of, role  # noqa: E402
from form_rules import load_config  # noqa: E402
from pipeline import run_clip  # noqa: E402

LOGS = DATA / "demo" / "heldout_replay"


def main():
    cfg = load_config()
    lat = {m[1]: (float(m[2]), float(m[3])) for m in
           re.finditer(r"^(\S+) frames \d+\s+reps \d+\s+latency median ([\d.]+) ms\s+p95 ([\d.]+) ms", open(LOGS / "run.txt").read(), re.M)}
    test = [ci for ci in range(136) if role(ci) == "test"]
    per_clip, rule_rows = [], {f: [] for f in FAULTS}
    cm = np.zeros((4, 4), int)
    for ci in test:
        info = clip_info(ci)
        d = json.load(open(LOGS / f"{info['clip']}.json"))
        recs, _ = run_clip(kpts_of(info["clip"]), info["view"], cfg)             # Phase 4 path, cached pose
        ph_demo = np.array([f["phase"] for f in d["frames"]])
        ph_p4 = np.array([r["phase"] for r in recs])
        reps_demo = [(r["start"], r["end"], r["form"]) for r in d["reps"] if not r.get("rejected")]
        reps_p4 = [(r["rep"]["start"], r["rep"]["end"]) for r in recs if r["rep"]]
        cm += np.bincount(info["phase"] * 4 + ph_demo, minlength=16).reshape(4, 4)
        # match demo reps to reference reps (IoU >= 0.3, greedy) for rule flags vs tags
        pairs = sorted((((min(b, e) - max(a, s) + 1) / (max(b, e) - min(a, s) + 1)), gi, pi)
                       for gi, (a, b, _) in enumerate(info["reps"]) for pi, (s, e, _) in enumerate(reps_demo))
        mg, mp = {}, set()
        for iou, gi, pi in reversed(pairs):
            if iou >= 0.3 and gi not in mg and pi not in mp:
                mg[gi] = pi; mp.add(pi)
        for gi, (a, b, fv) in enumerate(info["reps"]):
            for k, f in enumerate(FAULTS):
                if fv[k] in (0, 1) and gi in mg:
                    rule_rows[f].append((int(fv[k]), bool(reps_demo[mg[gi]][2][f]), info["view"]))
        per_clip.append({"clip": info["clip"], "view": info["view"], "frames": len(ph_demo),
                         "phase_agreement_vs_phase4": float((ph_demo == ph_p4).mean()) if len(ph_demo) == len(ph_p4) else None,
                         "reps_demo": len(reps_demo), "reps_phase4": len(reps_p4), "reps_reference": len(info["reps"]),
                         "identical_rep_segments_vs_phase4": [(s, e) for s, e, _ in reps_demo] == reps_p4,
                         "matched_reference_reps": len(mg),
                         "latency_median_ms": lat.get(info["clip"], (None, None))[0],
                         "latency_p95_ms": lat.get(info["clip"], (None, None))[1]})

    def conf(rows):
        y = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
        tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
        fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
        return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                "fpr": fp / (fp + tn) if fp + tn else None}

    by_view = {}
    for v in ("front", "side", "diagonal"):
        g = [c for c in per_clip if c["view"] == v]
        e = np.array([c["reps_demo"] - c["reps_reference"] for c in g])
        by_view[v] = {"clips": len(g), "reps_reference": int(sum(c["reps_reference"] for c in g)),
                      "reps_demo": int(sum(c["reps_demo"] for c in g)), "exact_clips": int((e == 0).sum()),
                      "mae_per_clip": float(np.abs(e).mean())}
    f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
    out = {
        "scope": "run_demo.py --video on all 30 held-out clips (fresh per-frame YOLO, full demo code path); no tuning",
        "vs_phase4_pipeline": {
            "clips_identical_rep_segments": int(sum(c["identical_rep_segments_vs_phase4"] for c in per_clip)),
            "min_phase_agreement": min(c["phase_agreement_vs_phase4"] for c in per_clip),
            "mean_phase_agreement": float(np.mean([c["phase_agreement_vs_phase4"] for c in per_clip]))},
        "phase_vs_labels": {"acc": float(np.diag(cm).sum() / cm.sum()), "macro_f1": float(f1.mean()), "confusion": cm.tolist()},
        "rep_count_vs_auto_reference": by_view,
        "rule_flags_vs_author_tags_matched_reps": {f: {"all": conf(rule_rows[f]),
                                                       "by_view": {v: conf([r for r in rule_rows[f] if r[2] == v])
                                                                   for v in sorted(OBSERVABLE[f])}} for f in FAULTS},
        "latency_ms": {"median_of_clip_medians": float(np.median([c["latency_median_ms"] for c in per_clip])),
                       "max_p95": float(max(c["latency_p95_ms"] for c in per_clip)), "budget": 1000 / 15},
        "per_clip": per_clip,
    }
    json.dump(out, open(ARTIFACTS / "replay_metrics.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_clip"}, indent=1))


if __name__ == "__main__":
    main()
