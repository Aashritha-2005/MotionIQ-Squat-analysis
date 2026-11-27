"""Phase 4 driver: streaming parity test, rep-counter tuning (val subjects), form-rule tuning (training-split
subjects), then the held-out evaluation. Writes artifacts/pipeline_config.json and artifacts/form_metrics.json.

Held-out subjects are touched only in the parity test (feature equality, no tuning) and the final evaluation.
Usage: .venv/bin/python scripts/evaluate_phase4.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS, DATA, FAULTS, OBSERVABLE, POSE_DIR  # noqa: E402
from form_rules import (CANDIDATES, CONFIG_PATH, KNEE_VALGUS_RULE, MIN_POS_PER_VIEW, best_threshold,  # noqa: E402
                        evaluate_rep, rep_stat)
from pipeline import METRICS, PhaseStream, run_clip  # noqa: E402
from rep_counter import REQUIRE, RepCounter  # noqa: E402

B = dict(np.load(ARTIFACTS / "train_data.npz"))
META = json.load(open(METRICS))
VAL_SUBJ = set(META["splits"]["val_subjects"])
CLIPS = list(B["clip_id"])


def clip_info(ci):
    s, T = int(B["clip_start"][ci]), int(B["clip_len"][ci])
    reps = [(int(a), int(b), B["rep_faults"][k]) for k, (c, a, b) in
            enumerate(zip(B["rep_clip"], B["rep_start"], B["rep_end"])) if c == ci]
    return dict(clip=CLIPS[ci], subject=str(B["subject_id"][ci]), view=str(B["view"][ci]), split=str(B["split"][ci]),
                s=s, T=T, reps=reps, geom=B["geom"][s:s + T], phase=B["phase"][s:s + T], fill=B["fill"][s:s + T])


def role(ci):
    info = clip_info(ci)
    return "test" if info["split"] == "heldout" else ("val" if info["subject"] in VAL_SUBJ else "train")


def kpts_of(clip):
    return np.load(POSE_DIR / f"{clip}.npz")["kpts"]


def stream_clip(clip, lookahead, fixed_scale=None):
    ps = PhaseStream(lookahead, fixed_scale=fixed_scale)
    recs = [r for k in kpts_of(clip) for r in ps.push(k)] + ps.flush()
    assert [r[0] for r in recs] == list(range(len(recs)))
    return (np.stack([r[1] for r in recs]), np.array([r[2] for r in recs]), np.stack([r[3] for r in recs]), ps)


def offline_pred(ps, geom, valid):
    X = np.concatenate([geom, valid[:, None]], 1).astype(np.float32)
    W, out = ps.W, []
    for t in range(len(X)):
        idx = np.maximum(np.arange(t - W + 1, t + 1), 0)
        out.append(ps.sess.run(None, {ps.inp: X[idx][None]})[0][0].argmax())
    return np.array(out)


# ------------------------------------------------------------------ 1. parity
TRAIN_STD = B["geom"][B["frame_split"] == 0].std(0)


def parity_clip(ci, lookahead, fixed):
    """fixed=True: stream is given the offline whole-clip scale (tests code/fill equivalence only).
    fixed=False: deployed setting, causal running-median scale."""
    info = clip_info(ci)
    off_scale = float(np.load(DATA / "geom_features" / f"{info['clip']}.npz")["scale"])
    F, V, P, ps = stream_clip(info["clip"], lookahead, off_scale if fixed else None)
    G, fill = info["geom"], info["fill"]
    assert F.shape == G.shape and (V == B["valid"][info["s"]:info["s"] + info["T"]]).all()
    z = np.abs(F - G) / TRAIN_STD
    near_fill = np.zeros(len(F), bool)
    for o in (-1, 0, 1):
        near_fill |= np.roll(fill > 0, o)
    on, off = P.argmax(1), offline_pred(ps, G, V.astype(np.float32))
    r = {"clip": info["clip"], "view": info["view"], "frames": len(F), "filled_frames": int((fill > 0).sum()),
         "max_abs_diff_all": float(np.abs(F - G).max()),
         "max_abs_diff_excl_fill_neighbourhood": float(np.abs(F - G)[~near_fill].max()) if (~near_fill).any() else 0.0,
         "frames_with_diff>1e-4": int((np.abs(F - G).max(1) > 1e-4).sum()),
         "z_diff_p99": float(np.percentile(z.max(1), 99)), "z_diff_max": float(z.max()),
         "phase_agreement_stream_vs_stored": float((on == off).mean())}
    return r, on, off


def main():
    t0 = time.time()
    roles = np.array([role(ci) for ci in range(len(CLIPS))])
    test = np.flatnonzero(roles == "test")
    par_clips = [next(ci for ci in test if clip_info(ci)["view"] == v) for v in ("front", "side", "diagonal")]
    PAR = {}
    for name, D, fixed in (("A_offline_scale_lookahead7", 7, True), ("B_offline_scale_lookahead0", 0, True),
                           ("C_deployed_running_scale_lookahead0", 0, False)):
        PAR[name] = [parity_clip(ci, D, fixed)[0] for ci in par_clips]
        for r in PAR[name]:
            print(f"{name:38s} {r['clip']:34s} max|d| {r['max_abs_diff_all']:.2e} (excl. fill nbhd {r['max_abs_diff_excl_fill_neighbourhood']:.2e}) "
                  f"frames>1e-4 {r['frames_with_diff>1e-4']:3d}/{r['frames']}  z p99 {r['z_diff_p99']:.3f}  phase agree {r['phase_agreement_stream_vs_stored']:.4f}")
    # A: exact code equivalence (only clip-start fill can differ: offline borrows the next observed value)
    for r in PAR["A_offline_scale_lookahead7"]:
        assert r["max_abs_diff_excl_fill_neighbourhood"] <= 1e-4, r
    # C: deployed setting over ALL held-out clips -- downstream agreement (no tuning, agreement only)
    from rep_counter import RepCounter as _RC
    def _count(pred):
        rc, n = _RC(), 0
        for t, p in enumerate(pred):
            n += rc.update(t, p) is not None
        return n
    agree = tot = same = 0
    for ci in test:
        r, on, off = parity_clip(ci, 0, False)
        agree += int((on == off).sum()); tot += len(on); same += _count(on) == _count(off)
    DOWN = {"heldout_clips": len(test), "frame_phase_agreement": agree / tot, "clips_identical_rep_count(default RepCounter)": int(same)}
    print("deployed-scale downstream on all held-out clips:", DOWN)
    TOL = {"A_max_abs_diff_excl_fill_neighbourhood": 1e-4, "C_frame_phase_agreement_all_heldout": 0.98}
    assert DOWN["frame_phase_agreement"] >= TOL["C_frame_phase_agreement_all_heldout"], DOWN
    LOOKAHEAD = 0
    print(f"parity OK -> lookahead {LOOKAHEAD} (pipeline lag {LOOKAHEAD + 1} frame)")
    # ------------------------------------------------------------------ 2. stream all clips once
    S = {}
    for ci in range(len(CLIPS)):
        F, V, P, _ = stream_clip(CLIPS[ci], LOOKAHEAD)
        S[ci] = (F, V, P)
    print(f"streamed {len(S)} clips in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------ 3. tune rep counter on VAL subjects
    def count_clip(P, cfg):
        rc, evs = RepCounter(**cfg), []
        for t, p in enumerate(P):
            ph = int(p.argmax())
            ev = rc.update(t, ph)
            if ev:
                evs.append(ev)
        return evs

    def count_err(cis, cfg):
        e = np.array([len(count_clip(S[ci][2], cfg)) - len(clip_info(ci)["reps"]) for ci in cis])
        return {"clips": len(cis), "mae": float(np.abs(e).mean()), "exact": float((e == 0).mean()),
                "ref_reps": int(sum(len(clip_info(ci)["reps"]) for ci in cis)), "signed_total": int(e.sum())}

    grid = [dict(k_in=a, k_out=b, k_bounce=c, require=r) for a, b, c, r in
            itertools.product([1, 2, 3, 4, 5, 6], [1, 2, 3, 4], [0, 1, 2, 3, 4], list(REQUIRE))]
    val = np.flatnonzero(roles == "val")
    tr = np.flatnonzero(roles == "train")
    scored = [(count_err(val, g), g) for g in grid]
    rank = lambda x: (x[0]["mae"], -x[0]["exact"], abs(x[0]["signed_total"]), -x[1]["k_in"], -x[1]["k_out"])
    best_rc = min(scored, key=rank)
    RC = best_rc[1]
    rc_tune = {"objective": "min per-clip |count error| on val subjects; ties -> exact-match rate -> |signed error| -> more smoothing (larger k_in, k_out)",
               "best_per_requirement_val": {r: min([x for x in scored if x[1]["require"] == r], key=rank)[0] | {"cfg": min([x for x in scored if x[1]["require"] == r], key=rank)[1]} for r in REQUIRE},
               "grid_size": len(grid), "chosen": RC, "val": best_rc[0], "train_subjects_check": count_err(tr, RC),
               "top5_val": [{"cfg": g, **m} for m, g in sorted(scored, key=rank)[:5]]}
    print("rep counter:", RC, "val", best_rc[0], "train", rc_tune["train_subjects_check"])

    # ------------------------------------------------------------------ 4. tune form rules on TRAINING-SPLIT subjects
    trainsplit = np.flatnonzero(roles != "test")

    def rep_rows(cis, fault):
        rows = []
        fi = FAULTS.index(fault)
        for ci in cis:
            info = clip_info(ci)
            if info["view"] not in OBSERVABLE[fault]:
                continue
            F, V, _ = S[ci]
            for a, b, fv in info["reps"]:
                if fv[fi] in (0, 1):
                    rows.append((F[a:b + 1], V[a:b + 1], int(fv[fi]), info["view"]))
        return rows

    rules, rule_tune = {}, {}
    for fault, cands in CANDIDATES.items():
        rows = rep_rows(trainsplit, fault)
        res = []
        for feat, stat, op in cands:
            vals = [(rep_stat(F, V, feat, stat), y, v) for F, V, y, v in rows]
            vals = [x for x in vals if x[0] is not None]
            thr, f1, tp, fp, fn, tn = best_threshold([x[0] for x in vals], [x[1] for x in vals], op)
            res.append({"feature": feat, "stat": stat, "op": op, "threshold": thr, "train_f1": f1,
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "_vals": vals})
        best = max(res, key=lambda r: r["train_f1"])
        per_view = {}
        for v in sorted(OBSERVABLE[fault]):
            vv = [x for x in best["_vals"] if x[2] == v]
            npos = sum(x[1] for x in vv)
            if npos >= MIN_POS_PER_VIEW:
                t, f1v, *_ = best_threshold([x[0] for x in vv], [x[1] for x in vv], best["op"])
                per_view[v] = t
            rule_tune.setdefault(fault, {}).setdefault("train_positives_per_view", {})[v] = int(npos)
        rules[fault] = {"feature": best["feature"], "stat": best["stat"], "op": best["op"],
                        "threshold": best["threshold"], "threshold_per_view": per_view, "tuned": True,
                        "tuned_on": "training-split subjects (20), ground-truth rep segments, streaming features"}
        rule_tune[fault].update({"train_reps": len(rows), "train_positives": int(sum(r[2] for r in rows)),
                                 "candidates": [{k: v for k, v in r.items() if k != "_vals"} for r in res],
                                 "chosen": rules[fault]})
        print(f"{fault}: {best['feature']}[{best['stat']}] {best['op']} {best['threshold']:.3f} train F1 {best['train_f1']:.3f} per-view {per_view}")

    CFG = {"lookahead": LOOKAHEAD, "lag_frames": LOOKAHEAD + 1, "rep_counter": RC,
           "form_rules": rules, "knee_valgus_rule": KNEE_VALGUS_RULE,
           "note": "rep counter tuned on internal-val subjects; rules tuned on training-split subjects; nothing tuned on held-out"}
    json.dump(CFG, open(CONFIG_PATH, "w"), indent=1)

    # ------------------------------------------------------------------ 5. held-out evaluation with the full pipeline
    per_clip, gt_rows, pr_rows, e2e_extra = [], {f: [] for f in FAULTS}, {f: [] for f in FAULTS}, {f: {"fn_missed": 0, "fp_spurious": 0} for f in FAULTS}
    ph_true, ph_pred = [], []
    for ci in test:
        info = clip_info(ci)
        recs, p = run_clip(kpts_of(info["clip"]), info["view"], CFG)
        F, V = np.stack(p.F), np.array(p.V)
        assert np.allclose(F, S[ci][0])
        ph_true.append(info["phase"]); ph_pred.append(np.array([r["phase"] for r in recs]))
        pred = [r["rep"] for r in recs if r["rep"] and not r["rep"].get("rejected")]
        per_clip.append({"clip": info["clip"], "view": info["view"], "ref_reps": len(info["reps"]), "pred_reps": len(pred)})
        # GT segmentation
        for a, b, fv in info["reps"]:
            fr = evaluate_rep(F[a:b + 1], V[a:b + 1], info["view"], CFG)
            for k, f in enumerate(FAULTS):
                if fv[k] in (0, 1):
                    gt_rows[f].append((int(fv[k]), fr[f]["flag"], info["view"]))
        # predicted segmentation: match by IoU >= 0.3 (greedy, one-to-one)
        pairs = sorted(((min(b, ev["end"]) - max(a, ev["start"]) + 1) / (max(b, ev["end"]) - min(a, ev["start"]) + 1), gi, pi)
                       for gi, (a, b, _) in enumerate(info["reps"]) for pi, ev in enumerate(pred))
        mg, mp = {}, set()
        for iou, gi, pi in reversed(pairs):
            if iou >= 0.3 and gi not in mg and pi not in mp:
                mg[gi] = pi; mp.add(pi)
        per_clip[-1].update(matched=len(mg), missed=len(info["reps"]) - len(mg), spurious=len(pred) - len(mp))
        for gi, (a, b, fv) in enumerate(info["reps"]):
            for k, f in enumerate(FAULTS):
                if fv[k] not in (0, 1):
                    continue
                if gi in mg:
                    pr_rows[f].append((int(fv[k]), pred[mg[gi]]["form"][f]["flag"], info["view"]))
                elif fv[k] == 1:
                    e2e_extra[f]["fn_missed"] += 1
        for pi, ev in enumerate(pred):
            if pi not in mp:
                for f in FAULTS:
                    if ev["form"][f]["flag"]:
                        e2e_extra[f]["fp_spurious"] += 1

    def conf(rows):
        y = np.array([r[0] for r in rows]); fl = [r[1] for r in rows]
        p = np.array([bool(x) for x in fl])
        tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
        fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
        return {"n": len(rows), "positives": int((y == 1).sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "abstained": int(sum(x is None for x in fl)),
                "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                "fpr": fp / (fp + tn) if fp + tn else None}

    def rule_report(rows_by_fault):
        out = {}
        for f in FAULTS:
            rows = rows_by_fault[f]
            out[f] = {"all_observable_views": conf(rows),
                      "by_view": {v: conf([r for r in rows if r[2] == v]) for v in sorted(OBSERVABLE[f])}}
        return out

    yt, yp = np.concatenate(ph_true), np.concatenate(ph_pred)
    cm = np.bincount(yt * 4 + yp, minlength=16).reshape(4, 4)
    f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)
    errs = np.array([c["pred_reps"] - c["ref_reps"] for c in per_clip])
    GT, PR = rule_report(gt_rows), rule_report(pr_rows)
    E2E = {}
    for f in FAULTS:
        c = dict(PR[f]["all_observable_views"])
        c["fn"] += e2e_extra[f]["fn_missed"]; c["fp"] += e2e_extra[f]["fp_spurious"]
        c.update(missed_positive_reps=e2e_extra[f]["fn_missed"], spurious_rep_firings=e2e_extra[f]["fp_spurious"],
                 precision=c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None,
                 recall=c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None)
        E2E[f] = c
    for d in (GT, PR):
        d["knee_valgus"]["status"] = KNEE_VALGUS_RULE["status"]
    out = {
        "scope": "held-out subjects only: " + ", ".join(sorted({clip_info(ci)["subject"] for ci in test})) + f" ({len(test)} clips), cached YOLO pose streamed frame by frame",
        "model": "model_geom.onnx", "pipeline_config": CFG,
        "streaming_parity": {"clips": [CLIPS[c] for c in par_clips], "tolerance": TOL,
                             "definitions": {"A": "stream given the offline scale, lookahead 7: code/fill equivalence",
                                             "B": "stream given the offline scale, lookahead 0: + causal hold instead of interpolation",
                                             "C": "deployed: lookahead 0 + causal running-median scale",
                                             "z_diff": "|stream - stored| / training-split std per feature"},
                             "tests": PAR, "deployed_downstream_all_heldout": DOWN, "chosen_lookahead": LOOKAHEAD,
                             "residual": "only the causal running-median body scale differs from the offline whole-clip median; it is largest in the first seconds of a clip"},
        "phase_streaming_heldout": {"acc": float(np.diag(cm).sum() / cm.sum()), "macro_f1": float(f1.mean()),
                                    "f1_per_phase": dict(zip(META["label_mapping"].values(), f1.round(4).tolist())),
                                    "confusion": cm.tolist()},
        "rep_count": {"reference": "auto-generated reps from labels.csv (1.3d) -- NOT an independent count",
                      "clips": len(per_clip), "ref_reps_total": int(sum(c["ref_reps"] for c in per_clip)),
                      "pred_reps_total": int(sum(c["pred_reps"] for c in per_clip)),
                      "exact_match_clips": int((errs == 0).sum()), "within_1_clips": int((np.abs(errs) <= 1).sum()),
                      "mae_per_clip": float(np.abs(errs).mean()),
                      "segment_match_iou>=0.3": {k: int(sum(c[k] for c in per_clip)) for k in ("matched", "missed", "spurious")},
                      "per_clip": per_clip},
        "form_rules_gt_segmentation": GT,
        "form_rules_predicted_segmentation_matched_reps": PR,
        "form_rules_predicted_segmentation_end_to_end": E2E,
        "notes": ["Excluded per rule: reps tagged other/unclear (excl) and views where the fault is not observable (n/a).",
                  "Abstained reps (too few valid frames) count as 'not fired'.",
                  "Form-rule numbers measure agreement with the project author's own per-rep tags, not external ground truth.",
                  "knee_valgus: unvalidated (0 positive examples in the dataset); only its false-positive rate is meaningful."],
        "tuning": {"rep_counter": rc_tune, "form_rules": rule_tune},
    }
    json.dump(out, open(ARTIFACTS / "form_metrics.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("phase_streaming_heldout", "rep_count")}, default=float)[:1500])
    for name, d in (("GT seg", GT), ("pred seg (matched)", PR)):
        for f in FAULTS:
            print(name, f, d[f]["all_observable_views"], {v: (x["tp"], x["fp"], x["fn"], x["tn"]) for v, x in d[f]["by_view"].items()})
    for f in FAULTS:
        print("end-to-end", f, E2E[f])
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
