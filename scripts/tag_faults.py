"""Task 1.3b: clip-by-clip fault tagging for "bad" squat clips. You watch, you type; nothing is guessed.

Run:   .venv/bin/python scripts/tag_faults.py            (resumes at the first untagged bad clip)
       .venv/bin/python scripts/tag_faults.py --redo subject_004_squat_bad_side

Video window keys:  space = pause/resume   r = restart   any other key / end of clip = go to the prompt
Prompt syntax (rep numbers are the "REP n" shown on the video; abbreviations kv / sd / fl allowed):
    all: kv                    every rep has knee_valgus
    all: good                  no fault visible on any rep
    3: ot                      "other": clearly bad but not kv/sd/fl (any view; cannot be combined with
                               kv/sd/fl or good; excluded from per-rule precision/recall in 4.3)
    5-8: uc                    "unclear": you cannot judge the rep (e.g. privacy blur over the body); any view;
                               cannot be combined with anything; excluded from per-rule precision/recall in 4.3
    1-3: sd;fl, 4: good, 5: sd per-rep ("," between reps, ";" between faults on one rep)
    r     replay the clip       x   auto rep count/boundaries look wrong -> flag clip, leave untagged
    s     skip for now          q   save and quit
Faults that are not observable from the clip's view are refused (they stay "n/a").
CSV encoding for an "other" / "unclear" rep: other=1 / unclear=1, observable fault columns = "excl",
form_tags=other / unclear.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FAULTS, LABELS_CSV, OBSERVABLE, RAW_DIR  # noqa: E402

ABBR = {"kv": "knee_valgus", "sd": "shallow_depth", "fl": "forward_lean", "good": "good", "ot": "other", "other": "other", "uc": "unclear", "unclear": "unclear",
        **{f: f for f in FAULTS}}


def load():
    with open(LABELS_CSV) as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, list(rd)


def save(fields, rows):
    tmp = LABELS_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(LABELS_CSV)


def play(path, reps):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay, paused, win = max(1, int(1000 / fps)), False, "MotionIQ tagging"
    while True:
        if not paused:
            ok, fr = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            lab = next((f"REP {i + 1}" for i, (a, b) in enumerate(reps) if a <= t <= b), "standing")
            scale = 720 / fr.shape[0]
            fr = cv2.resize(fr, (int(fr.shape[1] * scale), 720))
            cv2.rectangle(fr, (0, 0), (fr.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(fr, f"{lab}   ({len(reps)} reps)  t={t:5.1f}s", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if lab != "standing" else (200, 200, 200), 2)
            cv2.imshow(win, fr)
        k = cv2.waitKey(delay) & 0xFF
        if k == 255:
            continue
        if k == ord(" "):
            paused = not paused
        elif k == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def parse(text, n_reps, view):
    """Return {rep_index0: set_of_faults} covering every rep, or raise ValueError."""
    out = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if ":" not in part:
            raise ValueError(f"'{part}': expected '<reps>: <tags>'")
        who, tags = (x.strip().lower() for x in part.split(":", 1))
        if who == "all":
            idx = list(range(n_reps))
        else:
            m = re.fullmatch(r"(\d+)(?:-(\d+))?", who)
            if not m:
                raise ValueError(f"bad rep spec '{who}'")
            a, b = int(m[1]), int(m[2] or m[1])
            if not 1 <= a <= b <= n_reps:
                raise ValueError(f"reps must be within 1..{n_reps}")
            idx = list(range(a - 1, b))
        fs = set()
        for t in [t.strip() for t in tags.split(";") if t.strip()]:
            if t not in ABBR:
                raise ValueError(f"unknown tag '{t}' (use kv, sd, fl, ot, uc, good)")
            fs.add(ABBR[t])
        if not fs:
            raise ValueError(f"no tags for '{who}'")
        if "good" in fs and len(fs) > 1:
            raise ValueError("'good' cannot be combined with a fault")
        for x in ("other", "unclear"):
            if x in fs and len(fs) > 1:
                raise ValueError(f"'{x}' cannot be combined with any other tag")
        fs.discard("good")
        bad = [f for f in fs if f in OBSERVABLE and view not in OBSERVABLE[f]]
        if bad:
            raise ValueError(f"{bad} not observable from {view} view (allowed: "
                             f"{[f for f in FAULTS if view in OBSERVABLE[f]]})")
        for i in idx:
            out[i] = fs
    missing = [i + 1 for i in range(n_reps) if i not in out]
    if missing:
        raise ValueError(f"no tags given for rep(s) {missing}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", help="clip_id to re-tag even if already tagged")
    args = ap.parse_args()
    fields, rows = load()

    clips = {}
    for r in rows:
        if r["quality"] == "bad":
            clips.setdefault(r["clip_id"], []).append(r)
    todo = [c for c, rs in clips.items()
            if (args.redo == c) or (not args.redo and rs[0]["tag_source"] == "pending")]
    done = sum(rs[0]["tag_source"] == "user" for rs in clips.values())
    print(f"{len(clips)} bad clips, {done} tagged, {len(todo)} to go.\n")

    for n, cid in enumerate(todo, 1):
        rs = sorted(clips[cid], key=lambda r: int(r["rep_idx"]))
        view = rs[0]["view"]
        reps = [(float(r["descent_start_s"]), float(r["ascent_end_s"])) for r in rs if r["rep_idx"] != "-1"]
        n_reps = max(len(reps), 1)
        allowed = [f for f in FAULTS if view in OBSERVABLE[f]] + ["other", "unclear"]
        while True:
            print(f"[{n}/{len(todo)}] {cid}  view={view}  auto reps={len(reps)}  taggable={allowed}")
            play(RAW_DIR / rs[0]["filename"], reps)
            while True:
                ans = input("tags> ").strip()
                if ans in ("r", "s", "q", "x") or ans:
                    break
            if ans == "r":
                continue
            break
        if ans == "q":
            break
        if ans == "s":
            continue
        if ans == "x":
            for r in rs:
                r["notes"] = (r["notes"] + ";auto_reps_wrong").strip(";")
            save(fields, rows)
            print("  flagged auto_reps_wrong, left untagged\n")
            continue
        while True:
            try:
                tags = parse(ans, n_reps, view)
                break
            except ValueError as e:
                print(f"  ! {e}")
                ans = input("tags> ").strip()
        for i, r in enumerate(rs):
            fs = tags[i if r["rep_idx"] != "-1" else 0]
            excl = bool(fs & {"other", "unclear"})
            for f in FAULTS:
                if view in OBSERVABLE[f]:
                    r[f] = "excl" if excl else ("1" if f in fs else "0")
            r["other"] = "1" if "other" in fs else "0"
            r["unclear"] = "1" if "unclear" in fs else "0"
            r["form_tags"] = ";".join(sorted(fs)) or "good"
            r["tag_source"] = "user"
        save(fields, rows)
        print("  saved: " + ", ".join(f"rep{i + 1}={r['form_tags']}" for i, r in enumerate(rs)) + "\n")
    print("Progress saved to", LABELS_CSV)


if __name__ == "__main__":
    main()
