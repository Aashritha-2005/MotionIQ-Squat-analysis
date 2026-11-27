"""Task 5.1: live squat demo. Webcam (or a recorded clip) -> YOLO11-pose -> StreamingFeatures -> model_geom.onnx
-> rep counter -> form rules -> overlay. Same code path for --camera and --video (scripts/pipeline.Pipeline).

  .venv/bin/python scripts/run_demo.py --view side                       # webcam 0, side view
  .venv/bin/python scripts/run_demo.py --view front --camera 1
  .venv/bin/python scripts/run_demo.py --view diagonal --video data/raw/subject_004_squat_bad_diagonal.mp4 --record out.mp4
  .venv/bin/python scripts/run_demo.py --view side --selftest-camera     # 10 s camera smoke test, nobody needed

Frames are sampled at config.FPS (15) by timestamp (camera: wall clock; video: the same source-frame indices as
Phase 2's extract_pose), so the model sees the training frame rate even if the camera runs at 30 fps.
Processing uses the un-mirrored frame; --mirror only flips what is displayed (default on for the camera).
Keys: q / Esc = quit.
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FAULTS, FPS, OBSERVABLE, POSE_MODEL  # noqa: E402
from form_rules import load_config, rep_stat, rule_for  # noqa: E402
from pipeline import MODEL, Pipeline  # noqa: E402
from prepare_labels import pick_person  # noqa: E402
from rep_counter import PHASES  # noqa: E402
from extract_features import FEATURE_NAMES  # noqa: E402
SW_IDX = FEATURE_NAMES.index("shoulder_width")

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
PHASE_COL = {"standing": (200, 200, 200), "descent": (0, 165, 255), "bottom": (60, 60, 230), "ascent": (90, 200, 90)}
CALIB_MIN_VALID, CALIB_WINDOW, CALIB_TOL = 45, 15, 0.02   # >= 3 s of valid frames and scale stable within 2% over 1 s
FEEDBACK_TTL = 10.0      # s: per-rep verdicts are hidden this long after the last rep (Phase 7, F8)
UPPER_MISSING_WARN = 15  # processed frames (1 s) with the lower body visible but shoulders not observed (F3)
VIEW_WINDOW_S = 2.0      # s of recent shoulder-width readings used by the view check
VIEW_SIDE_MAX = 0.245    # shoulder width / torso: midpoint between training side max (0.16) and diagonal min (0.33) (F4)
LABEL = {"shallow_depth": "depth", "forward_lean": "lean", "knee_valgus": "knee valgus (experimental)"}
VERDICT = {"shallow_depth": ("depth OK", "too shallow"), "forward_lean": ("torso OK", "leaning"),
           "knee_valgus": ("knees OK", "knees caving (exp.)")}

CAMERA_HELP = """
Could not read from camera {idx}. On macOS this is almost always the camera permission:
  System Settings -> Privacy & Security -> Camera -> turn ON the app you launched this from
  (Terminal, iTerm, VS Code, or your IDE), then QUIT and reopen that app and run the command again.
Also check: no other app (Zoom, FaceTime, Photo Booth) is using the camera, and try --camera 1 for an
external / Continuity camera.
"""


def open_source(a):
    if a.video:
        cap = cv2.VideoCapture(a.video)
        if not cap.isOpened():
            sys.exit(f"cannot open video {a.video}")
        return cap
    cap = cv2.VideoCapture(a.camera, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(a.camera)
    ok = cap.isOpened()
    if ok:
        for _ in range(20):                     # first frames can be empty while the camera warms up
            ok, fr = cap.read()
            if ok and fr is not None:
                break
            time.sleep(0.05)
    if not ok:
        sys.exit(CAMERA_HELP.format(idx=a.camera))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


CAM = {"raw": 0, "read_fail": 0, "elapsed": 0.0, "shape": None, "t_capture_ms": 0.0, "skipped": 0, "raw_index": 0}
SAVE_AT = {30, 75, 140}                  # processed-frame indices whose annotated frame the self-test saves
SELFTEST_DIR = Path(__file__).resolve().parent.parent / "data" / "demo" / "selftest"   # git-ignored


def frames(cap, a):
    """Yield (frame_bgr, t_seconds) at FPS. Video: Phase 2 source-frame indices. Camera: wall-clock sampling."""
    if a.video:
        # streaming form of extract_pose's targets round(k / FPS * src_fps); does not trust the container's frame
        # count (unreliable for webm). Identical frames for the dataset clips.
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = k = skipped = 0
        tc = time.perf_counter()
        while True:
            ok, fr = cap.read()
            if not ok:
                return
            while int(round(k / FPS * src_fps)) < i:   # source slower than FPS never happens here; keep k in step
                k += 1
            if int(round(k / FPS * src_fps)) == i:
                CAM.update(t_capture_ms=(time.perf_counter() - tc) * 1e3, skipped=skipped, raw_index=i)
                yield fr, k / FPS
                tc, skipped = time.perf_counter(), 0
                k += 1
            else:
                skipped += 1
            i += 1
    t0, nxt, misses, skipped = time.monotonic(), 0.0, 0, 0
    tc = time.perf_counter()
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            misses += 1
            CAM["read_fail"] += 1
            if misses > 30:
                sys.exit(CAMERA_HELP.format(idx=a.camera))
            continue
        misses = 0
        CAM["raw"] += 1
        CAM["shape"] = fr.shape
        t = time.monotonic() - t0
        CAM["elapsed"] = t
        if a.selftest_camera and t >= a.selftest_seconds:
            return
        if t + 1e-3 >= nxt:
            nxt = max(nxt + 1 / FPS, t)          # if processing falls behind, skip ahead instead of bursting
            CAM.update(t_capture_ms=(time.perf_counter() - tc) * 1e3, skipped=skipped, raw_index=CAM["raw"] - 1)
            yield fr, t
            tc, skipped = time.perf_counter(), 0
        else:
            skipped += 1


def detect(model, frame, device, info=None):
    """YOLO11-pose on one frame -> (17,3) keypoints of the main person (NaN if no person box).
    If `info` (dict) is given it is filled with detection diagnostics for the self-test."""
    r = model(frame, verbose=False, device=device)[0]
    k = pick_person(r)
    kp = np.full((17, 3), np.nan, np.float32)
    if k is not None and r.keypoints is not None:
        kp[:, :2] = r.keypoints.xy[k].cpu().numpy()
        kp[:, 2] = r.keypoints.conf[k].cpu().numpy()
    if info is not None:
        n = 0 if r.boxes is None else len(r.boxes)
        conf = r.boxes.conf.cpu().numpy() if n else np.zeros(0)
        xyxy = r.boxes.xyxy.cpu().numpy() if n else np.zeros((0, 4))
        H, W = frame.shape[:2]
        info.update(n_boxes=n, max_conf=float(conf.max()) if n else 0.0,
                    chosen_conf=float(conf[k]) if k is not None else 0.0,
                    area_frac=float((xyxy[k, 2] - xyxy[k, 0]) * (xyxy[k, 3] - xyxy[k, 1]) / (H * W)) if k is not None else 0.0,
                    boxes=[(*map(float, b), float(c)) for b, c in zip(xyxy, conf)])
    return kp


def put(img, text, org, scale=0.7, col=(255, 255, 255), th=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def draw(frame, kp, st, a):
    img = frame.copy()
    for i, j in SKELETON:
        if np.nan_to_num(kp[i, 2]) > 0.3 and np.nan_to_num(kp[j, 2]) > 0.3:
            cv2.line(img, tuple(map(int, kp[i, :2])), tuple(map(int, kp[j, :2])), (255, 200, 0), 3, cv2.LINE_AA)
    for x, y, c in kp[5:]:
        if np.nan_to_num(c) > 0.3:
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1, cv2.LINE_AA)
    if a.mirror:
        img = cv2.flip(img, 1)                  # display only; processing used the un-mirrored frame
    h, w = img.shape[:2]
    s = min(h / 720, w / 640)                  # fits landscape webcams and portrait phone clips
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (min(int(470 * s), w), h), (0, 0, 0), -1)
    img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
    y = int(45 * s)
    ph = st["phase"]
    put(img, f"REPS {st['count']}", (int(15 * s), y), 1.4 * s, (255, 255, 255), 3); y += int(50 * s)
    put(img, f"phase: {ph}", (int(15 * s), y), 0.9 * s, PHASE_COL.get(ph, (255, 255, 255))); y += int(35 * s)
    put(img, f"view: {a.view}", (int(15 * s), y), 0.6 * s, (200, 200, 200)); y += int(40 * s)
    for f in FAULTS:
        if a.view not in OBSERVABLE[f]:
            continue
        lv = st["live"].get(f)
        txt, col = f"{LABEL[f]}: --", (200, 200, 200)
        if lv is not None:
            txt = f"{LABEL[f]}: {lv['value']:+.2f} (thr {lv['thr']:+.2f})"
            col = (60, 60, 255) if lv["flag"] else (90, 220, 90)
        put(img, txt, (int(15 * s), y), 0.6 * s, col); y += int(30 * s)
    y += int(15 * s)
    fresh = [x for x in st["feedback"] if st["t"] - st["last_rep_t"] <= FEEDBACK_TTL]
    for j, (msg, col) in enumerate(fresh[::-1][:5]):
        put(img, ("> " if j == 0 else "  ") + msg, (int(15 * s), y), (0.7 if j == 0 else 0.55) * s, col); y += int(30 * s)
    yb = h - int(55 * s)
    if not st["valid"]:
        put(img, "lower body not visible - step back", (int(15 * s), yb), 0.7 * s, (0, 0, 255)); yb -= int(35 * s)
    elif st["upper_missing"] >= UPPER_MISSING_WARN:
        put(img, "upper body out of frame - move camera back/higher", (int(15 * s), yb), 0.65 * s, (0, 0, 255)); yb -= int(35 * s)
    if st["view_warn"]:
        put(img, st["view_warn"], (int(15 * s), yb), 0.6 * s, (0, 165, 255)); yb -= int(35 * s)
    if st["calibrating"]:
        put(img, "calibrating - stand still, head to feet in view", (int(15 * s), yb), 0.75 * s, (0, 215, 255))
    if a.video:   # recorded clip, not a live camera
        label = "REPLAY MODE - recorded clip"
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, 2)
        x0 = w - tw - int(20 * s)
        cv2.rectangle(img, (x0 - int(8 * s), int(10 * s)), (w - int(8 * s), int(20 * s) + th_ + int(8 * s)), (0, 0, 0), -1)
        put(img, label, (x0, int(20 * s) + th_), 0.7 * s, (0, 215, 255))
        if a.credit:
            put(img, a.credit, (int(15 * s), h - int(42 * s)), 0.42 * s, (200, 200, 200), 1)
    put(img, f"{st['fps']:.1f} fps   {st['lat_ms']:.0f} ms/frame (pose {st['pose_ms']:.0f})", (int(15 * s), h - int(20 * s)), 0.55 * s, (220, 220, 220), 1)
    return img


def selftest_report(log, a, L):
    n = len(log["latency_ms"])
    det = np.array(log.get("max_conf", []), float)
    has = np.array([not x for x in log.get("no_person", [])], bool)
    valid = np.array(log.get("valid", []), bool)
    reps = log["count"][-1] if log.get("count") else 0
    errs = log.get("errors", [])
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else 0.0
    print("\n=== self-test ===")
    if a.selftest_camera:
        el = max(CAM["elapsed"], 1e-6)
        print(f"camera: {CAM['raw']} frames in {el:.1f} s = {CAM['raw'] / el:.1f} fps native, {CAM['read_fail']} failed reads, "
              f"frame size {CAM['shape']}; processed {n} = {n / el:.1f} fps")
    print(f"person box detected (YOLO, conf >= 0.25): {has.mean():.0%} of {n} frames")
    if has.any():
        print(f"  max person-box confidence on frames with a box: median {np.median(det[has]):.2f}, "
              f"p10 {q(det[has], 10):.2f}, p90 {q(det[has], 90):.2f}, max {det.max():.2f}; "
              f"chosen box = {np.median(np.array(log['area_frac'])[has]):.0%} of the frame (median)")
    print(f"lower body visible (valid: person AND mean hip/knee/ankle keypoint conf >= 0.5): {valid.mean():.0%} of frames")
    print(f"state 'no person / lower body not visible': {(~valid).mean():.0%} of frames; reps counted: {reps}")
    print(f"latency: median {np.median(L):.1f} ms, p95 {q(L, 95):.1f} ms, max {L.max():.1f} ms; exceptions: {len(errs)}")
    checks = []
    if a.selftest_camera:
        el = max(CAM["elapsed"], 1e-6)
        checks += [("frames arrive from the camera", CAM["raw"] > 0),
                   (f"processed at >= {0.9 * FPS:.1f} fps", n / el >= 0.9 * FPS)]
    checks += [("no exceptions", not errs), (f"latency p95 < {1000 / FPS:.1f} ms", q(L, 95) < 1000 / FPS)]
    if a.expect == "empty":
        checks += [("empty scene -> 'no person / lower body not visible' on >= 90% of frames", (~valid).mean() >= 0.9),
                   ("empty scene -> 0 reps", reps == 0)]
        if has.mean() > 0.1:
            print(f"WARNING: a person box was found on {has.mean():.0%} of frames of an empty scene "
                  f"(median conf {np.median(det[has]):.2f}) -- see the saved frames for what YOLO is detecting")
    elif a.expect == "person":
        checks += [("person in view -> person box on >= 90% of frames", has.mean() >= 0.9)]
        if valid.mean() < 0.8:
            print("NOTE: lower body not visible on many frames -- step back until hips, knees and ankles are in frame")
    else:
        print("(pass --expect empty or --expect person to check the scenario)")
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if a.selftest_camera and a.selftest_frames:
        print(f"annotated frames saved to {SELFTEST_DIR.relative_to(Path.cwd()) if SELFTEST_DIR.is_relative_to(Path.cwd()) else SELFTEST_DIR}")
    print("RESULT:", "PASS" if all(c[1] for c in checks) else "FAIL")


def upper_missing_update(run, valid, obs):
    """Consecutive frames with the lower body visible but the shoulders not observed (F3 framing warning)."""
    return run + 1 if (valid and obs is not None and not obs[[5, 6]].all()) else 0


def view_warning(sw_hist, t, view):
    """F4 view check: median shoulder width (torso units) over the last VIEW_WINDOW_S seconds of frames with both
    shoulders observed, vs VIEW_SIDE_MAX (midpoint between the training side-view max and diagonal min)."""
    recent = [v for tt, v in sw_hist if t - tt <= VIEW_WINDOW_S]
    if len(recent) < 8:
        return ""
    msw = float(np.median(recent))
    if view == "side" and msw > VIEW_SIDE_MAX:
        return "you do not look side-on - turn side-on or use --view front/diagonal"
    if view != "side" and msw <= VIEW_SIDE_MAX:
        return "you look side-on - use --view side"
    return ""


def calibrating(feat):
    v = sum(feat.valid)
    if v < CALIB_MIN_VALID or len(feat.torso_fifo) < CALIB_WINDOW + 1:
        return True
    # scale stability: median of all-but-last-second vs now
    old = np.median(feat.torso_fifo[:-CALIB_WINDOW])
    return abs(feat.scale() / old - 1) > CALIB_TOL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", required=True, choices=["side", "front", "diagonal"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="replay a recorded clip through the same code path")
    ap.add_argument("--record", help="save the overlay video (mp4) at 15 fps")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=None,
                    help="flip the DISPLAY horizontally (default: on for camera, off for video)")
    ap.add_argument("--headless", action="store_true", help="no window (testing)")
    ap.add_argument("--log", help="write per-frame phases and rep events as JSON (testing)")
    ap.add_argument("--credit", help="source credit line drawn in replay mode (e.g. dataset attribution)")
    ap.add_argument("--selftest-camera", action="store_true",
                    help="camera smoke test: run the full loop on the webcam for --selftest-seconds (nobody needs to be in"
                         " frame), then print frame-rate / no-person / crash / latency checks and exit")
    ap.add_argument("--selftest-seconds", type=float, default=10.0)
    ap.add_argument("--expect", choices=["empty", "person"],
                    help="self-test scenario: 'empty' = nobody in frame, 'person' = someone standing in full view")
    ap.add_argument("--selftest-frames", action=argparse.BooleanOptionalAction, default=True,
                    help="save 3 annotated frames (YOLO boxes + confidence) to data/demo/selftest/ (local, git-ignored)")
    ap.add_argument("--report", action="store_true", help="print the detection/state summary also for --video (testing)")
    ap.add_argument("--record-raw", help="save the raw processed frames (un-mirrored, no overlay) as mp4 at 15 fps, so a "
                                         "live session can be replayed exactly with --video")
    ap.add_argument("--log-session", help="save a per-frame debug log (.npz): timestamps, raw keypoints, detection info,"
                                          " scale, features, phase probabilities, counter state, rule values, per-stage timings")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing existing --record/--record-raw/--log-session files")
    a = ap.parse_args()
    existing = [p for p in (a.record, a.record_raw, a.log_session, a.log) if p and Path(p).exists()]
    if existing and not a.overwrite:   # earlier live runs were silently overwritten (Phase 7 F14)
        sys.exit("refusing to overwrite existing output file(s): " + ", ".join(existing)
                 + "\nuse a new run id in the file names, or pass --overwrite")
    if a.selftest_camera and a.video:
        sys.exit("--selftest-camera tests the webcam; do not combine it with --video")
    if a.mirror is None:
        a.mirror = a.video is None

    from ultralytics import YOLO
    yolo = YOLO(POSE_MODEL)
    detect(yolo, np.zeros((720, 1280, 3), np.uint8), a.device)   # warm-up so the first frame's latency is not ~1 s
    cfg = load_config()
    pipe = Pipeline(a.view, cfg, model=a.model)
    cap = open_source(a)
    writer, win = None, "MotionIQ"
    st = {"phase": "standing", "count": 0, "live": {}, "feedback": deque(maxlen=5), "calibrating": True,
          "valid": True, "fps": 0.0, "lat_ms": 0.0, "pose_ms": 0.0, "t": 0.0, "last_rep_t": -1e9,
          "upper_missing": 0, "view_warn": ""}
    sw_hist = deque(maxlen=30)   # (t, shoulder width) on finalised frames with both shoulders observed (view check, F4)
    log = {"frames": [], "reps": [], "latency_ms": [], "pose_ms": []}
    S = {k: [] for k in ("t", "raw_index", "skipped", "kpts", "n_boxes", "max_conf", "chosen_conf", "area_frac", "scale",
                         "valid", "calibrating", "count", "moving", "smooth", "live_depth", "live_lean", "live_valgus",
                         "t_capture", "t_pose", "t_features", "t_onnx", "t_logic", "t_draw", "t_display", "t_record",
                         "view_warn", "upper_missing_warn")}
    SF = {k: [] for k in ("frame", "feats", "probs", "phase")}   # per finalised frame (pipeline lag = 1 frame)
    raw_writer = None
    t_prev = None
    try:
        for frame, t in frames(cap, a):
            t0 = time.perf_counter()
            try:
                info = {}
                kp = detect(yolo, frame, a.device, info)
            except Exception as e:                      # never crash on a bad frame
                print("pose error:", e)
                log.setdefault("errors", []).append(repr(e))
                kp = np.full((17, 3), np.nan, np.float32)
                info = {"n_boxes": 0, "max_conf": 0.0, "chosen_conf": 0.0, "area_frac": 0.0, "boxes": []}
            log.setdefault("no_person", []).append(bool(np.isnan(kp[0, 0])))
            for key in ("n_boxes", "max_conf", "chosen_conf", "area_frac"):
                log.setdefault(key, []).append(info.get(key, 0))
            tp = time.perf_counter()
            try:
                recs = pipe.push(kp)
            except Exception as e:                      # report, keep running (self-test counts these)
                print("pipeline error:", e)
                log.setdefault("errors", []).append(repr(e))
                recs = []
            for r in recs:
                if a.log_session:
                    SF["frame"].append(r["frame"]); SF["feats"].append(pipe.F[r["frame"]]); SF["probs"].append(r["prob"])
                    SF["phase"].append(r["phase"])
                st["phase"] = PHASES[r["smooth"]]
                st["count"] = r["count"]
                log["frames"].append({"frame": r["frame"], "phase": r["phase"], "smooth": r["smooth"], "count": r["count"]})
                if pipe.stream.feat.obs[r["frame"]][[5, 6]].all():
                    sw_hist.append((t, float(pipe.F[r["frame"]][SW_IDX])))
                if r["rep"] and r["rep"].get("rejected"):
                    ev = r["rep"]
                    st["feedback"].append(("Not counted: " + ev["rejected"].split(" (")[0], (0, 165, 255)))
                    st["last_rep_t"] = t
                    log["reps"].append({"rep": None, "start": ev["start"], "end": ev["end"], "rejected": ev["rejected"],
                                        "upper_share": ev["upper_share"], "lower_share": ev["lower_share"],
                                        "form": {f: ev["form"][f]["flag"] for f in FAULTS}})
                elif r["rep"]:
                    ev = r["rep"]
                    st["last_rep_t"] = t
                    parts, bad = [], False
                    for f in FAULTS:
                        fl = ev["form"][f]["flag"]
                        if fl is None:
                            continue
                        parts.append(VERDICT[f][int(fl)]); bad |= fl
                    st["feedback"].append((f"Rep {ev['rep']}: " + (", ".join(parts) or "no rule for this view"),
                                           (60, 60, 255) if bad else (90, 220, 90)))
                    log["reps"].append({"rep": ev["rep"], "start": ev["start"], "end": ev["end"],
                                        "upper_share": ev["upper_share"], "lower_share": ev["lower_share"],
                                        "form": {f: ev["form"][f]["flag"] for f in FAULTS}})
            # live flags: rule statistic over the current rep so far (final verdict comes at rep end)
            rc, live = pipe.counter, {}
            if rc.moving and rc.start is not None and len(pipe.F) > rc.start:
                Fs, Vs = np.stack(pipe.F[rc.start:]), np.array(pipe.V[rc.start:])
                for f in FAULTS:
                    if a.view in OBSERVABLE[f]:
                        r_ = rule_for(f, a.view, cfg)
                        v = rep_stat(Fs, Vs, r_["feature"], r_["stat"])
                        if v is not None:
                            live[f] = {"value": v, "thr": r_["threshold"],
                                       "flag": v < r_["threshold"] if r_["op"] == "lt" else v > r_["threshold"]}
            st["live"] = live
            feat = pipe.stream.feat
            st["calibrating"] = calibrating(feat)
            st["valid"] = bool(feat.valid[-1]) if feat.valid else False
            st["t"] = t
            st["upper_missing"] = upper_missing_update(st["upper_missing"], st["valid"], feat.obs[-1] if feat.obs else None)
            st["view_warn"] = view_warning(sw_hist, t, a.view)
            log.setdefault("view_warn", []).append(bool(st["view_warn"]))
            log.setdefault("upper_missing_warn", []).append(st["upper_missing"] >= UPPER_MISSING_WARN)
            log.setdefault("valid", []).append(st["valid"])
            log.setdefault("count", []).append(pipe.counter.count)
            if a.selftest_camera and a.selftest_frames and len(log["valid"]) in SAVE_AT:
                snap = frame.copy()
                for x0, y0, x1, y1, c in info.get("boxes", []):
                    cv2.rectangle(snap, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
                    put(snap, f"person {c:.2f}", (int(x0), max(int(y0) - 8, 20)), 0.7, (0, 255, 255))
                SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(SELFTEST_DIR / f"{a.expect or 'run'}_{len(log['valid']):03d}.jpg"), snap)
            lat = (time.perf_counter() - t0) * 1e3
            now = time.perf_counter()
            if t_prev is not None:
                st["fps"] = 0.9 * st["fps"] + 0.1 / max(now - t_prev, 1e-6) if st["fps"] else 1 / max(now - t_prev, 1e-6)
            t_prev = now
            st["lat_ms"] = 0.9 * st["lat_ms"] + 0.1 * lat if st["lat_ms"] else lat
            st["pose_ms"] = (tp - t0) * 1e3
            log["latency_ms"].append(lat); log["pose_ms"].append((tp - t0) * 1e3)
            td0 = time.perf_counter()
            img = draw(frame, kp, st, a)
            td1 = time.perf_counter()
            if a.record_raw:
                if raw_writer is None:
                    for cc in ("avc1", "mp4v"):
                        raw_writer = cv2.VideoWriter(a.record_raw, cv2.VideoWriter_fourcc(*cc), FPS, (frame.shape[1], frame.shape[0]))
                        if raw_writer.isOpened():
                            break
                raw_writer.write(frame)
            if a.record:
                if writer is None:
                    for cc in ("avc1", "mp4v"):          # H.264 plays in QuickTime; mp4v as fallback
                        writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*cc), FPS, (img.shape[1], img.shape[0]))
                        if writer.isOpened():
                            break
                writer.write(img)
            td2 = time.perf_counter()
            quit_ = False
            if not a.headless:
                cv2.imshow(win, img)
                quit_ = cv2.waitKey(1) & 0xFF in (ord("q"), 27)
            td3 = time.perf_counter()
            if a.log_session:
                fv = pipe.stream.feat
                for k_, v_ in (("t", t), ("raw_index", CAM["raw_index"]), ("skipped", CAM["skipped"]), ("kpts", kp),
                               ("n_boxes", info.get("n_boxes", 0)), ("max_conf", info.get("max_conf", 0.0)),
                               ("chosen_conf", info.get("chosen_conf", 0.0)), ("area_frac", info.get("area_frac", 0.0)),
                               ("scale", fv.scale() if fv.torso_all else np.nan), ("valid", st["valid"]),
                               ("calibrating", st["calibrating"]), ("count", pipe.counter.count),
                               ("moving", pipe.counter.moving), ("smooth", pipe.counter.state),
                               ("live_depth", live.get("shallow_depth", {}).get("value", np.nan)),
                               ("live_lean", live.get("forward_lean", {}).get("value", np.nan)),
                               ("live_valgus", live.get("knee_valgus", {}).get("value", np.nan)),
                               ("t_capture", CAM["t_capture_ms"]), ("t_pose", (tp - t0) * 1e3),
                               ("t_features", getattr(pipe.stream, "t_features_ms", np.nan)),
                               ("t_onnx", getattr(pipe.stream, "t_onnx_ms", np.nan)),
                               ("t_logic", getattr(pipe, "t_logic_ms", np.nan)), ("t_draw", (td1 - td0) * 1e3),
                               ("t_record", (td2 - td1) * 1e3), ("t_display", (td3 - td2) * 1e3),
                               ("view_warn", bool(st["view_warn"])),    # displayed warnings (logging only)
                               ("upper_missing_warn", st["upper_missing"] >= UPPER_MISSING_WARN)):
                    S[k_].append(v_)
            if quit_:
                break
        for r in pipe.flush() if a.video else []:      # end of a recorded clip: finalise the last frame
            log["frames"].append({"frame": r["frame"], "phase": r["phase"], "smooth": r["smooth"], "count": r["count"]})
            if r["rep"]:
                log["reps"].append({"rep": None if r["rep"].get("rejected") else r["rep"]["rep"],
                                    "start": r["rep"]["start"], "end": r["rep"]["end"], "rejected": r["rep"].get("rejected"),
                                    "form": {f: r["rep"]["form"][f]["flag"] for f in FAULTS}})
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    L = np.array(log["latency_ms"]) if log["latency_ms"] else np.zeros(1)
    print(f"frames {len(log['latency_ms'])}  reps {pipe.counter.count}  latency median {np.median(L):.1f} ms  "
          f"p95 {np.percentile(L, 95):.1f} ms  (budget {1000 / FPS:.1f} ms at {FPS} fps)")
    if a.log:
        json.dump(log, open(a.log, "w"))
    if raw_writer is not None:
        raw_writer.release()
    if a.log_session:
        np.savez_compressed(a.log_session, **{k: np.array(v) for k, v in S.items()},
                            **{f"fin_{k}": np.array(v) for k, v in SF.items()},
                            reps=json.dumps(log["reps"]), args=json.dumps(vars(a)), fps=FPS)
        print("session log:", a.log_session)
    if a.selftest_camera or a.report:
        selftest_report(log, a, L)


if __name__ == "__main__":
    main()
