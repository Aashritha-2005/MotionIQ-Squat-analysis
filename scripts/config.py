"""Shared constants for every MotionIQ script (offline extraction and live demo)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
POSE_DIR = DATA / "pose"
LABELS_CSV = DATA / "labels.csv"
SPOTCHECK_DIR = DATA / "spotcheck"
ARTIFACTS = ROOT / "artifacts"

# All clips are resampled to this rate; per-frame labels and features live on this grid.
FPS = 15

POSE_MODEL = "yolo11s-pose.pt"

PHASES = ["standing", "descent", "bottom", "ascent"]
FAULTS = ["knee_valgus", "shallow_depth", "forward_lean"]

# A fault is only tagged/evaluated on views where it is visible in 2D.
OBSERVABLE = {
    "knee_valgus": {"front", "diagonal"},
    "shallow_depth": {"side", "diagonal"},
    "forward_lean": {"side", "diagonal"},
}

# COCO-17 keypoint indices used by YOLO-pose.
KP = dict(nose=0, l_sho=5, r_sho=6, l_hip=11, r_hip=12, l_knee=13, r_knee=14, l_ank=15, r_ank=16)
