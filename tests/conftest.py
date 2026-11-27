import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Local-only data (dataset pose cache, dev-session keypoint traces). Tests that need it are skipped when absent,
# e.g. on a fresh clone: no videos or traces of people are committed.
POSE_DIR = ROOT / "data" / "pose"
DEV_FIXTURE = ROOT / "data" / "debug" / "fixtures" / "live_side_dev_kpts.npz"
RUN2_FIXTURE = ROOT / "data" / "debug" / "fixtures" / "test_side_run2_dev_kpts.npz"   # 2nd live dev run (keypoints only)
TEST_RUN1 = ROOT / "data" / "debug" / "fixtures" / "test_side_run1_kpts.npz"   # scripted live test run 1 (keypoints only)
TEST_RUN2 = ROOT / "data" / "debug" / "fixtures" / "test_side_run2_kpts.npz"   # scripted live test run 2 (keypoints only)
