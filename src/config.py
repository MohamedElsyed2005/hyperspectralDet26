"""Central configuration for the Hyperspectral Object Detection Challenge 2026 pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths - Configured to match your exact directory layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRAIN_IMG_DIR = DATA_DIR / "data_train" / "data_train" / "VIS"
TRAIN_ANN_DIR = DATA_DIR / "data_train" / "data_train" / "Annotations" / "VIS"
if not TRAIN_ANN_DIR.exists():
    # Fallback in case Annotations are directly under data_train/Annotations
    TRAIN_ANN_DIR = DATA_DIR / "data_train" / "data_train" / "Annotations"

TEST_IMG_DIR = DATA_DIR / "data_test" / "data_test" / "VIS"
RANKING_IMG_DIR = DATA_DIR / "ranking_images" / "VIS"

CLASS_FILE = DATA_DIR / "class.txt"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
NUM_BANDS = 16          # 16 spectral bands (460-600nm)
SFA_CELL_SIZE = 4       # 4x4 spectral filter array

if CLASS_FILE.exists():
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        CLASSES = [line.strip() for line in f if line.strip()]
else:
    # Falls back to the known 18-class list so this module can still be
    # imported (e.g. for testing) without the data/ directory present.
    CLASSES = [
        "apple", "apple_plastic", "badminton", "banana", "banana_plastic",
        "car", "car_toy", "charger_head", "e-bike", "egg", "egg_plastic",
        "egg_wood", "orange", "orange_plastic", "people", "rubik",
        "stone_block", "table_tennis",
    ]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}
NUM_CLASSES = len(CLASSES)  # 18

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 30
LEARNING_RATE = 2e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_STEP_SIZE = 10
LR_GAMMA = 0.1
VAL_SPLIT = 0.1
SEED = 42

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
USE_AUGMENTATION = True   # applied to the train split only, never val/test
AUG_MIN_BOX_VISIBILITY = 0.2   # drop boxes cropped below this fraction of original area
AUG_BRIGHTNESS_JITTER = 0.15   # +/- fraction, applied identically across all 16 bands

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
SCORE_THRESH = 0.05      # mAP evaluation sweeps confidence thresholds
NMS_THRESH = 0.5
MAX_DETECTIONS_PER_IMAGE = 100
