"""
Central configuration for the durian leaf disease classification pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

DATASET_ROOT = Path(
    os.environ.get("DURIAN_DATASET_ROOT", RAW_DATA_DIR / "Durian_Leaf_Diseases")
)
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_REPORT_DIR = REPORTS_DIR / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EDA_REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CLASSES
# =============================================================================

CLASS_NAMES = [
    "Leaf_Algal",
    "Leaf_Blight",
    "Leaf_Colletotrichum",
    "Leaf_Healthy",
    "Leaf_Phomopsis",
    "Leaf_Rhizoctonia",
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_LABELS_VI = {
    "Leaf_Algal": "Đốm rong",
    "Leaf_Blight": "Cháy lá",
    "Leaf_Colletotrichum": "Thán thư",
    "Leaf_Healthy": "Lá khỏe",
    "Leaf_Phomopsis": "Phomopsis",
    "Leaf_Rhizoctonia": "Rhizoctonia",
}


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 224
NUM_EPOCHS = 50          # Tăng lên 50 để model hội tụ đầy đủ
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING = 7       # Dừng sớm nếu val_loss không cải thiện sau 7 epoch
MIXED_PRECISION = True

PHASE1_EPOCHS = 10
PHASE2_EPOCHS = NUM_EPOCHS - PHASE1_EPOCHS  # 40 epochs fine-tuning

MODELS_TO_TRAIN = [
    "mobilenet_v2",
    "efficientnet_b0",
    "resnet50",
]


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device      : {DEVICE}")
    print(f"GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Num classes : {NUM_CLASSES}")
    print(f"Classes     : {CLASS_NAMES}")
    print(f"Dataset     : {DATASET_ROOT}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Reports dir : {REPORTS_DIR}")
