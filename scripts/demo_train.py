"""
demo_train.py - Chạy train mẫu nhanh (3 epochs) trên 1 model.

Dùng để:
  * Kiểm chứng pipeline chạy đúng trên máy thật
  * In output chi tiết gửi cho ngườii hướng dẫn đánh giá
  * Ước lượng thời gian trước khi chạy full, tối đa 50 epochs

Chạy:
    python scripts/demo_train.py

Output sẽ lưu vào:
    outputs/demo/mobilenet_v2/best_model.pth
    outputs/demo/mobilenet_v2/history.json
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

# Monkey-patch config để chạy demo nhanh (trước khi import trainer)
from durian_leaf_disease import config

config.NUM_EPOCHS = 3
config.PHASE1_EPOCHS = 1
config.PHASE2_EPOCHS = 2
config.MODELS_TO_TRAIN = ["mobilenet_v2"]
config.EARLY_STOPPING = 10  # Tắt early stopping trong demo
config.OUTPUT_DIR = config.OUTPUT_DIR / "demo"
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from durian_leaf_disease.training.trainer import main

if __name__ == "__main__":
    main()
