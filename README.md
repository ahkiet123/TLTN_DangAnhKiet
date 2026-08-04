# Durian Leaf Disease Classification

Python project for classifying durian leaf disease images with transfer learning.

## Structure

```text
data/raw/Durian_Leaf_Diseases/      Raw image dataset split into train/val/test
docs/                               Project reports and proposal files
outputs/                            Training checkpoints, histories, model summaries
reports/eda/                        EDA figures and summaries
scripts/                            Thin command-line entrypoints
src/durian_leaf_disease/            Importable Python package
```

## Setup

```powershell
# Cài dependencies (nên dùng virtualenv)
pip install -r requirements.txt

# Kiểm tra dataset load đúng
python scripts/run_eda.py

# Chạy smoke test trước khi train full
python scripts/smoke_test.py

# Demo train nhanh 3 epochs (1 model) để kiểm chứng pipeline
python scripts/demo_train.py

# Train 3 models (30 epochs each)
python scripts/train.py

# So sánh cấu hình / forward pass
python scripts/compare_models.py

# Đánh giá trên tập test + vẽ confusion matrix
python scripts/evaluate.py --model all
```

The dataset path is configured in `src/durian_leaf_disease/config.py`.
Set `DURIAN_DATASET_ROOT` if you need to point the project at another dataset location.
