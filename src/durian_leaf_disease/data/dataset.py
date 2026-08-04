"""
dataset.py — Dataset & DataLoader cho bài toán phân loại bệnh lá sầu riêng
=============================================================================
Mô-đun này cung cấp:
  • get_transforms(phase)  — pipeline augmentation / preprocessing
  • get_dataloaders(...)   — DataLoader cho train / val / test
  • get_class_weights(...) — class weights để cân bằng CrossEntropyLoss
  • verify_dataset()       — in thống kê chi tiết để xác nhận load đúng

Thiết kế augmentation (giải thích):
───────────────────────────────────
  Train:
    1. Resize(256)        — phóng to hơn 224 để tạo khoảng crop
    2. RandomCrop(224)    — cắt ngẫu nhiên → model học nhiều vùng ảnh
    3. HorizontalFlip(0.5)— lá có thể xuất hiện chiều ngang bất kỳ
    4. VerticalFlip(0.3)  — lá có thể bị chụp ngược, nhưng ít phổ biến hơn
    5. Rotation(±20°)     — nghiêng nhẹ, mô phỏng lá trên cây thật
    6. ColorJitter         — thay đổi sáng/tương phản/bão hòa/hue nhẹ
       (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
       → mô phỏng điều kiện ánh sáng khác nhau ngoài đồng
    7. ToTensor + Normalize(ImageNet mean/std)
    8. RandomErasing(p=0.15) — che ngẫu nhiên 1 vùng nhỏ trên ảnh
       → buộc model không dựa vào 1 vùng duy nhất, tăng generalization

  ⚠ KHÔNG dùng RandomGrayscale — bệnh lá sầu riêng phân biệt chủ yếu
    bằng MÀU SẮC (R/G ratio khác nhau giữa các lớp, EDA Bước 1 đã chứng minh).
    Xóa thông tin màu = xóa đặc trưng quan trọng nhất.

  Val / Test:
    1. Resize(256) → CenterCrop(224) — deterministic, không augment
    2. ToTensor + Normalize(ImageNet)
"""

import os

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Import config
# ─────────────────────────────────────────────────────────────────────────────
from durian_leaf_disease.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    CLASS_NAMES, NUM_CLASSES,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Giá trị chuẩn hóa ImageNet — dùng chung cho tất cả model pretrained
# Lý do dùng ImageNet thay vì tính riêng: backbone đã học features từ ImageNet,
# dùng cùng mean/std giúp đầu vào match phân phối mà model đã quen.
# Dataset này có G cao hơn (0.506 vs 0.456) — sai lệch nhỏ, fine-tuning bù được.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Seed cố định để kết quả reproducible
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    """
    Đặt seed cho tất cả nguồn random để kết quả có thể tái tạo.
    Gọi hàm này 1 lần ở đầu chương trình.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Đảm bảo cuDNN deterministic (chậm hơn ~5-10% nhưng reproducible)
    # Trong production/deploy, nên đặt benchmark=True để tăng tốc.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Seed cho mỗi DataLoader worker — đảm bảo augmentation giống nhau
    giữa các lần chạy, kể cả khi dùng num_workers > 0.
    Truyền vào DataLoader(worker_init_fn=seed_worker).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─────────────────────────────────────────────────────────────────────────────
# PATH VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_dirs() -> None:
    """Kiểm tra tất cả thư mục dataset tồn tại trước khi load."""
    for name, path in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"❌ Thư mục {name} không tồn tại: {path}\n"
                f"   Kiểm tra lại DATASET_ROOT trong config.py"
            )
        # Kiểm tra có ít nhất 1 subfolder (class)
        subdirs = [d for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d))]
        if len(subdirs) == 0:
            raise FileNotFoundError(
                f"❌ Thư mục {name} không có class nào: {path}\n"
                f"   Cần cấu trúc: {path}/Leaf_Algal/, Leaf_Blight/, ..."
            )


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(phase: str) -> transforms.Compose:
    """
    Trả về transform pipeline phù hợp cho từng phase.

    Args:
        phase: "train", "val", hoặc "test"

    Returns:
        transforms.Compose — pipeline biến đổi ảnh

    Lý do thiết kế:
    ────────────────
    • Resize(256) → RandomCrop(224): thay vì resize thẳng 224,
      cách này cho model thấy nhiều vùng ảnh khác nhau mỗi epoch.
    • HFlip(0.5) + VFlip(0.3): lá trên cây có thể ở hướng bất kỳ,
      nhưng lật dọc ít phổ biến hơn nên p thấp hơn.
    • Rotation(±20°): mô phỏng lá nghiêng tự nhiên, không quá lớn
      vì >30° có thể tạo ảnh phi thực tế.
    • ColorJitter: mô phỏng ánh sáng ngoài đồng (sáng/mờ/ngược sáng).
      Hue chỉ ±0.1 vì màu sắc LÀ đặc trưng quan trọng (EDA đã chứng minh).
    • RandomErasing: che 1 vùng nhỏ → model phải dùng nhiều vùng để
      quyết định, giảm overfitting. p=0.15 = vừa phải.
    • ⚠ KHÔNG dùng RandomGrayscale vì xóa thông tin màu.
    """
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),          # Phóng to để tạo khoảng crop
            transforms.RandomCrop(IMAGE_SIZE),       # Crop ngẫu nhiên 224×224
            transforms.RandomHorizontalFlip(p=0.5),  # Lật ngang — lá hướng bất kỳ
            transforms.RandomVerticalFlip(p=0.3),    # Lật dọc — ít phổ biến hơn
            transforms.RandomRotation(degrees=20),   # Xoay ±20° — lá nghiêng
            transforms.ColorJitter(                  # Mô phỏng ánh sáng khác nhau
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1,                             # Hue thấp: giữ đặc trưng màu
            ),
            # ⚠ KHÔNG dùng RandomGrayscale — xóa thông tin màu = xóa đặc trưng
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(                # Che ngẫu nhiên 1 vùng
                p=0.15,                              # 15% ảnh bị che
                scale=(0.02, 0.20),                  # Che 2-20% diện tích
                ratio=(0.3, 3.3),                    # Tỷ lệ hình chữ nhật
                value='random',                      # Che bằng pixel ngẫu nhiên
            ),
        ])
    else:  # val / test — deterministic, KHÔNG augment
        return transforms.Compose([
            transforms.Resize((256, 256)),           # Nhất quán với train
            transforms.CenterCrop(IMAGE_SIZE),       # Crop chính giữa 224×224
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# DATASET & DATALOADER
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """
    Tạo DataLoader cho train/val/test dùng torchvision.datasets.ImageFolder.

    ImageFolder tự đọc cấu trúc thư mục:
        train/
            Leaf_Algal/          ← class 0 (theo alphabet)
            Leaf_Blight/         ← class 1
            Leaf_Colletotrichum/ ← class 2
            Leaf_Healthy/        ← class 3
            Leaf_Phomopsis/      ← class 4
            Leaf_Rhizoctonia/    ← class 5

    Args:
        batch_size:  số ảnh mỗi batch (mặc định 16, an toàn cho 4GB VRAM)
        num_workers: số worker song song (mặc định 2, tránh lỗi Windows)

    Returns:
        dataloaders:  dict {"train": DataLoader, "val": DataLoader, "test": DataLoader}
        class_to_idx: dict {"Leaf_Algal": 0, "Leaf_Blight": 1, ...}
    """
    # Validate đường dẫn trước khi load
    _validate_dirs()

    dataset_dirs = {
        "train": TRAIN_DIR,
        "val":   VAL_DIR,
        "test":  TEST_DIR,
    }

    image_datasets = {
        phase: datasets.ImageFolder(dir_, transform=get_transforms(phase))
        for phase, dir_ in dataset_dirs.items()
    }

    # Generator riêng cho DataLoader — đảm bảo reproducible
    g = torch.Generator()
    g.manual_seed(SEED)

    dataloaders = {}
    for phase, ds in image_datasets.items():
        is_train = (phase == "train")
        dataloaders[phase] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,                     # Chỉ shuffle tập train
            num_workers=num_workers,
            pin_memory=True,                      # Tăng tốc copy CPU → GPU
            drop_last=is_train,                   # Bỏ batch cuối nếu không đủ
            persistent_workers=(num_workers > 0), # Giữ worker sống giữa các epoch
            worker_init_fn=seed_worker,           # Seed worker để reproducible
            generator=g if is_train else None,    # Generator cho shuffle
        )

    class_to_idx = image_datasets["train"].class_to_idx

    # Validate thứ tự class khớp với config
    expected_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    if class_to_idx != expected_idx:
        raise ValueError(
            f"❌ class_to_idx không khớp config!\n"
            f"   ImageFolder: {class_to_idx}\n"
            f"   Config:      {expected_idx}\n"
            f"   → Kiểm tra lại CLASS_NAMES trong config.py"
        )

    return dataloaders, class_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(
    train_dir: str = TRAIN_DIR,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Tính class weights theo công thức cân bằng:
        weight[c] = total_samples / (num_classes × count_class[c])

    Lớp ít ảnh hơn → weight cao hơn → CrossEntropyLoss phạt nặng hơn
    khi model sai lớp này → model chú ý đều các lớp.

    Ví dụ (dataset này):
        Healthy (338 ảnh, nhiều nhất)  → weight = 0.895 (thấp nhất)
        Rhizoctonia (278 ảnh, ít nhất) → weight = 1.088 (cao nhất)

    Args:
        train_dir: đường dẫn thư mục train
        device:    thiết bị (cuda/cpu) để chuyển tensor

    Returns:
        torch.Tensor shape (NUM_CLASSES,) — truyền vào CrossEntropyLoss(weight=...)
    """
    # Chỉ cần đếm file, KHÔNG load ảnh → nhanh hơn nhiều
    counts = {}
    class_dirs = sorted(os.listdir(train_dir))  # Sorted = giữ thứ tự alphabet
    for idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            # Đếm file ảnh (bỏ qua file ẩn, thư mục con)
            # Chỉ đếm file ảnh (lọc extension), tránh đếm nhầm .txt, .DS_Store
            n_files = len([
                f for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
                and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            counts[idx] = n_files

    total = sum(counts.values())
    n_cls = len(counts)

    if n_cls != NUM_CLASSES:
        raise ValueError(
            f"❌ Số class trong thư mục ({n_cls}) ≠ NUM_CLASSES ({NUM_CLASSES})\n"
            f"   Thư mục: {class_dirs}"
        )

    weights = torch.tensor(
        [total / (n_cls * counts[c]) for c in range(n_cls)],
        dtype=torch.float32,
    )

    if device is not None:
        weights = weights.to(device)

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY — in thống kê dataset chi tiết
# ─────────────────────────────────────────────────────────────────────────────

def verify_dataset() -> None:
    """
    In thông tin tổng quan dataset để xác nhận mọi thứ load đúng.
    Bao gồm: số ảnh mỗi class, batch shape, class mapping, class weights,
    và tóm tắt augmentation pipeline.
    """
    set_seed()

    print("=" * 60)
    print("  XÁC NHẬN DATASET — Phân loại bệnh lá sầu riêng")
    print("=" * 60)

    # ── 1. Thống kê số ảnh ──
    total_all = 0
    for phase, dir_ in [("TRAIN", TRAIN_DIR), ("VAL", VAL_DIR), ("TEST", TEST_DIR)]:
        class_dirs = sorted([
            d for d in os.listdir(dir_)
            if os.path.isdir(os.path.join(dir_, d))
        ])
        total_phase = 0
        print(f"\n📂 [{phase}] — {dir_}")
        print(f"   {'Lớp bệnh':<25} {'Số ảnh':>8}")
        print(f"   {'─' * 25} {'─' * 8}")
        for cls in class_dirs:
            cls_path = os.path.join(dir_, cls)
            n = len([f for f in os.listdir(cls_path)
                     if os.path.isfile(os.path.join(cls_path, f))
                     and not f.startswith('.')])
            print(f"   {cls:<25} {n:>8}")
            total_phase += n
        print(f"   {'─' * 25} {'─' * 8}")
        print(f"   {'TỔNG':<25} {total_phase:>8}")
        total_all += total_phase

    print(f"\n📊 Tổng toàn bộ: {total_all} ảnh")

    # ── 2. Thử load 1 batch ──
    print(f"\n{'─' * 60}")
    print("🔄 Thử load 1 batch từ train set...")
    loaders, c2i = get_dataloaders()
    imgs, lbls = next(iter(loaders["train"]))

    print(f"   Batch shape : {imgs.shape}")
    print(f"                 (batch={imgs.shape[0]}, "
          f"channels={imgs.shape[1]}, H={imgs.shape[2]}, W={imgs.shape[3]})")
    print(f"   Labels      : {lbls.tolist()}")
    print(f"   Pixel range : [{imgs.min():.3f}, {imgs.max():.3f}]"
          f"  (sau normalize)")
    print(f"   dtype       : {imgs.dtype}")

    # ── 3. Class mapping ──
    print(f"\n{'─' * 60}")
    print("🏷️  Class mapping (ImageFolder, thứ tự alphabet):")
    for name, idx in c2i.items():
        print(f"   {idx} → {name}")

    # ── 4. Class weights ──
    weights = get_class_weights()
    print(f"\n{'─' * 60}")
    print("⚖️  Class weights (cho CrossEntropyLoss):")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(weights[i].item() * 20)
        print(f"   {name:<25} {weights[i]:.4f}  {bar}")

    # ── 5. Augmentation summary ──
    print(f"\n{'─' * 60}")
    print("🎨 Augmentation pipeline (đọc trực tiếp từ get_transforms):")
    print("   TRAIN:")
    for t in get_transforms("train").transforms:
        print(f"     → {t}")
    print("   VAL/TEST:")
    for t in get_transforms("val").transforms:
        print(f"     → {t}")
    print("   ⚠ KHÔNG dùng RandomGrayscale (bệnh lá phân biệt bằng màu sắc)")

    # ── 6. Config check ──
    print(f"\n{'─' * 60}")
    print("⚙️  Config:")
    print(f"   batch_size       = {BATCH_SIZE}")
    print(f"   num_workers      = {NUM_WORKERS}")
    print(f"   image_size       = {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"   persistent_workers = True")
    print(f"   pin_memory       = True")
    print(f"   seed             = {SEED}")
    print(f"   normalize        = ImageNet (mean={IMAGENET_MEAN}, std={IMAGENET_STD})")

    print(f"\n{'=' * 60}")
    print("✅ Dataset load thành công — sẵn sàng cho training!")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    verify_dataset()
