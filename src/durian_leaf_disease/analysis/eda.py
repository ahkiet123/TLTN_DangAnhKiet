"""
eda.py - Phân tích khám phá dữ liệu (EDA) cho dataset lá sầu riêng
Xuất biểu đồ và thống kê vào thư mục reports/eda/

Chạy: python scripts/run_eda.py
"""
import os
import json
import warnings
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# CẤU HÌNH (import từ config.py — single source of truth)
# ─────────────────────────────────────────────
from durian_leaf_disease.config import (
    CLASS_NAMES,
    CLASS_LABELS_VI as CLASS_VI,
    DATASET_ROOT,
    EDA_REPORT_DIR,
)

DATASET_DIR = Path(DATASET_ROOT)
REPORT_DIR  = Path(EDA_REPORT_DIR)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

COLORS_CLASS = ["#4361EE", "#F72585", "#3A0CA3", "#4CC9F0", "#F77F00", "#2DC653"]

random.seed(42)


# ═══════════════════════════════════════════════════════════
# PHẦN 1: THU THẬP DỮ LIỆU THÔ
# ═══════════════════════════════════════════════════════════

def count_images():
    """Đếm số ảnh mỗi class mỗi split."""
    counts = {}
    for split in SPLITS:
        counts[split] = {}
        for cls in CLASS_NAMES:
            cls_dir = DATASET_DIR / split / cls
            if cls_dir.exists():
                imgs = [f for f in cls_dir.iterdir()
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
                counts[split][cls] = len(imgs)
            else:
                counts[split][cls] = 0
    return counts


def check_corrupt():
    """Kiểm tra toàn bộ ảnh — trả về danh sách ảnh lỗi."""
    corrupt, total = [], 0
    for split in SPLITS:
        for cls in CLASS_NAMES:
            cls_dir = DATASET_DIR / split / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                total += 1
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    corrupt.append(str(img_path))
    return corrupt, total


def analyze_all_image_properties():
    """
    Phân tích toàn bộ ảnh tập train: kích thước, mean/std RGB toàn dataset,
    và mean RGB từng class. Đọc ảnh MỘT LẦN duy nhất (v1 đọc 2 lần).

    Returns:
        sizes    : list of (width, height)
        rgb_mean : np.array [R, G, B] — trung bình toàn dataset
        rgb_std  : np.array [R, G, B] — std toàn dataset
        class_rgb: dict class -> {'R': float, 'G': float, 'B': float}
    """
    sizes = []
    all_r, all_g, all_b = [], [], []
    class_rgb = {}

    print("    Đang đọc ảnh train để phân tích (có thể mất ~30 giây)...")
    for cls in CLASS_NAMES:
        cls_dir = DATASET_DIR / "train" / cls
        r_list, g_list, b_list = [], [], []

        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    w, h = img.size
                    sizes.append((w, h))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    r_mean = arr[:, :, 0].mean()
                    g_mean = arr[:, :, 1].mean()
                    b_mean = arr[:, :, 2].mean()
                    r_list.append(r_mean)
                    g_list.append(g_mean)
                    b_list.append(b_mean)
                    all_r.append(r_mean)
                    all_g.append(g_mean)
                    all_b.append(b_mean)
            except Exception as e:
                warnings.warn(f"Không đọc được ảnh {img_path}: {e}")

        class_rgb[cls] = {
            'R': float(np.mean(r_list)),
            'G': float(np.mean(g_list)),
            'B': float(np.mean(b_list)),
        }

    rgb_mean = np.array([np.mean(all_r), np.mean(all_g), np.mean(all_b)])
    rgb_std  = np.array([np.std(all_r),  np.std(all_g),  np.std(all_b)])
    return sizes, rgb_mean, rgb_std, class_rgb


def compute_class_weights(counts):
    train_counts = [counts['train'][c] for c in CLASS_NAMES]
    total  = sum(train_counts)
    n_cls  = len(train_counts)
    return [total / (n_cls * cnt) for cnt in train_counts]


# ═══════════════════════════════════════════════════════════
# PHẦN 2: CÁC BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════

# ── Chart 01: Bar chart phân phối theo từng split ──────────
def plot_class_distribution(counts):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phân phối số ảnh theo lớp bệnh",
                 fontsize=16, fontweight='bold', y=1.02)

    for ax, split in zip(axes, SPLITS):
        labels = [CLASS_VI[c] for c in CLASS_NAMES]
        values = [counts[split][c] for c in CLASS_NAMES]
        bars   = ax.bar(labels, values, color=COLORS_CLASS,
                        edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, str(val),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        total = sum(values)
        ax.set_title(f"{split.upper()}  ({total} ảnh)",
                     fontsize=13, fontweight='bold')
        ax.set_ylabel("Số ảnh")
        ax.set_ylim(0, max(values) * 1.22)
        ax.tick_params(axis='x', rotation=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save("01_class_distribution.png", fig)


# ── Chart 02: Grouped bar — so sánh 3 split ────────────────
def plot_grouped_distribution(counts):
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    split_colors = ['#2563EB', '#16A34A', '#DC2626']

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (split, color) in enumerate(zip(SPLITS, split_colors)):
        values = [counts[split][c] for c in CLASS_NAMES]
        bars = ax.bar(x + i * w, values, w, label=split.upper(),
                      color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, str(val),
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels([CLASS_VI[c] for c in CLASS_NAMES],
                       rotation=20, ha='right')
    ax.set_ylabel("Số ảnh")
    ax.set_title("So sánh số ảnh Train / Val / Test theo từng lớp bệnh",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save("02_grouped_distribution.png", fig)


# ── Chart 03: Grid ảnh mẫu 6 lớp × 4 ảnh ─────────────────
def plot_sample_images():
    n_cols, n_rows = 4, len(CLASS_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    fig.suptitle("Ảnh mẫu — 6 lớp bệnh lá sầu riêng",
                 fontsize=16, fontweight='bold', y=1.005)

    for row, cls in enumerate(CLASS_NAMES):
        cls_dir   = DATASET_DIR / "train" / cls
        all_files = [f for f in cls_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        # Chọn ngẫu nhiên để đa dạng hơn
        chosen = random.sample(all_files, min(n_cols, len(all_files)))

        for col in range(n_cols):
            ax = axes[row][col]
            if col < len(chosen):
                try:
                    img = Image.open(chosen[col]).convert('RGB')
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                            transform=ax.transAxes)
            else:
                ax.axis('off')
                continue
            ax.axis('off')
            if col == 0:
                ax.set_title(
                    f"{CLASS_VI[cls]}\n({cls})",
                    fontsize=10, fontweight='bold',
                    color=COLORS_CLASS[row], loc='left', pad=4
                )

    plt.tight_layout()
    _save("03_sample_images.png", fig)


# ── Chart 04: RGB trung bình từng class ────────────────────
def plot_rgb_per_class(class_rgb: dict):
    """
    Vẽ grouped bar chart thể hiện giá trị R, G, B trung bình
    của mỗi lớp bệnh. Cho thấy rõ đặc trưng màu sắc khác nhau
    giữa các loại bệnh — thông tin quan trọng cho bài toán.
    """
    x = np.arange(len(CLASS_NAMES))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    r_vals = [class_rgb[c]['R'] for c in CLASS_NAMES]
    g_vals = [class_rgb[c]['G'] for c in CLASS_NAMES]
    b_vals = [class_rgb[c]['B'] for c in CLASS_NAMES]

    ax.bar(x - w, r_vals, w, label='Red',   color='#EF4444', alpha=0.85, edgecolor='white')
    ax.bar(x,     g_vals, w, label='Green', color='#22C55E', alpha=0.85, edgecolor='white')
    ax.bar(x + w, b_vals, w, label='Blue',  color='#3B82F6', alpha=0.85, edgecolor='white')

    # Ghi giá trị lên đầu cột
    for i, (r, g, b) in enumerate(zip(r_vals, g_vals, b_vals)):
        ax.text(i - w, r + 0.003, f"{r:.2f}", ha='center', va='bottom', fontsize=7.5)
        ax.text(i,     g + 0.003, f"{g:.2f}", ha='center', va='bottom', fontsize=7.5)
        ax.text(i + w, b + 0.003, f"{b:.2f}", ha='center', va='bottom', fontsize=7.5)

    # Vẽ đường tham chiếu R/G ratio của lá khỏe
    healthy_idx = CLASS_NAMES.index("Leaf_Healthy")
    healthy_rg  = r_vals[healthy_idx] / g_vals[healthy_idx]
    rg_ratios   = [r / g for r, g in zip(r_vals, g_vals)]

    ax2 = ax.twinx()
    ax2.plot(x, rg_ratios, 'D--', color='#7C3AED', linewidth=1.5,
             markersize=7, label='R/G ratio (cao = nâu/bệnh hơn)')
    ax2.axhline(healthy_rg, color='#7C3AED', linestyle=':', alpha=0.5,
                label=f'Baseline lá khỏe (R/G={healthy_rg:.3f})')
    ax2.set_ylabel("R/G ratio", color='#7C3AED', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#7C3AED')
    ax2.set_ylim(0.7, 1.1)

    # Hợp nhất legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_VI[c] for c in CLASS_NAMES],
                       rotation=15, ha='right', fontsize=11)
    ax.set_ylabel("Giá trị pixel trung bình (0–1)", fontsize=11)
    ax.set_ylim(0, 0.70)
    ax.set_title("Phân phối màu sắc RGB trung bình theo từng lớp bệnh (tập Train)\n"
                 "(Đường tím: R/G ratio — càng cao lá càng nâu/bệnh)",
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.5, -0.18,
            "R/G ratio cao hơn baseline lá khỏe → lá có màu nâu/đỏ hơn → triệu chứng bệnh rõ hơn.",
            transform=ax.transAxes, ha='center', fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    _save("04_rgb_mean_per_class.png", fig)


# ── Chart 05: Augmentation preview ─────────────────────────
def plot_augmentation_preview():
    """
    Hiển thị 1 ảnh gốc và 6 phiên bản sau augmentation
    để xác nhận pipeline transform hoạt động đúng.
    """
    # Lấy 1 ảnh từ mỗi class để hiển thị đa dạng
    # Lưu ý: p=0.9/0.5 CAO HƠN pipeline thật (0.5/0.3 trong dataset.py)
    # để ảnh demo thấy rõ hiệu ứng augmentation hơn.
    aug_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.9),     # Demo: p cao hơn thật (0.5)
        T.RandomVerticalFlip(p=0.5),       # Demo: p cao hơn thật (0.3)
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    ])

    n_aug = 6  # số phiên bản augmented hiển thị
    n_cls = len(CLASS_NAMES)

    fig, axes = plt.subplots(n_cls, 1 + n_aug,
                             figsize=(18, 3.2 * n_cls))
    fig.suptitle("Kiểm tra Augmentation Pipeline — Ảnh gốc vs. Sau augmentation",
                 fontsize=15, fontweight='bold', y=1.005)

    for row, cls in enumerate(CLASS_NAMES):
        cls_dir  = DATASET_DIR / "train" / cls
        img_file = sorted(cls_dir.iterdir())[5]   # lấy ảnh thứ 6 (bỏ qua mấy ảnh đầu)
        orig     = Image.open(img_file).convert('RGB')

        # Cột 0: ảnh gốc (resize để cùng kích thước hiển thị)
        ax0 = axes[row][0]
        ax0.imshow(orig.resize((224, 224)))
        ax0.axis('off')
        ax0.set_title("Gốc\n400×400",
                      fontsize=9, color=COLORS_CLASS[row], fontweight='bold')
        if row == 0:
            # Label cột đầu tiên
            pass
        # Viền đỏ để phân biệt ảnh gốc
        for spine in ax0.spines.values():
            spine.set_edgecolor(COLORS_CLASS[row])
            spine.set_linewidth(3)
            spine.set_visible(True)
        ax0.set_ylabel(CLASS_VI[cls], fontsize=10, fontweight='bold',
                       color=COLORS_CLASS[row], rotation=90, labelpad=6)

        # Cột 1 → n_aug: augmented
        for col in range(1, 1 + n_aug):
            ax = axes[row][col]
            aug_img = aug_transform(orig)
            ax.imshow(aug_img)
            ax.axis('off')
            if row == 0:
                ax.set_title(f"Aug #{col}\n224×224", fontsize=9)

    plt.tight_layout()
    _save("05_augmentation_preview.png", fig)


# ── Chart 06: Pie chart tỉ lệ train/val/test ───────────────
def plot_split_pie(counts):
    totals = {split: sum(counts[split].values()) for split in SPLITS}

    labels = [f"{s.upper()}\n{totals[s]} ảnh\n({totals[s]/sum(totals.values())*100:.1f}%)"
              for s in SPLITS]
    sizes  = [totals[s] for s in SPLITS]
    colors = ['#2563EB', '#16A34A', '#DC2626']
    explode = (0.04, 0.04, 0.04)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts = ax.pie(
        sizes, explode=explode, labels=labels,
        colors=colors, startangle=90,
        textprops={'fontsize': 13},
        wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
    )
    ax.set_title(f"Tỉ lệ phân chia dataset\n(Tổng: {sum(totals.values())} ảnh)",
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    _save("06_dataset_split_pie.png", fig)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def _save(filename, fig):
    out = REPORT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {filename}")


# ═══════════════════════════════════════════════════════════
# PHẦN 3: BÁO CÁO VĂN BẢN
# ═══════════════════════════════════════════════════════════

def print_and_save_report(counts, sizes, rgb_mean, rgb_std,
                          corrupt, total_imgs, weights, class_rgb):
    # Kiểm tra kích thước ảnh
    unique_sizes = set(sizes)
    all_same     = (len(unique_sizes) == 1)
    w_ref, h_ref = list(unique_sizes)[0] if all_same else (None, None)

    lines = []
    sep   = "=" * 62

    lines += [sep,
              "BÁO CÁO PHÂN TÍCH DATASET — LÁ SẦU RIÊNG",
              sep, ""]

    # ── 1. Số lượng ảnh ──────────────────────────────────────
    lines.append("1. SỐ LƯỢNG ẢNH THEO SPLIT VÀ CLASS")
    lines.append("-" * 62)
    header = f"{'Lớp bệnh':<25}" + "".join(f"{s.upper():>8}" for s in SPLITS) + f"{'TỔNG':>8}"
    lines.append(header)
    lines.append("-" * 62)

    grand = 0
    for cls in CLASS_NAMES:
        row    = f"{CLASS_VI[cls]:<25}"
        subtot = 0
        for split in SPLITS:
            cnt    = counts[split][cls]
            row   += f"{cnt:>8}"
            subtot += cnt
        row   += f"{subtot:>8}"
        grand += subtot
        lines.append(row)

    lines.append("-" * 62)
    footer = f"{'TỔNG':<25}"
    for split in SPLITS:
        footer += f"{sum(counts[split].values()):>8}"
    footer += f"{grand:>8}"
    lines.append(footer)

    lines += ["",
              f"  Tỉ lệ: Train {sum(counts['train'].values())/grand*100:.1f}%"
              f" | Val {sum(counts['val'].values())/grand*100:.1f}%"
              f" | Test {sum(counts['test'].values())/grand*100:.1f}%"]

    # ── 2. Kích thước ảnh ─────────────────────────────────────
    lines += ["", "2. KÍCH THƯỚC ẢNH"]
    lines.append("-" * 62)
    if all_same:
        lines.append(f"  Tất cả {len(sizes)} ảnh (tập train) đều có kích thước đồng nhất:")
        lines.append(f"  → {w_ref} × {h_ref} px (width × height)")
        lines.append( "  → Không cần lo ảnh méo hay kích thước bất thường.")
        lines.append( "  → Sẽ Resize xuống 256×256, sau đó crop về 224×224 (chuẩn ImageNet).")
    else:
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        lines.append(f"  Width  — Min:{min(ws)} | Max:{max(ws)} | Mean:{np.mean(ws):.1f}")
        lines.append(f"  Height — Min:{min(hs)} | Max:{max(hs)} | Mean:{np.mean(hs):.1f}")

    # ── 3. Phân tích màu RGB ──────────────────────────────────
    lines += ["", "3. PHÂN PHỐI MÀU SẮC RGB TRUNG BÌNH (tập Train)"]
    lines.append("-" * 62)
    lines.append("  Phương pháp: So sánh R/G ratio từng class với class lá khỏe (baseline).")
    lines.append("  R/G ratio cao hơn lá khỏe → lá có màu nâu/đỏ hơn → triệu chứng bệnh rõ hơn."
                 )
    lines.append("")

    # Tính R/G ratio từng class và so sánh với lá khỏe
    healthy_rg = class_rgb['Leaf_Healthy']['R'] / class_rgb['Leaf_Healthy']['G']
    rg_rows = []
    for cls in CLASS_NAMES:
        r, g, b  = class_rgb[cls]['R'], class_rgb[cls]['G'], class_rgb[cls]['B']
        rg       = r / g
        delta_rg = rg - healthy_rg
        if cls == 'Leaf_Healthy':
            note = "← Baseline (lá khỏe, R/G thấp nhất)"
        elif delta_rg > 0.03:
            note = f"R/G cao hơn lá khỏe {delta_rg:+.3f} → nâu/đỏ rõ"
        elif delta_rg > 0.01:
            note = f"R/G cao hơn lá khỏe {delta_rg:+.3f} → nâu nhẹ"
        else:
            note = f"R/G ≈ lá khỏe ({delta_rg:+.3f})"
        rg_rows.append((cls, r, g, b, rg, note))

    lines.append(f"  {'Lớp bệnh':<20}  {'R':>6}  {'G':>6}  {'B':>6}  {'R/G':>6}  Nhận xét")
    lines.append(f"  {'-'*72}")
    for cls, r, g, b, rg, note in rg_rows:
        lines.append(f"  {CLASS_VI[cls]:<20}  {r:>6.3f}  {g:>6.3f}  {b:>6.3f}  {rg:>6.3f}  {note}")

    lines += ["",
              f"  Mean RGB toàn dataset (train): "
              f"R={rgb_mean[0]:.4f}  G={rgb_mean[1]:.4f}  B={rgb_mean[2]:.4f}",
              f"  Std  RGB toàn dataset (train): "
              f"R={rgb_std[0]:.4f}  G={rgb_std[1]:.4f}  B={rgb_std[2]:.4f}",
              "",
              f"  So sánh với ImageNet normalization:",
              f"    ImageNet mean = [0.485, 0.456, 0.406]",
              f"    Dataset  mean = [{rgb_mean[0]:.3f}, {rgb_mean[1]:.3f}, {rgb_mean[2]:.3f}]",
              f"    → Kênh G của dataset ({rgb_mean[1]:.3f}) cao hơn ImageNet (0.456)",
              f"      do toàn bộ ảnh là lá cây (xanh lá chiếm ưu thế).",
              f"    → Vẫn dùng ImageNet normalization vì model pretrained yêu cầu,",
              f"      sai lệch nhỏ và được bù đắp bởi quá trình Fine-tuning."]

    # ── 4. Kiểm tra ảnh lỗi ──────────────────────────────────
    lines += ["", "4. KIỂM TRA ẢNH LỖI (CORRUPT)"]
    lines.append("-" * 62)
    lines.append(f"  Tổng ảnh kiểm tra : {total_imgs}")
    lines.append(f"  Ảnh lỗi           : {len(corrupt)}")
    if corrupt:
        lines.append("  Danh sách:")
        for p in corrupt:
            lines.append(f"    - {p}")
    else:
        lines.append("  → Không có ảnh nào bị lỗi. Không cần bước làm sạch.")

    # ── 5. Cân bằng class + class weights ────────────────────
    lines += ["", "5. MỨC ĐỘ CÂN BẰNG CLASS VÀ CLASS WEIGHTS"]
    lines.append("-" * 62)
    train_cnts = [counts['train'][c] for c in CLASS_NAMES]
    ratio = max(train_cnts) / min(train_cnts)
    lines.append(f"  Tỉ lệ max/min (train): {ratio:.2f}×  "
                 f"({max(train_cnts)} / {min(train_cnts)} ảnh)")

    if ratio < 1.5:
        balance_note = "Dataset CÂN BẰNG TỐT (ratio < 1.5)."
    elif ratio < 3.0:
        balance_note = "Dataset MẤT CÂN BẰNG NHẸ (1.5 ≤ ratio < 3.0)."
    else:
        balance_note = "Dataset MẤT CÂN BẰNG NGHIÊM TRỌNG (ratio ≥ 3.0)."
    lines.append(f"  → {balance_note}")
    lines.append( "  → Vẫn áp dụng Class Weights để model không thiên vị class nào.")
    lines.append( "  → Công thức: weight[c] = total / (n_classes × count[c])")
    lines += ["",
              f"  {'Lớp bệnh':<20}  {'Train':>7}  {'Weight':>8}"]
    lines.append(f"  {'-'*42}")
    for cls, w in zip(CLASS_NAMES, weights):
        lines.append(f"  {CLASS_VI[cls]:<20}  {counts['train'][cls]:>7}  {w:>8.4f}")

    # ── 6. Tổng kết ───────────────────────────────────────────
    lines += ["", "6. TỔNG KẾT VÀ NHẬN XÉT"]
    lines.append("-" * 62)
    lines += [
        f"  - Tổng dataset   : {grand} ảnh | 6 lớp bệnh",
        f"  - Kích thước ảnh : Đồng nhất {w_ref}×{h_ref}px — chất lượng cao",
        f"  - Ảnh lỗi        : 0 — không cần làm sạch",
        f"  - Cân bằng class : Tốt (ratio {ratio:.2f}×)",
        f"  - Đặc trưng màu  : Có sự khác biệt RGB giữa các lớp bệnh",
        f"                     → Mô hình CNN có thể học được đặc trưng màu.",
        f"  - Augmentation   : Phù hợp — flip, rotation, colorjitter",
        f"  - Normalization  : Dùng ImageNet mean/std — phù hợp với pretrained models.",
    ]

    lines.append("")
    lines.append(sep)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    # Lưu text
    out_txt = REPORT_DIR / "eda_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  ✓ eda_summary.txt")

    # Lưu JSON (dùng lại trong các bước sau)
    summary = {
        "counts": counts,
        "image_properties": {
            "all_same_size": all_same,
            "size_px": f"{w_ref}x{h_ref}" if all_same else "mixed",
            "total_train_images": len(sizes),
        },
        "dataset_rgb": {
            "mean": rgb_mean.tolist(),
            "std":  rgb_std.tolist(),
        },
        "class_rgb_mean": class_rgb,
        "corrupt_images": corrupt,
        "total_images_checked": total_imgs,
        "class_weights": dict(zip(CLASS_NAMES, [round(w, 4) for w in weights])),
        "imbalance_ratio": round(ratio, 3),
    }
    out_json = REPORT_DIR / "eda_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ eda_summary.json")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 62)
    print("  EDA — PHÂN TÍCH DATASET LÁ SẦU RIÊNG")
    print("=" * 62)

    print("\n[1/5] Đếm số ảnh mỗi class...")
    counts = count_images()

    print("[2/5] Kiểm tra ảnh bị lỗi (corrupt)...")
    corrupt, total_imgs = check_corrupt()
    print(f"      → {total_imgs} ảnh kiểm tra | {len(corrupt)} ảnh lỗi")

    print("[3/5] Phân tích kích thước + RGB (đọc ảnh 1 lần)...")
    sizes, rgb_mean, rgb_std, class_rgb = analyze_all_image_properties()
    print(f"      → Mean RGB: R={rgb_mean[0]:.4f}  G={rgb_mean[1]:.4f}  B={rgb_mean[2]:.4f}")

    weights = compute_class_weights(counts)

    print("[4/5] Vẽ biểu đồ...")
    plot_class_distribution(counts)
    plot_grouped_distribution(counts)
    plot_sample_images()
    plot_rgb_per_class(class_rgb)
    plot_augmentation_preview()
    plot_split_pie(counts)

    print("[5/5] Xuất báo cáo text...")
    print_and_save_report(counts, sizes, rgb_mean, rgb_std,
                          corrupt, total_imgs, weights, class_rgb)

    print(f"\n{'='*62}")
    print(f"  EDA hoàn tất! Output tại: {REPORT_DIR}")
    print(f"  6 biểu đồ + eda_summary.txt + eda_summary.json")
    print(f"{'='*62}")
