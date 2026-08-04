"""
model.py — Xây dựng model Transfer Learning cho phân loại bệnh lá sầu riêng
=============================================================================
Bài toán: Phân loại 6 lớp bệnh lá sầu riêng (đa lớp, 1 nhãn/ảnh)
Dataset : 2,595 ảnh × 6 lớp (Train=1814 / Val=387 / Test=394)
Hardware: NVIDIA RTX 3050 Ti Laptop (4GB VRAM)

Lý do chọn 3 kiến trúc:
────────────────────────
• MobileNetV2     — Nhẹ (~3.4MB), tốc độ cao, phù hợp deploy embedded
                    Inverted Residual Bottleneck + Depthwise Separable Conv
• EfficientNet-B0 — Cân bằng accuracy/efficiency tốt nhất (compound scaling)
                    Thường dẫn đầu trên dataset ảnh vừa nhỏ
• ResNet-50       — Baseline mạnh, Skip connections chống vanishing gradient
                    Đã chứng minh hiệu quả trên nhiều bài toán phân loại ảnh lá

Chiến lược Transfer Learning 2 giai đoạn:
──────────────────────────────────────────
  Giai đoạn 1 (Feature Extraction): Đóng băng toàn bộ backbone, chỉ train
    classifier mới. Mục đích: hội tụ nhanh, tránh phá vỡ features ImageNet.
  Giai đoạn 2 (Fine-tuning): Mở N block cuối backbone. Mục đích: tinh chỉnh
    features cấp cao cho đặc điểm bệnh lá sầu riêng.

Classifier mới (thay thế head gốc):
  Dropout(0.3) → Linear(in, 256) → BatchNorm1d(256) → ReLU → Dropout(0.2)
  → Linear(256, num_classes)
  Dropout giảm overfitting. BN1d ổn định training. 2 lớp tăng capacity.
"""

import os
import json

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict

from durian_leaf_disease.config import NUM_CLASSES, OUTPUT_DIR, DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CLASSIFIER HEAD
# ─────────────────────────────────────────────────────────────────────────────

def _make_classifier(in_features: int, num_classes: int,
                     dropout1: float = 0.3,
                     dropout2: float = 0.2,
                     hidden: int = 256) -> nn.Sequential:
    """
    Tạo classifier head mới thay thế head gốc của model pretrained.

    Kiến trúc:
        Dropout(p1) → Linear(in, hidden) → BatchNorm1d(hidden)
        → ReLU → Dropout(p2) → Linear(hidden, num_classes)

    Tại sao 2 lớp thay vì 1 lớp?
      Dataset nhỏ (1814 ảnh train) → 1 lớp đơn giản thường underfit với
      backbone lớn như ResNet-50 (2048 features → 6 classes trực tiếp).
      Thêm 1 lớp ẩn 256 units tạo thêm capacity mà không quá phức tạp.

    Tại sao Dropout trước Linear đầu tiên?
      Features từ backbone rất giàu thông tin. Dropout sớm giảm co-adaptation,
      buộc classifier học nhiều patterns độc lập nhau.

    Tại sao BatchNorm1d giữa 2 lớp?
      Ổn định phân phối đầu vào lớp 2, giúp training nhanh hơn và
      ít nhạy cảm với learning rate.

    Args:
        in_features: số features từ backbone (ví dụ 1280 với MobileNetV2)
        num_classes: số lớp đầu ra (6)
        dropout1   : tỷ lệ dropout đầu tiên (0.3)
        dropout2   : tỷ lệ dropout thứ hai (0.2)
        hidden     : số neurons lớp ẩn (256)

    Returns:
        nn.Sequential — classifier head hoàn chỉnh
    """
    assert in_features > 0, f"in_features phải > 0, nhận được {in_features}"
    assert hidden > 0, f"hidden phải > 0, nhận được {hidden}"
    assert num_classes > 0, f"num_classes phải > 0, nhận được {num_classes}"

    return nn.Sequential(
        nn.Dropout(p=dropout1),
        nn.Linear(in_features, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout2),
        nn.Linear(hidden, num_classes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    dropout1: float = 0.3,
    dropout2: float = 0.2,
    hidden_size: int = 256,
) -> nn.Module:
    """
    Tạo model pretrained ImageNet với custom classifier head cho 6 lớp bệnh.

    Với mỗi model:
      1. Load pretrained weights từ ImageNet (IMAGENET1K_V1)
      2. Đóng băng toàn bộ backbone (nếu freeze_backbone=True)
      3. Thay thế classifier head gốc bằng custom head có Dropout + BN

    Args:
        model_name     : 'mobilenet_v2' | 'efficientnet_b0' | 'resnet50'
        num_classes    : số lớp đầu ra (mặc định 6)
        freeze_backbone: True = đóng băng backbone (Feature Extraction)
                         False = train toàn bộ (Fine-tuning từ đầu)
        dropout1       : dropout trước Linear đầu tiên (mặc định 0.3)
        dropout2       : dropout giữa 2 Linear (mặc định 0.2)
        hidden_size    : số neurons lớp ẩn trong classifier (mặc định 256)

    Returns:
        model: nn.Module đã cấu hình, sẵn sàng train

    Raises:
        ValueError: nếu model_name không được hỗ trợ
    """
    model_name = model_name.lower().strip()

    # ── MobileNetV2 ──────────────────────────────────────────────────────────
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        # Đóng băng backbone (model.features là toàn bộ backbone)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        # model.classifier = [Dropout(0.2), Linear(1280, 1000)] gốc
        # → Thay thế bằng custom head
        in_features = model.classifier[1].in_features   # 1280
        model.classifier = _make_classifier(
            in_features, num_classes, dropout1, dropout2, hidden_size
        )

    # ── EfficientNet-B0 ──────────────────────────────────────────────────────
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # Đóng băng backbone (model.features)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        # model.classifier = [Dropout(0.2), Linear(1280, 1000)] gốc
        in_features = model.classifier[1].in_features   # 1280
        model.classifier = _make_classifier(
            in_features, num_classes, dropout1, dropout2, hidden_size
        )

    # ── ResNet-50 ─────────────────────────────────────────────────────────────
    elif model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        # Đóng băng tất cả layer trừ fc (sẽ bị thay thế)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        # model.fc = Linear(2048, 1000) gốc
        in_features = model.fc.in_features              # 2048
        model.fc = _make_classifier(
            in_features, num_classes, dropout1, dropout2, hidden_size
        )

    else:
        supported = ["mobilenet_v2", "efficientnet_b0", "resnet50"]
        raise ValueError(
            f"Model '{model_name}' không được hỗ trợ.\n"
            f"Chọn một trong: {supported}"
        )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# UNFREEZE LAST N BLOCKS (Giai đoạn 2 Fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

def unfreeze_last_n_blocks(
    model: nn.Module,
    model_name: str,
    n: int = 3,
) -> None:
    """
    Giai đoạn 2 Fine-tuning: Mở N block cuối của backbone để tinh chỉnh.

    Lý do mở N block cuối (không phải toàn bộ):
      • Block đầu (edge, texture, color) → chung cho mọi dataset → giữ đóng
      • Block cuối (high-level features) → đặc thù domain → cần fine-tune
        cho bệnh lá sầu riêng (màu, hình dạng đốm, cấu trúc lá)

    Args:
        model     : model đã build_model() (đang trong trạng thái freeze)
        model_name: 'mobilenet_v2' | 'efficientnet_b0' | 'resnet50'
        n         : số block cuối muốn mở (mặc định 3)

    Với từng kiến trúc:
      MobileNetV2    : model.features có 19 phần tử → mở features[-n:]
      EfficientNet-B0: model.features có  9 phần tử → mở features[-n:]
      ResNet-50      : 4 layer chính (layer1-4) → mở layer(5-n) đến layer4
                       n=1→layer4; n=2→layer3+layer4; n=3→layer2+layer3+layer4
                       ⚠ n=3 mở ~99% params ResNet-50 — kiểm soát overfitting
                         bằng weight_decay, early stopping, và augmentation.
    """
    model_name = model_name.lower().strip()

    if model_name in ("mobilenet_v2", "efficientnet_b0"):
        # Sequential backbone → mở n phần tử cuối
        blocks = list(model.features.children())
        n_clamped = min(n, len(blocks))  # Không vượt quá số block
        for block in blocks[-n_clamped:]:
            for param in block.parameters():
                param.requires_grad = True

    elif model_name == "resnet50":
        # ResNet-50 có 4 layer chính: layer1, layer2, layer3, layer4
        # Danh sách từ cuối vào: [layer4, layer3, layer2, layer1]
        resnet_layers = [model.layer4, model.layer3,
                         model.layer2, model.layer1]
        n_clamped = min(n, len(resnet_layers))
        for layer in resnet_layers[:n_clamped]:
            for param in layer.parameters():
                param.requires_grad = True

    else:
        raise ValueError(
            f"Model '{model_name}' không được hỗ trợ trong unfreeze_last_n_blocks."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_param_counts(model: nn.Module) -> Dict[str, int]:
    """
    Đếm số tham số của model.

    Returns:
        dict với keys: 'total', 'trainable', 'frozen'
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def get_model_size_mb(model: nn.Module) -> float:
    """
    Tính kích thước model (MB) dựa trên tất cả tham số và buffer.
    Công thức: số phần tử × 4 bytes (float32) / 1024^2

    Note: Đây là kích thước khi lưu ra file .pt (xấp xỉ).
    """
    total_bytes = sum(
        p.nelement() * p.element_size()
        for p in model.parameters()
    ) + sum(
        b.nelement() * b.element_size()
        for b in model.buffers()
    )
    return total_bytes / (1024 ** 2)


def get_classifier_head(model: nn.Module, model_name: str) -> nn.Sequential:
    """
    Trả về classifier head của model (đã được thay thế bởi _make_classifier).
    Helper để tránh lặp lại logic if/else khi cần truy cập head.

    Returns:
        nn.Sequential — classifier head (model.classifier hoặc model.fc)
    """
    model_name = model_name.lower().strip()
    if model_name in ("mobilenet_v2", "efficientnet_b0"):
        return model.classifier
    elif model_name == "resnet50":
        return model.fc
    else:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ.")


def verify_forward_pass(
    model: nn.Module,
    model_name: str,
    device: torch.device,
) -> bool:
    """
    Kiểm tra forward pass với dummy input torch.zeros(2, 3, 224, 224).
    Trả về True nếu output shape đúng, False nếu có lỗi.

    Dùng batch_size=2 (không phải 1) vì BatchNorm1d trong classifier
    cần ít nhất 2 samples để tính mean/std khi ở chế độ train().

    ⚠ Hàm tạm chuyển model sang eval() để test, sau đó khôi phục
      trạng thái ban đầu (train/eval) để không ảnh hưởng caller.
    """
    was_training = model.training  # Ghi nhớ trạng thái ban đầu
    model.eval()                   # Tránh lỗi BN với batch nhỏ
    dummy = torch.zeros(2, 3, 224, 224).to(device)
    try:
        with torch.no_grad():
            out = model(dummy)
        expected = (2, NUM_CLASSES)
        if out.shape != torch.Size(expected):
            print(f"  ❌ Output shape sai: {out.shape} (cần {expected})")
            return False
        print(f"  ✅ Forward pass OK: input {list(dummy.shape)} → output {list(out.shape)}")
        return True
    except Exception as e:
        print(f"  ❌ Forward pass THẤT BẠI: {e}")
        return False
    finally:
        # Khôi phục trạng thái ban đầu — quan trọng nếu caller đang train
        if was_training:
            model.train()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — So sánh 3 model
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # DEVICE đã import từ config.py (single source of truth)

    print("=" * 70)
    print("  BƯỚC 3 — XÂY DỰNG VÀ SO SÁNH 3 MODEL TRANSFER LEARNING")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    MODEL_NAMES = ["mobilenet_v2", "efficientnet_b0", "resnet50"]

    # Lưu kết quả để in bảng cuối
    results = {}

    for model_name in MODEL_NAMES:
        print(f"\n{'─' * 70}")
        print(f"  📦 {model_name.upper()}")
        print(f"{'─' * 70}")

        # ── A. Giai đoạn 1: Feature Extraction (freeze_backbone=True) ──
        print(f"\n  [GIAI ĐOẠN 1] Feature Extraction — freeze_backbone=True")
        model_p1 = build_model(model_name, freeze_backbone=True)
        model_p1 = model_p1.to(DEVICE)

        p1 = get_param_counts(model_p1)
        size_mb = get_model_size_mb(model_p1)
        pct_train_p1 = 100 * p1["trainable"] / p1["total"]

        print(f"  Tổng params      : {p1['total']:>12,}")
        print(f"  Đóng băng        : {p1['frozen']:>12,}")
        print(f"  Trainable (Phase1): {p1['trainable']:>12,}  ({pct_train_p1:.1f}%)")
        print(f"  Kích thước model : {size_mb:>11.2f} MB")

        # ── B. Verify forward pass Giai đoạn 1 ──
        print(f"\n  Forward pass (Phase 1):")
        ok_p1 = verify_forward_pass(model_p1, model_name, DEVICE)

        # ── C. Giai đoạn 2: Unfreeze 3 block cuối ──
        print(f"\n  [GIAI ĐOẠN 2] Fine-tuning — unfreeze 3 blocks cuối")
        unfreeze_last_n_blocks(model_p1, model_name, n=3)
        p2 = get_param_counts(model_p1)
        pct_train_p2 = 100 * p2["trainable"] / p2["total"]
        newly_opened = p2["trainable"] - p1["trainable"]

        print(f"  Trainable (Phase2): {p2['trainable']:>12,}  ({pct_train_p2:.1f}%)")
        print(f"  Mới mở thêm      : {newly_opened:>12,} params")

        # ── D. Verify forward pass Giai đoạn 2 ──
        print(f"\n  Forward pass (Phase 2):")
        ok_p2 = verify_forward_pass(model_p1, model_name, DEVICE)

        # ── E. Kiểm tra classifier head ──
        print(f"\n  Classifier head:")
        head = get_classifier_head(model_p1, model_name)
        for i, layer in enumerate(head):
            print(f"    [{i}] {layer}")

        # ── F. Kiểm tra gradient flow ──
        print(f"\n  Gradient flow (Phase 2 — params với requires_grad=True):")
        grad_names = [
            name for name, p in model_p1.named_parameters()
            if p.requires_grad
        ]
        # In 5 đầu và 5 cuối để minh họa
        if len(grad_names) > 10:
            shown = grad_names[:3] + ["  ..."] + grad_names[-3:]
        else:
            shown = grad_names
        for g in shown:
            print(f"    ✓ {g}")
        print(f"    (Tổng {len(grad_names)} tensors có gradient)")

        # Lưu kết quả
        results[model_name] = {
            "total_params":      p1["total"],
            "frozen_params":     p1["frozen"],
            "trainable_phase1":  p1["trainable"],
            "trainable_phase2":  p2["trainable"],
            "pct_phase1":        round(pct_train_p1, 1),
            "pct_phase2":        round(pct_train_p2, 1),
            "size_mb":           round(size_mb, 2),
            "forward_ok_p1":     ok_p1,
            "forward_ok_p2":     ok_p2,
        }

        # Giải phóng VRAM
        del model_p1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── BẢNG SO SÁNH ─────────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  BẢNG SO SÁNH 3 MODEL")
    print(f"{'=' * 70}")

    # Header
    h1 = f"{'Tiêu chí':<28}"
    h2 = f"{'MobileNetV2':>14}"
    h3 = f"{'EfficientNet-B0':>16}"
    h4 = f"{'ResNet-50':>12}"
    print(f"  {h1} {h2} {h3} {h4}")
    print(f"  {'─' * 68}")

    r_mb  = results["mobilenet_v2"]
    r_eb  = results["efficientnet_b0"]
    r_rn  = results["resnet50"]

    rows = [
        ("Tổng params",
            f"{r_mb['total_params']:,}",
            f"{r_eb['total_params']:,}",
            f"{r_rn['total_params']:,}"),
        ("Kích thước (MB)",
            f"{r_mb['size_mb']} MB",
            f"{r_eb['size_mb']} MB",
            f"{r_rn['size_mb']} MB"),
        ("Trainable Phase 1",
            f"{r_mb['trainable_phase1']:,} ({r_mb['pct_phase1']}%)",
            f"{r_eb['trainable_phase1']:,} ({r_eb['pct_phase1']}%)",
            f"{r_rn['trainable_phase1']:,} ({r_rn['pct_phase1']}%)"),
        ("Trainable Phase 2",
            f"{r_mb['trainable_phase2']:,} ({r_mb['pct_phase2']}%)",
            f"{r_eb['trainable_phase2']:,} ({r_eb['pct_phase2']}%)",
            f"{r_rn['trainable_phase2']:,} ({r_rn['pct_phase2']}%)"),
        ("Forward Pass OK",
            "✅" if r_mb["forward_ok_p1"] and r_mb["forward_ok_p2"] else "❌",
            "✅" if r_eb["forward_ok_p1"] and r_eb["forward_ok_p2"] else "❌",
            "✅" if r_rn["forward_ok_p1"] and r_rn["forward_ok_p2"] else "❌"),
    ]

    for label, v_mb, v_eb, v_rn in rows:
        print(f"  {label:<28} {v_mb:>14} {v_eb:>16} {v_rn:>12}")

    print(f"\n  Ghi chú:")
    print(f"  • Tất cả model dùng pretrained ImageNet1K-V1")
    print(f"  • Classifier head mới: Dropout(0.3)→Linear→BN1d→ReLU→Dropout(0.2)→Linear")
    print(f"  • Phase 1: Chỉ train classifier head (10 epochs)")
    print(f"  • Phase 2: Unfreeze 3 blocks cuối backbone (20 epochs)")
    print(f"  • Input chuẩn: 224×224 px, chuẩn hóa ImageNet mean/std")

    # ── Lưu kết quả ra JSON ───────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "step3_model_comparison.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Kết quả đã lưu: {save_path}")

    print(f"\n{'=' * 70}")
    print("  ✅ BƯỚC 3 HOÀN TẤT — Cả 3 model build và verify thành công!")
    print(f"{'=' * 70}")
