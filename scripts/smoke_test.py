"""
smoke_test.py - Chạy thử nghiệm nhanh trước khi train full.

Mục tiêu:
  * Kiểm tra pipeline train/val chạy được trên GPU/CPU
  * Phát hiện OOM sớm (đặc biệt ResNet-50 phase 2)
  * Ước lượng thờii gian mỗi epoch để lập kế hoạch chạy full

Chạy:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from durian_leaf_disease.config import (
    DEVICE,
    LR_BACKBONE,
    LR_HEAD,
    MIXED_PRECISION,
    NUM_CLASSES,
    WEIGHT_DECAY,
)
from durian_leaf_disease.data.dataset import get_class_weights, get_dataloaders
from durian_leaf_disease.models.transfer import build_model, unfreeze_last_n_blocks


def smoke_train(model_name: str, dataloaders: dict, class_weights: torch.Tensor):
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST: {model_name.upper()}")
    print(f"{'='*60}")

    model = build_model(model_name, num_classes=NUM_CLASSES, freeze_backbone=True)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler(device=DEVICE.type, enabled=MIXED_PRECISION and DEVICE.type == "cuda")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY,
    )
    amp_enabled = MIXED_PRECISION and DEVICE.type == "cuda"

    # Phase 1: 1 epoch
    print("\n[Phase 1] Feature extraction - 1 epoch")
    model.train()
    t0 = time.time()
    for images, labels in dataloaders["train"]:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=DEVICE.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    elapsed_train = time.time() - t0

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for images, labels in dataloaders["val"]:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=amp_enabled):
                outputs = model(images)
    elapsed_val = time.time() - t0

    print(f"  Train 1 epoch: {elapsed_train:.1f}s")
    print(f"  Val   1 epoch: {elapsed_val:.1f}s")

    # Phase 2: unfreeze 3 blocks + 1 epoch
    print("\n[Phase 2] Fine-tuning - unfreeze 3 blocks + 1 epoch")
    unfreeze_last_n_blocks(model, model_name, n=3)

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ("classifier" in name) or ("fc" in name):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": head_params, "lr": LR_HEAD},
        {"params": backbone_params, "lr": LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    model.train()
    t0 = time.time()
    for images, labels in dataloaders["train"]:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=DEVICE.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    elapsed_train2 = time.time() - t0
    print(f"  Train 1 epoch (phase 2): {elapsed_train2:.1f}s")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "phase1_train_s": elapsed_train,
        "phase1_val_s": elapsed_val,
        "phase2_train_s": elapsed_train2,
    }


def main() -> None:
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    dataloaders, _ = get_dataloaders()
    class_weights = get_class_weights(device=DEVICE)

    timings = {}
    for model_name in ["mobilenet_v2", "efficientnet_b0", "resnet50"]:
        try:
            timings[model_name] = smoke_train(model_name, dataloaders, class_weights)
        except RuntimeError as exc:
            print(f"\n  ❌ Lỗi với {model_name}: {exc}")
            timings[model_name] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("  TỔNG KẾT SMOKE TEST")
    print(f"{'='*60}")
    for name, t in timings.items():
        if t is None:
            print(f"  {name:<20}: FAILED")
        else:
            total_per_epoch = t["phase1_train_s"] + t["phase1_val_s"]
            print(f"  {name:<20}: phase1 ~{total_per_epoch:.0f}s/epoch | "
                  f"phase2 ~{t['phase2_train_s']:.0f}s/epoch")
            print(f"  {'':20}  (ước tính full 30 epochs: "
                  f"~{((total_per_epoch*10 + t['phase2_train_s']*20)/3600):.1f}h)")


if __name__ == "__main__":
    main()
