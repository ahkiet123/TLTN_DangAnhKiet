"""
train.py - Pipeline huấn luyện 2 giai đoạn với Transfer Learning
Giai đoạn 1 (Feature Extraction): Backbone đóng băng, chỉ train classifier
Giai đoạn 2 (Fine-tuning)       : Mở thêm các block cuối, train toàn bộ

Chạy: python scripts/train.py
"""
import os
import time
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from durian_leaf_disease.config import (
    DEVICE, OUTPUT_DIR, NUM_CLASSES, CLASS_NAMES,
    NUM_EPOCHS, PHASE1_EPOCHS, LR_HEAD, LR_BACKBONE,
    WEIGHT_DECAY, EARLY_STOPPING, MIXED_PRECISION,
    MODELS_TO_TRAIN
)
from durian_leaf_disease.data.dataset import (
    get_class_weights,
    get_dataloaders,
    set_seed,
)
from durian_leaf_disease.models.transfer import build_model, unfreeze_last_n_blocks


# =============================================================================
# HELPERS
# =============================================================================

class EarlyStopping:
    """Dừng sớm nếu val_loss không cải thiện sau `patience` epoch."""
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = None
        self.stop      = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def run_epoch(model, loader, criterion, optimizer, scaler,
              phase: str, device) -> tuple[float, float]:
    """
    Chạy 1 epoch train hoặc val.
    Returns: (avg_loss, accuracy)
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    amp_enabled = MIXED_PRECISION and device.type == "cuda"

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model(model_name: str, dataloaders: dict, class_weights: torch.Tensor):
    """
    Train 1 model qua 2 giai đoạn, lưu best checkpoint.
    """
    print(f"\n{'#'*60}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"{'#'*60}")

    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "best_model.pth")

    # ---------- Build model ----------
    model = build_model(model_name, num_classes=NUM_CLASSES, freeze_backbone=True)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler    = GradScaler(device=DEVICE.type, enabled=MIXED_PRECISION and DEVICE.type == "cuda")
    early_stopper = EarlyStopping(patience=EARLY_STOPPING)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc  = float("-inf")
    best_epoch: int | None = None
    best_weights  = copy.deepcopy(model.state_dict())
    training_started_at = time.perf_counter()

    def save_checkpoint(epoch: int) -> None:
        elapsed_seconds = time.perf_counter() - training_started_at
        torch.save({
            "epoch":       epoch,
            "best_epoch":  epoch,
            "model_name":  model_name,
            "model_state": best_weights,
            "val_acc":     best_val_acc,
            "class_names": CLASS_NAMES,
            "training_time_seconds": elapsed_seconds,
        }, ckpt_path)

    # ===================================================
    # GIAI ĐOẠN 1: Feature Extraction (epoch 1 -> PHASE1_EPOCHS)
    # ===================================================
    print(f"\n[Giai đoạn 1] Feature Extraction - {PHASE1_EPOCHS} epoch, "
          f"backbone đóng băng, lr={LR_HEAD}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, dataloaders["train"], criterion,
                                          optimizer, scaler, "train", DEVICE)
        val_loss, val_acc     = run_epoch(model, dataloaders["val"],   criterion,
                                          optimizer, scaler, "val",   DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = " ← best" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:02d}/{PHASE1_EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc*100:.1f}% | "
              f"val_loss={val_loss:.4f} acc={val_acc*100:.1f}% | "
              f"{elapsed:.0f}s{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            save_checkpoint(epoch)

        if early_stopper.step(val_loss):
            print(f"  Early stopping tại epoch {epoch}")
            break

    # ===================================================
    # GIAI ĐOẠN 2: Fine-tuning (tiếp theo)
    # ===================================================
    print(f"\n[Giai đoạn 2] Fine-tuning - {NUM_EPOCHS - PHASE1_EPOCHS} epoch, "
          f"mở 3 block cuối, lr_backbone={LR_BACKBONE}, lr_head={LR_HEAD}")

    # Khôi phục best weights từ giai đoạn 1
    model.load_state_dict(best_weights)
    unfreeze_last_n_blocks(model, model_name, n=3)

    # Tạo optimizer với 2 param group: head (lr cao) + backbone (lr thấp)
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
        {"params": head_params,     "lr": LR_HEAD},
        {"params": backbone_params, "lr": LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)

    phase2_epochs = NUM_EPOCHS - PHASE1_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING)

    for epoch in range(PHASE1_EPOCHS + 1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, dataloaders["train"], criterion,
                                          optimizer, scaler, "train", DEVICE)
        val_loss, val_acc     = run_epoch(model, dataloaders["val"],   criterion,
                                          optimizer, scaler, "val",   DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = " ← best" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc*100:.1f}% | "
              f"val_loss={val_loss:.4f} acc={val_acc*100:.1f}% | "
              f"{elapsed:.0f}s{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            # Lưu checkpoint tốt nhất
            save_checkpoint(epoch)

        if early_stopper.step(val_loss):
            print(f"  Early stopping tại epoch {epoch}")
            break

    # Lưu history
    training_time_seconds = time.perf_counter() - training_started_at
    history["training_time_seconds"] = training_time_seconds
    history["training_time_minutes"] = training_time_seconds / 60.0
    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_val_acc

    # Rewrite the best checkpoint once so its metadata contains the full run time.
    if best_epoch is not None:
        save_checkpoint(best_epoch)

    history_path = os.path.join(model_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"  Training time    : {training_time_seconds / 60.0:.2f} minutes")
    print(f"\n  ✓ Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"  ✓ Checkpoint: outputs/{model_name}/best_model.pth")
    print(f"  ✓ History   : outputs/{model_name}/history.json")
    return best_val_acc


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(model_names: list[str] | None = None) -> None:
    set_seed()

    selected_models = MODELS_TO_TRAIN if model_names is None else model_names
    unsupported = set(selected_models) - set(MODELS_TO_TRAIN)
    if unsupported:
        raise ValueError(f"Model không được hỗ trợ: {sorted(unsupported)}")
    if not selected_models:
        raise ValueError("Danh sách model cần huấn luyện đang rỗng")

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # Load dataloaders và class weights 1 lần dùng chung
    print("\nLoading dataloaders...")
    dataloaders, class_to_idx = get_dataloaders()
    class_weights = get_class_weights(device=DEVICE)
    print(f"Class weights: {class_weights.cpu().tolist()}")

    results = {}
    for model_name in selected_models:
        val_acc = train_model(model_name, dataloaders, class_weights)
        results[model_name] = val_acc

    # Tổng kết
    print(f"\n{'='*60}")
    print("TỔNG KẾT - Best Val Accuracy:")
    print(f"{'='*60}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<20}: {acc*100:.2f}%")
    best_model = max(results, key=results.get)
    print(f"\n  🏆 Model tốt nhất: {best_model} ({results[best_model]*100:.2f}%)")
