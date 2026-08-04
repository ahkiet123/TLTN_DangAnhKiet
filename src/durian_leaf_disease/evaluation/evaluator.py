"""
Evaluate trained checkpoints on the test split.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.amp import autocast

from durian_leaf_disease.config import (
    CLASS_LABELS_VI,
    CLASS_NAMES,
    DEVICE,
    MIXED_PRECISION,
    MODELS_TO_TRAIN,
    NUM_CLASSES,
    OUTPUT_DIR,
)
from durian_leaf_disease.data.dataset import get_dataloaders
from durian_leaf_disease.models.transfer import build_model


def load_checkpoint(model_name: str) -> torch.nn.Module:
    ckpt_path = Path(OUTPUT_DIR) / model_name / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(model_name, num_classes=NUM_CLASSES, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()

    print(
        f"  Loaded: {ckpt_path} "
        f"(epoch {ckpt['epoch']}, val_acc={ckpt['val_acc'] * 100:.2f}%)"
    )
    return model


def predict(model: torch.nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    all_preds, all_labels = [], []
    amp_enabled = MIXED_PRECISION and DEVICE.type == "cuda"

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=amp_enabled):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, model_name: str, save_dir: Path) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    class_labels = [CLASS_LABELS_VI.get(c, c) for c in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Dự đoán", fontsize=12)
    ax.set_ylabel("Thực tế", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = save_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix: {out_path}")


def plot_training_curves(model_name: str, save_dir: Path) -> None:
    history_path = Path(OUTPUT_DIR) / model_name / "history.json"
    if not history_path.exists():
        return

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2563EB")
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#DC2626")
    axes[0].axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="Phase 2")
    axes[0].set_title("Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        [acc * 100 for acc in history["train_acc"]],
        label="Train",
        color="#2563EB",
    )
    axes[1].plot(
        epochs,
        [acc * 100 for acc in history["val_acc"]],
        label="Val",
        color="#DC2626",
    )
    axes[1].axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="Phase 2")
    axes[1].set_title("Accuracy (%)", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Training Curves - {model_name}", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = save_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Training curves: {out_path}")


def evaluate_model(model_name: str, dataloaders: dict) -> dict:
    print(f"\n{'=' * 55}")
    print(f"  ĐÁNH GIÁ: {model_name.upper()}")
    print(f"{'=' * 55}")

    save_dir = Path(OUTPUT_DIR) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint(model_name)
    labels, preds = predict(model, dataloaders["test"])

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    class_labels = [CLASS_LABELS_VI.get(c, c) for c in CLASS_NAMES]
    report = classification_report(
        labels,
        preds,
        labels=list(range(NUM_CLASSES)),
        target_names=class_labels,
        digits=4,
        zero_division=0,
    )

    print(f"\n  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Weighted F1   : {f1 * 100:.2f}%")
    print(f"\n{report}")

    report_path = save_dir / "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy : {acc * 100:.2f}%\n")
        f.write(f"Weighted F1   : {f1 * 100:.2f}%\n\n")
        f.write(report)
    print(f"  Report saved : {report_path}")

    plot_confusion_matrix(labels, preds, model_name, save_dir)
    plot_training_curves(model_name, save_dir)

    return {"accuracy": acc, "f1": f1}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    args = parser.parse_args(argv)

    dataloaders, _ = get_dataloaders()
    model_list = MODELS_TO_TRAIN if args.model == "all" else [args.model]

    results = {}
    for name in model_list:
        try:
            results[name] = evaluate_model(name, dataloaders)
        except FileNotFoundError as exc:
            print(f"  SKIP {name}: {exc}")

    if results:
        print(f"\n{'=' * 55}")
        print("TỔNG KẾT TEST:")
        for name, result in sorted(results.items(), key=lambda item: -item[1]["accuracy"]):
            print(
                f"  {name:<20}: "
                f"Acc={result['accuracy'] * 100:.2f}%  "
                f"F1={result['f1'] * 100:.2f}%"
            )


if __name__ == "__main__":
    main()
