"""
Evaluate trained checkpoints on the test split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.amp import autocast

from durian_leaf_disease.config import (
    CLASS_LABELS_VI,
    CLASS_NAMES,
    DEVICE,
    IMAGE_SIZE,
    MIXED_PRECISION,
    MODELS_TO_TRAIN,
    NUM_CLASSES,
    OUTPUT_DIR,
    PHASE1_EPOCHS,
)
from durian_leaf_disease.data.dataset import get_dataloaders
from durian_leaf_disease.evaluation.gradcam import generate_gradcam_grid
from durian_leaf_disease.models.transfer import (
    build_model,
    get_model_size_mb,
    get_param_counts,
)


def _class_labels() -> list[str]:
    """Return display labels in the same order as the configured classes."""
    return [CLASS_LABELS_VI.get(name, name) for name in CLASS_NAMES]


def load_checkpoint(model_name: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a checkpoint and return the model together with its metadata."""
    ckpt_path = Path(OUTPUT_DIR) / model_name / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Checkpoint không hợp lệ: {ckpt_path}")

    model = build_model(model_name, num_classes=NUM_CLASSES, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()

    epoch = ckpt.get("epoch")
    val_acc = ckpt.get("val_acc")
    checkpoint_info = []
    if epoch is not None:
        checkpoint_info.append(f"epoch {epoch}")
    if val_acc is not None:
        try:
            checkpoint_info.append(f"val_acc={float(val_acc) * 100:.2f}%")
        except (TypeError, ValueError):
            checkpoint_info.append("val_acc=unavailable")
    suffix = f" ({', '.join(checkpoint_info)})" if checkpoint_info else ""
    print(f"  Loaded: {ckpt_path}{suffix}")

    return model, {
        "epoch": epoch,
        "best_epoch": ckpt.get("best_epoch", epoch),
        "val_acc": val_acc,
        "training_time_seconds": ckpt.get("training_time_seconds"),
    }


def predict(
    model: torch.nn.Module,
    loader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect labels, argmax predictions, and softmax class probabilities."""
    all_preds, all_labels, all_probabilities = [], [], []
    amp_enabled = MIXED_PRECISION and DEVICE.type == "cuda"

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=amp_enabled):
                outputs = model(images)
            probabilities = torch.softmax(outputs.float(), dim=1)
            preds = probabilities.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    if not all_labels:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, NUM_CLASSES), dtype=np.float32),
        )

    return (
        np.concatenate(all_labels).astype(np.int64, copy=False),
        np.concatenate(all_preds).astype(np.int64, copy=False),
        np.concatenate(all_probabilities).astype(np.float32, copy=False),
    )


def _synchronize_if_cuda(device: torch.device = DEVICE) -> None:
    """Wait for queued CUDA work before and after a timed region."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def compute_inference_time(
    model: torch.nn.Module,
    device: torch.device = DEVICE,
    num_runs: int = 50,
    warmup: int = 10,
) -> dict[str, Any]:
    """Measure batch-one inference with a dummy tensor and no data-loading I/O."""
    num_runs = max(int(num_runs), 1)
    warmup = min(max(int(warmup), 0), num_runs - 1)
    measured_runs = num_runs - warmup
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    amp_enabled = MIXED_PRECISION and device.type == "cuda"
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                with autocast(device_type=device.type, enabled=amp_enabled):
                    model(dummy)

            elapsed_seconds = 0.0
            for _ in range(measured_runs):
                _synchronize_if_cuda(device)
                start = time.perf_counter()
                with autocast(device_type=device.type, enabled=amp_enabled):
                    model(dummy)
                _synchronize_if_cuda(device)
                elapsed_seconds += time.perf_counter() - start
    finally:
        if was_training:
            model.train()

    avg_ms = (elapsed_seconds / measured_runs) * 1000.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else None
    return {
        "batch_size": 1,
        "warmup_iterations": warmup,
        "measured_iterations": measured_runs,
        "scope": "model_only",
        "avg_ms": round(float(avg_ms), 2),
        "fps": round(float(fps), 1) if fps is not None else None,
    }


def plot_confusion_matrix(labels, preds, model_name: str, save_dir: Path) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    class_labels = _class_labels()

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
        print(f"  Training curves skipped: {history_path} not found")
        return

    try:
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        keys = ("train_loss", "val_loss", "train_acc", "val_acc")
        if not all(key in history for key in keys):
            raise ValueError("history.json is missing one or more curve fields")
        n_epochs = min(len(history[key]) for key in keys)
        if n_epochs == 0:
            raise ValueError("history.json contains no epochs")
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        print(f"  Training curves skipped: {exc}")
        return

    epochs = range(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"][:n_epochs], label="Train", color="#2563EB")
    axes[0].plot(epochs, history["val_loss"][:n_epochs], label="Val", color="#DC2626")
    axes[0].axvline(
        x=PHASE1_EPOCHS,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Phase 2",
    )
    axes[0].set_title("Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        [acc * 100 for acc in history["train_acc"][:n_epochs]],
        label="Train",
        color="#2563EB",
    )
    axes[1].plot(
        epochs,
        [acc * 100 for acc in history["val_acc"][:n_epochs]],
        label="Val",
        color="#DC2626",
    )
    axes[1].axvline(
        x=PHASE1_EPOCHS,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Phase 2",
    )
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


def _compute_roc_data(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Compute one-vs-rest curves and AUC values, tolerating absent classes."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected probability shape (N, {NUM_CLASSES}), got {probabilities.shape}"
        )

    per_class_auc: dict[str, float | None] = {}
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    supports: dict[int, int] = {}

    for class_index, class_name in enumerate(CLASS_NAMES):
        binary_labels = (labels == class_index).astype(np.int64)
        supports[class_index] = int(binary_labels.sum())
        if np.unique(binary_labels).size < 2:
            per_class_auc[class_name] = None
            continue

        per_class_auc[class_name] = float(
            roc_auc_score(binary_labels, probabilities[:, class_index])
        )
        fpr, tpr, _ = roc_curve(binary_labels, probabilities[:, class_index])
        curves[class_index] = (fpr, tpr)

    valid_indices = [index for index, name in enumerate(CLASS_NAMES) if per_class_auc[name] is not None]
    macro_auc: float | None = None
    weighted_auc: float | None = None
    if valid_indices:
        if len(valid_indices) == NUM_CLASSES:
            try:
                macro_auc = float(
                    roc_auc_score(
                        labels,
                        probabilities,
                        labels=list(range(NUM_CLASSES)),
                        multi_class="ovr",
                        average="macro",
                    )
                )
                weighted_auc = float(
                    roc_auc_score(
                        labels,
                        probabilities,
                        labels=list(range(NUM_CLASSES)),
                        multi_class="ovr",
                        average="weighted",
                    )
                )
            except ValueError:
                # Fall back to the same one-vs-rest columns if sklearn rejects
                # an edge case in the multiclass representation.
                macro_auc = None
                weighted_auc = None

        if macro_auc is None or weighted_auc is None:
            target_matrix = np.column_stack(
                [(labels == index).astype(np.int64) for index in valid_indices]
            )
            score_matrix = probabilities[:, valid_indices]
            try:
                macro_auc = float(
                    roc_auc_score(target_matrix, score_matrix, average="macro")
                )
                weighted_auc = float(
                    roc_auc_score(target_matrix, score_matrix, average="weighted")
                )
            except ValueError:
                valid_aucs = [
                    float(per_class_auc[CLASS_NAMES[index]]) for index in valid_indices
                ]
                weights = np.asarray([supports[index] for index in valid_indices])
                macro_auc = float(np.mean(valid_aucs))
                weighted_auc = float(np.average(valid_aucs, weights=weights))

    return {
        "per_class_auc": per_class_auc,
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "curves": curves,
    }


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    save_dir: Path,
) -> dict[str, Any]:
    """Save multiclass one-vs-rest ROC curves and return their metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    roc_data = _compute_roc_data(labels, probabilities)
    class_labels = _class_labels()

    fig, ax = plt.subplots(figsize=(9, 7))
    plotted = False
    for class_index, class_name in enumerate(CLASS_NAMES):
        curve = roc_data["curves"].get(class_index)
        auc_value = roc_data["per_class_auc"][class_name]
        if curve is None or auc_value is None:
            continue
        fpr, tpr = curve
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_labels[class_index]} (AUC={auc_value:.4f})",
        )
        plotted = True

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"One-vs-Rest ROC Curves - {model_name}", fontweight="bold")
    if plotted:
        ax.legend(loc="lower right", fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "AUC unavailable: test labels need both positive and negative samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = save_dir / "roc_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ROC curves: {out_path}")
    return roc_data


def _jsonify_report(report: dict[str, Any]) -> dict[str, Any]:
    """Convert sklearn report values to JSON-native numbers."""
    result: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, dict):
            result[key] = {
                metric: int(metric_value)
                if metric == "support"
                else float(metric_value)
                for metric, metric_value in value.items()
            }
        elif key == "support":
            result[key] = int(value)
        else:
            result[key] = float(value)
    return result


def _build_per_class_report(
    report: dict[str, Any],
    class_labels: list[str],
) -> dict[str, dict[str, Any]]:
    """Key the per-class report by config class names and retain display labels."""
    per_class: dict[str, dict[str, Any]] = {}
    for class_name, display_label in zip(CLASS_NAMES, class_labels):
        values = report.get(display_label, {})
        per_class[class_name] = {
            "label": display_label,
            "precision": float(values.get("precision", 0.0)),
            "recall": float(values.get("recall", 0.0)),
            "f1-score": float(values.get("f1-score", 0.0)),
            "support": int(values.get("support", 0)),
        }
    return per_class


def _build_per_class_metrics(
    report: dict[str, Any],
    class_labels: list[str],
    per_class_auc: dict[str, float | None],
) -> dict[str, dict[str, Any]]:
    """Build the stable per-class schema used by metrics.json."""
    per_class: dict[str, dict[str, Any]] = {}
    for class_name, display_label in zip(CLASS_NAMES, class_labels):
        values = report.get(display_label, {})
        per_class[class_name] = {
            "label": display_label,
            "precision": float(values.get("precision", 0.0)),
            "recall": float(values.get("recall", 0.0)),
            "f1": float(values.get("f1-score", 0.0)),
            "auc": per_class_auc.get(class_name),
            "support": int(values.get("support", 0)),
        }
    return per_class


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _device_description() -> str:
    """Return the exact CUDA device name or the active CPU device."""
    if DEVICE.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(DEVICE)
    return str(DEVICE)


def _write_optional_error(save_dir: Path, output_name: str, error: Exception) -> None:
    """Document an optional output failure without failing the evaluation."""
    error_path = save_dir / f"{output_name}_error.txt"
    try:
        error_path.write_text(
            f"{output_name} generation skipped: {error}\n",
            encoding="utf-8",
        )
        print(f"  {output_name} error: {error_path}")
    except OSError:
        print(f"  {output_name} generation skipped: {error}")


def evaluate_model(model_name: str, dataloaders: dict) -> dict[str, Any]:
    print(f"\n{'=' * 55}")
    print(f"  ĐÁNH GIÁ: {model_name.upper()}")
    print(f"{'=' * 55}")

    save_dir = Path(OUTPUT_DIR) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint(model_name)
    labels, preds, probabilities = predict(model, dataloaders["test"])
    inference = compute_inference_time(model, DEVICE)

    class_labels = _class_labels()
    if labels.size:
        acc = float(accuracy_score(labels, preds))
        precision_macro = float(
            precision_score(
                labels,
                preds,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        )
        precision_weighted = float(
            precision_score(
                labels,
                preds,
                labels=list(range(NUM_CLASSES)),
                average="weighted",
                zero_division=0,
            )
        )
        recall_macro = float(
            recall_score(
                labels,
                preds,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        )
        recall_weighted = float(
            recall_score(
                labels,
                preds,
                labels=list(range(NUM_CLASSES)),
                average="weighted",
                zero_division=0,
            )
        )
        f1_macro = float(
            f1_score(
                labels,
                preds,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        )
        f1_weighted = float(
            f1_score(
                labels,
                preds,
                labels=list(range(NUM_CLASSES)),
                average="weighted",
                zero_division=0,
            )
        )
        report = classification_report(
            labels,
            preds,
            labels=list(range(NUM_CLASSES)),
            target_names=class_labels,
            digits=4,
            zero_division=0,
        )
        report_dict = classification_report(
            labels,
            preds,
            labels=list(range(NUM_CLASSES)),
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
    else:
        acc = 0.0
        precision_macro = precision_weighted = 0.0
        recall_macro = recall_weighted = 0.0
        f1_macro = f1_weighted = 0.0
        report = "No test samples available.\n"
        report_dict = {
            label: {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 0,
            }
            for label in class_labels
        }

    print(f"\n  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Weighted F1   : {f1_weighted * 100:.2f}%")
    print(f"\n{report}")

    report_path = save_dir / "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy : {acc * 100:.2f}%\n")
        f.write(f"Weighted F1   : {f1_weighted * 100:.2f}%\n\n")
        f.write(report)
    print(f"  Report saved : {report_path}")

    try:
        plot_confusion_matrix(labels, preds, model_name, save_dir)
    except Exception as exc:
        _write_optional_error(save_dir, "confusion_matrix", exc)
    try:
        plot_training_curves(model_name, save_dir)
    except Exception as exc:
        _write_optional_error(save_dir, "training_curves", exc)

    empty_roc_data = {
        "per_class_auc": {name: None for name in CLASS_NAMES},
        "macro_auc": None,
        "weighted_auc": None,
    }
    try:
        roc_data = plot_roc_curves(labels, probabilities, model_name, save_dir)
    except Exception as exc:
        _write_optional_error(save_dir, "roc_curves", exc)
        roc_data = empty_roc_data

    try:
        gradcam_path = generate_gradcam_grid(
            model,
            dataloaders["test"],
            model_name,
            save_dir / "gradcam_grid.png",
            class_names=CLASS_NAMES,
            class_labels=class_labels,
        )
        print(f"  Grad-CAM: {gradcam_path}")
    except Exception as exc:
        _write_optional_error(save_dir, "gradcam", exc)

    per_class_report = _build_per_class_report(report_dict, class_labels)
    per_class = _build_per_class_metrics(
        report_dict,
        class_labels,
        roc_data["per_class_auc"],
    )
    checkpoint_epoch = _optional_int(checkpoint.get("epoch"))
    best_epoch = _optional_int(checkpoint.get("best_epoch"))
    if best_epoch is None:
        best_epoch = checkpoint_epoch
    checkpoint_val_acc = _optional_float(checkpoint.get("val_acc"))
    training_time_seconds = _optional_float(checkpoint.get("training_time_seconds"))
    training_time_minutes = (
        round(training_time_seconds / 60.0, 2)
        if training_time_seconds is not None
        else None
    )
    param_counts = get_param_counts(model)
    metrics = {
        "model_name": model_name,
        "model": model_name,
        "class_names": list(CLASS_NAMES),
        "class_labels": class_labels,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        # Preserve the field used by the existing CLI summary.
        "f1": f1_weighted,
        "per_class": per_class,
        "per_class_report": per_class_report,
        "classification_report": _jsonify_report(report_dict),
        "per_class_auc": roc_data["per_class_auc"],
        "auc_macro": roc_data["macro_auc"],
        "auc_weighted": roc_data["weighted_auc"],
        "macro_auc": roc_data["macro_auc"],
        "weighted_auc": roc_data["weighted_auc"],
        "auc": {
            "per_class": roc_data["per_class_auc"],
            "macro": roc_data["macro_auc"],
            "weighted": roc_data["weighted_auc"],
        },
        "total_params": param_counts["total"],
        "model_size_mb": round(float(get_model_size_mb(model)), 2),
        "inference_ms": inference["avg_ms"],
        "inference_fps": inference["fps"],
        "training_time_seconds": training_time_seconds,
        "training_time_minutes": training_time_minutes,
        "best_epoch": best_epoch,
        "device": _device_description(),
        "num_test_samples": int(labels.size),
        "inference": inference,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_val_acc": checkpoint_val_acc,
        "checkpoint": {
            "epoch": checkpoint_epoch,
            "val_acc": checkpoint_val_acc,
        },
    }

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"  Metrics saved : {metrics_path}")
    print(
        f"  Inference     : {inference['avg_ms']:.3f} ms/batch, "
        f"{inference['fps']:.2f} FPS"
        if inference["avg_ms"] is not None and inference["fps"] is not None
        else "  Inference     : unavailable"
    )

    return metrics


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
