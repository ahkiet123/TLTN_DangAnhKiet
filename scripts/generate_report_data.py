"""
generate_report_data.py - Tong hop metrics.json tu 3 model thanh bang so sanh.

Chay:
    python scripts/generate_report_data.py

Output:
    outputs/final_comparison.json
    outputs/model_comparison_chart.png
"""
from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from durian_leaf_disease.config import MODELS_TO_TRAIN, OUTPUT_DIR


DISPLAY_NAMES = {
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
}


def _pct(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, digits)


def load_metrics(model_name: str) -> dict:
    path = Path(OUTPUT_DIR) / model_name / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay metrics: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_model(model_name: str, metrics: dict) -> dict:
    inference = metrics.get("inference") or {}
    auc = metrics.get("auc") or {}
    checkpoint = metrics.get("checkpoint") or {}

    accuracy = metrics.get("accuracy")
    f1_macro = metrics.get("f1_macro")
    f1_weighted = metrics.get("f1_weighted", metrics.get("f1"))
    auc_macro = metrics.get("auc_macro", metrics.get("macro_auc", auc.get("macro")))
    auc_weighted = metrics.get(
        "auc_weighted", metrics.get("weighted_auc", auc.get("weighted"))
    )

    inference_ms = metrics.get("inference_ms", inference.get("avg_ms"))
    inference_fps = metrics.get("inference_fps", inference.get("fps"))
    best_epoch = metrics.get("best_epoch", metrics.get("checkpoint_epoch", checkpoint.get("epoch")))
    val_acc = metrics.get("checkpoint_val_acc", checkpoint.get("val_acc"))
    train_min = metrics.get("training_time_minutes")

    return {
        "model_name": model_name,
        "display_name": DISPLAY_NAMES.get(model_name, model_name),
        "accuracy": accuracy,
        "accuracy_pct": _pct(accuracy),
        "f1_macro": f1_macro,
        "f1_macro_pct": _pct(f1_macro),
        "f1_weighted": f1_weighted,
        "f1_weighted_pct": _pct(f1_weighted),
        "auc_macro": auc_macro,
        "auc_macro_pct": _pct(auc_macro),
        "auc_weighted": auc_weighted,
        "auc_weighted_pct": _pct(auc_weighted),
        "precision_macro": metrics.get("precision_macro"),
        "precision_macro_pct": _pct(metrics.get("precision_macro")),
        "recall_macro": metrics.get("recall_macro"),
        "recall_macro_pct": _pct(metrics.get("recall_macro")),
        "total_params": metrics.get("total_params"),
        "model_size_mb": metrics.get("model_size_mb"),
        "inference_ms": inference_ms,
        "inference_fps": inference_fps,
        "training_time_minutes": train_min,
        "best_epoch": best_epoch,
        "best_val_acc": val_acc,
        "best_val_acc_pct": _pct(val_acc),
        "device": metrics.get("device"),
        "per_class": metrics.get("per_class") or metrics.get("per_class_report") or {},
        "per_class_auc": metrics.get("per_class_auc") or auc.get("per_class") or {},
    }


def rank_models(rows: list[dict]) -> dict:
    def best_by(key: str, reverse: bool = True) -> str | None:
        valid = [row for row in rows if row.get(key) is not None]
        if not valid:
            return None
        return max(valid, key=lambda row: row[key] if reverse else -row[key])["model_name"]

    return {
        "best_accuracy": best_by("accuracy"),
        "best_f1_weighted": best_by("f1_weighted"),
        "best_f1_macro": best_by("f1_macro"),
        "best_auc_macro": best_by("auc_macro"),
        "fastest_inference": best_by("inference_ms", reverse=False),
        "smallest_model": best_by("model_size_mb", reverse=False),
        "fastest_training": best_by("training_time_minutes", reverse=False),
    }


def plot_comparison_chart(rows: list[dict], save_path: Path) -> None:
    labels = [row["display_name"] for row in rows]
    metrics = [
        ("Accuracy (%)", [row["accuracy_pct"] for row in rows], "#2563EB"),
        ("Weighted F1 (%)", [row["f1_weighted_pct"] for row in rows], "#16A34A"),
        ("Macro AUC (%)", [row["auc_macro_pct"] for row in rows], "#DC2626"),
    ]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))

    for index, (title, values, color) in enumerate(metrics):
        offset = (index - 1) * width
        bars = ax.bar(x + offset, values, width, label=title, color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            if value is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(80, 102)
    ax.set_title("So sánh Accuracy / Weighted F1 / Macro AUC trên tập Test", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_chart(rows: list[dict], save_path: Path) -> None:
    labels = [row["display_name"] for row in rows]
    sizes = [row["model_size_mb"] for row in rows]
    fps = [row["inference_fps"] for row in rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    bars1 = ax1.bar(x - width / 2, sizes, width, label="Model size (MB)", color="#7C3AED", alpha=0.9)
    ax1.set_ylabel("Model size (MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, fps, width, label="FPS", color="#F59E0B", alpha=0.9)
    ax2.set_ylabel("FPS")

    for bar, value in zip(bars1, sizes):
        if value is None:
            continue
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, value in zip(bars2, fps):
        if value is None:
            continue
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("So sánh kích thước mô hình và tốc độ suy luận", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for model_name in MODELS_TO_TRAIN:
        try:
            metrics = load_metrics(model_name)
            rows.append(summarize_model(model_name, metrics))
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if not rows:
        raise SystemExit("Không có metrics.json nào để tổng hợp.")

    ranking = rank_models(rows)
    comparison = {
        "models": rows,
        "ranking": ranking,
        "missing": missing,
        "notes": {
            "source": "metrics.json of each model under outputs/",
            "primary_metric": "f1_weighted on test set",
            "target_f1_pct": 85.0,
        },
    }

    json_path = output_dir / "final_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    chart_path = output_dir / "model_comparison_chart.png"
    plot_comparison_chart(rows, chart_path)

    efficiency_path = output_dir / "model_efficiency_chart.png"
    plot_efficiency_chart(rows, efficiency_path)

    print("=" * 60)
    print("TỔNG HỢP SO SÁNH 3 MODEL")
    print("=" * 60)
    header = (
        f"{'Model':<18} {'Acc%':>7} {'F1w%':>7} {'AUC%':>7} "
        f"{'ms':>7} {'FPS':>6} {'MB':>7} {'TrainMin':>9} {'Epoch':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda item: -(item["f1_weighted"] or 0)):
        print(
            f"{row['display_name']:<18} "
            f"{row['accuracy_pct']:>7.2f} "
            f"{row['f1_weighted_pct']:>7.2f} "
            f"{row['auc_macro_pct']:>7.2f} "
            f"{(row['inference_ms'] or 0):>7.2f} "
            f"{(row['inference_fps'] or 0):>6.1f} "
            f"{(row['model_size_mb'] or 0):>7.2f} "
            f"{(row['training_time_minutes'] or 0):>9.2f} "
            f"{(row['best_epoch'] or 0):>6}"
        )

    print("\nRanking:")
    for key, value in ranking.items():
        print(f"  {key}: {DISPLAY_NAMES.get(value, value)}")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {chart_path}")
    print(f"Saved: {efficiency_path}")
    if missing:
        print("\nMissing:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
