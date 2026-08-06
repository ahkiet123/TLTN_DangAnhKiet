"""Grad-CAM utilities for representative test-set visualizations."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from durian_leaf_disease.config import DEVICE, NUM_CLASSES


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Return a final spatial feature layer suitable for each supported model."""
    model_name = model_name.lower().strip()
    if model_name in ("mobilenet_v2", "efficientnet_b0"):
        try:
            return model.features[-1]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(f"Cannot locate features[-1] for {model_name}") from exc
    if model_name == "resnet50":
        try:
            return model.layer4[-1]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("Cannot locate layer4[-1] for resnet50") from exc
    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


class GradCAM:
    """Compute Grad-CAM maps with forward and full-backward hooks."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        model_name: str,
    ):
        self.model = model
        self.target_layer = target_layer
        self.model_name = model_name
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._activation_shape_reported = False
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_inputs, grad_outputs) -> None:
        gradient = grad_outputs[0]
        if gradient is not None:
            self.gradients = gradient.detach()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return normalized CAMs and predicted classes for a batch.

        This method deliberately builds a gradient graph and must run in the
        normal gradient-enabled execution path.
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.ndim != 4:
            raise ValueError(f"Expected input shape (N, C, H, W), got {input_tensor.shape}")

        self.activations = None
        self.gradients = None
        was_training = self.model.training
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)

        try:
            logits = self.model(input_tensor)
            if not torch.is_tensor(logits) or logits.ndim != 2:
                raise ValueError("Grad-CAM requires model output shaped (N, classes)")

            predicted = logits.argmax(dim=1)
            if target_class is None:
                target_indices = predicted
            elif isinstance(target_class, torch.Tensor):
                target_indices = target_class.to(logits.device, dtype=torch.long).reshape(-1)
            else:
                target_indices = torch.full(
                    (logits.shape[0],),
                    int(target_class),
                    device=logits.device,
                    dtype=torch.long,
                )
            if target_indices.numel() == 1 and logits.shape[0] > 1:
                target_indices = target_indices.expand(logits.shape[0])
            if target_indices.numel() != logits.shape[0]:
                raise ValueError("target_class must contain one class per input")

            score = logits.gather(1, target_indices.view(-1, 1)).sum()
            torch.autograd.backward(score)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
            if self.activations.ndim != 4 or self.gradients.ndim != 4:
                raise ValueError("The selected target layer must produce spatial features")
            if not self._activation_shape_reported:
                print(
                    "  Grad-CAM activation map "
                    f"({self.model_name}): {tuple(self.activations.shape)}"
                )
                self._activation_shape_reported = True

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cams = torch.relu((weights * self.activations).sum(dim=1))
            cams = cams.flatten(1)
            mins = cams.min(dim=1, keepdim=True).values
            maxs = cams.max(dim=1, keepdim=True).values
            cams = (cams - mins) / (maxs - mins).clamp_min(1e-8)
            cams = cams.reshape(-1, self.activations.shape[2], self.activations.shape[3])
            return cams.detach().cpu().numpy(), predicted.detach().cpu().numpy()
        finally:
            if was_training:
                self.model.train()


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalized tensor to an RGB array in [0, 1]."""
    image = image_tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return DEVICE


def _representative_samples(
    loader,
    samples_per_class: int,
) -> dict[int, list[torch.Tensor]]:
    """Select deterministic test examples, preserving two samples per class."""
    samples_per_class = max(int(samples_per_class), 1)
    selected: dict[int, list[torch.Tensor]] = {
        class_index: [] for class_index in range(NUM_CLASSES)
    }

    for images, labels in loader:
        for image, label in zip(images, labels):
            class_index = int(label)
            if class_index in selected and len(selected[class_index]) < samples_per_class:
                selected[class_index].append(image.detach().cpu())
        if all(len(samples) >= samples_per_class for samples in selected.values()):
            break
    return selected


def _resize_cam(cam: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Resize a normalized CAM to the source image dimensions with PIL."""
    height, width = image_shape
    cam_image = Image.fromarray(np.uint8(np.clip(cam, 0.0, 1.0) * 255), mode="L")
    resampling = getattr(Image, "Resampling", Image)
    cam_image = cam_image.resize((width, height), resampling.BILINEAR)
    return np.asarray(cam_image, dtype=np.float32) / 255.0


def generate_gradcam_grid(
    model: torch.nn.Module,
    test_loader,
    model_name: str,
    save_path: str | Path,
    class_names: Iterable[str] | None = None,
    class_labels: Iterable[str] | None = None,
    samples_per_class: int = 2,
) -> Path:
    """Save a 6 x 2 grid of Grad-CAM overlays from deterministic test samples."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(class_names) if class_names is not None else [str(i) for i in range(NUM_CLASSES)]
    labels = list(class_labels) if class_labels is not None else names
    samples_per_class = max(int(samples_per_class), 1)
    samples_by_class = _representative_samples(test_loader, samples_per_class)
    target_layer = get_target_layer(model, model_name)
    cam_extractor = GradCAM(model, target_layer, model_name)
    device = _model_device(model)
    rows = len(names)
    fig, axes = plt.subplots(
        rows,
        samples_per_class,
        figsize=(5 * samples_per_class, max(3.5 * rows, 4)),
        squeeze=False,
    )

    try:
        for class_index in range(rows):
            actual_label = (
                labels[class_index] if class_index < len(labels) else str(class_index)
            )
            class_samples = samples_by_class.get(class_index, [])
            for column in range(samples_per_class):
                axis = axes[class_index, column]
                if column >= len(class_samples):
                    axis.text(0.5, 0.5, "No test sample", ha="center", va="center")
                    axis.set_title(f"Actual: {actual_label}")
                    axis.axis("off")
                    continue

                image_tensor = class_samples[column]
                input_tensor = image_tensor.unsqueeze(0).to(device)
                cams, predicted = cam_extractor(input_tensor)
                image = denormalize_image(image_tensor)
                cam = _resize_cam(cams[0], image.shape[:2])
                predicted_index = int(predicted[0])
                predicted_label = (
                    labels[predicted_index]
                    if predicted_index < len(labels)
                    else str(predicted_index)
                )

                axis.imshow(image)
                axis.imshow(cam, cmap="jet", alpha=0.4, vmin=0.0, vmax=1.0)
                axis.set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}")
                axis.axis("off")

        fig.suptitle(f"Grad-CAM - {model_name}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    finally:
        cam_extractor.remove_hooks()
        plt.close(fig)
    return save_path
