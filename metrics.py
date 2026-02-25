"""
utils/metrics.py
Computes and packages all evaluation metrics:
  - Accuracy, per-class precision, recall, F1
  - Confusion matrix figure
  - Training/inference time tracking
  - Model size reporting
"""

import time
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for both Colab and local

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ── Epoch metrics ─────────────────────────────────────────────────────────────

class MetricTracker:
    """Accumulates loss and predictions across an epoch then computes everything."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._loss_sum   = 0.0
        self._n_samples  = 0
        self._all_preds  = []
        self._all_labels = []
        self._start_time = time.time()

    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor):
        batch_size        = labels.size(0)
        self._loss_sum   += loss * batch_size
        self._n_samples  += batch_size
        self._all_preds.extend(preds.cpu().numpy())
        self._all_labels.extend(labels.cpu().numpy())

    def compute(self) -> dict:
        y_true = np.array(self._all_labels)
        y_pred = np.array(self._all_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        return {
            "loss":              self._loss_sum / max(self._n_samples, 1),
            "accuracy":          accuracy_score(y_true, y_pred),
            "macro_precision":   precision,
            "macro_recall":      recall,
            "macro_f1":          f1,
            "per_class_precision": per_class_precision.tolist(),
            "per_class_recall":    per_class_recall.tolist(),
            "per_class_f1":        per_class_f1.tolist(),
            "epoch_time_sec":    time.time() - self._start_time,
            "y_true":            y_true,
            "y_pred":            y_pred,
        }


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Returns a matplotlib Figure of the confusion matrix.
    For large class counts (>30) class names are omitted for readability.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    n_classes = cm.shape[0]
    fig_size  = max(10, n_classes // 4)
    fig, ax   = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    if class_names and n_classes <= 30:
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ── Inference time ────────────────────────────────────────────────────────────

def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 10,
) -> dict:
    """
    Measures average per-image inference time over n_batches.
    Returns dict with total_ms and per_image_ms.
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            start  = time.perf_counter()
            _      = model(images)
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = (time.perf_counter() - start) * 1000   # ms
            times.append((elapsed, images.size(0)))

    total_ms  = sum(t for t, _ in times)
    total_img = sum(n for _, n in times)
    return {
        "inference_total_ms":     total_ms,
        "inference_per_image_ms": total_ms / max(total_img, 1),
    }


# ── Model size ────────────────────────────────────────────────────────────────

def get_model_size(model: nn.Module) -> dict:
    """Returns parameter count and estimated size in MB."""
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb          = total_params * 4 / (1024 ** 2)   # float32 = 4 bytes

    return {
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "model_size_mb":    round(size_mb, 2),
    }


# ── Classification report (text) ──────────────────────────────────────────────

def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list | None = None,
) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
