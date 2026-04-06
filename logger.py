"""
utils/logger.py
Thin wrapper around wandb that handles initialisation, metric logging,
image logging, and graceful teardown.
All wandb calls are centralised here so the rest of the codebase stays clean.
"""

import io
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[logger] wandb not installed — logging disabled. Run: pip install wandb")


class WandbLogger:
    """
    Wraps wandb for experiment tracking.
    If wandb is unavailable or disabled, all methods become no-ops.
    """

    def __init__(self, cfg: dict, enabled: bool = True):
        self.enabled = enabled and WANDB_AVAILABLE
        self._run    = None

        if not self.enabled:
            return

        log_cfg = cfg["logging"]

        self._run = wandb.init(
            project = log_cfg["wandb_project"],
            entity  = log_cfg.get("wandb_entity") or None,
            name    = cfg["experiment_name"],
            config  = cfg,       # entire config dict stored on the run
            reinit  = True,
        )

    # ── Metric logging ────────────────────────────────────────────────────────

    def log_epoch(self, split: str, metrics: dict, epoch: int):
        """Log scalar metrics for a train or val epoch."""
        if not self.enabled:
            return

        payload = {f"{split}/{k}": v
                   for k, v in metrics.items()
                   if isinstance(v, (int, float, np.floating))}
        payload["epoch"] = epoch
        wandb.log(payload, step=epoch)

    def log_model_info(self, model_info: dict):
        """Log model size and parameter counts (logged once)."""
        if not self.enabled:
            return
        wandb.log(model_info)

    def log_inference_time(self, timing: dict):
        """Log inference timing metrics."""
        if not self.enabled:
            return
        wandb.log(timing)

    # ── Figure / image logging ────────────────────────────────────────────────

    def log_figure(self, key: str, fig: plt.Figure, epoch: int | None = None):
        """Log a matplotlib Figure as a wandb Image."""
        if not self.enabled:
            return
        # Convert to PIL first to avoid wandb's Windows temp-file path bug.
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        pil_img = PILImage.open(buf).copy()
        buf.close()
        plt.close(fig)
        payload = {key: wandb.Image(pil_img)}
        if epoch is not None:
            payload["epoch"] = epoch
            wandb.log(payload, step=epoch)
        else:
            wandb.log(payload)

    def log_confusion_matrix(self, fig: plt.Figure, split: str, epoch: int):
        self.log_figure(f"{split}/confusion_matrix", fig, epoch)

    def log_gradcam(self, fig: plt.Figure, epoch: int):
        self.log_figure("visualizations/grad_cam", fig, epoch)

    def log_tsne(self, fig: plt.Figure, epoch: int):
        self.log_figure("visualizations/tsne", fig, epoch)

    def log_per_class_metrics(self, metrics: dict, class_names: list | None, split: str, epoch: int):
        """
        Log per-class F1, precision, recall as a wandb Table for easy inspection.
        """
        if not self.enabled:
            return

        columns = ["class", "precision", "recall", "f1"]
        data    = []
        n       = len(metrics["per_class_f1"])

        for i in range(n):
            name = class_names[i] if class_names else str(i)
            data.append([
                name,
                round(metrics["per_class_precision"][i], 4),
                round(metrics["per_class_recall"][i],    4),
                round(metrics["per_class_f1"][i],        4),
            ])

        table = wandb.Table(columns=columns, data=data)
        wandb.log({f"{split}/per_class_metrics": table, "epoch": epoch}, step=epoch)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_model_artifact(self, path: str, name: str, metadata: dict | None = None):
        """Upload a model checkpoint as a wandb Artifact."""
        if not self.enabled:
            return
        artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    # ── Teardown ──────────────────────────────────────────────────────────────

    def finish(self):
        if self.enabled and self._run is not None:
            wandb.finish()
