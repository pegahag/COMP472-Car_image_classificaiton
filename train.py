"""
train.py
Main training script. Accepts a config YAML and runs a complete experiment.

Usage:
    python train.py --config configs/resnet50_car_brand_classification_transfer.yaml

    # Disable wandb (useful for quick debugging):
    python train.py --config configs/alexnet_five_car_models_scratch.yaml --no-wandb
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# ── Local imports ─────────────────────────────────────────────────────────────
from model_factory import build_model, unfreeze_backbone, get_param_groups
from data_factory import build_dataloaders
from metrics import (
    MetricTracker,
    plot_confusion_matrix,
    measure_inference_time,
    get_model_size,
    get_classification_report,
)
from visualizations import GradCAM, plot_gradcam_grid, extract_embeddings, plot_tsne
from logger import WandbLogger


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Optimizer / scheduler factory ────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    train_cfg  = cfg["training"]
    model_cfg  = cfg["model"]
    optimizer  = train_cfg["optimizer"].lower()
    lr         = train_cfg["learning_rate"]
    wd         = train_cfg["weight_decay"]

    params = model.parameters()   # default: all params

    if optimizer == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    elif optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'.")


def build_scheduler(optimizer: optim.Optimizer, cfg: dict, n_epochs: int):
    sched = cfg["training"].get("scheduler", "none").lower()
    if sched == "cosine":
        return CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif sched == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        return None


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, cfg, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"epoch_{epoch:03d}.pt"
    torch.save({
        "epoch":      epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics":    metrics,
        "config":     cfg,
    }, path)
    return str(path)


def load_best_checkpoint(model, output_dir: Path, device: torch.device):
    best = output_dir / "best_model.pt"
    if best.exists():
        try:
            ckpt = torch.load(best, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return ckpt.get("metrics", {})
    return {}


# ── Single epoch ──────────────────────────────────────────────────────────────

def run_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device:    torch.device,
    split:     str,
) -> dict:
    is_train = (split == "train")
    model.train() if is_train else model.eval()
    tracker = MetricTracker()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            tracker.update(loss.item(), preds, labels)

    return tracker.compute()


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, use_wandb: bool = True):
    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(cfg["env"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Experiment : {cfg['experiment_name']}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = Path(cfg["env"]["output_root"]) / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = WandbLogger(cfg, enabled=use_wandb)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders     = build_dataloaders(cfg)
    class_names = loaders.get("class_names")
    train_loader = loaders["train"]
    val_loader   = loaders["val"]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg["model"]).to(device)

    model_info = get_model_size(model)
    print(f"  Model size : {model_info['model_size_mb']} MB  |  "
          f"Trainable params: {model_info['trainable_params']:,}")
    logger.log_model_info(model_info)

    # ── Optimizer / scheduler / loss ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, cfg["training"]["epochs"])

    # ── Training config ───────────────────────────────────────────────────────
    train_cfg            = cfg["training"]
    n_epochs             = train_cfg["epochs"]
    patience             = train_cfg["early_stopping_patience"]
    unfreeze_after       = train_cfg.get("unfreeze_after_epoch", 0)
    finetune_lr          = train_cfg.get("finetune_lr", 1e-4)
    log_every            = cfg["logging"]["save_checkpoint_every"]
    log_gradcam          = cfg["logging"].get("log_gradcam", True)
    log_tsne_flag        = cfg["logging"].get("log_tsne", True)
    arch                 = cfg["model"]["architecture"]

    # ── Early stopping state ──────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    backbone_unfrozen = False

    # ── Grad-CAM setup ────────────────────────────────────────────────────────
    gradcam = None
    if log_gradcam:
        try:
            gradcam = GradCAM(model, arch)
        except Exception as e:
            print(f"[warn] Grad-CAM setup failed: {e}")

    # ── Training loop ─────────────────────────────────────────────────────────
    training_start = time.time()

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")

        # ── Backbone unfreezing ───────────────────────────────────────────────
        if (
            cfg["model"].get("pretrained") and
            cfg["model"].get("freeze_backbone") and
            unfreeze_after > 0 and
            epoch == unfreeze_after + 1 and
            not backbone_unfrozen
        ):
            print(f"  [!] Unfreezing backbone — switching to finetune lr={finetune_lr}")
            unfreeze_backbone(model)
            param_groups = get_param_groups(model, arch, finetune_lr, train_cfg["learning_rate"])
            optimizer    = optim.AdamW(param_groups, weight_decay=train_cfg["weight_decay"])
            scheduler    = build_scheduler(optimizer, cfg, n_epochs - epoch)
            backbone_unfrozen = True

        # ── Forward passes ────────────────────────────────────────────────────
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, "train")
        val_metrics   = run_epoch(model, val_loader,   criterion, None,      device, "val")

        if scheduler:
            scheduler.step()

        # ── Print summary ──────────────────────────────────────────────────────
        print(f"  Train — loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}  "
              f"F1: {train_metrics['macro_f1']:.4f}")
        print(f"  Val   — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}  "
              f"F1: {val_metrics['macro_f1']:.4f}")

        # ── Log scalars ───────────────────────────────────────────────────────
        logger.log_epoch("train", train_metrics, epoch)
        logger.log_epoch("val",   val_metrics,   epoch)

        # ── Log per-class metrics (every 5 epochs to keep wandb clean) ────────
        if epoch % 5 == 0 or epoch == n_epochs:
            logger.log_per_class_metrics(val_metrics, class_names, "val", epoch)

        # ── Confusion matrix ──────────────────────────────────────────────────
        if epoch % log_every == 0 or epoch == n_epochs:
            cm_fig = plot_confusion_matrix(
                val_metrics["y_true"],
                val_metrics["y_pred"],
                class_names,
                title=f"Val Confusion Matrix — Epoch {epoch}",
            )
            logger.log_confusion_matrix(cm_fig, "val", epoch)

        # ── Grad-CAM (log at epoch 1, midpoint, and final) ───────────────────
        if gradcam and epoch in {1, n_epochs // 2, n_epochs}:
            try:
                sample_images, sample_labels = next(iter(val_loader))
                n_samples = cfg["logging"].get("num_gradcam_samples", 8)
                sample_images = sample_images[:n_samples].to(device)
                sample_labels = sample_labels[:n_samples]

                heatmaps = gradcam.generate(sample_images)
                with torch.no_grad():
                    preds = model(sample_images).argmax(dim=1).cpu().tolist()

                cam_fig = plot_gradcam_grid(
                    sample_images.detach(), heatmaps,
                    sample_labels.tolist(), preds,
                    class_names,
                    title=f"Grad-CAM Epoch {epoch}",
                )
                logger.log_gradcam(cam_fig, epoch)
            except Exception as e:
                print(f"[warn] Grad-CAM logging failed at epoch {epoch}: {e}")

        # ── Checkpointing ─────────────────────────────────────────────────────
        if epoch % log_every == 0:
            ckpt_path = save_checkpoint(model, optimizer, epoch, val_metrics, cfg, output_dir)

        # ── Best model ────────────────────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            best_path = output_dir / "best_model.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "metrics":     val_metrics,
                "config":      cfg,
            }, best_path)
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

    total_training_time = time.time() - training_start
    completed_epochs = epoch
    print(f"\nTraining complete in {total_training_time/60:.1f} min.")

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\nRunning final evaluation on test set...")
    load_best_checkpoint(model, output_dir, device)
    test_loader = loaders.get("test", val_loader)
    test_metrics = run_epoch(model, test_loader, criterion, None, device, "test")

    print(f"\n  Test accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test macro F1 : {test_metrics['macro_f1']:.4f}")
    logger.log_epoch("test", test_metrics, completed_epochs)
    logger.log_per_class_metrics(test_metrics, class_names, "test", completed_epochs)

    # Final confusion matrix on test set
    cm_fig = plot_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"],
        class_names, title="Test Confusion Matrix (Best Model)"
    )
    logger.log_confusion_matrix(cm_fig, "test", completed_epochs)

    # ── t-SNE (run once at the end on best model) ──────────────────────────
    if log_tsne_flag:
        print("\nGenerating t-SNE embeddings...")
        try:
            embeddings, emb_labels = extract_embeddings(model, val_loader, device, arch)
            tsne_fig = plot_tsne(embeddings, emb_labels, class_names,
                                 title=f"t-SNE — {cfg['experiment_name']}")
            logger.log_tsne(tsne_fig, completed_epochs)
        except Exception as e:
            print(f"[warn] t-SNE failed: {e}")

    # ── Inference time ────────────────────────────────────────────────────────
    print("\nMeasuring inference time...")
    timing = measure_inference_time(model, test_loader, device)
    print(f"  Per-image inference: {timing['inference_per_image_ms']:.2f} ms")
    logger.log_inference_time(timing)

    # ── Classification report (console + saved to file) ───────────────────────
    report = get_classification_report(test_metrics["y_true"], test_metrics["y_pred"], class_names)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Experiment: {cfg['experiment_name']}\n\n")
        f.write(report)
    print(f"\nClassification report saved to {report_path}")

    # ── Save test metrics as JSON (read by run_all_experiments.py) ────────────
    import json
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
        }, f)

    # ── Upload best model artifact ────────────────────────────────────────────
    logger.save_model_artifact(
        str(output_dir / "best_model.pt"),
        name=cfg["experiment_name"],
        metadata={"test_accuracy": test_metrics["accuracy"], "test_f1": test_metrics["macro_f1"]},
    )

    logger.finish()
    return test_metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN for car brand classification.")
    parser.add_argument("--config",   required=True, help="Path to experiment YAML config.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, use_wandb=not args.no_wandb)
