"""
models/model_factory.py
Builds any supported CNN architecture with the correct output head.
Handles both scratch and transfer learning setups from config.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ── Supported architectures ───────────────────────────────────────────────────
SUPPORTED = ["alexnet", "resnet50", "mobilenet_v2"]


def build_model(model_cfg: dict) -> nn.Module:
    """
    Build and return a model ready for training.

    Args:
        model_cfg: the 'model' section of the experiment config dict.
                   Keys: architecture, pretrained, freeze_backbone, num_classes

    Returns:
        nn.Module with the final layer replaced to match num_classes.
    """
    arch        = model_cfg["architecture"]
    pretrained  = model_cfg["pretrained"]
    freeze      = model_cfg.get("freeze_backbone", False)
    num_classes = model_cfg["num_classes"]

    if arch not in SUPPORTED:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {SUPPORTED}.")

    # ── Load base model ───────────────────────────────────────────────────────
    model = _load_base_model(arch, pretrained)

    # ── Freeze backbone if requested ──────────────────────────────────────────
    if pretrained and freeze:
        _freeze_backbone(model)

    # ── Replace classification head ───────────────────────────────────────────
    _replace_head(model, arch, num_classes)

    return model


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_base_model(arch: str, pretrained: bool) -> nn.Module:
    if arch == "alexnet":
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        return models.alexnet(weights=weights)

    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet50(weights=weights)

    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        return models.mobilenet_v2(weights=weights)


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters — the new head will be unfrozen after replacement."""
    for param in model.parameters():
        param.requires_grad = False


def _replace_head(model: nn.Module, arch: str, num_classes: int) -> None:
    """Swap the final classification layer and ensure it is trainable."""
    if arch == "alexnet":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif arch == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "mobilenet_v2":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    # The newly created layer always has requires_grad=True by default,
    # so no explicit unfreeze needed here.


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all parameters for full fine-tuning.
    Call this after N warm-up epochs as specified in config unfreeze_after_epoch.
    """
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(model: nn.Module, arch: str, finetune_lr: float, head_lr: float) -> list:
    """
    Return separate param groups so the head trains at head_lr
    and the backbone trains at the lower finetune_lr after unfreezing.
    """
    if arch == "alexnet":
        head_params = list(model.classifier[6].parameters())
    elif arch == "resnet50":
        head_params = list(model.fc.parameters())
    elif arch == "mobilenet_v2":
        head_params = list(model.classifier[1].parameters())

    head_ids     = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    return [
        {"params": backbone_params, "lr": finetune_lr},
        {"params": head_params,     "lr": head_lr},
    ]
