"""
generate_ablation_configs.py
Generates ablation study configs to identify the best hyperparameters
before running the full 18-experiment comparison.

Five ablation groups (one variable changed at a time):
  A — Learning Rate       (5 configs)
  B — Batch Size          (4 configs)
  C — Number of Classes   (3 configs, via dataset choice)
  D — Images per Class    (5 configs, via max_samples_per_class)
  E — Model Architecture  (3 configs)

Total: 20 configs in configs/ablation/

Run once from the project root: python generate_ablation_configs.py
"""

import os
import yaml

# ── Base (control) configuration ──────────────────────────────────────────────
# All ablation groups hold these values fixed except the one being varied.
BASE = {
    "architecture":        "resnet50",
    "dataset":             "car_brand_classification",
    "num_classes":         33,
    "batch_size":          32,
    "learning_rate":       0.001,
    "finetune_lr":         0.0001,
    "pretrained":          True,
    "freeze_backbone":     True,
    "unfreeze_after_epoch": 5,
    "epochs":              15,
    "early_stopping_patience": 5,
}

SHARED_ENV = {
    "local_data_root":  ".",
    "colab_data_root":  "/content/drive/MyDrive/COMP472-Car_image_classification",
    "output_root":      "./outputs",
    "seed":             42,
}

SHARED_AUGMENTATION = {
    "horizontal_flip":    True,
    "rotation_degrees":   15,
    "brightness":         0.2,
    "contrast":           0.2,
    "random_crop":        True,
    "random_crop_scale":  [0.8, 1.0],
}

SHARED_LOGGING = {
    "wandb_project":        "comp472_ablation",
    "wandb_entity":         None,
    "save_checkpoint_every": 5,
    "log_gradcam":          False,   # disabled for speed during ablation
    "log_tsne":             False,
    "num_gradcam_samples":  0,
}

# ── Helper ─────────────────────────────────────────────────────────────────────

def make_config(
    exp_name: str,
    architecture: str,
    dataset: str,
    num_classes: int,
    batch_size: int,
    learning_rate: float,
    finetune_lr: float,
    max_samples_per_class=None,
) -> dict:
    """Assemble a full config dict from the provided overrides and shared base."""
    cfg = {
        "experiment_name": exp_name,
        "env": SHARED_ENV,
        "model": {
            "architecture":        architecture,
            "pretrained":          BASE["pretrained"],
            "freeze_backbone":     BASE["freeze_backbone"],
            "num_classes":         num_classes,
        },
        "dataset": {
            "name":       dataset,
            "image_size": 224,
            "batch_size": batch_size,
            "num_workers": 4,
        },
        "augmentation": SHARED_AUGMENTATION,
        "training": {
            "epochs":                    BASE["epochs"],
            "optimizer":                 "adamw",
            "learning_rate":             learning_rate,
            "weight_decay":              0.01,
            "scheduler":                 "cosine",
            "early_stopping_patience":   BASE["early_stopping_patience"],
            "unfreeze_after_epoch":      BASE["unfreeze_after_epoch"],
            "finetune_lr":               finetune_lr,
        },
        "logging": SHARED_LOGGING,
    }
    if max_samples_per_class is not None:
        cfg["dataset"]["max_samples_per_class"] = max_samples_per_class
    return cfg


def write_config(cfg: dict, path: str):
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Created: {path}")


# ── Output directory ───────────────────────────────────────────────────────────
os.makedirs("configs/ablation", exist_ok=True)
count = 0

# ── Group A: Learning Rate ─────────────────────────────────────────────────────
# Varies: learning_rate (finetune_lr is always 1/10th to keep the ratio fixed)
# Fixed:  arch=resnet50, dataset=car_brand_classification, batch=32
LR_VALUES = {
    "1e-4": (0.0001,  0.00001),
    "5e-4": (0.0005,  0.00005),
    "1e-3": (0.001,   0.0001),    # baseline
    "5e-3": (0.005,   0.0005),
    "1e-2": (0.01,    0.001),
}
for lr_label, (lr, ft_lr) in LR_VALUES.items():
    exp_name = f"ablation_lr_{lr_label}"
    cfg = make_config(
        exp_name       = exp_name,
        architecture   = BASE["architecture"],
        dataset        = BASE["dataset"],
        num_classes    = BASE["num_classes"],
        batch_size     = BASE["batch_size"],
        learning_rate  = lr,
        finetune_lr    = ft_lr,
    )
    write_config(cfg, f"configs/ablation/{exp_name}.yaml")
    count += 1

# ── Group B: Batch Size ────────────────────────────────────────────────────────
# Varies: batch_size
# Fixed:  arch=resnet50, dataset=car_brand_classification, lr=0.001
BATCH_SIZES = [8, 16, 32, 64]
for bs in BATCH_SIZES:
    exp_name = f"ablation_bs_{bs}"
    cfg = make_config(
        exp_name       = exp_name,
        architecture   = BASE["architecture"],
        dataset        = BASE["dataset"],
        num_classes    = BASE["num_classes"],
        batch_size     = bs,
        learning_rate  = BASE["learning_rate"],
        finetune_lr    = BASE["finetune_lr"],
    )
    write_config(cfg, f"configs/ablation/{exp_name}.yaml")
    count += 1

# ── Group C: Number of Classes ─────────────────────────────────────────────────
# Varies: dataset (which determines num_classes and task complexity)
# Fixed:  arch=resnet50, lr=0.001, batch=32 (or 16 for five_car_models)
# Note:   stanford_cars manifests use brand-level labels → 49 classes
CLASS_DATASETS = {
    "5":  ("five_car_models",          5,  16),
    "33": ("car_brand_classification", 33, 32),  # baseline
    "49": ("stanford_cars",            49, 32),
}
for nc_label, (ds_name, nc, bs) in CLASS_DATASETS.items():
    exp_name = f"ablation_nclasses_{nc_label}"
    cfg = make_config(
        exp_name       = exp_name,
        architecture   = BASE["architecture"],
        dataset        = ds_name,
        num_classes    = nc,
        batch_size     = bs,
        learning_rate  = BASE["learning_rate"],
        finetune_lr    = BASE["finetune_lr"],
    )
    write_config(cfg, f"configs/ablation/{exp_name}.yaml")
    count += 1

# ── Group D: Images per Class ──────────────────────────────────────────────────
# Varies: max_samples_per_class (null = no cap, uses full ~399 imgs/class)
# Fixed:  arch=resnet50, dataset=car_brand_classification, lr=0.001, batch=32
# car_brand_classification is balanced at ~499 imgs/class, making the cap clean.
SAMPLES_PER_CLASS = [25, 50, 100, 200, None]
for spc in SAMPLES_PER_CLASS:
    label    = str(spc) if spc is not None else "all"
    exp_name = f"ablation_maxsamples_{label}"
    cfg = make_config(
        exp_name              = exp_name,
        architecture          = BASE["architecture"],
        dataset               = BASE["dataset"],
        num_classes           = BASE["num_classes"],
        batch_size            = BASE["batch_size"],
        learning_rate         = BASE["learning_rate"],
        finetune_lr           = BASE["finetune_lr"],
        max_samples_per_class = spc,
    )
    write_config(cfg, f"configs/ablation/{exp_name}.yaml")
    count += 1

# ── Group E: Model Architecture ───────────────────────────────────────────────
# Varies: architecture
# Fixed:  dataset=car_brand_classification, lr=0.001, batch=32
ARCHITECTURES = ["alexnet", "resnet50", "mobilenet_v2"]
for arch in ARCHITECTURES:
    exp_name = f"ablation_arch_{arch}"
    cfg = make_config(
        exp_name       = exp_name,
        architecture   = arch,
        dataset        = BASE["dataset"],
        num_classes    = BASE["num_classes"],
        batch_size     = BASE["batch_size"],
        learning_rate  = BASE["learning_rate"],
        finetune_lr    = BASE["finetune_lr"],
    )
    write_config(cfg, f"configs/ablation/{exp_name}.yaml")
    count += 1

print(f"\nDone — {count} ablation configs generated in configs/ablation/")
print("\nGroups:")
print("  A  Learning Rate     : ablation_lr_*       (5 configs)")
print("  B  Batch Size        : ablation_bs_*        (4 configs)")
print("  C  Number of Classes : ablation_nclasses_*  (3 configs)")
print("  D  Images per Class  : ablation_maxsamples_*(5 configs)")
print("  E  Architecture      : ablation_arch_*      (3 configs)")
