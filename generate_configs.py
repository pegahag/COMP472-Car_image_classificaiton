"""
generate_configs.py
Generates all 18 experiment config YAML files (3 architectures × 3 datasets × 2 strategies).
Run once from the project root: python generate_configs.py
"""

import os
import yaml

# ── Experiment matrix ──────────────────────────────────────────────────────────
ARCHITECTURES = ["alexnet", "resnet50", "mobilenet_v2"]

DATASETS = {
    "stanford_cars": {
        "num_classes": 196,
        "image_size": 224,
        "batch_size": 32,
    },
    "car_brand_classification": {
        "num_classes": 33,
        "image_size": 224,
        "batch_size": 32,
    },
    "five_car_models": {
        "num_classes": 5,
        "image_size": 224,
        "batch_size": 16,
    },
}

STRATEGIES = {
    "scratch": {
        "pretrained": False,
        "freeze_backbone": False,
        "unfreeze_after_epoch": 0,
        "learning_rate": 0.001,
    },
    "transfer": {
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_after_epoch": 5,
        "learning_rate": 0.001,
    },
}

# AlexNet has no pretrained transfer variant that makes sense to fine-tune
# but we still include it for completeness / comparison
SHARED = {
    "augmentation": {
        "horizontal_flip": True,
        "rotation_degrees": 15,
        "brightness": 0.2,
        "contrast": 0.2,
        "random_crop": True,
        "random_crop_scale": [0.8, 1.0],
    },
    "training_shared": {
        "epochs": 50,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "early_stopping_patience": 7,
        "finetune_lr": 0.0001,
    },
    "env": {
        "local_data_root": ".",
        "colab_data_root": "/content/drive/MyDrive/COMP472-Car_image_classification",
        "output_root": "./outputs",
        "seed": 42,
    },
    "logging": {
        "wandb_project": "comp472_car_detection_final",
        "wandb_entity": None,
        "save_checkpoint_every": 50,
        "log_gradcam": True,
        "log_tsne": True,
        "num_gradcam_samples": 8,
    },
}

# ── Generator ──────────────────────────────────────────────────────────────────
os.makedirs("configs", exist_ok=True)

for arch in ARCHITECTURES:
    for ds_name, ds_cfg in DATASETS.items():
        for strategy_name, strategy_cfg in STRATEGIES.items():

            exp_name = f"{arch}_{ds_name}_{strategy_name}"

            config = {
                "experiment_name": exp_name,
                "env": SHARED["env"],
                "model": {
                    "architecture": arch,
                    "pretrained": strategy_cfg["pretrained"],
                    "freeze_backbone": strategy_cfg["freeze_backbone"],
                    "num_classes": ds_cfg["num_classes"],
                },
                "dataset": {
                    "name": ds_name,
                    "image_size": ds_cfg["image_size"],
                    "batch_size": ds_cfg["batch_size"],
                    "num_workers": 4,
                },
                "augmentation": SHARED["augmentation"],
                "training": {
                    "epochs": SHARED["training_shared"]["epochs"],
                    "optimizer": SHARED["training_shared"]["optimizer"],
                    "learning_rate": strategy_cfg["learning_rate"],
                    "weight_decay": SHARED["training_shared"]["weight_decay"],
                    "scheduler": SHARED["training_shared"]["scheduler"],
                    "early_stopping_patience": SHARED["training_shared"]["early_stopping_patience"],
                    "unfreeze_after_epoch": strategy_cfg["unfreeze_after_epoch"],
                    "finetune_lr": SHARED["training_shared"]["finetune_lr"],
                },
                "logging": SHARED["logging"],
            }

            filename = f"configs/{exp_name}.yaml"
            with open(filename, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print(f"  Created: {filename}")

print(f"\nDone — {len(ARCHITECTURES) * len(DATASETS) * len(STRATEGIES)} configs generated.")
