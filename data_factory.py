"""
datasets/data_factory.py
Builds DataLoaders for all three datasets using the manifest CSVs
produced by the preprocessing pipeline.

Manifest format (same across all three datasets):
    path      — relative path from the repo root to the processed image
    brand     — human-readable class name  (e.g. "audi")
    brand_id  — original integer ID (NOT necessarily 0-indexed — remapped here)

Expected folder structure per dataset:
    processed_data/
    └── <dataset_name>/
        ├── manifest_train.csv
        ├── manifest_val.csv
        ├── manifest_test.csv
        ├── class_map.csv
        ├── brand_counts.csv
        └── images_processed/
            └── <brand>/
                └── *.jpg / *.png
"""

import os
import csv
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Environment detection ─────────────────────────────────────────────────────

def is_colab() -> bool:
    return "COLAB_GPU" in os.environ


def resolve_repo_root(env_cfg: dict) -> Path:
    """
    Returns the repo root so that the relative paths stored
    in the manifest CSVs resolve correctly on both local and Colab.
    """
    root = env_cfg["colab_data_root"] if is_colab() else env_cfg["local_data_root"]
    return Path(root)


# ── Transform builders ────────────────────────────────────────────────────────

def build_transforms(aug_cfg: dict, image_size: int, split: str) -> transforms.Compose:
    """
    Build the transform pipeline for a given split.
    Augmentation is only applied during training.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        t = []

        if aug_cfg.get("random_crop", False):
            scale = tuple(aug_cfg.get("random_crop_scale", [0.8, 1.0]))
            t.append(transforms.RandomResizedCrop(image_size, scale=scale))
        else:
            t.append(transforms.Resize((image_size, image_size)))

        if aug_cfg.get("horizontal_flip", False):
            t.append(transforms.RandomHorizontalFlip())

        rot = aug_cfg.get("rotation_degrees", 0)
        if rot:
            t.append(transforms.RandomRotation(rot))

        brightness = aug_cfg.get("brightness", 0)
        contrast   = aug_cfg.get("contrast", 0)
        if brightness or contrast:
            t.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

        t += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(t)

    else:  # val / test — deterministic, no augmentation
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Manifest dataset ──────────────────────────────────────────────────────────

class ManifestDataset(Dataset):
    """
    A generic Dataset that reads from a manifest CSV.

    The manifest has columns: path, brand, brand_id
    - `path` is relative to repo_root
      e.g. "processed_data/five_car_models/images_processed/audi/audi q8-10.jpg"
    - `brand_id` values may not be contiguous (e.g. 3, 4, 5, 33, 46), so we
      remap them to 0-indexed class indices using the class_to_idx mapping
      built from the training manifest.

    Args:
        manifest_path : path to the manifest CSV (train / val / test)
        repo_root     : absolute path to the repo root so image paths resolve
        class_to_idx  : dict mapping brand_id (int) → contiguous class index
        classes       : ordered list of brand names (index == class index)
        transform     : torchvision transform pipeline
    """

    def __init__(
        self,
        manifest_path: Path,
        repo_root: Path,
        class_to_idx: dict,
        classes: list,
        transform=None,
    ):
        self.repo_root    = repo_root
        self.class_to_idx = class_to_idx
        self.classes      = classes
        self.transform    = transform
        self.samples      = []   # list of (absolute_image_path, class_index)

        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path  = repo_root / row["path"]
                brand_id  = int(row["brand_id"])
                class_idx = class_to_idx[brand_id]
                self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")   # handles 1-ch and 4-ch edge cases
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Class index builder ───────────────────────────────────────────────────────

def build_class_index(train_manifest: Path) -> tuple[dict, list]:
    """
    Reads the training manifest and builds:
      - class_to_idx : {brand_id (int) → contiguous 0-based index}
      - classes      : list of brand names ordered by class index

    Using the training manifest as the single source of truth ensures
    val and test use the exact same mapping.
    """
    brand_id_to_name = {}

    with open(train_manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            brand_id_to_name[int(row["brand_id"])] = row["brand"]

    # Sort by brand_id for deterministic ordering across runs
    sorted_ids   = sorted(brand_id_to_name.keys())
    class_to_idx = {brand_id: idx for idx, brand_id in enumerate(sorted_ids)}
    classes      = [brand_id_to_name[bid] for bid in sorted_ids]

    return class_to_idx, classes


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> dict:
    """
    Returns a dict:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader,
         "class_names": list[str]}

    Everything is driven by the config. All three datasets share the same
    manifest format so no dataset-specific branching is needed here.
    """
    ds_cfg      = cfg["dataset"]
    aug_cfg     = cfg["augmentation"]
    env_cfg     = cfg["env"]

    name        = ds_cfg["name"]
    image_size  = ds_cfg["image_size"]
    batch_size  = ds_cfg["batch_size"]
    num_workers = ds_cfg.get("num_workers", 4)

    repo_root    = resolve_repo_root(env_cfg)
    manifest_dir = repo_root / "processed_data" / name

    train_manifest = manifest_dir / "manifest_train.csv"
    val_manifest   = manifest_dir / "manifest_val.csv"
    test_manifest  = manifest_dir / "manifest_test.csv"

    # ── Build class index from training manifest ───────────────────────────────
    class_to_idx, classes = build_class_index(train_manifest)
    print(f"  [{name}] {len(classes)} classes: {classes}")

    # ── Build one DataLoader per split ────────────────────────────────────────
    loaders = {}
    split_manifests = {"train": train_manifest, "val": val_manifest}

    if test_manifest.exists():
        split_manifests["test"] = test_manifest

    for split, manifest_path in split_manifests.items():
        transform = build_transforms(aug_cfg, image_size, split)
        dataset   = ManifestDataset(
            manifest_path = manifest_path,
            repo_root     = repo_root,
            class_to_idx  = class_to_idx,
            classes       = classes,
            transform     = transform,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = True,
        )
        print(f"  [{split}] {len(dataset)} samples loaded from {manifest_path.name}")

    # Fall back to val if no test manifest exists
    if "test" not in loaders:
        loaders["test"] = loaders["val"]
        print(f"  [test] No manifest_test.csv found — using val set for final evaluation.")

    loaders["class_names"] = classes
    return loaders
