"""
inference.py
Run a pre-trained model checkpoint on the provided sample test dataset
(or any folder of images) and print per-image predictions and overall accuracy.

Usage
-----
# Run on the included 100-image sample test dataset:
python inference.py --checkpoint outputs/resnet50_car_brand_classification_transfer/best_model.pt \
                    --dataset sample_test_dataset/car_brand_classification

# Run on a different dataset folder (flat or class-subfolder layout):
python inference.py --checkpoint outputs/mobilenet_v2_five_car_models_transfer/best_model.pt \
                    --dataset sample_test_dataset/five_car_models

# Suppress per-image output, show only the summary:
python inference.py --checkpoint outputs/resnet50_stanford_cars_transfer/best_model.pt \
                    --dataset sample_test_dataset/stanford_cars --quiet
"""

import os
import csv
import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from model_factory import build_model


# ── Image extensions accepted ─────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Preprocessing (matches test-time transform in data_factory.py) ────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_class_names(cfg: dict) -> list:
    """
    Rebuild the ordered class name list from the training manifest.
    Mirrors the logic in data_factory.build_class_index so that
    predicted indices align with the same class ordering used during training.
    """
    repo_root    = Path(cfg["env"]["local_data_root"])
    dataset_name = cfg["dataset"]["name"]
    train_manifest = repo_root / "processed_data" / dataset_name / "manifest_train.csv"

    if not train_manifest.exists():
        return None

    brand_id_to_name = {}
    with open(train_manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            brand_id_to_name[int(row["brand_id"])] = row["brand"]

    sorted_ids  = sorted(brand_id_to_name.keys())
    class_names = [brand_id_to_name[bid] for bid in sorted_ids]
    return class_names


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load a model from a checkpoint produced by train.py.
    The checkpoint stores the full config so architecture and num_classes
    are recovered automatically — no separate config file needed.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    class_names = load_class_names(cfg)
    return model, class_names, cfg


def collect_images(dataset_dir: str):
    """
    Walk dataset_dir and return a list of (image_path, label_str) tuples.
    Supports two layouts:
      - Flat:          dataset_dir/<image>.jpg          → label = "unknown"
      - Class folders: dataset_dir/<class>/<image>.jpg  → label = <class>
    """
    root = Path(dataset_dir)
    samples = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            # If parent is the root itself → flat layout
            label = p.parent.name if p.parent != root else "unknown"
            samples.append((p, label))
    return samples


def predict(model, image_path: Path, device: torch.device, class_names):
    """Run a single image through the model and return (pred_idx, pred_label, confidence)."""
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
    pred_idx  = pred_idx.item()
    conf      = conf.item()
    pred_label = class_names[pred_idx] if class_names else str(pred_idx)
    return pred_idx, pred_label, conf


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained car brand model.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a best_model.pt checkpoint "
                             "(e.g. outputs/resnet50_car_brand_classification_transfer/best_model.pt)")
    parser.add_argument("--dataset", required=True,
                        help="Path to the image folder "
                             "(e.g. sample_test_dataset/car_brand_classification)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-image output; show summary only.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    model, class_names, cfg = load_model(args.checkpoint, device)
    arch       = cfg["model"]["architecture"]
    num_classes = cfg["model"]["num_classes"]
    print(f"Architecture : {arch}  |  Classes : {num_classes}")

    samples = collect_images(args.dataset)
    if not samples:
        print(f"\nNo images found in '{args.dataset}'. Check the path.")
        return

    print(f"Images found : {len(samples)}\n")
    if not args.quiet:
        print(f"{'Image':<55} {'True Label':<22} {'Prediction':<22} {'Confidence':>10}  Match")
        print("-" * 115)

    correct = 0
    for img_path, true_label in samples:
        _, pred_label, conf = predict(model, img_path, device, class_names)
        # Normalize for comparison: lowercase, hyphens == underscores
        def _norm(s): return s.lower().replace("-", "_")
        match   = (_norm(pred_label) == _norm(true_label))
        correct += int(match)
        if not args.quiet:
            tick = "OK" if match else "XX"
            print(f"{str(img_path.name):<55} {true_label:<22} {pred_label:<22} {conf:>9.1%}  {tick}")

    total    = len(samples)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'-'*50}")
    print(f"Results  :  {correct} / {total} correct")
    print(f"Accuracy :  {accuracy:.1%}")
    print(f"{'-'*50}\n")


if __name__ == "__main__":
    main()
