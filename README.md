# Car Brand Detection — COMP 472

CNN-based car classification system comparing AlexNet, ResNet-50, and MobileNetV2
across three datasets with both scratch and transfer learning strategies.

---

## Project Structure

```
car_classification/
├── configs/                  ← One YAML per experiment (18 total)
│   └── base_config.yaml      ← Reference template
├── datasets/
│   └── data_factory.py       ← DataLoader builder for all 3 datasets
├── models/
│   └── model_factory.py      ← Model builder (AlexNet / ResNet50 / MobileNetV2)
├── utils/
│   ├── metrics.py            ← Accuracy, F1, confusion matrix, timing, model size
│   ├── visualizations.py     ← Grad-CAM and t-SNE
│   └── logger.py             ← wandb wrapper
├── train.py                  ← Main training script
├── run_all_experiments.py    ← Runs all 18 experiments sequentially
├── generate_configs.py       ← Generates all 18 config YAMLs
└── requirements.txt
```

---

## Setup

### Local

```bash
pip install -r requirements.txt
wandb login   # paste your API key from wandb.ai/settings
```

### Google Colab

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd car_classification
!pip install -r requirements.txt
!wandb login   # paste API key when prompted

from google.colab import drive
drive.mount('/content/drive')
```

---

## Data Setup

### Expected folder structure under your data root

```
data/
├── car_brand_classification/
│   ├── train/   ← 33 class subfolders
│   ├── val/
│   └── test/
├── five_car_models/
│   ├── train/   ← 5 class subfolders (Audi, Bentley, BMW, Mercedes, Toyota)
│   └── val/
└── stanford_cars/
    ├── cars_train/
    ├── cars_test/
    └── devkit/
        ├── cars_train_annos.mat
        ├── cars_test_annos_withlabels.mat
        └── cars_meta.mat
```

For **Colab**, upload this same structure to Google Drive under:
`My Drive/car_datasets/`

The code auto-detects whether it's running in Colab and uses the right path.

---

## Usage

### 1. Generate all 18 experiment configs

```bash
python generate_configs.py
```

### 2. Run a single experiment

```bash
python train.py --config configs/resnet50_car_brand_classification_transfer.yaml

# Without wandb (debugging):
python train.py --config configs/alexnet_five_car_models_scratch.yaml --no-wandb
```

### 3. Run all experiments

```bash
python run_all_experiments.py

# Only run ResNet50 experiments:
python run_all_experiments.py --filter resnet50

# Skip wandb:
python run_all_experiments.py --no-wandb
```

Already-completed experiments (those with a `best_model.pt` checkpoint) are
automatically skipped so you can safely resume after an interruption.

---

## What Gets Tracked in wandb

| Category | Metrics |
|---|---|
| Scalars (per epoch) | train/val loss, accuracy, macro precision, recall, F1 |
| Tables | Per-class precision, recall, F1 (val + test) |
| Images | Confusion matrix, Grad-CAM overlays, t-SNE embedding plot |
| Model info | Total params, trainable params, size (MB) |
| Timing | Per-image inference time (ms) |
| Artifacts | Best model checkpoint uploaded per run |

All runs appear in the same wandb project for easy cross-experiment comparison.

---

## Config Reference

Key fields in each YAML:

| Field | Description |
|---|---|
| `model.architecture` | `alexnet` \| `resnet50` \| `mobilenet_v2` |
| `model.pretrained` | `true` = transfer learning, `false` = from scratch |
| `model.freeze_backbone` | Freeze all layers except head during warm-up |
| `training.unfreeze_after_epoch` | Epoch at which full fine-tuning begins (0 = never) |
| `training.early_stopping_patience` | Stop if val loss doesn't improve for N epochs |
| `logging.log_gradcam` | Log Grad-CAM visualizations to wandb |
| `logging.log_tsne` | Log t-SNE plot to wandb |
