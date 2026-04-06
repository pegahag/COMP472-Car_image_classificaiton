# Car Brand Detection using CNN Architectures — COMP 472

A systematic study of CNN-based car brand classification using three architectures
(AlexNet, ResNet-50, MobileNetV2) trained across three datasets of increasing
complexity, comparing training from scratch versus ImageNet transfer learning.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Requirements](#3-requirements)
4. [Dataset Setup](#4-dataset-setup)
5. [Training a Model](#5-training-a-model)
6. [Running All Experiments](#6-running-all-experiments)
7. [Running the Ablation Study](#7-running-the-ablation-study)
8. [Running Inference on the Sample Test Dataset](#8-running-inference-on-the-sample-test-dataset)
9. [Experiment Tracking (WandB)](#9-experiment-tracking-wandb)
10. [Config Reference](#10-config-reference)

---

## 1. Project Overview

### Problem
Automatically identify the make (brand) of a car from an image. This task is
challenging due to variations in lighting, viewing angle, background, and strong
visual similarities between brands.

### Datasets

| Dataset | Source | Classes | Images | Complexity |
|---|---|---|---|---|
| Five Car Models | Kaggle (B. Law) | 5 | 1,646 | Low |
| Car Brand Classification | Kaggle (A. Elsany) | 33 | 16,467 | Medium |
| Stanford Cars | Kaggle / Krause et al. | 49 (brand-level) | 16,185 | High |

### Architectures

| Model | Parameters | Role |
|---|---|---|
| AlexNet | ~61M | Baseline |
| ResNet-50 | ~25M | High accuracy |
| MobileNetV2 | ~3.4M | Lightweight / efficient |

### Experiment Design
- **18 main experiments**: 3 architectures × 3 datasets × 2 strategies (scratch + transfer)
- **20 ablation experiments**: sensitivity analysis across learning rate, batch size,
  dataset complexity, training set size, and architecture
- All models trained with AdamW, CosineAnnealingLR, and early stopping
- Evaluation via accuracy, macro F1, per-class precision/recall/F1, confusion matrices,
  Grad-CAM, and t-SNE visualizations

---

## 2. Repository Structure

```
comp472/
├── configs/
│   ├── base_config.yaml               ← Reference template with all fields documented
│   ├── alexnet_five_car_models_scratch.yaml
│   ├── resnet50_car_brand_classification_transfer.yaml
│   ├── ...                            ← 18 experiment configs total
│   └── ablation/
│       ├── ablation_lr_1e-3.yaml
│       ├── ablation_bs_32.yaml
│       └── ...                        ← 20 ablation configs total
├── processed_data/
│   ├── five_car_models/
│   │   ├── manifest_train.csv
│   │   ├── manifest_val.csv
│   │   ├── manifest_test.csv
│   │   └── images_processed/<brand>/*.jpg
│   ├── car_brand_classification/      ← same layout
│   └── stanford_cars/                 ← same layout
├── sample_test_dataset/               ← 100-image sample for quick inference testing
│   ├── five_car_models/<brand>/*.jpg  (18 images, 5 classes)
│   ├── car_brand_classification/<brand>/*.jpg  (33 images, 33 classes)
│   ├── stanford_cars/<brand>/*.jpg    (49 images, 49 classes)
│   └── labels.csv                     ← ground-truth labels for all 100 images
├── outputs/
│   └── <experiment_name>/
│       ├── best_model.pt              ← best checkpoint (by val loss)
│       ├── test_metrics.json          ← {accuracy, macro_f1}
│       └── classification_report.txt
├── train.py                           ← main training script
├── inference.py                       ← run a checkpoint on the sample test dataset
├── run_all_experiments.py             ← sequential runner for all 18 main experiments
├── run_ablation.py                    ← sequential runner for all 20 ablation experiments
├── generate_configs.py                ← regenerate the 18 main experiment YAMLs
├── generate_ablation_configs.py       ← regenerate the 20 ablation YAMLs
├── data_factory.py                    ← DataLoader builder (manifest-based)
├── model_factory.py                   ← model builder (AlexNet / ResNet-50 / MobileNetV2)
├── metrics.py                         ← accuracy, F1, confusion matrix, timing, model size
├── visualizations.py                  ← Grad-CAM and t-SNE
├── logger.py                          ← WandB wrapper
├── create_sample_test_dataset.py      ← script that created the 100-image sample dataset
├── requirements.txt
└── colab/
    └── COMP472_Training.ipynb         ← Google Colab notebook
```

---

## 3. Requirements

**Python**: 3.10 or higher recommended.

Install all dependencies:

```bash
pip install -r requirements.txt
```

Full list (`requirements.txt`):

```
torch>=2.0.0
torchvision>=0.15.0
wandb>=0.16.0
pyyaml>=6.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
```

> **GPU**: A CUDA-capable GPU is strongly recommended for training. Inference
> runs on CPU if no GPU is detected.

> **WandB** (optional): experiment tracking. Create a free account at
> https://wandb.ai, then run `wandb login` and paste your API key.
> All training commands accept `--no-wandb` to skip tracking entirely.

---

## 4. Dataset Setup

The `processed_data/` folder is **included in the repository** and contains the
pre-processed manifest CSVs and images for all three datasets. No additional
download is required to reproduce the experiments.

### Downloading the original raw datasets

If you need the original unprocessed images, download them from Kaggle:

| Dataset | Download Link |
|---|---|
| Five Car Models Images | https://www.kaggle.com/datasets/itsahmad/five-cars-models-images |
| Car Brand Classification | https://www.kaggle.com/datasets/ahmedelsany/car-brand-classification |
| Stanford Cars | https://www.kaggle.com/datasets/eduardopalmans/stanford-cars-dataset |

Each dataset downloads as a ZIP archive. Extract to a folder of your choice.
The raw datasets are **not required** to run training (the processed manifests
and images in `processed_data/` are sufficient).

---

## 5. Training a Model

### Step 1 — Generate experiment configs (only needed once)

```bash
python generate_configs.py
```

This writes all 18 YAML configs to `configs/`. They are already committed to
the repository, so this step is only needed if you delete or modify them.

### Step 2 — Run a single experiment

```bash
# Transfer learning — ResNet-50 on Car Brand Classification:
python train.py --config configs/resnet50_car_brand_classification_transfer.yaml

# From scratch — AlexNet on Five Car Models:
python train.py --config configs/alexnet_five_car_models_scratch.yaml

# Disable WandB logging (useful for debugging):
python train.py --config configs/mobilenet_v2_stanford_cars_transfer.yaml --no-wandb
```

**What happens:**
- Loads data from `processed_data/<dataset>/manifest_{train,val,test}.csv`
- Trains for up to 50 epochs with early stopping (patience = 7)
- Saves the best checkpoint to `outputs/<experiment_name>/best_model.pt`
- Saves test metrics to `outputs/<experiment_name>/test_metrics.json`
- Logs per-epoch metrics, confusion matrices, Grad-CAM, and t-SNE to WandB

### Available config names

```
alexnet_car_brand_classification_scratch
alexnet_car_brand_classification_transfer
alexnet_five_car_models_scratch
alexnet_five_car_models_transfer
alexnet_stanford_cars_scratch
alexnet_stanford_cars_transfer
mobilenet_v2_car_brand_classification_scratch
mobilenet_v2_car_brand_classification_transfer
mobilenet_v2_five_car_models_scratch
mobilenet_v2_five_car_models_transfer
mobilenet_v2_stanford_cars_scratch
mobilenet_v2_stanford_cars_transfer
resnet50_car_brand_classification_scratch
resnet50_car_brand_classification_transfer
resnet50_five_car_models_scratch
resnet50_five_car_models_transfer
resnet50_stanford_cars_scratch
resnet50_stanford_cars_transfer
```

---

## 6. Running All Experiments

```bash
# Run all 18 experiments sequentially:
python run_all_experiments.py

# Run only ResNet-50 experiments:
python run_all_experiments.py --filter resnet50

# Skip WandB:
python run_all_experiments.py --no-wandb
```

Experiments that already have a `best_model.pt` checkpoint are **automatically
skipped**, so it is safe to interrupt and resume at any time.

A summary table is saved to `outputs/experiment_summary.yaml` after all runs complete.

---

## 7. Running the Ablation Study

```bash
# Generate ablation configs (already committed, only needed if deleted):
python generate_ablation_configs.py

# Run all 20 ablation experiments:
python run_ablation.py

# Skip WandB:
python run_ablation.py --no-wandb
```

Ablation results are saved to `outputs/ablation_summary.yaml`.

**Ablation groups:**

| Group | Variable | Values tested |
|---|---|---|
| A | Learning rate | 1e-4, 5e-4, 1e-3, 5e-3, 1e-2 |
| B | Batch size | 8, 16, 32, 64 |
| C | Number of classes | 5, 33, 49 |
| D | Training set size | 25, 50, 100, 200, all images/class |
| E | Architecture | AlexNet, ResNet-50, MobileNetV2 |

Baseline for all groups: ResNet-50 with transfer learning on Car Brand Classification
(33 classes, lr=1e-3, batch size=32, full dataset, 15 epochs).

---

## 8. Running Inference on the Sample Test Dataset

A 100-image sample test dataset is included at `sample_test_dataset/`, containing
images from all three datasets with ground-truth labels in `sample_test_dataset/labels.csv`.

Pre-trained checkpoints are stored in the `outputs/` directory.

### Run inference on each dataset's sample:

```bash
# Car Brand Classification — ResNet-50 transfer learning (33 classes):
python inference.py \
  --checkpoint outputs/resnet50_car_brand_classification_transfer/best_model.pt \
  --dataset sample_test_dataset/car_brand_classification

# Five Car Models — MobileNetV2 transfer learning (5 classes):
python inference.py \
  --checkpoint outputs/mobilenet_v2_five_car_models_transfer/best_model.pt \
  --dataset sample_test_dataset/five_car_models

# Stanford Cars — ResNet-50 transfer learning (49 classes):
python inference.py \
  --checkpoint outputs/resnet50_stanford_cars_transfer/best_model.pt \
  --dataset sample_test_dataset/stanford_cars
```

### Output

The script prints a per-image prediction table followed by a summary:

```
Image                             True Label     Prediction     Confidence  Match
acura ilx-1.jpg                   acura          acura              89.3%   OK
aston martin db9-2.jpg            aston martin   aston martin       76.1%   OK
audi a4-7.jpg                     audi           bmw                54.2%   XX
...
--------------------------------------------------
Results  :  21 / 33 correct
Accuracy :  63.6%
--------------------------------------------------
```

### Options

```
--checkpoint   Path to a best_model.pt checkpoint file          (required)
--dataset      Path to image folder (flat or class-subfolder layout)  (required)
--quiet        Show summary only, suppress per-image table      (optional)
```

### Running on your own images

The script accepts any folder of images — either flat or organized into
class subfolders. If using class subfolders, the subfolder name is used as
the true label for accuracy calculation.

```bash
# Flat folder (no accuracy calculated, just predictions):
python inference.py \
  --checkpoint outputs/resnet50_car_brand_classification_transfer/best_model.pt \
  --dataset /path/to/your/images
```

> **Note**: The checkpoint file encodes the architecture and number of classes
> automatically — no separate config file is needed for inference.

---

## 9. Experiment Tracking (WandB)

All training runs log to WandB under two projects:

| Project | Contents |
|---|---|
| `comp472_car_detection_final` | 18 main experiments |
| `comp472_ablation` | 20 ablation experiments |

**What is logged per run:**

| Category | Details |
|---|---|
| Scalars | Train/val loss, accuracy, macro precision, recall, F1 (per epoch) |
| Tables | Per-class precision, recall, F1 (val + test) |
| Images | Confusion matrix, Grad-CAM overlays (epochs 1, mid, final), t-SNE plot |
| Model info | Total params, trainable params, size (MB) |
| Timing | Per-image inference latency (ms) |
| Artifacts | `best_model.pt` uploaded per run |

To skip WandB on any command, append `--no-wandb`.

---

## 10. Config Reference

All experiments are controlled by YAML files. The annotated template is at
`configs/base_config.yaml`. Key fields:

| Field | Options / Type | Description |
|---|---|---|
| `model.architecture` | `alexnet` \| `resnet50` \| `mobilenet_v2` | CNN backbone |
| `model.pretrained` | `true` / `false` | Transfer learning vs. from scratch |
| `model.freeze_backbone` | `true` / `false` | Freeze backbone during warm-up phase |
| `model.num_classes` | integer | Number of output classes |
| `dataset.name` | `five_car_models` \| `car_brand_classification` \| `stanford_cars` | Dataset to use |
| `dataset.batch_size` | integer | Mini-batch size (use 16 for five_car_models) |
| `training.epochs` | integer | Maximum training epochs |
| `training.learning_rate` | float | Initial LR for head / full training |
| `training.finetune_lr` | float | Backbone LR after unfreezing (transfer only) |
| `training.weight_decay` | float | L2 regularization coefficient |
| `training.early_stopping_patience` | integer | Stop if val loss stagnates for N epochs |
| `training.unfreeze_after_epoch` | integer | Epoch to unfreeze backbone (0 = never) |
| `training.scheduler` | `cosine` \| `step` \| `none` | LR schedule |
| `logging.log_gradcam` | `true` / `false` | Log Grad-CAM images to WandB |
| `logging.log_tsne` | `true` / `false` | Log t-SNE embedding plot to WandB |

---
## Contributions

Our team worked collaboratively on Google Colab throughout the development of this project, as reflected in the team contribution report. Project files were then maintained and pushed locally to GitHub as part of the final repository organization and submission workflow.

## Contributors

## Contributors

| Name | GitHub | Contributions |
|---|---|---|
| Pegah Aghili | [pegahag](https://github.com/pegahag) | Preprocessed and prepared the three datasets used in the project; contributed to the writing and editing of the final report; assisted with the dataset description and related documentation. |
| Arash Shafiee | [Arashcito](https://github.com/Arashcito) | Contributed to the transfer learning component (ResNet + MobileNet); assisted in the preparation of the final report; worked on the dataset description section. |
| Joyal Biju Kulangara | [Joyal99](https://github.com/Joyal99) | Led the evaluation and metrics analysis of the datasets and models; calculated and interpreted accuracy, precision, recall, F1-score, and confusion matrices; contributed to the analysis-related coding tasks; designed and prepared the presentation. |
| Youssef Ajam | [YoussefA0807](https://github.com/YoussefA0807) | Worked on the ablation study; developed training scripts and configuration files; contributed to the literature review and evaluation metrics; helped write the results section; assisted with the overall preparation and formatting of the final report. |
| Andrew Kamami | [AndrewKaranu](https://github.com/AndrewKaranu) | Developed the training script and baseline model implementation using AlexNet; contributed to the implementation and optimization of the CNN models; helped write the methodology section of the report; assisted with the experimental setup write-up. |
| Dr. Mahdi S. Hosseini (Supervisor) | [AtlasAnalyticsLab](https://github.com/AtlasAnalyticsLab) | Project supervision and academic guidance. |
| Rose Rostami (Lead TA) | TBD | Teaching assistant support and project guidance. |
