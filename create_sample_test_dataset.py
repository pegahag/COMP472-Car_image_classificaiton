"""
create_sample_test_dataset.py

Builds a 100-image sample test dataset from the three held-out test manifests.

Sampling strategy (stratified by class, seeded for reproducibility):
  - Five Car Models        :  18 images  (~3-4 per class, 5 classes)
  - Car Brand Classification:  33 images  (1 per class, 33 classes)
  - Stanford Cars          :  49 images  (1 per class, 49 classes)
  Total                    : 100 images

Output layout:
  sample_test_dataset/
    five_car_models/
      <brand>/
        <image>.jpg
    car_brand_classification/
      <brand>/
        <image>.jpg
    stanford_cars/
      <brand>/
        <image>.jpg
    labels.csv          <- path, brand, brand_id, dataset
"""

import os
import shutil
import random
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(ROOT, "processed_data")
OUTPUT_DIR  = os.path.join(ROOT, "sample_test_dataset")
SEED        = 42

DATASETS = {
    "five_car_models":          18,   # ~3-4 per class across 5 classes
    "car_brand_classification": 33,   # 1 per class across 33 classes
    "stanford_cars":            49,   # 1 per class across 49 classes
}
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

all_records = []

for dataset_name, n_total in DATASETS.items():
    manifest_path = os.path.join(DATA_ROOT, dataset_name, "manifest_test.csv")
    df = pd.read_csv(manifest_path)

    classes = sorted(df["brand"].unique())
    n_classes = len(classes)
    base_per_class = n_total // n_classes
    remainder = n_total % n_classes

    # Distribute remainder to first `remainder` classes (sorted)
    class_quota = {c: base_per_class for c in classes}
    for c in classes[:remainder]:
        class_quota[c] += 1

    sampled_rows = []
    for brand, quota in class_quota.items():
        rows = df[df["brand"] == brand].copy()
        quota = min(quota, len(rows))          # guard against small classes
        sampled_rows.append(rows.sample(n=quota, random_state=SEED))

    sampled_df = pd.concat(sampled_rows).reset_index(drop=True)

    # Copy images into output folder
    for _, row in sampled_df.iterrows():
        src = os.path.join(ROOT, row["path"])
        rel_within_dataset = os.path.relpath(row["path"],
                                             os.path.join("processed_data",
                                                          dataset_name,
                                                          "images_processed"))
        dest = os.path.join(OUTPUT_DIR, dataset_name, rel_within_dataset)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)

        all_records.append({
            "dataset":  dataset_name,
            "path":     os.path.join(dataset_name, rel_within_dataset),
            "brand":    row["brand"],
            "brand_id": row["brand_id"],
        })

    print(f"  {dataset_name}: sampled {len(sampled_df)} / {n_total} images "
          f"across {n_classes} classes")

# Write labels CSV
labels_df = pd.DataFrame(all_records)
labels_df.to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)

total = len(labels_df)
print(f"\nDone. {total} images written to: {OUTPUT_DIR}")
print(f"Labels saved to: {os.path.join(OUTPUT_DIR, 'labels.csv')}")
