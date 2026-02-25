"""
run_all_experiments.py
Runs all 18 experiments sequentially.
Skips any experiment that already has a best_model.pt checkpoint.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --no-wandb        # disable wandb
    python run_all_experiments.py --filter resnet50  # only run configs matching a string
"""

import os
import argparse
from pathlib import Path
import yaml

from train import train, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--filter",   default=None,
                        help="Only run configs whose filename contains this string.")
    args = parser.parse_args()

    config_dir = Path("configs")
    configs    = sorted(config_dir.glob("*.yaml"))

    # Exclude the base template
    configs = [c for c in configs if c.name != "base_config.yaml"]

    if args.filter:
        configs = [c for c in configs if args.filter in c.name]

    print(f"\nFound {len(configs)} experiment configs to run.\n")

    results = {}

    for config_path in configs:
        cfg = load_config(str(config_path))
        exp_name = cfg["experiment_name"]

        # Skip if already completed
        best_model_path = Path(cfg["env"]["output_root"]) / exp_name / "best_model.pt"
        if best_model_path.exists():
            print(f"  [skip] {exp_name} — checkpoint already exists.")
            continue

        print(f"\n{'='*60}")
        print(f"  Starting: {exp_name}")
        print(f"{'='*60}")

        try:
            metrics = train(cfg, use_wandb=not args.no_wandb)
            results[exp_name] = {
                "status":        "complete",
                "test_accuracy": metrics["accuracy"],
                "test_f1":       metrics["macro_f1"],
            }
        except Exception as e:
            print(f"\n  [ERROR] {exp_name} failed: {e}")
            results[exp_name] = {"status": "failed", "error": str(e)}

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Experiment':<45} {'Acc':>7}  {'F1':>7}  Status")
    print(f"  {'-'*45} {'-'*7}  {'-'*7}  {'-'*8}")

    for name, r in sorted(results.items()):
        if r["status"] == "complete":
            print(f"  {name:<45} {r['test_accuracy']:>7.4f}  {r['test_f1']:>7.4f}  ✓")
        else:
            print(f"  {name:<45} {'—':>7}  {'—':>7}  FAILED")

    # Save summary to file
    summary_path = Path("outputs") / "experiment_summary.yaml"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
