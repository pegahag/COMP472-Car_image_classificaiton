"""
run_all_experiments.py
Runs all 18 experiments sequentially, each in its own subprocess.
Skips any experiment that already has a best_model.pt checkpoint.

Running each experiment as a subprocess gives wandb a fresh service
connection per run, avoiding the Windows named-pipe crash ([WinError 64])
that kills the entire session when running multiple runs in-process.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --no-wandb        # disable wandb
    python run_all_experiments.py --filter resnet50  # only run configs matching a string
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--filter",   default=None,
                        help="Only run configs whose filename contains this string.")
    args = parser.parse_args()

    config_dir = Path("configs")
    configs    = sorted(config_dir.glob("*.yaml"))

    # Exclude the base template and ablation configs
    configs = [c for c in configs if c.name != "base_config.yaml"]

    if args.filter:
        configs = [c for c in configs if args.filter in c.name]

    print(f"\nFound {len(configs)} experiment configs to run.\n")

    results = {}

    for config_path in configs:
        cfg      = load_config(str(config_path))
        exp_name = cfg["experiment_name"]

        # Skip if already completed
        output_dir     = Path(cfg["env"]["output_root"]) / exp_name
        best_model_path = output_dir / "best_model.pt"
        if best_model_path.exists():
            print(f"  [skip] {exp_name} — checkpoint already exists.")
            # Load metrics if available so they appear in the summary
            metrics_path = output_dir / "test_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                results[exp_name] = {
                    "status":        "complete",
                    "test_accuracy": m["accuracy"],
                    "test_f1":       m["macro_f1"],
                }
            continue

        print(f"\n{'='*60}")
        print(f"  Starting: {exp_name}")
        print(f"{'='*60}")

        # Run in a fresh subprocess so wandb gets a clean service connection
        cmd = [sys.executable, "train.py", "--config", str(config_path)]
        if args.no_wandb:
            cmd.append("--no-wandb")

        result = subprocess.run(cmd)

        metrics_path = output_dir / "test_metrics.json"
        if result.returncode == 0 and metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            results[exp_name] = {
                "status":        "complete",
                "test_accuracy": m["accuracy"],
                "test_f1":       m["macro_f1"],
            }
        else:
            results[exp_name] = {
                "status": "failed",
                "error":  f"exit code {result.returncode}",
            }

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
