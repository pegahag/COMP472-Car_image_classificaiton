"""
run_ablation.py
Runs all ablation study experiments (configs/ablation/*.yaml) sequentially.
Skips any experiment that already has a best_model.pt checkpoint.

The ablation study varies one hyperparameter at a time to find the best
configuration for final training:
  Group A — Learning Rate
  Group B — Batch Size
  Group C — Number of Classes (via dataset)
  Group D — Images per Class (max_samples_per_class)
  Group E — Model Architecture

Usage:
    python run_ablation.py                      # run all 20 ablation configs
    python run_ablation.py --group lr           # run only learning-rate group
    python run_ablation.py --filter bs_32       # run configs matching string
    python run_ablation.py --no-wandb           # disable wandb logging
"""

import argparse
from pathlib import Path
import yaml

from train import train, load_config


# Maps group flag → config name prefix
GROUPS = {
    "lr":         "ablation_lr_",
    "bs":         "ablation_bs_",
    "nclasses":   "ablation_nclasses_",
    "maxsamples": "ablation_maxsamples_",
    "arch":       "ablation_arch_",
}

GROUP_LABELS = {
    "lr":         "Learning Rate",
    "bs":         "Batch Size",
    "nclasses":   "Number of Classes",
    "maxsamples": "Images per Class",
    "arch":       "Architecture",
}

GROUP_FIXED = {
    "lr":         "arch=resnet50, dataset=car_brand, batch=32",
    "bs":         "arch=resnet50, dataset=car_brand, lr=0.001",
    "nclasses":   "arch=resnet50, lr=0.001, batch=32",
    "maxsamples": "arch=resnet50, dataset=car_brand, lr=0.001, batch=32",
    "arch":       "dataset=car_brand, lr=0.001, batch=32",
}


def get_group(exp_name: str) -> str:
    """Return the group key for an experiment name, or 'unknown'."""
    for key, prefix in GROUPS.items():
        if exp_name.startswith(prefix):
            return key
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--group", default=None,
        choices=list(GROUPS.keys()),
        help="Run only one ablation group (lr, bs, nclasses, maxsamples, arch).",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Run configs whose filename contains this string.",
    )
    args = parser.parse_args()

    ablation_dir = Path("configs/ablation")
    if not ablation_dir.exists():
        print("No configs/ablation/ directory found. Run generate_ablation_configs.py first.")
        return

    configs = sorted(ablation_dir.glob("*.yaml"))

    if args.group:
        prefix = GROUPS[args.group]
        configs = [c for c in configs if c.stem.startswith(prefix)]

    if args.filter:
        configs = [c for c in configs if args.filter in c.name]

    print(f"\nFound {len(configs)} ablation configs to run.\n")

    results = {}

    for config_path in configs:
        cfg      = load_config(str(config_path))
        exp_name = cfg["experiment_name"]

        # Skip if already completed
        best_model_path = Path(cfg["env"]["output_root"]) / exp_name / "best_model.pt"
        if best_model_path.exists():
            print(f"  [skip] {exp_name} — checkpoint already exists.")
            # Load saved metrics if available for the summary
            summary_path = best_model_path.parent / "classification_report.txt"
            results[exp_name] = {"status": "skipped"}
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
                "group":         get_group(exp_name),
            }
        except MemoryError as e:
            print(f"\n  [OOM] {exp_name} — out of memory: {e}")
            results[exp_name] = {"status": "oom", "error": str(e)}
        except Exception as e:
            print(f"\n  [ERROR] {exp_name} failed: {e}")
            results[exp_name] = {"status": "failed", "error": str(e)}

    # ── Flat summary table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Experiment':<45} {'Acc':>7}  {'F1':>7}  Status")
    print(f"  {'-'*45} {'-'*7}  {'-'*7}  {'-'*8}")

    for name, r in sorted(results.items()):
        if r["status"] == "complete":
            print(f"  {name:<45} {r['test_accuracy']:>7.4f}  {r['test_f1']:>7.4f}  complete")
        elif r["status"] == "skipped":
            print(f"  {name:<45} {'—':>7}  {'—':>7}  skipped")
        else:
            label = r["status"].upper()
            print(f"  {name:<45} {'—':>7}  {'—':>7}  {label}")

    # ── Grouped summary ────────────────────────────────────────────────────────
    completed = {n: r for n, r in results.items() if r["status"] == "complete"}

    if completed:
        print(f"\n\n{'='*60}")
        print("  RESULTS BY GROUP")
        print(f"{'='*60}")

        best_per_group = {}

        for group_key, group_label in GROUP_LABELS.items():
            group_results = {
                n: r for n, r in completed.items()
                if n.startswith(GROUPS[group_key])
            }
            if not group_results:
                continue

            best_name = max(group_results, key=lambda n: group_results[n]["test_f1"])
            best_per_group[group_key] = (best_name, group_results[best_name]["test_f1"])

            print(f"\n  {'-'*58}")
            print(f"  GROUP: {group_label}  ({GROUP_FIXED[group_key]})")
            print(f"  {'-'*58}")
            print(f"  {'Config':<45} {'Acc':>7}  {'Macro-F1':>8}")

            for name in sorted(group_results):
                r      = group_results[name]
                marker = " <- best" if name == best_name else ""
                print(f"  {name:<45} {r['test_accuracy']:>7.4f}  {r['test_f1']:>8.4f}{marker}")

        # ── Best per group ─────────────────────────────────────────────────────
        if best_per_group:
            print(f"\n\n{'='*60}")
            print("  BEST CONFIG PER GROUP  (by Macro-F1)")
            print(f"{'='*60}")
            for group_key, (best_name, best_f1) in best_per_group.items():
                label = GROUP_LABELS[group_key]
                print(f"  {label:<20}: {best_name:<45} F1={best_f1:.4f}")

    # ── Save to file ───────────────────────────────────────────────────────────
    output_path = Path("outputs") / "ablation_summary.yaml"
    output_path.parent.mkdir(exist_ok=True)

    # Build nested structure grouped by ablation variable
    grouped_results = {key: {} for key in GROUPS}
    for name, r in results.items():
        group = get_group(name)
        if group in grouped_results:
            grouped_results[group][name] = r

    summary = {
        "results_by_group": grouped_results,
        "best_per_group":   {
            k: v[0] for k, v in best_per_group.items()
        } if completed else {},
    }

    with open(output_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"\n  Summary saved to {output_path}")


if __name__ == "__main__":
    main()
