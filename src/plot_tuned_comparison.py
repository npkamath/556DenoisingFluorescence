#!/usr/bin/env python3
"""
Plot 2: Grouped bar chart comparing untuned vs task-tuned AP@0.5.

For each classical method (Wiener, Poisson-TV, BM3D), shows side-by-side bars
for the default hyperparameters vs the task-aware tuned hyperparameters.

Both are evaluated on the same 53 held-out test images (val split excluded)
for a fair comparison. Untuned scores are filtered to the test split using
results/extension1/validation_split.csv.

Usage:
    python src/plot_tuned_comparison.py
    python src/plot_tuned_comparison.py --ap_dir results/ap_scores \
        --tuned_dir results/extension1 --out_dir figures/comparison
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

AP_DIR    = Path("results/ap_scores")
TUNED_DIR = Path("results/extension1")
OUT_DIR   = Path("figures/comparison")

METHODS = {
    "wiener":     "Wiener",
    "poisson_tv": "Poisson-TV",
    "bm3d":       "BM3D",
}

COLOR_UNTUNED = "#4c78a8"
COLOR_TUNED   = "#f58518"


def load_ap50_dict(csv_path: Path) -> dict[str, float]:
    """Load per-image AP@0.5 as {stem: score} from a CSV."""
    scores = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        col = next((c for c in reader.fieldnames if "0.50" in c or "ap50" in c.lower()), None)
        if col is None:
            raise ValueError(f"No AP@0.5 column in {csv_path}")
        for row in reader:
            scores[row["image"]] = float(row[col])
    return scores


def load_test_stems(split_csv: Path) -> list[str]:
    """Load image stems assigned to 'test' split."""
    stems = []
    with open(split_csv) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                stems.append(row["image"])
    return stems


def main(ap_dir: Path, tuned_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    split_csv = tuned_dir / "validation_split.csv"
    if not split_csv.exists():
        print(f"validation_split.csv not found at {split_csv}. Run task_aware_tuning.py first.")
        return

    test_stems = set(load_test_stems(split_csv))
    print(f"Test split: {len(test_stems)} images")

    results = {}
    missing = []
    for method, label in METHODS.items():
        untuned_csv = ap_dir / f"{method}.csv"
        tuned_csv   = tuned_dir / "tuned_ap_scores" / f"{method}_tuned.csv"

        if not untuned_csv.exists():
            missing.append(f"untuned {label} ({untuned_csv})")
            continue
        if not tuned_csv.exists():
            missing.append(f"tuned {label} ({tuned_csv})")
            continue

        untuned_all = load_ap50_dict(untuned_csv)
        tuned_all   = load_ap50_dict(tuned_csv)

        # Filter both to the same test stems
        common = test_stems & set(untuned_all) & set(tuned_all)
        if not common:
            missing.append(f"{label}: no overlapping test images")
            continue

        untuned_scores = np.array([untuned_all[s] for s in sorted(common)])
        tuned_scores   = np.array([tuned_all[s]   for s in sorted(common)])
        results[method] = (label, untuned_scores, tuned_scores)

    if missing:
        print("Skipped (missing data):")
        for m in missing:
            print(f"  {m}")

    if not results:
        print("No complete method pairs found. Run evaluate.py first.")
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    n = len(results)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 3.2), 5))

    labels_list  = [v[0] for v in results.values()]
    untuned_means = [v[1].mean() for v in results.values()]
    tuned_means   = [v[2].mean() for v in results.values()]

    bars_u = ax.bar(x - width/2, untuned_means, width,
                    color=COLOR_UNTUNED, label="Default params",
                    edgecolor="white")
    bars_t = ax.bar(x + width/2, tuned_means, width,
                    color=COLOR_TUNED, label="Task-tuned params",
                    edgecolor="white")

    # Annotate with delta
    for i, (um, tm) in enumerate(zip(untuned_means, tuned_means)):
        delta = tm - um
        sign = "+" if delta >= 0 else ""
        ax.text(x[i] + width/2, tm + 0.008,
                f"{sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=9, color=COLOR_TUNED, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=11)
    ax.set_ylabel("Mean AP@0.5", fontsize=11)
    ax.set_title("Default vs Task-Tuned Hyperparameters — AP@0.5 (53 held-out images)", fontsize=11)
    all_means = untuned_means + tuned_means
    ax.set_ylim(0, min(1.05, max(all_means) + 0.08))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = out_dir / "tuned_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print table
    print(f"\n{'Method':<14} {'Default AP':>11} {'Tuned AP':>10} {'Delta':>8} {'N':>5}")
    print("-" * 52)
    for method, (label, u, t) in results.items():
        print(f"{label:<14} {u.mean():>11.4f} {t.mean():>10.4f} "
              f"{t.mean()-u.mean():>+8.4f} {len(u):>5}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ap_dir",    type=Path, default=AP_DIR)
    p.add_argument("--tuned_dir", type=Path, default=TUNED_DIR)
    p.add_argument("--out_dir",   type=Path, default=OUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.ap_dir, args.tuned_dir, args.out_dir)
