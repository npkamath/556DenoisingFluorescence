"""
src/analysis/bootstrap_ci.py
--------------------
Paired bootstrap confidence intervals for mean AP@0.5 differences.

Reads per-image AP CSVs produced by evaluate.py and computes:
  • Observed Δ = mean(method AP@0.5) − mean(baseline AP@0.5)
  • 95 % percentile bootstrap CI  (paired, resampling at the image level)
  • One-sided p-value for H₀: Δ ≤ 0  (i.e. no improvement)

This matches the "paired bootstrap confidence intervals over the 68-image set"
described in the proposal's Evaluation Protocol section.

Usage
-----
    # Noisy vs Cellpose3 (the P3 demo pair):
    python src/analysis/bootstrap_ci.py --baseline noisy --method cellpose3

    # All classical methods vs noisy in one call:
    python src/analysis/bootstrap_ci.py --baseline noisy --method wiener tv bm3d purelet

    # Custom AP score directory:
    python src/analysis/bootstrap_ci.py --baseline noisy --method cellpose3 \
        --ap_dir results/ap_scores --out_dir results/bootstrap

Input CSV format (from evaluate.py):
    image, ap@0.50, ap@0.55, ..., ap@0.95

Output (per method pair, under results/bootstrap/):
    noisy_vs_cellpose3_report.txt   — CI, p-value, interpretation
    noisy_vs_cellpose3_dist.png     — bootstrap Δ histogram
    bootstrap_summary.csv           — one-row-per-pair table for the report
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Default paths ─────────────────────────────────────────────────────────────
AP_DIR  = Path("results/ap_scores")
OUT_DIR = Path("results/bootstrap")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ap50(csv_path: Path) -> dict[str, float]:
    """
    Load per-image AP@0.5 from a CSV produced by evaluate.py.
    Returns {image_stem: ap_value}.
    """
    records = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            records[row["image"]] = float(row["ap@0.50"])
    if not records:
        raise ValueError(f"No data loaded from {csv_path}")
    return records


def align_pairs(
    base_dict: dict[str, float],
    method_dict: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return aligned arrays for images present in both dicts."""
    common = sorted(set(base_dict) & set(method_dict))
    dropped_b = set(method_dict) - set(base_dict)
    dropped_m = set(base_dict) - set(method_dict)
    if dropped_b:
        print(f"  [warn] {len(dropped_b)} images in method only — dropped")
    if dropped_m:
        print(f"  [warn] {len(dropped_m)} images in baseline only — dropped")
    ap_b = np.array([base_dict[k]   for k in common])
    ap_m = np.array([method_dict[k] for k in common])
    return ap_b, ap_m, common


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def paired_bootstrap_ci(
    ap_base   : np.ndarray,
    ap_method : np.ndarray,
    n_boot    : int   = 10_000,
    alpha     : float = 0.05,
    seed      : int   = 42,
) -> dict:
    """
    Percentile bootstrap CI for mean(Δ) where Δ = ap_method − ap_base,
    resampling image indices with replacement (paired).

    Returns dict with keys:
        delta_obs, ci_lo, ci_hi, p_value, boot_deltas, n_images, alpha
    """
    assert len(ap_base) == len(ap_method), "arrays must be the same length"
    rng  = np.random.default_rng(seed)
    n    = len(ap_base)
    diff = ap_method - ap_base
    delta_obs = float(diff.mean())

    boot_deltas = np.array([
        diff[rng.integers(0, n, size=n)].mean()
        for _ in range(n_boot)
    ])

    ci_lo   = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_hi   = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
    p_value = float((boot_deltas <= 0).mean())

    return dict(
        delta_obs   = delta_obs,
        ci_lo       = ci_lo,
        ci_hi       = ci_hi,
        p_value     = p_value,
        boot_deltas = boot_deltas,
        n_images    = n,
        alpha       = alpha,
    )


# ── Report & plot ─────────────────────────────────────────────────────────────

def write_report(
    result        : dict,
    baseline_name : str,
    method_name   : str,
    mean_base     : float,
    mean_method   : float,
    out_path      : Path,
) -> str:
    ci_pct = int((1 - result["alpha"]) * 100)
    sig    = result["p_value"] < result["alpha"]
    ci_has_zero = result["ci_lo"] <= 0

    lines = [
        "=" * 62,
        "  PAIRED BOOTSTRAP CI REPORT",
        f"  Baseline : {baseline_name}",
        f"  Method   : {method_name}",
        "=" * 62,
        "",
        f"  Images               : {result['n_images']}",
        f"  Bootstrap resamples  : {len(result['boot_deltas']):,}",
        f"  Significance level α : {result['alpha']}",
        "",
        f"  Mean AP@0.5 ({baseline_name:<12}) : {mean_base:.4f}",
        f"  Mean AP@0.5 ({method_name:<12}) : {mean_method:.4f}",
        f"  Observed Δ               : {result['delta_obs']:+.4f}",
        "",
        f"  {ci_pct}% Bootstrap CI (percentile) : "
        f"[{result['ci_lo']:+.4f},  {result['ci_hi']:+.4f}]",
        f"  CI excludes 0            : {'YES' if not ci_has_zero else 'NO'}",
        "",
        f"  One-sided p-value (H₀: Δ ≤ 0) : {result['p_value']:.4f}",
        f"  Significant at α={result['alpha']}        : {'YES' if sig else 'NO'}",
        "",
        "  Interpretation:",
    ]

    if sig and not ci_has_zero:
        lines += [
            f"    The improvement from {baseline_name} to {method_name} is",
            f"    statistically significant. The {ci_pct}% CI excludes 0,",
            f"    confirming the Δ = {result['delta_obs']:+.4f} is reliably non-zero.",
        ]
    elif sig:
        lines += [
            f"    p < α but the CI still contains 0 — borderline result.",
            f"    Interpret with caution.",
        ]
    else:
        lines += [
            f"    Improvement is NOT statistically significant at α={result['alpha']}.",
            f"    The CI contains 0.",
        ]

    lines += ["", "=" * 62]
    report = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    return report


def plot_distribution(
    result        : dict,
    baseline_name : str,
    method_name   : str,
    out_path      : Path,
) -> None:
    boot  = result["boot_deltas"]
    lo, hi, obs = result["ci_lo"], result["ci_hi"], result["delta_obs"]
    ci_pct = int((1 - result["alpha"]) * 100)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(boot, bins=80, color="#4878CF", edgecolor="none",
            alpha=0.75, density=True, label="Bootstrap Δ")

    # Shade the CI region
    ci_vals = boot[(boot >= lo) & (boot <= hi)]
    if len(ci_vals):
        ax.hist(ci_vals, bins=80, color="#4878CF", edgecolor="none",
                alpha=0.30, density=True)

    ax.axvline(lo,  color="#E05C2A", linestyle="--", linewidth=1.2,
               label=f"{ci_pct}% CI [{lo:+.4f}, {hi:+.4f}]")
    ax.axvline(hi,  color="#E05C2A", linestyle="--", linewidth=1.2)
    ax.axvline(obs, color="#2CA02C", linewidth=2.0,
               label=f"Observed Δ = {obs:+.4f}")
    ax.axvline(0,   color="black",   linewidth=0.8, linestyle=":")

    ax.set_xlabel(
        f"Δ AP@0.5  ({method_name} − {baseline_name})", fontsize=11
    )
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        f"Paired Bootstrap CI — {baseline_name} vs {method_name}\n"
        f"n={result['n_images']} images, {len(boot):,} resamples, "
        f"p={result['p_value']:.4f}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired bootstrap CI for AP@0.5 differences between methods."
    )
    p.add_argument("--baseline", type=str, default="noisy",
                   help="Baseline method name (must have <name>.csv in --ap_dir)")
    p.add_argument("--method",   type=str, nargs="+", default=["cellpose3"],
                   help="Method(s) to compare against baseline")
    p.add_argument("--ap_dir",   type=Path, default=AP_DIR)
    p.add_argument("--out_dir",  type=Path, default=OUT_DIR)
    p.add_argument("--n_boot",   type=int,  default=10_000)
    p.add_argument("--alpha",    type=float, default=0.05)
    p.add_argument("--seed",     type=int,  default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    baseline_csv = args.ap_dir / f"{args.baseline}.csv"
    if not baseline_csv.exists():
        raise FileNotFoundError(
            f"Baseline CSV not found: {baseline_csv}\n"
            f"Run evaluate.py for '{args.baseline}' first."
        )
    base_dict = load_ap50(baseline_csv)

    summary_rows = []

    for method_name in args.method:
        method_csv = args.ap_dir / f"{method_name}.csv"
        if not method_csv.exists():
            print(f"[skip] {method_csv} not found — skipping {method_name}")
            continue

        print(f"\n[*] Bootstrap: {args.baseline}  vs  {method_name}")
        method_dict = load_ap50(method_csv)
        ap_b, ap_m, common = align_pairs(base_dict, method_dict)
        print(f"    Paired images: {len(common)}")

        result = paired_bootstrap_ci(
            ap_base   = ap_b,
            ap_method = ap_m,
            n_boot    = args.n_boot,
            alpha     = args.alpha,
            seed      = args.seed,
        )

        tag = f"{args.baseline}_vs_{method_name}"

        report = write_report(
            result        = result,
            baseline_name = args.baseline,
            method_name   = method_name,
            mean_base     = float(ap_b.mean()),
            mean_method   = float(ap_m.mean()),
            out_path      = args.out_dir / f"{tag}_report.txt",
        )
        print(report)

        plot_distribution(
            result        = result,
            baseline_name = args.baseline,
            method_name   = method_name,
            out_path      = args.out_dir / f"{tag}_dist.png",
        )
        print(f"    Plot → {args.out_dir / f'{tag}_dist.png'}")

        summary_rows.append(dict(
            baseline     = args.baseline,
            method       = method_name,
            n_images     = result["n_images"],
            mean_baseline= round(float(ap_b.mean()),  4),
            mean_method  = round(float(ap_m.mean()),  4),
            delta_obs    = round(result["delta_obs"], 4),
            ci_lo        = round(result["ci_lo"],     4),
            ci_hi        = round(result["ci_hi"],     4),
            p_value      = round(result["p_value"],   4),
            significant  = result["p_value"] < args.alpha,
        ))

    # Summary CSV
    if summary_rows:
        summary_path = args.out_dir / "bootstrap_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

        ci_pct = int((1 - args.alpha) * 100)
        print(f"\n{'='*68}")
        print(f"  {'Method':<20} {'Δ AP':>8}  {f'{ci_pct}% CI':<26}  {'p':>8}  Sig?")
        print(f"  {'-'*62}")
        for s in summary_rows:
            ci_str = f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]"
            sig    = "✓" if s["significant"] else "✗"
            print(f"  {s['method']:<20} {s['delta_obs']:>+8.4f}  "
                  f"{ci_str:<26}  {s['p_value']:>8.4f}  {sig}")
        print(f"{'='*68}")
        print(f"\nSummary → {summary_path}")
