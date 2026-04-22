"""
src/analysis/scatter_delta_plots.py
---------------------------
Per-image scatter plots and Δ (delta) plots comparing each denoising method
against the noisy baseline and against Cellpose3.

Two figure types are produced for every method M:

  1. Scatter plot  (figures/scatter/<M>_vs_noisy.png)
     X-axis : per-image AP@0.5 for the noisy baseline
     Y-axis : per-image AP@0.5 for method M
     Diagonal = no-change line.  Points above = improvement.
     Coloured by Δ = AP_M − AP_noisy (red-white-blue diverging).

  2. Δ (delta) sorted bar plot  (figures/scatter/<M>_delta_sorted.png)
     Per-image Δ = AP_M − AP_noisy, sorted descending.
     Bars coloured green (improvement) / red (degradation).
     Horizontal line at 0.  95% bootstrap CI band shown as shading.

  3. Scatter vs Cellpose3  (figures/scatter/<M>_vs_cellpose3.png)  [optional]
     Same as (1) but X-axis = Cellpose3 AP@0.5.
     Only produced when cellpose3.csv exists.

  4. Summary multi-panel scatter  (figures/scatter/all_methods_scatter.png)
     All methods on one figure, one subplot per method, scatter vs noisy.

Reads:
    results/ap_scores/<method>.csv   — per-image AP@0.5 from evaluate.py
    results/bootstrap/               — bootstrap CI reports (optional, for shading)

Usage
-----
    # All methods found automatically:
    python src/analysis/scatter_delta_plots.py

    # Specific methods:
    python src/analysis/scatter_delta_plots.py --methods bm3d wiener tv purelet cellpose3

    # Change baseline (default: noisy_poisson):
    python src/analysis/scatter_delta_plots.py --baseline noisy_poisson

    # Also compare against cellpose3:
    python src/analysis/scatter_delta_plots.py --vs_cellpose3

Output
------
    figures/scatter/<method>_vs_<baseline>.png
    figures/scatter/<method>_delta_sorted.png
    figures/scatter/<method>_vs_cellpose3.png   (if --vs_cellpose3)
    figures/scatter/all_methods_scatter.png
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── Default paths ──────────────────────────────────────────────────────────────
AP_DIR       = Path("results/ap_scores")
BOOTSTRAP_DIR= Path("results/bootstrap")
FIGURES_DIR  = Path("figures/scatter")

# Pretty display names (mirrors generate_visuals.py)
PRETTY = {
    "clean"           : "Clean (upper bound)",
    "noisy"           : "Noisy baseline",
    "noisy_poisson"   : "Noisy (Poisson-only)",
    "noisy_cellpose3" : "Noisy (Cellpose3 noise)",
    "cellpose3"       : "Cellpose3 (DL)",
    "wiener"          : "Wiener",
    "tv"              : "Poisson-TV",
    "bm3d"            : "VST + BM3D",
    "purelet"         : "PURE-LET",
}

# Colour cycle for multi-panel figures (one colour per method)
METHOD_COLORS = {
    "bm3d"    : "#2196F3",   # blue
    "wiener"  : "#FF9800",   # orange
    "tv"      : "#4CAF50",   # green
    "purelet" : "#9C27B0",   # purple
    "cellpose3": "#F44336",  # red
}
DEFAULT_COLOR = "#607D8B"


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_ap50(csv_path: Path) -> dict[str, float]:
    """Load per-image AP@0.5 from an evaluate.py CSV. Returns {stem: value}."""
    records = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            # evaluate.py writes "ap@0.50"
            records[row["image"]] = float(row["ap@0.50"])
    return records


def align_to_baseline(
    base: dict[str, float],
    method: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return paired (ap_base, ap_method, stems) for images common to both."""
    common = sorted(set(base) & set(method))
    return (
        np.array([base[s]   for s in common]),
        np.array([method[s] for s in common]),
        common,
    )


def load_bootstrap_ci(baseline: str, method: str) -> tuple[float, float] | None:
    """
    Try to load the 95% CI [lo, hi] for (baseline, method) from bootstrap CSV.
    Returns None if the file / row is missing.
    """
    summary = BOOTSTRAP_DIR / "bootstrap_summary.csv"
    if not summary.exists():
        return None
    with open(summary, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["baseline"] == baseline and row["method"] == method:
                return float(row["ci_lo"]), float(row["ci_hi"])
    return None


def discover_methods(ap_dir: Path, baseline: str) -> list[str]:
    """Return all method names that have an ap_scores CSV, excluding baselines."""
    exclude = {"clean", baseline, "noisy", "noisy_poisson", "noisy_cellpose3",
               "noisy_poisson_summary", "noisy_cellpose3_summary",
               "clean_summary", "bm3d_summary", "baseline_validation"}
    methods = []
    for f in sorted(ap_dir.glob("*.csv")):
        name = f.stem
        if name.endswith("_summary"):
            continue
        if name not in exclude and name != baseline:
            methods.append(name)
    return methods


# ── Individual plot builders ───────────────────────────────────────────────────

def plot_scatter(
    ap_base   : np.ndarray,
    ap_method : np.ndarray,
    stems     : list[str],
    base_label: str,
    method_label: str,
    out_path  : Path,
    color     : str = "#2196F3",
    dpi       : int = 150,
) -> None:
    """
    Scatter: X = baseline AP@0.5, Y = method AP@0.5.
    Points coloured by Δ = ap_method − ap_base.
    """
    delta = ap_method - ap_base
    n = len(stems)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Diverging colormap centred at 0
    vmax = max(abs(delta.min()), abs(delta.max()), 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    sc = ax.scatter(
        ap_base, ap_method,
        c=delta, cmap=cmap, norm=norm,
        s=55, edgecolors="white", linewidths=0.5, zorder=3,
    )

    # Diagonal (no-change) line
    lim = [0, 1]
    ax.plot(lim, lim, color="gray", linewidth=1.0, linestyle="--",
            zorder=1, label="No change (y = x)")

    # Axis labels and formatting
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f"AP@0.5 — {base_label}", fontsize=11)
    ax.set_ylabel(f"AP@0.5 — {method_label}", fontsize=11)
    ax.set_title(
        f"{method_label}  vs.  {base_label}\n"
        f"n={n} images    mean Δ = {delta.mean():+.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Δ AP@0.5 (method − baseline)", fontsize=9)

    # Annotate n_above / n_below
    n_above = int((delta > 0).sum())
    n_below = int((delta < 0).sum())
    ax.text(0.98, 0.04,
            f"↑ improved: {n_above}/{n}  ↓ worse: {n_below}/{n}",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                      ec="gray", alpha=0.85))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Scatter → {out_path}")


def plot_delta_sorted(
    ap_base    : np.ndarray,
    ap_method  : np.ndarray,
    stems      : list[str],
    base_label : str,
    method_label: str,
    out_path   : Path,
    ci         : tuple[float, float] | None = None,
    color      : str = "#2196F3",
    dpi        : int = 150,
) -> None:
    """
    Sorted bar chart of per-image Δ = ap_method − ap_base.
    Green bars = improvement, red = degradation.
    Optional 95% CI shading (horizontal band).
    """
    delta = ap_method - ap_base
    order = np.argsort(delta)[::-1]
    delta_sorted = delta[order]
    stems_sorted = [stems[i] for i in order]
    n = len(delta_sorted)

    bar_colors = ["#4CAF50" if d >= 0 else "#F44336" for d in delta_sorted]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.18), 5))

    ax.bar(range(n), delta_sorted, color=bar_colors, edgecolor="none",
           width=0.85, zorder=2)
    ax.axhline(0, color="black", linewidth=0.8, zorder=3)
    ax.axhline(delta.mean(), color=color, linewidth=1.6, linestyle="--",
               zorder=3, label=f"Mean Δ = {delta.mean():+.4f}")

    # Bootstrap CI shading
    if ci is not None:
        ax.axhspan(ci[0], ci[1], alpha=0.15, color=color, zorder=1,
                   label=f"95% Bootstrap CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    # Reference lines at proposal prediction bounds (BM3D 0.56–0.60 → Δ ~ 0.10–0.14)
    # Omit for cleanliness — only mean and CI are drawn.

    ax.set_xlabel("Image (sorted by Δ AP@0.5, descending)", fontsize=10)
    ax.set_ylabel("Δ AP@0.5  (method − baseline)", fontsize=10)
    ax.set_title(
        f"Per-image Δ AP@0.5:  {method_label}  −  {base_label}\n"
        f"n={n}    mean={delta.mean():+.4f}    "
        f"n_positive={int((delta>0).sum())}    n_negative={int((delta<0).sum())}",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlim(-0.5, n - 0.5)
    y_abs_max = max(abs(delta_sorted.min()), abs(delta_sorted.max()), 0.05)
    ax.set_ylim(-y_abs_max * 1.25, y_abs_max * 1.25)

    # Tick labels: only show every k-th stem to avoid clutter
    step = max(1, n // 20)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([stems_sorted[i] for i in range(0, n, step)],
                       rotation=45, ha="right", fontsize=7)

    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Legend patches for colours
    legend_patches = [
        Patch(facecolor="#4CAF50", label=f"Improved ({int((delta>0).sum())})"),
        Patch(facecolor="#F44336", label=f"Degraded ({int((delta<0).sum())})"),
    ]
    leg2 = ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.add_artist(leg2)
    # Re-add main legend
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Δ-sorted bar → {out_path}")


def plot_all_methods_scatter(
    base_dict    : dict[str, float],
    method_dicts : dict[str, dict[str, float]],
    base_label   : str,
    out_path     : Path,
    dpi          : int = 150,
) -> None:
    """
    One subplot per method, scatter vs baseline.  All on one figure.
    """
    methods = list(method_dicts.keys())
    n_methods = len(methods)
    if n_methods == 0:
        return

    ncols = min(3, n_methods)
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5 * nrows),
                             squeeze=False)
    fig.suptitle(
        f"Per-image AP@0.5: all methods vs.  {base_label}",
        fontsize=13, fontweight="bold",
    )

    for idx, method in enumerate(methods):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        ap_b, ap_m, stems = align_to_baseline(base_dict, method_dicts[method])
        delta = ap_m - ap_b
        color = METHOD_COLORS.get(method, DEFAULT_COLOR)
        label = PRETTY.get(method, method)

        vmax = max(abs(delta.min()), abs(delta.max()), 0.05)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap("RdBu_r")

        ax.scatter(ap_b, ap_m, c=delta, cmap=cmap, norm=norm,
                   s=40, edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8,
                linestyle="--", zorder=1)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\nmean Δ = {delta.mean():+.4f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel(f"AP@0.5 ({base_label})", fontsize=8)
        ax.set_ylabel(f"AP@0.5 ({label})", fontsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.4)

        n_above = int((delta > 0).sum())
        n = len(stems)
        ax.text(0.97, 0.04, f"↑{n_above}/{n}", transform=ax.transAxes,
                fontsize=8, ha="right", va="bottom", color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                          ec="gray", alpha=0.8))

    # Hide unused subplots
    for idx in range(n_methods, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  All-methods scatter → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-image scatter and Δ plots for AP@0.5 comparison."
    )
    p.add_argument("--baseline",    type=str, default="noisy_poisson",
                   help="Name of the baseline method CSV in results/ap_scores/")
    p.add_argument("--methods",     nargs="+", default=None,
                   help="Methods to compare (default: all found automatically)")
    p.add_argument("--ap_dir",      type=Path, default=AP_DIR)
    p.add_argument("--out_dir",     type=Path, default=FIGURES_DIR)
    p.add_argument("--vs_cellpose3",action="store_true",
                   help="Also produce scatter of each method vs. Cellpose3")
    p.add_argument("--dpi",         type=int,  default=150)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Load baseline ────────────────────────────────────────────────────────
    baseline_csv = args.ap_dir / f"{args.baseline}.csv"
    if not baseline_csv.exists():
        # Fall back to "noisy" if the exact name isn't found
        fallback = args.ap_dir / "noisy.csv"
        if fallback.exists():
            print(f"[warn] {baseline_csv} not found, falling back to {fallback}")
            baseline_csv = fallback
            args.baseline = "noisy"
        else:
            raise FileNotFoundError(
                f"Baseline CSV not found: {baseline_csv}\n"
                f"Run evaluate.py for '{args.baseline}' first."
            )
    base_dict   = load_ap50(baseline_csv)
    base_label  = PRETTY.get(args.baseline, args.baseline)
    print(f"Baseline: {base_label}  ({len(base_dict)} images)\n")

    # ── Discover methods ─────────────────────────────────────────────────────
    methods = args.methods if args.methods else discover_methods(args.ap_dir, args.baseline)
    if not methods:
        raise SystemExit(
            f"No method CSVs found in {args.ap_dir} (excluding baseline).\n"
            f"Run evaluate.py for each denoising method first."
        )
    print(f"Methods: {methods}\n")

    # ── Load Cellpose3 dict for optional vs-cellpose3 plots ──────────────────
    c3_dict: dict[str, float] | None = None
    if args.vs_cellpose3:
        c3_csv = args.ap_dir / "cellpose3.csv"
        if c3_csv.exists():
            c3_dict = load_ap50(c3_csv)
            print(f"Cellpose3 reference loaded ({len(c3_dict)} images)")
        else:
            print("[warn] --vs_cellpose3 requested but cellpose3.csv not found — skipping")

    # ── Per-method plots ─────────────────────────────────────────────────────
    method_dicts: dict[str, dict[str, float]] = {}
    for method in methods:
        csv_path = args.ap_dir / f"{method}.csv"
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found")
            continue

        m_dict = load_ap50(csv_path)
        method_dicts[method] = m_dict
        label  = PRETTY.get(method, method)
        color  = METHOD_COLORS.get(method, DEFAULT_COLOR)
        ap_b, ap_m, stems = align_to_baseline(base_dict, m_dict)

        print(f"[*] {label}  (n={len(stems)}  mean AP = {ap_m.mean():.4f}  "
              f"mean Δ = {(ap_m - ap_b).mean():+.4f})")

        # 1. Scatter vs baseline
        plot_scatter(
            ap_b, ap_m, stems,
            base_label=base_label,
            method_label=label,
            out_path=args.out_dir / f"{method}_vs_{args.baseline}.png",
            color=color,
            dpi=args.dpi,
        )

        # 2. Δ sorted bar
        ci = load_bootstrap_ci(args.baseline, method)
        plot_delta_sorted(
            ap_b, ap_m, stems,
            base_label=base_label,
            method_label=label,
            out_path=args.out_dir / f"{method}_delta_sorted.png",
            ci=ci,
            color=color,
            dpi=args.dpi,
        )

        # 3. Scatter vs Cellpose3
        if c3_dict is not None:
            ap_c3, ap_m2, stems2 = align_to_baseline(c3_dict, m_dict)
            if len(stems2) > 0:
                plot_scatter(
                    ap_c3, ap_m2, stems2,
                    base_label=PRETTY.get("cellpose3", "Cellpose3"),
                    method_label=label,
                    out_path=args.out_dir / f"{method}_vs_cellpose3.png",
                    color=color,
                    dpi=args.dpi,
                )

        print()

    # ── Multi-panel all-methods scatter ──────────────────────────────────────
    if len(method_dicts) >= 2:
        plot_all_methods_scatter(
            base_dict=base_dict,
            method_dicts=method_dicts,
            base_label=base_label,
            out_path=args.out_dir / "all_methods_scatter.png",
            dpi=args.dpi,
        )

    print(f"\n[✓] All scatter/delta figures saved under {args.out_dir}/")
