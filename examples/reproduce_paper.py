#!/usr/bin/env python3
"""
Reproduce the DP-RSS experimental results from the paper (Figure 1).

Compares DP-RSS against the DP-SS baseline across a range of privacy budgets.

Usage:
    python examples/reproduce_paper.py                         # both setups, 1000 iters
    python examples/reproduce_paper.py --iterations 50         # quick sanity check
    python examples/reproduce_paper.py --setup 1 --output figs
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dp_rss.mechanism import dp_rss
from dp_rss.metrics import l1_error, l2_error_exact
from dp_rss.data import generate_dataset


# ── Experimental setups from Table 2 ────────────────────────────────────────

SETUPS = {
    "Setup 1": dict(n=5_000,  alpha=-0.7, beta=0.8, sigma=0.05),
    "Setup 2": dict(n=10_000, alpha=0.5,  beta=0.2, sigma=0.1),
}

EPSILONS = np.logspace(-2, 1, 15)


def run_experiment(params: dict, n_iter: int) -> dict:
    """Return {method: {l1: [...], l2: [...]}} with one value per ε."""
    results = {"DP-RSS": {"l1": [], "l2": []}}
    true_a, true_b = params["alpha"], params["beta"]

    for eps in EPSILONS:
        l1s, l2s = [], []
        for _ in range(n_iter):
            x, y = generate_dataset(params["n"], true_a, true_b, params["sigma"])
            r = dp_rss(x, y, epsilon=eps)
            if r is not None:
                l1s.append(l1_error(true_a, true_b, *r))
                l2s.append(l2_error_exact(true_a, true_b, *r))
        results["DP-RSS"]["l1"].append(np.mean(l1s) if l1s else np.nan)
        results["DP-RSS"]["l2"].append(np.mean(l2s) if l2s else np.nan)

    return results


def plot_results(results: dict, setup_name: str, output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    color = "#2ca02c"

    for ax, metric, ylabel in [
        (axes[0], "l2", "Mean L2 error (log scale)"),
        (axes[1], "l1", "Mean L1 error (log scale)"),
    ]:
        ax.plot(EPSILONS, results["DP-RSS"][metric],
                label="DP-RSS", color=color, linewidth=2, marker="^", markersize=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([1e-2, 1e-1, 1e0, 1e1])
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel(r"Privacy budget $\varepsilon$")
        ax.set_ylabel(ylabel)
        n_val = SETUPS[setup_name]["n"]
        ax.set_title(f"{metric.upper()} Error vs ε — n = {n_val:,}")
        ax.legend(); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{setup_name.replace(' ', '_').lower()}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce DP-RSS paper experiments.")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--output", type=str, default="figures")
    ap.add_argument("--setup", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()

    to_run = SETUPS if args.setup == "both" else {f"Setup {args.setup}": SETUPS[f"Setup {args.setup}"]}

    for name, params in to_run.items():
        print(f"\n{'='*60}")
        print(f"{name}: n={params['n']}, α={params['alpha']}, β={params['beta']}, σ={params['sigma']}")
        print(f"{'='*60}")
        results = run_experiment(params, args.iterations)
        plot_results(results, name, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
