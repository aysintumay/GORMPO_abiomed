"""
Plot sensitivity analysis across all models from saved CSVs.

Each model produces results/hyperparameter/{algo_name}_sensitivity.csv via tune_gormpo.py.
This script reads all available CSVs and overlays them in one figure with 3 subplots
(Return, ACP, WS) per reward_penalty_coef value.

Usage:
    python plot_sensitivity.py [--results_dir results/hyperparameter] [--out figures/sensitivity_all_models.png]
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLORS = {
    "mbpo":              "black",
    "GORMPO-KDE":        "tab:blue",
    "GORMPO-VAE":        "tab:orange",
    "GORMPO-RealNVP":    "tab:green",
    "GORMPO-NeuralODE":  "tab:purple",
    "GORMPO-Diffusion":  "tab:red",
}
MODEL_MARKERS = {
    "mbpo":              "s",
    "GORMPO-KDE":        "o",
    "GORMPO-VAE":        "^",
    "GORMPO-RealNVP":    "D",
    "GORMPO-NeuralODE":  "v",
    "GORMPO-Diffusion":  "P",
}


def plot_metric(ax, df, x_col, mean_col, ci_col, model, ylabel, ylim=None):
    color  = MODEL_COLORS.get(model, None)
    marker = MODEL_MARKERS.get(model, "o")
    x  = df[x_col].values
    y  = df[mean_col].values
    ci = df[ci_col].values
    ax.plot(x, y, marker=marker, linewidth=2, label=model, color=color)
    ax.fill_between(x, y - ci, y + ci, alpha=0.2, color=color)
    ax.set_xlabel("Reward Penalty Coefficient", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/hyperparameter")
    parser.add_argument("--out", default="figures/sensitivity_all_models.png")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.results_dir, "*_sensitivity.csv")))
    if not csv_files:
        print(f"No sensitivity CSVs found in {args.results_dir}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    ax_reward, ax_acp, ax_ws = axes

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        model = df["model"].iloc[0]
        df = df.sort_values("reward_penalty_coef")

        plot_metric(ax_reward, df, "reward_penalty_coef", "mean_reward", "ci95_reward",
                    model, "Episode Return", ylim=(0, 1.5))
        plot_metric(ax_acp,    df, "reward_penalty_coef", "mean_acp",    "ci95_acp",
                    model, "ACP Score")
        plot_metric(ax_ws,     df, "reward_penalty_coef", "mean_ws",     "ci95_ws",
                    model, "Weaning Score (gradient)")

    for ax, title in zip(axes, ["Return", "ACP", "Weaning Score"]):
        ax.set_title(f"Sensitivity: {title} vs Penalty Coef", fontsize=13)
        ax.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Combined sensitivity plot saved to {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
