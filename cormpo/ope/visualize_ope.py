import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def plot_value_estimates(csv_path="ope_value.csv", save_path="ope_value_estimates.png"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["model"] = df["model"].str.strip()

    ope_methods = {
        "On-Policy": ("mc_value_mean",  "mc_value_std"),
        "DM":        ("ope_dm_mean",    "ope_dm_std"),
        "TIS":       ("ope_tis_mean",   "ope_tis_std"),
        "PDIS":      ("ope_pdis_mean",  "ope_pdis_std"),
        "DR":        ("ope_dr_mean",    "ope_dr_std"),
    }

    model_order = ["mbpo", "gormpo-kde", "gormpo-vae", "gormpo-realnvp", "gormpo-diffusion", "gormpo-neuralode"]
    models = [m for m in model_order if m in df["model"].tolist()]
    n_models = len(models)
    method_labels = list(ope_methods.keys())
    n_methods = len(method_labels)
    x = np.arange(n_methods)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 9))

    for i, model in enumerate(models):
        row = df[df["model"] == model].iloc[0]
        means = [row[ope_methods[m][0]] for m in method_labels]
        stds  = [row[ope_methods[m][1]] for m in method_labels]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=model,
               capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=22)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("Value Estimate", fontsize=22)
    ax.set_xlabel("OPE Method", fontsize=22)
    ax.set_title("OPE Value Estimates (mean ± std)", fontsize=24)

    ax.legend(fontsize=22,
              loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, borderaxespad=0.)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_policy_mismatch_vs_rmse(ope_csv="ope_value.csv", rmse_csv="rmse.csv", save_path="ope_mismatch_vs_rmse.png"):
    ope_df = pd.read_csv(ope_csv)
    ope_df.columns = ope_df.columns.str.strip()
    ope_df["model"] = ope_df["model"].str.strip()

    rmse_df = pd.read_csv(rmse_csv)
    rmse_df.columns = rmse_df.columns.str.strip()
    rmse_df["model"] = rmse_df["model"].str.strip()

    merged = ope_df[["model", "log25_iw_max_mean"]].merge(rmse_df, on="model")
    merged = merged.sort_values("log25_iw_max_mean").reset_index(drop=True)

    ope_methods = ["dm", "tis", "pdis", "dr"]
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(ope_methods)))

    x_pos = np.arange(len(merged))
    x_labels = [
        f"{row['model']}\n({row['log25_iw_max_mean']:.2f})"
        for _, row in merged.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(14, 9))

    for method, color in zip(ope_methods, colors):
        ax.scatter(
            x_pos,
            merged[method],
            label=method.upper(),
            color=color,
            s=100,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=20, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=20)
    ax.set_xlabel("log25 policy mismatch", fontsize=22)
    ax.set_ylabel("RMSE", fontsize=22)
    ax.set_title("Policy Mismatch vs. OPE RMSE", fontsize=22)
    ax.legend( fontsize=20, loc="upper center", bbox_to_anchor=(0.4, 1.0), ncol=4, borderaxespad=0.)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()


def plot_rmse_by_policy(rmse_csv="rmse.csv", save_path="ope_rmse_by_policy.png"):
    rmse_df = pd.read_csv(rmse_csv)
    rmse_df.columns = rmse_df.columns.str.strip()
    rmse_df["model"] = rmse_df["model"].str.strip()

    ope_methods = ["dm", "tis", "pdis", "dr"]
    method_labels = [m.upper() for m in ope_methods]

    policies = rmse_df["model"].tolist()
    n_policies = len(policies)
    n_methods = len(ope_methods)
    x = np.arange(n_policies)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(14, 9))

    for i, (method, label) in enumerate(zip(ope_methods, method_labels)):
        values = rmse_df[method].tolist()
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=18, rotation=30, ha="right")
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("RMSE", fontsize=22)
    ax.set_xlabel("Policy", fontsize=22)
    ax.set_title("OPE RMSE per Policy", fontsize=24)
    ax.legend( fontsize=20, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, borderaxespad=0.)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()


def total_rmse_per_policy_table(rmse_csv="rmse.csv"):
    rmse_df = pd.read_csv(rmse_csv)
    rmse_df.columns = rmse_df.columns.str.strip()
    rmse_df["model"] = rmse_df["model"].str.strip()

    ope_methods = ["dm", "tis", "pdis", "dr"]
    rmse_df["total_rmse"] = rmse_df[ope_methods].sum(axis=1)

    table = rmse_df[["model", "total_rmse"]].rename(columns={"model": "Policy", "total_rmse": "Total RMSE"}).set_index("Policy")
    print(table.to_string())
    return table


def plot_total_rmse_vs_mismatch(ope_csv="ope_value.csv", rmse_csv="rmse.csv", save_path="ope_total_rmse_vs_mismatch.png"):
    ope_df = pd.read_csv(ope_csv)
    ope_df.columns = ope_df.columns.str.strip()
    ope_df["model"] = ope_df["model"].str.strip()

    rmse_df = pd.read_csv(rmse_csv)
    rmse_df.columns = rmse_df.columns.str.strip()
    rmse_df["model"] = rmse_df["model"].str.strip()

    ope_methods = ["dm", "tis", "pdis", "dr"]
    rmse_df["total_rmse"] = rmse_df[ope_methods].sum(axis=1)

    merged = ope_df[["model", "log25_iw_max_mean"]].merge(rmse_df[["model", "total_rmse"]], on="model")
    merged = merged.sort_values("log25_iw_max_mean").reset_index(drop=True)

    model_colors = {
        "mbpo":              "blue",
        "gormpo-kde":        "orange",
        "gormpo-vae":        "green",
        "gormpo-realnvp":    "red",
        "gormpo-diffusion":  "purple",
        "gormpo-neuralode":  "brown",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in merged.iterrows():
        color = model_colors.get(row["model"], "gray")
        ax.scatter(row["log25_iw_max_mean"], row["total_rmse"],
                   s=100, color=color, edgecolors="k", linewidths=0.5, alpha=0.85)

    for _, row in merged.iterrows():
        ax.annotate(row["model"], (row["log25_iw_max_mean"], row["total_rmse"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=16)

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("log25 Policy Mismatch", fontsize=14)
    ax.set_ylabel("Total RMSE", fontsize=14)
    ax.set_title("Total OPE RMSE vs. Policy Mismatch", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()


def plot_mc_value_vs_mismatch(ope_csv="ope_value.csv", save_path="ope_mc_value_vs_mismatch.png"):
    ope_df = pd.read_csv(ope_csv)
    ope_df.columns = ope_df.columns.str.strip()
    ope_df["model"] = ope_df["model"].str.strip()

    merged = ope_df[["model", "log25_iw_max_mean", "mc_value_mean"]].copy()
    merged = merged.sort_values("log25_iw_max_mean").reset_index(drop=True)

    model_colors = {
        "mbpo":              "blue",
        "gormpo-kde":        "orange",
        "gormpo-vae":        "green",
        "gormpo-realnvp":    "red",
        "gormpo-diffusion":  "purple",
        "gormpo-neuralode":  "brown",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in merged.iterrows():
        color = model_colors.get(row["model"], "gray")
        ax.scatter(row["log25_iw_max_mean"], row["mc_value_mean"],
                   s=100, color=color, edgecolors="k", linewidths=0.5, alpha=0.85)

    for _, row in merged.iterrows():
        ax.annotate(row["model"], (row["log25_iw_max_mean"], row["mc_value_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=16)

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("log25 Policy Mismatch", fontsize=22)
    ax.set_ylabel("On-Policy Value", fontsize=22)
    ax.set_title("On-Policy Value vs. Policy Mismatch", fontsize=22)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.show()


def plot_rmse_and_value_vs_optimism_penalty(
    ope_csv="ope_value.csv",
    rmse_csv="rmse.csv",
    save_path="ope_optimism_penalty_analysis.png",
):
    OPTIMISM = {
        "gormpo-neuralode":  1.0,
        "gormpo-vae":        1.0,
        "gormpo-realnvp":    0.5,
        "gormpo-kde":        0.2,
        "gormpo-diffusion":  0.0,
    }
    PENALTY_VAR = {
        "gormpo-realnvp":   1.00,
        "gormpo-kde":       0.40,
        "gormpo-vae":       0.03,
        "gormpo-neuralode": 0.07,
        "gormpo-diffusion": 0.02,
    }

    ope_df = pd.read_csv(ope_csv)
    ope_df.columns = ope_df.columns.str.strip()
    ope_df["model"] = ope_df["model"].str.strip()

    rmse_df = pd.read_csv(rmse_csv)
    rmse_df.columns = rmse_df.columns.str.strip()
    rmse_df["model"] = rmse_df["model"].str.strip()

    ope_methods = ["dm", "tis", "pdis", "dr"]
    rmse_df["total_rmse"] = rmse_df[ope_methods].sum(axis=1)

    merged = ope_df[["model", "mc_value_mean"]].merge(
        rmse_df[["model", "total_rmse"]], on="model"
    )
    merged = merged[merged["model"].isin(OPTIMISM)].copy()
    merged["optimism"] = merged["model"].map(OPTIMISM)
    merged["penalty_var"] = merged["model"].map(PENALTY_VAR)

    model_colors = {
        "gormpo-kde":        "orange",
        "gormpo-vae":        "green",
        "gormpo-realnvp":    "red",
        "gormpo-diffusion":  "purple",
        "gormpo-neuralode":  "brown",
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    plot_specs = [
        (axes[0], "optimism",    "Optimism",        "total_rmse", "Total RMSE",      "RMSE vs. Optimism"),
        (axes[1], "penalty_var", "Penalty Variance", "total_rmse", "Total RMSE",      "RMSE vs. Penalty Variance"),
        (axes[2], "optimism",    "Optimism",        "mc_value_mean", "On-Policy Value", "On-Policy Value vs. Optimism"),
    ]

    for ax, x_col, x_label, y_col, y_label, title in plot_specs:
        for _, row in merged.iterrows():
            color = model_colors.get(row["model"], "gray")
            ax.scatter(row[x_col], row[y_col],
                       s=120, color=color, edgecolors="k", linewidths=0.6, alpha=0.9)
            ax.annotate(row["model"], (row[x_col], row[y_col]),
                        textcoords="offset points", xytext=(6, 4), fontsize=13)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_title(title, fontsize=17)
        ax.tick_params(labelsize=13)
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_mismatch_vs_optimism_and_penalty_var(
    ope_csv="ope_value.csv",
    save_path="ope_mismatch_vs_optimism_penaltyvar.png",
):
    OPTIMISM = {
        "gormpo-neuralode":  1.0,
        "gormpo-vae":        1.0,
        "gormpo-realnvp":    0.5,
        "gormpo-kde":        0.2,
        "gormpo-diffusion":  0.0,
    }
    PENALTY_VAR = {
        "gormpo-realnvp":   1.00,
        "gormpo-kde":       0.40,
        "gormpo-vae":       0.03,
        "gormpo-neuralode": 0.07,
        "gormpo-diffusion": 0.02,
    }

    ope_df = pd.read_csv(ope_csv)
    ope_df.columns = ope_df.columns.str.strip()
    ope_df["model"] = ope_df["model"].str.strip()

    merged = ope_df[["model", "log25_iw_max_mean"]].copy()
    merged = merged[merged["model"].isin(OPTIMISM)].copy()
    merged["optimism"] = merged["model"].map(OPTIMISM)
    merged["penalty_var"] = merged["model"].map(PENALTY_VAR)

    model_colors = {
        "gormpo-kde":        "orange",
        "gormpo-vae":        "green",
        "gormpo-realnvp":    "red",
        "gormpo-diffusion":  "purple",
        "gormpo-neuralode":  "brown",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_specs = [
        (axes[0], "optimism",    "Optimism",         "Policy Mismatch vs. Optimism"),
        (axes[1], "penalty_var", "Penalty Variance",  "Policy Mismatch vs. Penalty Variance"),
    ]

    for ax, x_col, x_label, title in plot_specs:
        for _, row in merged.iterrows():
            color = model_colors.get(row["model"], "gray")
            ax.scatter(row[x_col], row["log25_iw_max_mean"],
                       s=120, color=color, edgecolors="k", linewidths=0.6, alpha=0.9)
            ax.annotate(row["model"], (row[x_col], row["log25_iw_max_mean"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=13)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel("log25 Policy Mismatch", fontsize=16)
        ax.set_title(title, fontsize=17)
        ax.tick_params(labelsize=13)
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_value_vs_penalty_var_and_optimism(
    ope_csv="ope_value.csv",
    save_path="ope_penalty_variance_vs_value.png",
):
    OPTIMISM = {
        "gormpo-neuralode":  1.0,
        "gormpo-vae":        1.0,
        "gormpo-realnvp":    0.5,
        "gormpo-kde":        0.2,
        "gormpo-diffusion":  0.0,
    }
    PENALTY_VAR = {
        "gormpo-realnvp":   1.00,
        "gormpo-kde":       0.40,
        "gormpo-vae":       0.03,
        "gormpo-neuralode": 0.07,
        "gormpo-diffusion": 0.02,
    }

    ope_df = pd.read_csv(ope_csv)
    ope_df.columns = ope_df.columns.str.strip()
    ope_df["model"] = ope_df["model"].str.strip()

    merged = ope_df[["model", "mc_value_mean"]].copy()
    merged = merged[merged["model"].isin(PENALTY_VAR)].copy()
    merged["penalty_var"] = merged["model"].map(PENALTY_VAR)
    merged["optimism"] = merged["model"].map(OPTIMISM)

    model_colors = {
        "gormpo-kde":        "orange",
        "gormpo-vae":        "green",
        "gormpo-realnvp":    "red",
        "gormpo-diffusion":  "purple",
        "gormpo-neuralode":  "brown",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_specs = [
        (axes[0], "penalty_var", "Penalty Variance", "On-Policy Value vs. Penalty Variance"),
        (axes[1], "optimism",    "Optimism",          "On-Policy Value vs. Optimism"),
    ]

    for ax, x_col, x_label, title in plot_specs:
        for _, row in merged.iterrows():
            color = model_colors.get(row["model"], "gray")
            ax.scatter(row[x_col], row["mc_value_mean"],
                       s=120, color=color, edgecolors="k", linewidths=0.6, alpha=0.9)
            ax.annotate(row["model"], (row[x_col], row["mc_value_mean"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=13)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel("On-Policy Value", fontsize=16)
        ax.set_title(title, fontsize=17)
        ax.tick_params(labelsize=13)
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def total_rmse_table(rmse_csv="rmse.csv"):
    rmse_df = pd.read_csv(rmse_csv)
    rmse_df.columns = rmse_df.columns.str.strip()
    rmse_df["model"] = rmse_df["model"].str.strip()

    ope_methods = ["dm", "tis", "pdis", "dr"]
    total_rmse = {m.upper(): round(rmse_df[m].sum(), 6) for m in ope_methods}

    table = pd.DataFrame(list(total_rmse.items()), columns=["OPE Method", "Total RMSE"]).set_index("OPE Method")
    print(table.to_string())
    return table


def spearman_correlation_table(csv_path="ope_value.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["model"] = df["model"].str.strip()

    ground_truth = df["mc_value_mean"]
    ope_methods = {
        "DM":   "ope_dm_mean",
        "TIS":  "ope_tis_mean",
        "PDIS": "ope_pdis_mean",
        "DR":   "ope_dr_mean",
    }

    results = []
    for label, col in ope_methods.items():
        corr, pval = spearmanr(ground_truth, df[col])
        results.append({"Method": label, "Spearman Correlation": round(corr, 4), "p-value": round(pval, 4)})

    table = pd.DataFrame(results).set_index("Method")
    print(table.to_string())
    return table


if __name__ == "__main__":
    plot_value_estimates()
    plot_policy_mismatch_vs_rmse()
    plot_rmse_by_policy()
    total_rmse_per_policy_table()
    plot_total_rmse_vs_mismatch()
    plot_mc_value_vs_mismatch()
    total_rmse_table()
    spearman_correlation_table()
    plot_rmse_and_value_vs_optimism_penalty()
    plot_value_vs_penalty_var_and_optimism()
    plot_mismatch_vs_optimism_and_penalty_var()
