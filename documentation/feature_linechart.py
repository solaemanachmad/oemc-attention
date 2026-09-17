import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Input CSV paths — edit if your paths differ
# ------------------------------------------------------------------ #
GAZECOM_CSV = r"results/ablation/conv_attention/features/feature_ablation_gazecom_summary.csv"
HMR_CSV     = r"results/ablation/conv_attention/features/feature_ablation_hmr_summary.csv"

OUT_PNG = "fig4_feature_ablation_line.png"
OUT_PDF = "fig4_feature_ablation_line.pdf"

# Progressive order (baseline -> full)
TAG_ORDER  = ["sp_dir", "sp_dir_std", "sp_dir_dis", "sp_dir_std_dis"]
TAG_LABELS = ["Sp+Dir", "Sp+Dir\n+Std", "Sp+Dir\n+Dis", "Sp+Dir+Std\n+Dis"]

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_ordered(csv_path):
    """
    Load a feature_ablation summary CSV and return rows in TAG_ORDER.
    Handles the tag column being named either 'feature_tag' or,
    inconsistently, 'Unnamed: 0' (seen in some HMR exports).
    """
    df = pd.read_csv(csv_path)

    tag_col = "feature_tag" if "feature_tag" in df.columns else df.columns[0]
    df = df.set_index(tag_col)

    missing = [t for t in TAG_ORDER if t not in df.index]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected variant(s): {missing}. "
            f"Found: {df.index.tolist()}"
        )

    df = df.loc[TAG_ORDER]
    return {
        "macro_sample": (df["F1_avg"] * 100).tolist(),
        "macro_event":  (df["ev_F1_avg"] * 100).tolist(),
        "sp_sample":    (df["F1_Pursuit"] * 100).tolist(),
        "sp_event":     (df["ev_F1_Pursuit"] * 100).tolist(),
    }


data = {
    "(a) GazeCom (250 Hz)": load_ordered(GAZECOM_CSV),
    "(b) HMR (200 Hz)":     load_ordered(HMR_CSV),
}

COLOR_MACRO = "#2a78d6"
COLOR_SP    = "#d95926"

x = np.arange(len(TAG_LABELS))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))

for ax, (title, d) in zip(axes, data.items()):
    ax.plot(x, d["macro_sample"], marker="o", markersize=4, linewidth=1.4,
             color=COLOR_MACRO, label="F1-macro, sample-level")
    ax.plot(x, d["macro_event"], marker="o", markersize=4, linewidth=1.4,
             linestyle="--", color=COLOR_MACRO, alpha=0.75, label="F1-macro, event-level")
    ax.plot(x, d["sp_sample"], marker="s", markersize=4, linewidth=1.4,
             color=COLOR_SP, label="F1 (SP), sample-level")
    ax.plot(x, d["sp_event"], marker="s", markersize=4, linewidth=1.4,
             linestyle="--", color=COLOR_SP, alpha=0.75, label="F1 (SP), event-level")

    # Highlight the full-feature point (last x position)
    for series in (d["macro_sample"], d["macro_event"], d["sp_sample"], d["sp_event"]):
        ax.plot(x[-1], series[-1], marker="o", markersize=8, markerfacecolor="none",
                 markeredgecolor="black", markeredgewidth=1.2, zorder=5)

    ax.set_title(title, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(TAG_LABELS, fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_xlim(-0.3, len(TAG_LABELS) - 0.7)

axes[0].set_ylabel("F1-score (%)")

fig.subplots_adjust(top=0.80, bottom=0.15, left=0.08, right=0.98, wspace=0.18)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.97), fontsize=7.5, columnspacing=1.2,
           handletextpad=0.4, borderaxespad=0.0)

fig.savefig(OUT_PNG, dpi=300, pad_inches=0.05)
fig.savefig(OUT_PDF, pad_inches=0.05)
print(f"Saved {OUT_PNG} and {OUT_PDF}")