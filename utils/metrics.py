"""
utils/metrics.py

Publication-ready evaluation figures — IEEE / Elsevier style.

Design system
─────────────
Palette   : colorblind-friendly (Wong 2011, 8-colour)
            BLUE   #0072B2   ORANGE #E69F00   GREEN  #009E73
            RED    #D55E00   PURPLE #CC79A7   TEAL   #56B4E9
            YELLOW #F0E442   BLACK  #000000
Font      : sans-serif throughout (Arial/Helvetica) — clean, consistent,
            works well at small sizes for publication figures
DPI       : 300 PNG + vector PDF (same call, two outputs)
Size      : single-column figure 3.5 × 3.0 in, double-column 7.0 × 3.0 in
Line width: 1.5 pt for data, 0.8 pt for frame
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
import wandb

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from torchmetrics import F1Score

from utils.logger import logger
from utils.helpers import set_folder_path, set_prefix


# ═══════════════════════════════════════════════════════════════════ #
# Global style  (applied once on import)
# ═══════════════════════════════════════════════════════════════════ #

# Wong (2011) colorblind-safe palette — 8 colours
_PALETTE = [
    "#0072B2",   # 0 blue
    "#E69F00",   # 1 orange
    "#009E73",   # 2 green
    "#D55E00",   # 3 vermillion
    "#CC79A7",   # 4 purple
    "#56B4E9",   # 5 sky-blue
    "#F0E442",   # 6 yellow
    "#000000",   # 7 black
]

_RC = {
    # Font
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.title_fontsize": 7,
    # Lines
    "axes.linewidth":     0.8,
    "lines.linewidth":    1.5,
    "patch.linewidth":    0.6,
    # Grid
    "axes.grid":          False,
    # Spines
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    # Save
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    # PDF backend — embed fonts
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
}

mpl.rcParams.update(_RC)


# figure sizes (inches) — IEEE column widths
_FIG_SINGLE = (3.5, 3.0)   # single column
_FIG_DOUBLE = (7.0, 3.8)   # double column (two square panels side by side)
_FIG_ROC    = (3.5, 3.2)   # ROC / PR curve — square-ish


def _save(fig, path_no_ext: str, wandb_run=None, wandb_key: str = None,
          caption: str = ""):
    """Save PNG (300 dpi) + PDF (vector) and optionally log to WandB."""
    png_path = path_no_ext + ".png"
    pdf_path = path_no_ext + ".pdf"
    fig.savefig(png_path, dpi=300, format="png")
    fig.savefig(pdf_path, format="pdf")
    if wandb_run is not None and wandb_key:
        wandb_run.log({wandb_key: wandb.Image(png_path, caption=caption)})
    plt.close(fig)
    return png_path          # primary path returned for artifact upload


# ═══════════════════════════════════════════════════════════════════ #
# Event-level aggregation
# ═══════════════════════════════════════════════════════════════════ #

def _get_event(preds, labels):
    preds_np  = preds.cpu().numpy()  if torch.is_tensor(preds)  else np.array(preds)
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    event_preds, event_gt = [], []
    i = 0
    while i < len(labels_np):
        g_0 = int(labels_np[i])
        ini = i
        while i < len(labels_np) and int(labels_np[i]) == g_0:
            i += 1
        if ini == i:
            continue
        majority = np.bincount(preds_np[ini:i].astype(int)).argmax()
        event_preds.append(majority)
        event_gt.append(g_0)

    return np.array(event_preds), np.array(event_gt)


# ═══════════════════════════════════════════════════════════════════ #
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════ #

def compute_roc_auc(total_label, total_probs, n_classes):
    y_true  = total_label.cpu().numpy()
    y_score = total_probs.cpu().numpy()
    y_bin   = label_binarize(y_true, classes=np.arange(n_classes))
    return (
        roc_auc_score(y_bin, y_score, average="micro"),
        roc_auc_score(y_bin, y_score, average="macro"),
    )


def evaluate_f1(total_pred, total_label, average="macro",
                num_classes=4, device="cpu"):
    metric = F1Score(task="multiclass", num_classes=num_classes,
                     average=average).to(device)
    return metric(total_pred.to(device), total_label.to(device)).item()


# ═══════════════════════════════════════════════════════════════════ #
# print_scores
# ═══════════════════════════════════════════════════════════════════ #

def print_scores(total_pred, total_probs, total_label,
                 val_loss, train_loss, name,
                 class_names=None, log_detail=False, device="cpu"):

    n_classes = len(class_names) if class_names else 4
    c_names   = class_names or [f"Class{i}" for i in range(n_classes)]

    f1_macro       = evaluate_f1(total_pred, total_label,
                                 average="macro",
                                 num_classes=n_classes, device=device)
    y_true = total_label.cpu().numpy()
    y_pred = total_pred.cpu().numpy()

    f1_per_class   = f1_score(y_true, y_pred, average=None,
                               labels=np.arange(n_classes), zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None,
                                     labels=np.arange(n_classes), zero_division=0)
    rec_per_class  = recall_score(y_true, y_pred, average=None,
                                  labels=np.arange(n_classes), zero_division=0)

    roc_auc_micro = roc_auc_macro = None
    try:
        roc_auc_micro, roc_auc_macro = compute_roc_auc(
            total_label, total_probs, n_classes
        )
    except Exception:
        pass

    # ── per-epoch compact log ────────────────────────────────────────
    if not log_detail:
        f1_str = " | ".join(
            [f"{c_names[i]}: {f1_per_class[i]*100:.1f}%"
             for i in range(n_classes)]
        )
        logger.info(
            f"{name} set: "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}"
        )
        logger.info(f"F1 Macro Average: {f1_macro*100:.2f}%")
        logger.info(f"F1 {f1_str}")

    # ── end-of-training detailed log ────────────────────────────────
    else:
        event_preds, event_labels = _get_event(total_pred, total_label)

        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL METRICS - {name}")
        logger.info(f"{'='*60}")
        logger.info(f"Train Loss : {train_loss:.4f} | Val Loss : {val_loss:.4f}")

        logger.info("\n[1] SAMPLE-LEVEL METRICS (per data point):")
        logger.info("-" * 60)
        for line in classification_report(
            y_true, y_pred, target_names=c_names,
            digits=4, zero_division=0
        ).split("\n"):
            if line.strip():
                logger.info(line)

        logger.info("\n    Confusion Matrix (Sample - counts):")
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
        col_w = max(len(n) for n in c_names) + 2
        header = " " * 14 + "".join(f"{n:>{col_w}}" for n in c_names)
        logger.info(header)
        for i, row in enumerate(cm):
            row_str = "".join(f"{int(x):>{col_w}}" for x in row)
            logger.info(f"    {c_names[i]:<10}{row_str}")
        logger.info("")

        logger.info("[2] EVENT-LEVEL METRICS (per movement segment):")
        logger.info("-" * 60)
        for line in classification_report(
            event_labels, event_preds, target_names=c_names,
            digits=4, zero_division=0
        ).split("\n"):
            if line.strip():
                logger.info(line)

        logger.info("\n    Confusion Matrix (Event - counts):")
        cm_ev = confusion_matrix(event_labels, event_preds,
                                 labels=np.arange(n_classes))
        logger.info(header)
        for i, row in enumerate(cm_ev):
            row_str = "".join(f"{int(x):>{col_w}}" for x in row)
            logger.info(f"    {c_names[i]:<10}{row_str}")
        logger.info("")
        logger.info(f"\n{'='*60}\n")

    pad = lambda arr, n: list(arr) + [0] * (n - len(arr))
    f1p  = pad(f1_per_class,   n_classes)
    prcp = pad(prec_per_class, n_classes)
    recp = pad(rec_per_class,  n_classes)

    return (
        f1_macro,
        f1p[0],  f1p[1],  f1p[2],  f1p[3],
        prcp[0], prcp[1], prcp[2], prcp[3],
        recp[0], recp[1], recp[2], recp[3],
        roc_auc_micro, roc_auc_macro,
    )


# ═══════════════════════════════════════════════════════════════════ #
# Confusion matrix — shared drawing helper
# ═══════════════════════════════════════════════════════════════════ #

def _draw_cm(ax, cm, class_names, fmt, cmap, vmin=None, vmax=None,
             title=None):
    """
    Draw a single confusion matrix panel on *ax*.

    Parameters
    ----------
    cm         : 2-D array, either raw counts (int) or row-normalised floats
    fmt        : "d" for counts, ".2f" for normalised (displayed as 0.98 style)
    cmap       : matplotlib colormap name
    title      : optional panel subtitle
    """
    n  = len(class_names)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal")   # equal = square cells

    # Colour bar — thin, right side
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)

    # Tick labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticklabels(class_names, rotation=0, fontsize=7.5)

    # Axis labels
    ax.set_xlabel("Predicted label", fontsize=8.5, labelpad=4)
    ax.set_ylabel("True label",      fontsize=8.5, labelpad=4)

    # Optional panel title
    if title:
        ax.set_title(title, fontsize=8, pad=4, style="italic")

    # Cell annotations — 0.98 style for normalised, integer for counts
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val   = cm[i, j]
            color = "white" if val > thresh else "black"
            if fmt == "d":
                text = f"{int(val):d}"
            else:
                # Show as decimal (0.98) not percent — cleaner for publication
                text = f"{val:.2f}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    fontsize=8.5, color=color,
                    fontweight="normal")

    # Force square axes extent
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    # Spine cleanup
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


# ═══════════════════════════════════════════════════════════════════ #
# plot_cmcount — count confusion matrices (sample + event)
# ═══════════════════════════════════════════════════════════════════ #

def plot_cmcount(
    prefix, y_true, y_pred, class_names,
    use_kfold=False, fold_idx=None, base_dir="results",
    wandb_run=None, model_type=None,
):
    folder   = set_folder_path(use_kfold, fold_idx, base_dir, model_type)
    out_base = os.path.join(folder, f"{prefix}_cmcount")

    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)

    cm_sample             = confusion_matrix(y_true_np, y_pred_np)
    ev_preds, ev_labels   = _get_event(y_pred_np, y_true_np)
    cm_event              = confusion_matrix(ev_labels, ev_preds,
                                             labels=np.arange(len(class_names)))

    # Shared colour scale across both panels
    vmax = max(cm_sample.max(), cm_event.max())

    fig, axes = plt.subplots(1, 2, figsize=_FIG_DOUBLE)
    fig.subplots_adjust(wspace=0.45)

    _draw_cm(axes[0], cm_sample, class_names, fmt="d",
             cmap="Blues", vmin=0, vmax=vmax,
             title="(a) Sample-level")
    _draw_cm(axes[1], cm_event,  class_names, fmt="d",
             cmap="Greens", vmin=0, vmax=vmax,
             title="(b) Event-level")

    return _save(fig, out_base, wandb_run,
                 wandb_key=f"{prefix}_cmcount",
                 caption="Confusion Matrix — Counts")


# ═══════════════════════════════════════════════════════════════════ #
# plot_cmpercent — normalised confusion matrices (sample + event)
# ═══════════════════════════════════════════════════════════════════ #

def plot_cmpercent(
    prefix, y_true, y_pred, class_names,
    use_kfold=False, fold_idx=None, base_dir="results",
    wandb_run=None, model_type=None,
):
    folder   = set_folder_path(use_kfold, fold_idx, base_dir, model_type)
    out_base = os.path.join(folder, f"{prefix}_cmpercent")

    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)

    def _row_norm(cm):
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        return cm.astype(float) / row_sum

    cm_sample           = _row_norm(confusion_matrix(y_true_np, y_pred_np))
    ev_preds, ev_labels = _get_event(y_pred_np, y_true_np)
    cm_event            = _row_norm(
        confusion_matrix(ev_labels, ev_preds,
                         labels=np.arange(len(class_names)))
    )

    fig, axes = plt.subplots(1, 2, figsize=_FIG_DOUBLE)
    fig.subplots_adjust(wspace=0.45)

    _draw_cm(axes[0], cm_sample, class_names, fmt=".2f",
             cmap="Blues", vmin=0.0, vmax=1.0,
             title="(a) Sample-level (row %)")
    _draw_cm(axes[1], cm_event,  class_names, fmt=".2f",
             cmap="Greens", vmin=0.0, vmax=1.0,
             title="(b) Event-level (row %)")

    return _save(fig, out_base, wandb_run,
                 wandb_key=f"{prefix}_cmpercent",
                 caption="Confusion Matrix — Normalised")


# ═══════════════════════════════════════════════════════════════════ #
# plot_roc — per-class + micro + macro ROC curves
# ═══════════════════════════════════════════════════════════════════ #

def plot_roc(
    prefix, labels, probs, class_names,
    use_kfold=False, fold_idx=None, base_dir="results",
    wandb_run=None, model_type=None,
):
    folder   = set_folder_path(use_kfold, fold_idx, base_dir, model_type)
    out_base = os.path.join(folder, f"{prefix}_roc")

    n_classes = len(class_names)
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
    probs_np  = probs.cpu().numpy()  if torch.is_tensor(probs)  else np.array(probs)
    y_bin     = label_binarize(labels_np, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=_FIG_ROC)
    ax.set_facecolor("#EEF4FB")          # light blue background
    ax.grid(color="white", lw=0.6, zorder=0)

    # Per-class curves — thin lines
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_np[:, i])
        roc_i       = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                color=_PALETTE[i % len(_PALETTE)],
                lw=0.7,
                label=f"{class_names[i]} (AUC={roc_i:.3f})")

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs_np.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro,
            color="#000000", lw=0.85, linestyle="--",
            label=f"Micro-avg (AUC={auc_micro:.3f})")

    # Macro-average (interpolated)
    all_fpr   = np.unique(np.concatenate(
        [roc_curve(y_bin[:, i], probs_np[:, i])[0] for i in range(n_classes)]
    ))
    mean_tpr  = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs_np[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= n_classes
    auc_macro  = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr,
            color="#000000", lw=0.85, linestyle=":",
            label=f"Macro-avg (AUC={auc_macro:.3f})")

    # Reference diagonal
    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=0.5,
            linestyle="-", zorder=0)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=8)
    ax.set_ylabel("True Positive Rate",  fontsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(loc="lower right", fontsize=6.5,
              framealpha=0.9, edgecolor="#CCCCCC",
              handlelength=1.8)
    fig.tight_layout()

    return _save(fig, out_base, wandb_run,
                 wandb_key=f"{prefix}_roc",
                 caption="ROC Curve")


# ═══════════════════════════════════════════════════════════════════ #
# plot_pr — per-class Precision-Recall curves
# ═══════════════════════════════════════════════════════════════════ #

def plot_pr(
    prefix, labels, probs, class_names,
    use_kfold=False, fold_idx=None, base_dir="results",
    wandb_run=None, model_type=None,
):
    """
    Precision-Recall curves — useful for class-imbalanced datasets.
    Saved as PNG + PDF alongside the ROC curve.
    """
    folder   = set_folder_path(use_kfold, fold_idx, base_dir, model_type)
    out_base = os.path.join(folder, f"{prefix}_pr")

    n_classes = len(class_names)
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
    probs_np  = probs.cpu().numpy()  if torch.is_tensor(probs)  else np.array(probs)
    y_bin     = label_binarize(labels_np, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=_FIG_ROC)
    ax.set_facecolor("#EEF4FB")          # light blue background
    ax.grid(color="white", lw=0.6, zorder=0)

    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs_np[:, i])
        ap           = average_precision_score(y_bin[:, i], probs_np[:, i])
        ax.plot(rec, prec,
                color=_PALETTE[i % len(_PALETTE)],
                lw=0.7,
                label=f"{class_names[i]} (AP={ap:.3f})")

    # Micro-average
    prec_micro, rec_micro, _ = precision_recall_curve(
        y_bin.ravel(), probs_np.ravel()
    )
    ap_micro = average_precision_score(y_bin, probs_np, average="micro")
    ax.plot(rec_micro, prec_micro,
            color="#000000", lw=0.85, linestyle="--",
            label=f"Micro-avg (AP={ap_micro:.3f})")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Recall",    fontsize=8)
    ax.set_ylabel("Precision", fontsize=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(loc="lower left", fontsize=6.5,
              framealpha=0.9, edgecolor="#CCCCCC",
              handlelength=1.8)
    fig.tight_layout()

    return _save(fig, out_base, wandb_run,
                 wandb_key=f"{prefix}_pr",
                 caption="Precision-Recall Curve")