"""
ablation/feature_ablation.py

Feature combination ablation study — fully isolated from main training code.

Key behaviours
──────────────
1. Output dir  : results/ablation/<model_type>/features/
2. Summary CSV : accumulates across multiple partial runs (merge strategy)
3. Resume      : skips combos that already have a metrics CSV on disk
4. Checkpoint  : controlled by --checkpoint flag (off by default for ablation)
5. WandB       : controlled by --use_wandb flag
6. Partial run : run sp_dir + sp_dir_std first, then sp_dir_dis + sp_dir_std_dis
                 — summary CSV is merged/updated each time, never overwritten

Feature naming convention:
  sp  = speed    dir = direction    std = stddev    dis = displacement
"""

import os
import sys
import glob
import datetime
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()   # load WANDB_API_KEY and other env vars from .env

import pandas as pd
import utils.helpers as _helpers
from data.preprocessor import Preprocessor
from train import main_kfold
from utils.logger import logger
from utils.helpers import set_randomness, set_folder_path


# ------------------------------------------------------------------ #
# Feature combinations registry
# ------------------------------------------------------------------ #
FEATURE_COMBOS = [
    # Single
    ("sp",             ["speed"]),
    ("dir",            ["direction"]),
    ("std",            ["stddev"]),
    ("dis",            ["displacement"]),
    # Pairs
    ("sp_dir",         ["speed", "direction"]),
    ("sp_std",         ["speed", "stddev"]),
    ("sp_dis",         ["speed", "displacement"]),
    ("dir_std",        ["direction", "stddev"]),
    ("dir_dis",        ["direction", "displacement"]),
    ("std_dis",        ["stddev", "displacement"]),
    # Triples
    ("sp_dir_std",     ["speed", "direction", "stddev"]),
    ("sp_dir_dis",     ["speed", "direction", "displacement"]),
    ("sp_std_dis",     ["speed", "stddev", "displacement"]),
    ("dir_std_dis",    ["direction", "stddev", "displacement"]),
    # Full (baseline)
    ("sp_dir_std_dis", ["speed", "direction", "stddev", "displacement"]),
]

CLASS_NAMES = ["Fixation", "Saccade", "Pursuit", "Blink"]


def _ablation_dir(model_type):
    """results/ablation/<model_type>/features/"""
    return os.path.join("results", "ablation", model_type, "features")


def _summary_path(model_type, dataset):
    return os.path.join(
        _ablation_dir(model_type),
        f"feature_ablation_{dataset}_summary.csv"
    )


# ------------------------------------------------------------------ #
# Context manager: redirect save/plot outputs to ablation folder
# ------------------------------------------------------------------ #

@contextmanager
def ablation_output_dir(base_dir):
    """
    Temporarily patch utils.helpers.set_folder_path and
    utils.metrics.set_folder_path so all save_* and plot_* calls
    write to base_dir instead of the default 'results/'.
    Restored automatically on exit — no permanent changes.
    """
    original_fn = _helpers.set_folder_path

    def _patched(use_kfold=False, fold_idx=None,
                 base_dir=base_dir, model_type=None):
        if use_kfold:
            path = os.path.join(base_dir, "kfold", model_type or "")
            if fold_idx is not None:
                path = os.path.join(path, f"fold_{fold_idx + 1}")
        else:
            path = os.path.join(base_dir, model_type or "")
        os.makedirs(path, exist_ok=True)
        return path

    import utils.metrics as _metrics
    _helpers.set_folder_path = _patched
    _metrics.set_folder_path = _patched

    try:
        yield
    finally:
        _helpers.set_folder_path = original_fn
        _metrics.set_folder_path = original_fn


# ------------------------------------------------------------------ #
# Summary CSV helpers — merge strategy for partial runs
# ------------------------------------------------------------------ #

def _load_summary(path):
    """Load existing summary CSV, or return empty DataFrame."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _merge_summary(existing_df, new_rows):
    """
    Merge new_rows (list of dicts) into existing_df.
    If a feature_tag already exists, overwrite it with the new result.
    This allows partial re-runs to update only the combos that ran.
    """
    if not new_rows:
        return existing_df

    new_df = pd.DataFrame(new_rows)

    if existing_df.empty:
        return new_df

    # Drop old rows for tags that were re-run
    new_tags   = set(new_df["feature_tag"].tolist())
    existing_df = existing_df[~existing_df["feature_tag"].isin(new_tags)]

    merged = pd.concat([existing_df, new_df], ignore_index=True)

    # Restore original order from FEATURE_COMBOS
    tag_order = [t for t, _ in FEATURE_COMBOS]
    merged["_order"] = merged["feature_tag"].map(
        {t: i for i, t in enumerate(tag_order)}
    )
    merged = merged.sort_values("_order").drop(columns=["_order"])
    return merged.reset_index(drop=True)


def _save_summary(df, path, dataset, model_type):
    """Sort by F1_avg desc (best first) and save."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if "F1_avg" in df.columns:
        df = df.sort_values("F1_avg", ascending=False)

    # Reorder columns — feature info first, then key metrics
    key_cols = [
        "feature_tag", "features", "n_features",
        "F1_avg", "F1_Fixation", "F1_Saccade", "F1_Pursuit", "F1_Blink",
        "ev_F1_avg", "ev_F1_Fixation", "ev_F1_Saccade",
        "ev_F1_Pursuit", "ev_F1_Blink",
        "roc_auc_macro", "roc_auc_micro",
        "val_loss", "train_loss", "epochs_run",
    ]
    other_cols = [c for c in df.columns if c not in key_cols]
    df = df[[c for c in key_cols if c in df.columns] + other_cols]
    df.to_csv(path, index=False, float_format="%.4f")
    return path


def _already_done(tag, model_type):
    """
    Check if this combo already has a metrics CSV on disk.
    Used for resume — skip combos that completed successfully.
    """
    folder  = os.path.join(_ablation_dir(model_type), model_type)
    pattern = os.path.join(folder, f"*_{tag}_*metrics.csv")
    return bool(glob.glob(pattern))


# ------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------ #

def run(args, combos=None):
    set_randomness(42)

    stride    = args.stride or (10 if args.dataset == "gazecom" else 8)
    freq      = args.frequency or (250 if args.dataset == "gazecom" else 200)
    data_path = args.data_path or os.path.join(
        "dataset", "processed",
        f"{args.dataset}_s{stride}_f{freq}_w{args.window_length}_o{args.offset}"
    )

    ablation_base = _ablation_dir(args.model_type)
    summary_path  = _summary_path(args.model_type, args.dataset)
    wandb_project = f"ablation_feature_{args.dataset}_{args.model_type}"
    date_str      = datetime.datetime.now().strftime("%Y%m%d")

    # Filter combos if subset requested
    selected = FEATURE_COMBOS
    if combos:
        selected = [(t, f) for t, f in FEATURE_COMBOS if t in combos]
        if not selected:
            logger.error(f"No combos matched: {combos}")
            logger.error(f"Available tags: {[t for t, _ in FEATURE_COMBOS]}")
            return

    # Separate skip list — combos already done (resume support)
    to_run  = []
    skipped = []
    for tag, feats in selected:
        if getattr(args, "resume", False) and _already_done(tag, args.model_type):
            skipped.append(tag)
        else:
            to_run.append((tag, feats))

    logger.info("=" * 60)
    logger.info("FEATURE ABLATION STUDY")
    logger.info(f"Dataset   : {args.dataset.upper()}")
    logger.info(f"Model     : {args.model_type}")
    logger.info(f"Output    : {ablation_base}/")
    logger.info(f"Summary   : {summary_path}")
    logger.info(f"Combos    : {len(to_run)} to run"
                + (f", {len(skipped)} skipped (already done)" if skipped else ""))
    if skipped:
        logger.info(f"Skipped   : {skipped}")
    logger.info(f"WandB     : {'ON — ' + wandb_project if args.use_wandb else 'OFF'}")
    logger.info(f"Checkpoint: {'ON' if args.checkpoint else 'OFF'}")
    logger.info("=" * 60)

    pprep       = Preprocessor()
    new_rows    = []    # results from this run only

    for i, (tag, selected_features) in enumerate(to_run, 1):
        logger.info(f"\n[{i}/{len(to_run)}] Feature combo: [{tag}] — {selected_features}")

        try:
            train_X, train_Y, _, _ = pprep.load_data(
                data_path,
                stride=stride,
                selected_features=selected_features,
            )
            logger.info(f"  Input shape : {train_X.shape}")

            run_name = f"{date_str}_{tag}"

            with ablation_output_dir(ablation_base):
                all_metrics, _, _, _, _, _ = main_kfold(
                    X=train_X,
                    Y=train_Y,
                    run_name=run_name,
                    model_type=args.model_type,
                    class_names=CLASS_NAMES,
                    timesteps=args.timesteps,
                    d_model=args.d_model,
                    num_heads=args.num_heads,
                    kernel_size=args.kernel_size,
                    dropout=args.dropout,
                    lr=args.lr,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    use_kfold=False,
                    n_splits=5,
                    start_fold=0,
                    max_folds=1,
                    wandb_project=wandb_project,
                    use_wandb=args.use_wandb,
                    checkpoint=args.checkpoint,
                    plot_result=True,
                )

            if all_metrics:
                row = {
                    "feature_tag": tag,
                    "features":    "+".join(selected_features),
                    "n_features":  train_X.shape[1],
                }
                row.update({k: v for k, v in all_metrics[0].items()
                            if k != "fold"})
                new_rows.append(row)

                logger.info(
                    f"  Done [{tag}] — "
                    f"F1={all_metrics[0].get('F1_avg', 0)*100:.2f}%  "
                    f"SP={all_metrics[0].get('F1_Pursuit', 0)*100:.2f}%"
                )

                # ── Update summary after EACH combo ──────────────────
                # Merge with any previous results so partial runs
                # accumulate into one CSV without overwriting prior work.
                existing = _load_summary(summary_path)
                merged   = _merge_summary(existing, new_rows)
                _save_summary(merged, summary_path, args.dataset, args.model_type)
                logger.info(f"  Summary updated: {summary_path} "
                            f"({len(merged)} total combos)")

        except Exception as e:
            logger.error(f"  FAILED [{tag}]: {e}")
            logger.error("  Continuing with next combo...")
            continue

    # ── Final summary log ─────────────────────────────────────────────
    final_df = _load_summary(summary_path)
    if not final_df.empty and "F1_avg" in final_df.columns:
        top = final_df.sort_values("F1_avg", ascending=False).head(5)
        logger.info("\nTop-5 feature combos by F1_avg (all runs):")
        logger.info(f"  {'Tag':<18} {'F1_avg':>8} {'Pursuit':>8} {'Saccade':>8} {'ROC-macro':>10}")
        logger.info("  " + "-" * 56)
        for _, r in top.iterrows():
            logger.info(
                f"  {r['feature_tag']:<18}"
                f"{r.get('F1_avg', 0)*100:>7.2f}%"
                f"{r.get('F1_Pursuit', 0)*100:>8.2f}%"
                f"{r.get('F1_Saccade', 0)*100:>8.2f}%"
                f"{r.get('roc_auc_macro', 0):>10.4f}"
            )

    logger.info("\n" + "=" * 60)
    logger.info(f"FEATURE ABLATION — {len(to_run)} combos ran, "
                f"{len(skipped)} skipped")
    logger.info(f"Summary  : {summary_path}")
    logger.info("=" * 60)