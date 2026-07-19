"""
ablation/feature_ablation.py

Feature combination ablation study — fully isolated from main training code.
Does NOT modify train.py or helpers.py.

Strategy: monkey-patch utils.helpers.set_folder_path temporarily to
redirect all save/plot outputs to results/ablation/feature/ instead of
the default results/ folder.

Feature naming convention:
  sp  = speed
  dir = direction
  std = stddev
  dis = displacement

Results saved to: results/ablation/feature/<model_type>/
"""

import os
import sys
import datetime
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import utils.helpers as _helpers
from data.preprocessor import Preprocessor
from train import main_kfold
from utils.logger import logger
from utils.helpers import set_randomness


# ------------------------------------------------------------------ #
# Feature combinations
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

ABLATION_BASE_DIR = os.path.join("results", "ablation", "feature")
CLASS_NAMES       = ["Fixation", "Saccade", "Pursuit", "Blink"]


# ------------------------------------------------------------------ #
# Context manager: redirect all save/plot outputs to ablation folder
# ------------------------------------------------------------------ #

@contextmanager
def ablation_output_dir(base_dir):
    """
    Temporarily patch utils.helpers.set_folder_path so all save_*
    and plot_* calls write to base_dir instead of the default 'results/'.

    Only active within the with-block — no permanent changes to helpers.py.
    """
    original_fn = _helpers.set_folder_path

    def _patched(use_kfold=False, fold_idx=None,
                 base_dir=base_dir, model_type=None):
        # Ignore whatever base_dir is passed in — always use ablation dir
        if use_kfold:
            path = os.path.join(base_dir, "kfold", model_type or "")
            if fold_idx is not None:
                path = os.path.join(path, f"fold_{fold_idx + 1}")
        else:
            path = os.path.join(base_dir, model_type or "")
        os.makedirs(path, exist_ok=True)
        return path

    # Patch all modules that imported set_folder_path
    import utils.metrics as _metrics
    _helpers.set_folder_path  = _patched
    _metrics.set_folder_path  = _patched

    try:
        yield
    finally:
        # Restore originals
        _helpers.set_folder_path  = original_fn
        _metrics.set_folder_path  = original_fn


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
    wandb_project = f"ablation_feature_{args.dataset}"
    date_str      = datetime.datetime.now().strftime("%Y%m%d")

    # Filter combos if subset requested
    selected = FEATURE_COMBOS
    if combos:
        selected = [(t, f) for t, f in FEATURE_COMBOS if t in combos]
        if not selected:
            logger.error(f"No combos matched: {combos}")
            logger.error(f"Available tags: {[t for t, _ in FEATURE_COMBOS]}")
            return

    logger.info("=" * 60)
    logger.info("FEATURE ABLATION STUDY")
    logger.info(f"Dataset  : {args.dataset.upper()}")
    logger.info(f"Model    : {args.model_type}")
    logger.info(f"Combos   : {len(selected)}")
    logger.info(f"Results  : {ABLATION_BASE_DIR}/")
    logger.info("=" * 60)

    pprep    = Preprocessor()
    summary  = []   # collects one dict per combo for the summary CSV

    for i, (tag, selected_features) in enumerate(selected, 1):
        logger.info(f"\n[{i}/{len(selected)}] Feature combo: [{tag}] — {selected_features}")

        train_X, train_Y, _, _ = pprep.load_data(
            data_path,
            stride=stride,
            selected_features=selected_features,
        )

        logger.info(f"  Input shape : {train_X.shape}")

        run_name = f"{date_str}_{tag}"

        with ablation_output_dir(ABLATION_BASE_DIR):
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
                use_kfold=False,            # ablation always single split
                n_splits=5,
                start_fold=0,
                max_folds=1,
                wandb_project=wandb_project,
                use_wandb=args.use_wandb,
                checkpoint=False,
                plot_result=True,
            )

        # Collect into summary
        if all_metrics:
            row = {
                "feature_tag":      tag,
                "features":         "+".join(selected_features),
                "n_features":       train_X.shape[1],
            }
            row.update({k: v for k, v in all_metrics[0].items()
                        if k != "fold"})
            summary.append(row)
            logger.info(
                f"  Done: [{tag}] — "
                f"F1={all_metrics[0].get('F1_avg', 0)*100:.2f}%  "
                f"SP={all_metrics[0].get('F1_Pursuit', 0)*100:.2f}%"
            )

    logger.info("\n" + "=" * 60)
    logger.info(f"FEATURE ABLATION COMPLETE — {len(selected)} combos")
    logger.info(f"Results : {ABLATION_BASE_DIR}/")
    logger.info("=" * 60)