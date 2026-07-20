"""
ablation/timestep_ablation.py

Timestep ablation study — fully isolated from main training code.
Uses the same monkey-patch strategy as feature_ablation.py.

Results saved to: results/ablation/timestep/<model_type>/
"""

import os
import sys
import datetime
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import utils.helpers as _helpers
from data.preprocessor import Preprocessor
from train import main_kfold
from utils.logger import logger
from utils.helpers import set_randomness


TIMESTEP_VALUES   = [1, 5, 10, 15, 20, 25]
ABLATION_BASE_DIR = os.path.join("results", "ablation", "timestep")
CLASS_NAMES       = ["Fixation", "Saccade", "Pursuit", "Blink"]
FULL_FEATURES     = ["speed", "direction", "stddev", "displacement"]


@contextmanager
def ablation_output_dir(base_dir):
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


def run(args, timesteps=None):
    set_randomness(42)

    stride    = args.stride or (10 if args.dataset == "gazecom" else 8)
    freq      = args.frequency or (250 if args.dataset == "gazecom" else 200)
    data_path = args.data_path or os.path.join(
        "dataset", "processed",
        f"{args.dataset}_s{stride}_f{freq}_w{args.window_length}_o{args.offset}"
    )
    wandb_project = f"ablation_timestep_{args.dataset}"
    date_str      = datetime.datetime.now().strftime("%Y%m%d")

    selected = TIMESTEP_VALUES
    if timesteps:
        selected = [t for t in TIMESTEP_VALUES if t in timesteps]
        if not selected:
            logger.error(f"No timesteps matched: {timesteps}")
            logger.error(f"Available: {TIMESTEP_VALUES}")
            return

    logger.info("=" * 60)
    logger.info("TIMESTEP ABLATION STUDY")
    logger.info(f"Dataset   : {args.dataset.upper()}")
    logger.info(f"Model     : {args.model_type}")
    logger.info(f"Timesteps : {selected}")
    logger.info(f"Features  : {FULL_FEATURES}")
    logger.info(f"Results   : {ABLATION_BASE_DIR}/")
    logger.info("=" * 60)

    pprep = Preprocessor()

    # Load data once — features fixed to full set
    train_X, train_Y, _, _ = pprep.load_data(
        data_path,
        stride=stride,
        selected_features=FULL_FEATURES,
    )
    logger.info(f"Input shape: {train_X.shape}")

    for i, t in enumerate(selected, 1):
        tag = f"t{t}"
        logger.info(f"\n[{i}/{len(selected)}] Timesteps: {t}")

        run_name = f"{date_str}_{tag}"

        with ablation_output_dir(ABLATION_BASE_DIR):
            main_kfold(
                X=train_X,
                Y=train_Y,
                run_name=run_name,
                model_type=args.model_type,
                class_names=CLASS_NAMES,
                timesteps=t,
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
                checkpoint=False,
                plot_result=True,
            )

        logger.info(f"  Done: [{tag}]")

    logger.info("\n" + "=" * 60)
    logger.info(f"TIMESTEP ABLATION COMPLETE — {len(selected)} values")
    logger.info(f"Results : {ABLATION_BASE_DIR}/")
    logger.info("=" * 60)