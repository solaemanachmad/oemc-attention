"""
ablation/architecture_ablation.py

Architecture / component ablation study for the proposed Conv-Attention
model — fully isolated from main training code. Uses the same
monkey-patch strategy as feature_ablation.py and timestep_ablation.py.

Only applies to model_type == "conv_attention" (the toggled components
— attention, positional encoding, conv refinement depth — don't exist
in the baseline architectures).

Key behaviours (mirrors feature_ablation.py / timestep_ablation.py)
─────────────────────────────────────────────────────────────────
1. Output dir  : results/ablation/conv_attention/architecture/
2. Summary CSV : accumulates across multiple partial runs (merge strategy)
3. Resume      : skips variants that already have a metrics CSV on disk
4. Checkpoint  : controlled by --checkpoint flag (off by default)
5. WandB       : controlled by --use_wandb flag
6. Validation  : always hold-out 80/20 (stratified) — same as the other
                 two ablation scripts, never k-fold, for speed.

Unlike feature/timestep ablation, this script calls train_model()
directly instead of main_kfold(), because the toggle flags
(use_attention, use_positional_encoding, encoder_layers) need to be
injected straight into model_params — main_kfold() builds model_params
internally per model_type and has no hook for extra kwargs.
"""

import os
import sys
import glob
import datetime
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import utils.helpers as _helpers
from data.preprocessor import Preprocessor
from train import train_model
from utils.logger import logger
from utils.helpers import set_randomness


# ------------------------------------------------------------------ #
# Architecture variants registry
# ------------------------------------------------------------------ #
# Each variant is (tag, flags_dict). flags_dict is merged into
# model_params on top of the shared base hyperparams (d_model,
# num_heads, kernel_size, dropout).
#
# Designed to support TWO narratives in the paper from ONE set of runs
# (some variants are shared between both tables — no duplicate runs):
#
#   (A) Cumulative build-up (primary table — shows each component's
#       marginal contribution as it's progressively added):
#         cnn -> cnn_attention -> cnn_attention_pos -> full
#
#   (B) Leave-one-out from the full model (secondary table — isolates
#       each component's contribution when removed from the full model):
#         full -> no_attention / no_pos_encoding / no_attn_no_pos
#         (cnn_attention_pos IS "full minus encoder", reused from table A)
#
#   Encoder depth sweep (attention + pos_encoding always on):
#         cnn_attention_pos (0 layers) -> encoder_1layer -> encoder_2layer -> full (3 layers)
ARCHITECTURE_VARIANTS = [
    # --- (A) cumulative build-up, in order ---
    ("cnn",               dict(use_attention=False, use_positional_encoding=False, encoder_layers=0)),
    ("cnn_attention",     dict(use_attention=True,  use_positional_encoding=False, encoder_layers=0)),
    ("cnn_attention_pos", dict(use_attention=True,  use_positional_encoding=True,  encoder_layers=0)),
    ("full",              dict(use_attention=True,  use_positional_encoding=True,  encoder_layers=3)),

    # --- (B) leave-one-out from full (cnn_attention_pos above = "full minus encoder") ---
    ("no_attention",      dict(use_attention=False, use_positional_encoding=True,  encoder_layers=3)),
    ("no_pos_encoding",   dict(use_attention=True,  use_positional_encoding=False, encoder_layers=3)),
    ("no_attn_no_pos",    dict(use_attention=False, use_positional_encoding=False, encoder_layers=3)),

    # --- encoder depth sweep (attention + pos_encoding always on) ---
    ("encoder_1layer",    dict(use_attention=True,  use_positional_encoding=True,  encoder_layers=1)),
    ("encoder_2layer",    dict(use_attention=True,  use_positional_encoding=True,  encoder_layers=2)),

    # --- (C) 3-line depth sweep: isolate encoder's effect WITH and WITHOUT
    # attention, at every depth (0-3), to properly test whether encoder
    # depth is the sole driver or whether attention still matters at
    # every depth level, not just depth=3.
    # "no_attention" (depth=3) and "cnn" (depth=0) already cover two
    # points of the "no-attention, pos-encoding-on" and "cnn-only" lines
    # respectively — these fill in the missing depths.
    ("no_attn_pos_encoder0", dict(use_attention=False, use_positional_encoding=True,  encoder_layers=0)),  # = cnn_attention_pos minus attention
    ("no_attn_pos_encoder1", dict(use_attention=False, use_positional_encoding=True,  encoder_layers=1)),
    ("no_attn_pos_encoder2", dict(use_attention=False, use_positional_encoding=True,  encoder_layers=2)),
    ("cnn_encoder1",         dict(use_attention=False, use_positional_encoding=False, encoder_layers=1)),
    ("cnn_encoder2",         dict(use_attention=False, use_positional_encoding=False, encoder_layers=2)),
]

CLASS_NAMES   = ["Fixation", "Saccade", "Pursuit", "Blink"]
FULL_FEATURES = ["speed", "direction", "stddev", "displacement"]


def _ablation_dir():
    """results/ablation/conv_attention/architecture/"""
    return os.path.join("results", "ablation", "conv_attention", "architecture")


def _summary_path(dataset):
    return os.path.join(
        _ablation_dir(),
        f"architecture_ablation_{dataset}_summary.csv"
    )


# ------------------------------------------------------------------ #
# Context manager: redirect save/plot outputs to ablation folder
# (identical pattern to feature_ablation.py / timestep_ablation.py)
# ------------------------------------------------------------------ #

@contextmanager
def ablation_output_dir(base_dir, dataset=None):
    """
    IMPORTANT: incoming base_dir/dataset from callers (train.py always
    passes train_model()'s own base_dir="results" positionally) are
    deliberately IGNORED — captured as plain closure variables instead
    of default parameter values, so they cannot be overridden.
    """
    _ablation_base    = base_dir
    _ablation_dataset = dataset
    original_fn = _helpers.set_folder_path

    def _patched(use_kfold=False, fold_idx=None, base_dir=None, model_type=None,
                 dataset=None):
        path = os.path.join(_ablation_base, _ablation_dataset or "")
        if use_kfold:
            path = os.path.join(path, "kfold", model_type or "")
            if fold_idx is not None:
                path = os.path.join(path, f"fold_{fold_idx + 1}")
        else:
            path = os.path.join(path, model_type or "")
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
# (identical pattern to feature_ablation.py)
# ------------------------------------------------------------------ #

def _load_summary(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _merge_summary(existing_df, new_rows):
    if not new_rows:
        return existing_df

    new_df = pd.DataFrame(new_rows)

    if existing_df.empty:
        merged = new_df
    else:
        new_tags = set(new_df["variant_tag"].tolist())
        existing_df = existing_df[~existing_df["variant_tag"].isin(new_tags)]
        merged = pd.concat([existing_df, new_df], ignore_index=True)

    tag_order = [t for t, _ in ARCHITECTURE_VARIANTS]
    merged["_order"] = merged["variant_tag"].map(
        {t: i for i, t in enumerate(tag_order)}
    )
    merged = merged.sort_values("_order").drop(columns=["_order"])
    return merged.reset_index(drop=True)


def _save_summary(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if "F1_avg" in df.columns:
        df = df.sort_values("F1_avg", ascending=False)

    key_cols = [
        "variant_tag", "use_attention", "use_positional_encoding",
        "encoder_layers", "timesteps", "loader_mode",
        "d_model", "num_heads", "kernel_size", "dropout", "params",
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


def _already_done(tag, dataset):
    """
    Check if this variant already has a metrics CSV on disk.
    Folder: results/ablation/conv_attention/architecture/<dataset>/conv_attention/
    Pattern: *_<tag>_*_metrics.csv
    """
    folder  = os.path.join(_ablation_dir(), dataset, "conv_attention")
    pattern = os.path.join(folder, f"*_{tag}_*_metrics.csv")
    found   = glob.glob(pattern)
    if found:
        logger.info(f"  [resume] Found existing result for [{tag}]: "
                    f"{os.path.basename(found[0])}")
    return bool(found)


# ------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------ #

def run(args, variants=None):
    set_randomness(42)

    if args.model_type != "conv_attention":
        logger.error(
            "Architecture ablation only applies to model_type='conv_attention' "
            "(the toggled components don't exist in the baseline architectures)."
        )
        return

    stride    = args.stride or (10 if args.dataset == "gazecom" else 8)
    freq      = args.frequency or (250 if args.dataset == "gazecom" else 200)
    data_path = args.data_path or os.path.join(
        "dataset", "processed",
        f"{args.dataset}_s{stride}_f{freq}_w{args.window_length}_o{args.offset}"
    )

    ablation_base = _ablation_dir()
    summary_path  = _summary_path(args.dataset)
    wandb_project = f"ablation_architecture_{args.dataset}"
    date_str      = datetime.datetime.now().strftime("%Y%m%d")

    selected = ARCHITECTURE_VARIANTS
    if variants:
        selected = [(t, f) for t, f in ARCHITECTURE_VARIANTS if t in variants]
        if not selected:
            logger.error(f"No variants matched: {variants}")
            logger.error(f"Available tags: {[t for t, _ in ARCHITECTURE_VARIANTS]}")
            return

    to_run  = []
    skipped = []
    for tag, flags in selected:
        if getattr(args, "resume", False) and _already_done(tag, args.dataset):
            skipped.append(tag)
        else:
            to_run.append((tag, flags))

    logger.info("=" * 60)
    logger.info("ARCHITECTURE ABLATION STUDY (Conv-Attention components)")
    logger.info(f"Dataset      : {args.dataset.upper()}")
    logger.info(f"Timesteps    : {args.timesteps}")
    logger.info(f"Output       : {ablation_base}/")
    logger.info(f"Summary      : {summary_path}")
    logger.info(f"Variants     : {len(to_run)} to run"
                + (f", {len(skipped)} skipped (already done)" if skipped else ""))
    if skipped:
        logger.info(f"Skipped      : {skipped}")
    logger.info(f"Loader mode  : {args.loader_mode}")
    logger.info(f"Validation   : hold-out 80/20 (stratified) — ablation always "
                f"uses hold-out, never k-fold, for speed across many runs")
    if getattr(args, "use_kfold", False):
        logger.warning(
            "--use_kfold was passed but is IGNORED by this ablation script. "
            "Ablation studies intentionally always use a single stratified "
            "hold-out split for speed. Run main.py directly if you need "
            "k-fold results for the final reported numbers."
        )
    logger.info(f"WandB        : {'ON — ' + wandb_project if args.use_wandb else 'OFF'}")
    logger.info("=" * 60)

    pprep = Preprocessor()

    train_X, train_Y, _, _ = pprep.load_data(
        data_path,
        stride=stride,
        selected_features=FULL_FEATURES,
    )
    logger.info(f"Input shape: {train_X.shape}")

    new_rows = []

    for i, (tag, flags) in enumerate(to_run, 1):
        logger.info(f"\n[{i}/{len(to_run)}] Variant: [{tag}] — {flags}")

        run_name = f"{date_str}_{tag}"

        # Base hyperparams shared by all variants, + this variant's
        # toggle flags. output_size / input_size / timesteps are
        # injected automatically inside train_model() (same as
        # main_kfold does), so they don't need to be set here.
        model_params = dict(
            d_model=args.d_model,
            num_heads=args.num_heads,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            **flags,
        )

        try:
            with ablation_output_dir(ablation_base, dataset=args.dataset):
                all_metrics, _, _, _, _, _ = train_model(
                    X=train_X,
                    Y=train_Y,
                    class_names=CLASS_NAMES,
                    model_type="conv_attention",
                    model_params=model_params,
                    run_name=run_name,
                    dataset=args.dataset,
                    timesteps=args.timesteps,
                    d_model=args.d_model,
                    dropout=args.dropout,
                    num_heads=args.num_heads,
                    kernel_size=args.kernel_size,
                    lr=args.lr,
                    loader_mode=args.loader_mode,
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
                    "variant_tag": tag,
                    "use_attention": flags["use_attention"],
                    "use_positional_encoding": flags["use_positional_encoding"],
                    "encoder_layers": flags["encoder_layers"],
                    # Config metadata — see feature_ablation.py for why
                    # this is kept explicit here rather than relying on
                    # WandB alone.
                    "timesteps":   args.timesteps,
                    "loader_mode": args.loader_mode,
                    "d_model":     args.d_model,
                    "num_heads":   args.num_heads,
                    "kernel_size": args.kernel_size,
                    "dropout":     args.dropout,
                }
                row.update({k: v for k, v in all_metrics[0].items()
                            if k != "fold"})
                new_rows.append(row)

                logger.info(
                    f"  Done [{tag}] — "
                    f"F1={all_metrics[0].get('F1_avg', 0)*100:.2f}%  "
                    f"Pursuit={all_metrics[0].get('F1_Pursuit', 0)*100:.2f}%"
                )

                existing = _load_summary(summary_path)
                merged   = _merge_summary(existing, new_rows)
                _save_summary(merged, summary_path)
                logger.info(f"  Summary updated: {summary_path} "
                            f"({len(merged)} total variants)")

        except Exception as e:
            logger.error(f"  FAILED [{tag}]: {e}")
            logger.error("  Continuing with next variant...")
            continue

    final_df = _load_summary(summary_path)
    if not final_df.empty and "F1_avg" in final_df.columns:
        top = final_df.sort_values("F1_avg", ascending=False)
        logger.info("\nAll variants by F1_avg (best first):")
        logger.info(f"  {'Tag':<18} {'F1_avg':>8} {'Pursuit':>8} {'ROC-macro':>10}")
        logger.info("  " + "-" * 48)
        for _, r in top.iterrows():
            logger.info(
                f"  {r['variant_tag']:<18}"
                f"{r.get('F1_avg', 0)*100:>7.2f}%"
                f"{r.get('F1_Pursuit', 0)*100:>8.2f}%"
                f"{r.get('roc_auc_macro', 0):>10.4f}"
            )

    logger.info("\n" + "=" * 60)
    logger.info(f"ARCHITECTURE ABLATION — {len(to_run)} variants ran, "
                f"{len(skipped)} skipped")
    logger.info(f"Summary  : {summary_path}")
    logger.info("=" * 60)