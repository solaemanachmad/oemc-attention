"""
ablation/timestep_ablation.py

Timestep / temporal-context ablation study — fully isolated from main
training code. Uses the same monkey-patch strategy as feature_ablation.py.

Key behaviours (mirrors feature_ablation.py)
─────────────────────────────────────────────
1. Output dir  : results/ablation/<model_type>/timesteps/
2. Summary CSV : accumulates across multiple partial runs (merge strategy)
3. Resume      : skips timestep values that already have a metrics CSV on disk
4. Checkpoint  : controlled by --checkpoint flag (off by default for ablation)
5. WandB       : controlled by --use_wandb flag
6. Partial run : run 20ms first, then 40ms 80ms later
                 — summary CSV is merged/updated each time, never overwritten

Timestep grid is defined PER DATASET, so that the same *duration in ms*
can be compared fairly across GazeCom (250 Hz, 4 ms/step) and HMR
(200 Hz, 5 ms/step). Use --values to run only a subset of timesteps.

Target ms  ~ GazeCom timesteps ~ HMR timesteps
  4-5 ms   ->  1                ->  1
  10 ms    ->  2  (8 ms)        ->  2  (10 ms, exact)
  20 ms    ->  5  (exact)       ->  4  (exact)
  40 ms    ->  10 (exact)       ->  8  (exact)
  80 ms    ->  20 (exact)       ->  16 (exact)
  100 ms   ->  25 (exact)       ->  20 (exact)
  full win ->  250 (~1000 ms)   ->  200 (~1000 ms)   [anchor point,
                                                       matches original
                                                       Elmadjian window]
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
from train import main_kfold
from utils.logger import logger
from utils.helpers import set_randomness


# ------------------------------------------------------------------ #
# Per-dataset timestep grid — free to edit / subset via --values
# ------------------------------------------------------------------ #
FREQ_MS = {
    "gazecom": 4,   # 250 Hz -> 4 ms / sample
    "hmr":     5,   # 200 Hz -> 5 ms / sample
}

TIMESTEP_VALUES = {
    "gazecom": [1, 2, 5, 10, 20, 25, 250],
    "hmr":     [1, 2, 4, 8, 16, 20, 200],
}

CLASS_NAMES   = ["Fixation", "Saccade", "Pursuit", "Blink"]
FULL_FEATURES = ["speed", "direction", "stddev", "displacement"]


def _ablation_dir(model_type):
    """results/ablation/<model_type>/timesteps/"""
    return os.path.join("results", "ablation", model_type, "timesteps")


def _summary_path(model_type, dataset):
    return os.path.join(
        _ablation_dir(model_type),
        f"timestep_ablation_{dataset}_summary.csv"
    )


def _tag(t, ms_per_step):
    """
    e.g. timesteps=1, ms_per_step=4 -> '4ms'
    e.g. timesteps=5, ms_per_step=4 -> '20ms'

    Format: '{context_ms}ms' only — NOT '{ms}ms_t{t}'.

    train.py's build_config_tag() already appends 't{timesteps}' right
    after run_name (alongside h/d/k/do/lr/b), so adding our own 't{t}'
    here produced a duplicated '20ms_t5_t5_h4_d256_...' in the final
    run name / WandB name. Keeping only the ms part here gives a clean
    '20ms_t5_h4_d256_k3_do02_lr001_b2048_convattention' — ms up front
    for readability, timesteps (and everything else) still visible
    exactly once via build_config_tag.
    """
    return f"{t * ms_per_step}ms"


# ------------------------------------------------------------------ #
# Context manager: redirect save/plot outputs to ablation folder
# (identical pattern to feature_ablation.py)
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
# Summary CSV helpers — merge strategy for partial runs
# (identical pattern to feature_ablation.py)
# ------------------------------------------------------------------ #

def _load_summary(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def _merge_summary(existing_df, new_rows, tag_order):
    if not new_rows:
        return existing_df

    new_df = pd.DataFrame(new_rows)

    if existing_df.empty:
        merged = new_df
    else:
        new_tags = set(new_df["timestep_tag"].tolist())
        existing_df = existing_df[~existing_df["timestep_tag"].isin(new_tags)]
        merged = pd.concat([existing_df, new_df], ignore_index=True)

    merged["_order"] = merged["timesteps"].astype(int)
    merged = merged.sort_values("_order").drop(columns=["_order"])
    return merged.reset_index(drop=True)


def _save_summary(df, path):
    """Sort by timesteps ascending (context length) and save."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    key_cols = [
        "timestep_tag", "timesteps", "context_ms",
        "F1_avg", "F1_Fixation", "F1_Saccade", "F1_Pursuit", "F1_Blink",
        "ev_F1_avg", "ev_F1_Fixation", "ev_F1_Saccade",
        "ev_F1_Pursuit", "ev_F1_Blink",
        "roc_auc_macro", "roc_auc_micro",
        "params", "flops",
        "val_loss", "train_loss", "epochs_run",
    ]
    other_cols = [c for c in df.columns if c not in key_cols]
    df = df.sort_values("timesteps")
    df = df[[c for c in key_cols if c in df.columns] + other_cols]
    df.to_csv(path, index=False, float_format="%.4f")
    return path


def _already_done(tag, model_type):
    """
    Check if this timestep value already has a metrics CSV on disk.
    Folder: results/ablation/<model_type>/timesteps/<model_type>/
    Pattern: *_<tag>_*_metrics.csv
    """
    folder  = os.path.join(_ablation_dir(model_type), model_type)
    pattern = os.path.join(folder, f"*_{tag}_*_metrics.csv")
    found   = glob.glob(pattern)
    if found:
        logger.info(f"  [resume] Found existing result for [{tag}]: "
                    f"{os.path.basename(found[0])}")
    return bool(found)


# ------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------ #

def run(args, timesteps=None):
    set_randomness(42)

    if args.dataset not in TIMESTEP_VALUES:
        logger.error(f"No timestep grid defined for dataset: {args.dataset}")
        return

    ms_per_step = FREQ_MS[args.dataset]
    full_grid   = TIMESTEP_VALUES[args.dataset]

    stride    = args.stride or (10 if args.dataset == "gazecom" else 8)
    freq      = args.frequency or (250 if args.dataset == "gazecom" else 200)
    data_path = args.data_path or os.path.join(
        "dataset", "processed",
        f"{args.dataset}_s{stride}_f{freq}_w{args.window_length}_o{args.offset}"
    )

    ablation_base = _ablation_dir(args.model_type)
    summary_path  = _summary_path(args.model_type, args.dataset)
    wandb_project = f"ablation_timestep_{args.dataset}_{args.model_type}"
    date_str      = datetime.datetime.now().strftime("%Y%m%d")

    # Free choice of which timestep values to run — mirrors feature_ablation's
    # --combos behaviour. If a requested value isn't in the predefined grid
    # for this dataset, it still runs (grid is a suggestion, not a hard cap),
    # but a warning is logged so typos don't silently no-op.
    if timesteps:
        selected = timesteps
        unknown  = [t for t in selected if t not in full_grid]
        if unknown:
            logger.warning(
                f"Timesteps {unknown} are not in the predefined grid for "
                f"{args.dataset} ({full_grid}), but will still be run."
            )
    else:
        selected = full_grid

    # Resume support — skip values already done on disk
    to_run  = []
    skipped = []
    for t in selected:
        tag = _tag(t, ms_per_step)
        if getattr(args, "resume", False) and _already_done(tag, args.model_type):
            skipped.append(tag)
        else:
            to_run.append(t)

    logger.info("=" * 60)
    logger.info("TIMESTEP / CONTEXT-LENGTH ABLATION STUDY")
    logger.info(f"Dataset      : {args.dataset.upper()}  ({ms_per_step} ms/step)")
    logger.info(f"Model        : {args.model_type}")
    logger.info(f"Full grid    : {full_grid}")
    logger.info(f"Context (ms) : {[t * ms_per_step for t in full_grid]}")
    logger.info(f"To run       : {len(to_run)}"
                + (f", {len(skipped)} skipped (already done)" if skipped else ""))
    if skipped:
        logger.info(f"Skipped      : {skipped}")
    logger.info(f"Features     : {FULL_FEATURES}")
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
    logger.info(f"Summary      : {summary_path}")
    logger.info(f"WandB        : {'ON — ' + wandb_project if args.use_wandb else 'OFF'}")
    logger.info("=" * 60)

    pprep = Preprocessor()

    # Load data once — features fixed to full set
    train_X, train_Y, _, _ = pprep.load_data(
        data_path,
        stride=stride,
        selected_features=FULL_FEATURES,
    )
    logger.info(f"Input shape: {train_X.shape}")

    new_rows = []  # results from this run only

    for i, t in enumerate(to_run, 1):
        tag = _tag(t, ms_per_step)
        context_ms = t * ms_per_step
        logger.info(f"\n[{i}/{len(to_run)}] Timesteps: {t}  (~{context_ms} ms)")

        run_name = f"{date_str}_{tag}"

        try:
            with ablation_output_dir(ablation_base):
                all_metrics, _, _, _, _, _ = main_kfold(
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
                    "timestep_tag": tag,
                    "timesteps":    t,
                    "context_ms":   context_ms,
                }
                row.update({k: v for k, v in all_metrics[0].items()
                            if k != "fold"})
                new_rows.append(row)

                logger.info(
                    f"  Done [{tag}] (~{context_ms} ms) — "
                    f"F1={all_metrics[0].get('F1_avg', 0)*100:.2f}%  "
                    f"Pursuit={all_metrics[0].get('F1_Pursuit', 0)*100:.2f}%"
                )

                # Update summary after EACH timestep value — merge strategy
                # so partial runs accumulate into one CSV without overwriting.
                existing = _load_summary(summary_path)
                merged   = _merge_summary(existing, new_rows, full_grid)
                _save_summary(merged, summary_path)
                logger.info(f"  Summary updated: {summary_path} "
                            f"({len(merged)} total values)")

        except Exception as e:
            logger.error(f"  FAILED [{tag}]: {e}")
            logger.error("  Continuing with next timestep value...")
            continue

    # ── Final summary log ─────────────────────────────────────────────
    final_df = _load_summary(summary_path)
    if not final_df.empty and "F1_avg" in final_df.columns:
        logger.info("\nAll results by context length (ascending):")
        logger.info(f"  {'Tag':<8} {'ms':>6} {'F1_avg':>8} {'Pursuit':>8} {'ROC-macro':>10}")
        logger.info("  " + "-" * 46)
        for _, r in final_df.sort_values("timesteps").iterrows():
            logger.info(
                f"  {r['timestep_tag']:<8}"
                f"{r.get('context_ms', 0):>5.0f}ms"
                f"{r.get('F1_avg', 0)*100:>8.2f}%"
                f"{r.get('F1_Pursuit', 0)*100:>8.2f}%"
                f"{r.get('roc_auc_macro', 0):>10.4f}"
            )

    logger.info("\n" + "=" * 60)
    logger.info(f"TIMESTEP ABLATION — {len(to_run)} values ran, "
                f"{len(skipped)} skipped")
    logger.info(f"Summary  : {summary_path}")
    logger.info("=" * 60)