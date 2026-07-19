"""
ablation/run_ablation.py

Entry point for all ablation studies.
Run from project root:

    python ablation/run_ablation.py feature -d gazecom
    python ablation/run_ablation.py timestep -d gazecom

See --help for each ablation type.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ------------------------------------------------------------------ #
# Shared args (used by all ablation types)
# ------------------------------------------------------------------ #

def add_shared_args(parser):
    # Dataset
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, choices=["gazecom", "hmr"]
    )
    parser.add_argument("--data_path",     type=str,   default=None)
    parser.add_argument("--stride",        type=int,   default=None)
    parser.add_argument("--frequency",     type=int,   default=None)
    parser.add_argument("--window_length", type=float, default=1.0)
    parser.add_argument("--offset",        type=int,   default=0)

    # Model — fixed to conv_attention for ablation
    parser.add_argument(
        "--model_type", type=str, default="conv_attention",
        choices=["conv_attention", "tcn", "cnn_lstm", "cnn_bilstm"],
    )

    # Fixed hyperparams (best known config for conv_attention)
    parser.add_argument("--timesteps",   type=int,   default=5)
    parser.add_argument("--d_model",     type=int,   default=256)
    parser.add_argument("--num_heads",   type=int,   default=4)
    parser.add_argument("--kernel_size", type=int,   default=3)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--batch_size",  type=int,   default=2048)
    parser.add_argument("--patience",    type=int,   default=10)

    # K-Fold
    parser.add_argument("--use_kfold",  action="store_true")
    parser.add_argument("--n_splits",   type=int, default=5)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--max_folds",  type=int, default=5)

    # WandB
    parser.add_argument("--use_wandb", action="store_true")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation types:
  feature   — compare different feature combinations (speed, direction, stddev, displacement)
  timestep  — compare different timestep values [1, 5, 10, 15, 20, 25]

Examples:
  # Run all feature combos on GazeCom
  python ablation/run_ablation.py feature -d gazecom

  # Run only specific combos
  python ablation/run_ablation.py feature -d gazecom --combos sp dir sp_dir sp_dir_std_dis

  # Run timestep ablation (planned)
  python ablation/run_ablation.py timestep -d gazecom

  # With kfold and WandB
  python ablation/run_ablation.py feature -d gazecom --use_kfold --use_wandb
        """
    )

    subparsers = parser.add_subparsers(dest="ablation_type", required=True)

    # ── feature ablation ──────────────────────────────────────────────
    feat_parser = subparsers.add_parser(
        "feature",
        help="Feature combination ablation study"
    )
    add_shared_args(feat_parser)
    feat_parser.add_argument(
        "--combos", type=str, nargs="+", default=None, metavar="TAG",
        help=(
            "Run only specific combos by tag. "
            "E.g. --combos sp dir sp_dir sp_dir_std_dis. "
            "If omitted, runs all 15 combos."
        )
    )

    # ── timestep ablation ────────────────────────────────────────────
    ts_parser = subparsers.add_parser(
        "timestep",
        help="Timestep ablation study"
    )
    add_shared_args(ts_parser)
    ts_parser.add_argument(
        "--values", type=int, nargs="+", default=None, metavar="T",
        help=(
            "Run only specific timestep values. "
            "E.g. --values 5 10 25. "
            "If omitted, runs all values [1, 5, 10, 15, 20, 25]."
        )
    )

    args = parser.parse_args()

    if args.ablation_type == "feature":
        from ablation.feature_ablation import run
        run(args, combos=args.combos)

    elif args.ablation_type == "timestep":
        from ablation.timestep_ablation import run
        run(args, timesteps=args.values)


if __name__ == "__main__":
    main()