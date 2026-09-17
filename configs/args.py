import argparse
import os


def data_defaults(args):
    stride = args.stride or (10 if args.dataset == "gazecom" else 8)
    freq = args.frequency or (250 if args.dataset == "gazecom" else 200)
    data_path = args.data_path or os.path.join(
        "dataset", "processed",
        f"{args.dataset}_s{stride}_f{freq}_w{args.window_length}_o{args.offset}"
    )
    return stride, freq, data_path


def get_args():
    parser = argparse.ArgumentParser(
        description="Eye Movement Classification - Preprocessing and Training"
    )

    # ------------------------------------------------------------------ #
    # DATASET & PREPROCESSING
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--raw_data_path', type=str, default="dataset/data_gazecom",
        help="Path to the raw dataset folder (contains TSV/CSV files)"
    )
    parser.add_argument(
        '--processed_data_path', type=str, default="dataset/processed",
        help="Path to save or load extracted features (.npz)"
    )
    parser.add_argument(
        '-d', '--dataset', type=str, required=True, choices=["gazecom", "hmr"],
        help="Dataset to use (required)"
    )
    parser.add_argument(
        '--window_length', type=float, default=1.0,
        help="Sliding window length in seconds"
    )
    parser.add_argument(
        '--offset', type=int, default=0,
        help="Label offset relative to the last sample of each window"
    )
    parser.add_argument(
        '--stride', type=int, default=None,
        help="Preprocessing stride; overrides dataset default when set"
    )
    parser.add_argument(
        '--frequency', type=int, default=None,
        help="Sampling frequency in Hz; overrides dataset default when set"
    )

    # ------------------------------------------------------------------ #
    # RUN / CHECKPOINT
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '-r', '--run_name', type=str, required=False, default=None,
        help=(
            "Name for this run (used for file naming and WandB). "
            "If omitted, auto-generated from date: YYYYMMDD_HHMMSS. "
            "Tip: use a config string like 'nh4_d256_lr001' for easy comparison."
        )
    )
    parser.add_argument(
        '--checkpoint', action=argparse.BooleanOptionalAction, default=True,
        help="Save model checkpoints per epoch (use --no-checkpoint to disable)"
    )

    # ------------------------------------------------------------------ #
    # MODEL SELECTION
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '-m', '--model_type', type=str, required=True,
        choices=["conv_attention", "tcn", "cnn_lstm", "cnn_bilstm", "skip_attseqnet"],
        help="Model architecture to train (required)"
    )
    parser.add_argument(
        '--data_path', type=str, default=None,
        help="Direct path to preprocessed dataset; skips auto-path construction"
    )

    # ------------------------------------------------------------------ #
    # SHARED MODEL HYPERPARAMETERS
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--timesteps', type=int, default=5,
        help="Number of timesteps (sequence length) fed to the model"
    )
    parser.add_argument(
        '--d_model', type=int, default=256,
        help="Hidden / filter dimension (used by conv_attention, cnn_transformer)"
    )
    parser.add_argument(
        '--num_heads', type=int, default=4,
        help="Number of attention heads (conv_attention, cnn_transformer)"
    )
    parser.add_argument(
        '--kernel_size', type=int, default=3,
        help="Convolution kernel size for non-TCN models"
    )
    parser.add_argument(
        '--dropout', type=float, default=0.2,
        help="Dropout rate applied across all models"
    )

    # ------------------------------------------------------------------ #
    # TCN-SPECIFIC HYPERPARAMETERS  (Bai et al. 2018)
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--tcn_kernel_size', type=int, default=5,
        help="Convolution kernel size for TCN (paper default: 5, source argparser default)"
    )
    parser.add_argument(
        '--tcn_channel_size', type=int, default=30,
        help="Number of filters per TCN level (paper default: 30)"
    )
    parser.add_argument(
        '--tcn_num_levels', type=int, default=4,
        help="Number of TCN levels; produces num_channels=[size]*levels (paper default: 4)"
    )

    # ------------------------------------------------------------------ #
    # SKIP-ATTSEQNET-SPECIFIC HYPERPARAMETERS  (Wang et al. 2025, BSPC)
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--skip_cnn_dropout', type=float, default=0.2,
        help="Dropout after each encoder conv block for Skip-AttSeqNet (paper default: 0.2)"
    )
    parser.add_argument(
        '--skip_rnn_dropout', type=float, default=0.3,
        help="Dropout on the BLSTM decoder for Skip-AttSeqNet (paper default: 0.3)"
    )

    # ------------------------------------------------------------------ #
    # TRAINING HYPERPARAMETERS
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '-e', '--epochs', type=int, default=300,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        '-b', '--batch_size', type=int, default=2048,
        help="Mini-batch size"
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help="Early-stopping patience (epochs without val_loss improvement)"
    )
    parser.add_argument(
        '--grad_clip_norm', type=float, default=1.0,
        help=(
            "Max gradient norm for gradient clipping (applied every "
            "training step, all models). Stabilizes training for "
            "LSTM-based models (cnn_lstm, cnn_bilstm, skip_attseqnet), "
            "which can otherwise collapse to predicting only the "
            "majority class on some folds. Set to a large value "
            "(e.g. 1e9) to effectively disable clipping."
        )
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=1,
        help=(
            "Number of initial epochs to linearly ramp up the learning "
            "rate from a small fraction of --lr to --lr, before the "
            "normal scheduler takes over. Mitigates adaptive-optimizer "
            "(e.g. RMSprop) cold-start instability, which can push an "
            "LSTM-based model's gates into saturation early on and "
            "cause a fold-dependent collapse to the majority class — a "
            "failure mode that gradient clipping alone does not fix, "
            "since it is caused by an overly large *effective* step "
            "size early in training, not by a large raw gradient norm. "
            "Set to 0 to disable warmup entirely."
        )
    )

    # ------------------------------------------------------------------ #
    # OPTIMIZER
    # Defaults per model family:
    #   conv_attention / cnn / bilstm / cnn_* : adamw
    #   tcn (paper replication)               : adamax
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--optimizer', type=str, default=None,
        choices=["adamw", "adamax", "rmsprop", "adam"],
        help=(
            "Optimizer to use. When omitted the default is chosen per model: "
            "adamw for all models except TCN paper-replication (adamax). "
            "Override explicitly to experiment across models."
        )
    )

    # ------------------------------------------------------------------ #
    # LR SCHEDULER
    # Defaults per model family:
    #   most models  : cosine  (CosineAnnealingLR)
    #   tcn + paper  : elmadjian (exact paper replication of the
    #                  manual lr/=2-on-plateau rule; see train.py)
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--scheduler', type=str, default=None,
        choices=["cosine", "plateau", "step", "elmadjian"],
        help=(
            "LR scheduler. When omitted the default is chosen per model: "
            "cosine for conv_attention, elmadjian for TCN/CNN-LSTM/"
            "CNN-BiLSTM paper-replication baselines. Override explicitly "
            "to experiment across models."
        )
    )

    # ------------------------------------------------------------------ #
    # LOSS FUNCTION
    # Default for all models: nll (plain NLLLoss, no class weights)
    # Proven best — plain NLLLoss achieved 87% SP on conv_attention.
    # FocalLoss and CrossEntropyLoss removed: hurt performance on this task.
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--loss', type=str, default=None,
        choices=["nll", "nll_w"],
        help=(
            "Loss function. When omitted the default is chosen per model. "
            "'nll': plain NLLLoss, no class weights (default, proven best at 87%% SP). "
            "'nll_w': NLLLoss with balanced class weights (ablation option)."
        )
    )

    # ------------------------------------------------------------------ #
    # DATA LOADER MODE
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--loader_mode', type=str, default="lookahead",
        choices=["lookahead", "lookback"],
        help=(
            "Windowing strategy for the data loader. "
            "'lookahead' (default, proven best): forward window [i : i+timesteps], label = Y[i+timesteps-1]. "
            "'lookback': lookback window [i-timesteps : i], label = Y[i-1], "
            "matching the create_batches() behaviour of the Bai et al. reference code."
        )
    )

    # ------------------------------------------------------------------ #
    # CROSS-VALIDATION & WANDB
    # ------------------------------------------------------------------ #
    parser.add_argument(
        '--use_kfold', action='store_true',
        help="Use Stratified K-Fold cross-validation instead of a single holdout split"
    )
    parser.add_argument(
        '--n_splits', type=int, default=5,
        help="Number of folds for K-Fold (ignored when --use_kfold is not set)"
    )
    parser.add_argument(
        '--start_fold', type=int, default=0,
        help="First fold index to run (useful for resuming)"
    )
    parser.add_argument(
        '--max_folds', type=int, default=None,
        help=(
            "Maximum number of folds to actually run out of --n_splits. "
            "Defaults to --n_splits (i.e. run ALL folds) when omitted — "
            "set this lower only if you deliberately want a partial run "
            "(e.g. quick smoke test on fold 1-2 only)."
        )
    )
    parser.add_argument(
        '--wandb_project', type=str, default="oemc_project",
        help="WandB project name"
    )
    parser.add_argument(
        '--use_wandb', action='store_true',
        help="Enable experiment logging to Weights & Biases"
    )

    args = parser.parse_args()

    # max_folds defaults to n_splits (run ALL folds) unless explicitly
    # overridden — previously this silently defaulted to a hardcoded 4,
    # causing kfold runs to stop early even when --n_splits 5 was passed.
    if args.max_folds is None:
        args.max_folds = args.n_splits

    return args