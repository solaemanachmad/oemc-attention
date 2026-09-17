import os
import datetime
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from configs.args import get_args, data_defaults
from data.preprocessor import Preprocessor
from train import main_kfold, _resolve
from utils.logger import logger
from utils.helpers import set_randomness


def main():
    args = get_args()
    set_randomness(42)

    stride, freq, data_path = data_defaults(args)

    # Auto run_name: user value or YYYYMMDD_HHMMSS
    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # WandB project: oemc_<dataset>_<model_type>
    # Override with --wandb_project if a custom name is needed.
    wandb_project = (
        args.wandb_project
        if args.wandb_project != "oemc_project"   # user explicitly set it
        else f"oemc_{args.dataset}_{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d')}"
    )

    # Resolve None -> actual model-family default (e.g. tcn -> adamax/plateau/nll)
    # so the log shows what is really used, not a placeholder.
    eff_optimizer = _resolve(args.optimizer, args.model_type, "optimizer")
    eff_scheduler = _resolve(args.scheduler, args.model_type, "scheduler")
    eff_loss      = _resolve(args.loss,      args.model_type, "loss")

    logger.info(f"Dataset       : {args.dataset.upper()}")
    logger.info(f"Data path     : {data_path}")
    logger.info(f"Stride        : {stride}")
    logger.info(f"Run name      : {run_name}")
    logger.info(f"WandB project : {wandb_project}")
    logger.info(
        f"Training config - "
        f"optimizer={eff_optimizer}  "
        f"scheduler={eff_scheduler}  "
        f"loss={eff_loss}  "
        f"loader={args.loader_mode}"
    )

    pprep = Preprocessor()
    train_X, train_Y, test_X, test_Y = pprep.load_data(
        data_path,
        stride=stride,
        selected_features=['speed', 'direction', 'stddev', 'displacement']
    )

    class_names = ["Fixation", "Saccade", "Pursuit", "Blink"]

    main_kfold(
        X=train_X,
        Y=train_Y,

        # Identity
        run_name=run_name,
        model_type=args.model_type,
        class_names=class_names,
        dataset=args.dataset,

        # Shared hyperparams
        timesteps=args.timesteps,
        d_model=args.d_model,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        warmup_epochs=args.warmup_epochs,

        # TCN-specific
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_channel_size=args.tcn_channel_size,
        tcn_num_levels=args.tcn_num_levels,

        # Skip-AttSeqNet-specific
        skip_cnn_dropout=args.skip_cnn_dropout,
        skip_rnn_dropout=args.skip_rnn_dropout,

        # Training config (None = use model-family default)
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        loss_type=args.loss,
        loader_mode=args.loader_mode,

        # CV & logging
        checkpoint=args.checkpoint,
        use_kfold=args.use_kfold,
        n_splits=args.n_splits,
        start_fold=args.start_fold,
        max_folds=args.max_folds,
        wandb_project=wandb_project,
        use_wandb=args.use_wandb,
    )


if __name__ == "__main__":
    main()