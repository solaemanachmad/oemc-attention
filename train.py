import os
import gc
import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score as sk_f1
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import datetime

from utils.logger import logger
from data.dataset import LOADER_REGISTRY
from models import get_model, init_weights_normal, print_summary
from utils.helpers import (EarlyStopping, log_flops, log_env,
                           save_checkpoint, save_model, save_results,
                           save_csv, save_json, set_prefix, log_config,
                           build_config_tag)
from utils.metrics import (print_scores, plot_cmcount,
                           plot_cmpercent, plot_roc, plot_pr, _get_event)
from utils.helpers import get_device

device = get_device()


# ------------------------------------------------------------------ #
# Default training config per model family
#
# All models default to:
#   optimizer : adamw
#   scheduler : cosine
#   loss      : nll  (plain NLLLoss, no class weights)
#               — proven best for eye movement classification
#
# TCN defaults to adamax + plateau to match Bai et al. (2018).
# Override any of these via CLI flags (--optimizer, --scheduler, --loss).
# ------------------------------------------------------------------ #
_MODEL_DEFAULTS = {
    #                  optimizer    scheduler  loss
    # Source paper (main.py get_optimizer):
    #   TCN          -> Adamax
    #   CNN_LSTM/BiLSTM -> RMSprop
    # Scheduler: paper uses manual lr/=2 on plateau — closest is "plateau"
    "conv_attention":  ("adamw",    "cosine",  "nll"),   # proposed model, not in paper
    "cnn_lstm":        ("rmsprop",  "plateau", "nll"),   # matches source paper
    "cnn_bilstm":      ("rmsprop",  "plateau", "nll"),   # matches source paper
    "tcn":             ("adamax",   "plateau", "nll"),   # matches source paper
}


def _resolve(value, model_type, key):
    """Return explicit CLI value if given, otherwise use model-family default."""
    if value is not None:
        return value
    return _MODEL_DEFAULTS[model_type][{"optimizer": 0, "scheduler": 1, "loss": 2}[key]]


# ------------------------------------------------------------------ #
# Train / eval steps
# ------------------------------------------------------------------ #

def train_step(model, optimizer, criterion, x, y):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss   = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(model, x, y):
    model.eval()
    with torch.no_grad():
        output = model(x)
        loss   = F.nll_loss(output, y).item()
        probs  = torch.exp(output)
        preds  = output.argmax(dim=1)
        return preds.cpu(), probs.cpu(), y.cpu(), loss


# ------------------------------------------------------------------ #
# Factories
# ------------------------------------------------------------------ #

def _make_loader(X, Y, timesteps, stride, batch_size, shuffle,
                 loader_mode="lookback"):
    """
    lookahead : forward window [i : i+timesteps], label = Y[i+timesteps-1]
    lookback  : backward window [i-timesteps : i], label = Y[i-1]
    """
    if loader_mode not in LOADER_REGISTRY:
        raise ValueError(f"Unknown loader_mode '{loader_mode}'. "
                         f"Choose from: {list(LOADER_REGISTRY.keys())}")
    dataset = LOADER_REGISTRY[loader_mode](X, Y, timesteps=timesteps, stride=stride)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=4)


def _make_optimizer(model, optimizer_type, lr):
    """
    adamw   : AdamW + weight_decay=1e-4  (default)
    adamax  : Adamax — TCN paper default
    rmsprop : RMSprop — legacy option
    """
    if optimizer_type == "adamax":
        return torch.optim.Adamax(model.parameters(), lr=lr)
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def _make_scheduler(optimizer, scheduler_type, epochs):
    """
    cosine  : CosineAnnealingLR    — step()
    plateau : ReduceLROnPlateau    — step(metric)
    step    : StepLR every 10 ep   — step()
    """
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    return CosineAnnealingLR(optimizer, T_max=epochs)


def _make_criterion(loss_type):
    """
    nll   : plain NLLLoss, no class weights — proven best (87% SP)
    nll_w : NLLLoss with balanced class weights — ablation option
    """
    if loss_type == "nll_w":
        # class_weights injected at call site
        return None   # sentinel — handled in train_model
    return torch.nn.NLLLoss()


# ------------------------------------------------------------------ #
# Core training loop
# ------------------------------------------------------------------ #

def train_model(
    X, Y, class_names, use_kfold=False,
    fold_idx=None, n_splits=5, start_fold=0, max_folds=4,
    timesteps=5, stride=1,
    d_model=256, dropout=0.2, epochs=300, batch_size=2048,
    lr=0.001, kernel_size=3, num_heads=4, patience=10,
    tcn_kernel_size=5, tcn_channel_size=30, tcn_num_levels=4,
    optimizer_type=None, scheduler_type=None, loss_type=None,
    loader_mode="lookback",
    close_wandb=True, model_type="conv_attention", model_params=None,
    wandb_project=None, run_name=None, checkpoint=True,
    plot_result=True, save_model_artifact=True,
    use_wandb=True, resume_path=None,
    base_dir="results",
):
    eff_optimizer = _resolve(optimizer_type, model_type, "optimizer")
    eff_scheduler = _resolve(scheduler_type, model_type, "scheduler")
    eff_loss      = _resolve(loss_type,      model_type, "loss")
    eff_loader    = loader_mode

    wandb_config = dict(
        timesteps=timesteps, epochs=epochs, batch_size=batch_size, lr=lr,
        patience=patience, class_names=class_names, use_kfold=use_kfold,
        n_splits=n_splits if use_kfold else None, model_type=model_type,
        optimizer=eff_optimizer, scheduler=eff_scheduler,
        loss=eff_loss, loader_mode=eff_loader,
        **(model_params or {})
    )

    if use_kfold:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=42).split(X, Y)
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(Y)), test_size=0.2, stratify=Y, random_state=42
        )
        folds = [(train_idx, val_idx)]

    all_metrics  = []
    best_labels = best_preds = best_probs = None
    fold_counter = 0

    if not use_kfold and use_wandb and wandb_project is not None:
        wandb.init(project=wandb_project, name=run_name,
                   config=wandb_config, reinit=True)
        # Note: prefix (with config_tag) is built inside the fold loop

    log_env()

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        epoch_logs = []
        if fold_idx < start_fold:
            continue
        if fold_counter >= max_folds:
            break
        fold_counter += 1

        config_tag = build_config_tag(
            model_type=model_type,
            timesteps=timesteps,
            # conv_attention
            num_heads=num_heads,
            d_model=d_model,
            kernel_size=kernel_size if model_type == "conv_attention"
                        else (tcn_kernel_size if model_type == "tcn"
                        else kernel_size),
            # tcn
            tcn_channel_size=tcn_channel_size if model_type == "tcn" else None,
            tcn_num_levels=tcn_num_levels if model_type == "tcn" else None,
            # cnn_lstm / cnn_bilstm
            lstm_layers=2 if model_type in ("cnn_lstm", "cnn_bilstm") else None,
            conv_filters=(32, 16, 8) if model_type in ("cnn_lstm", "cnn_bilstm") else None,
            # shared
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
        )
        prefix = set_prefix(fold_idx, run_name, model_type,
                               use_kfold, config_tag=config_tag)

        if use_wandb and wandb_project is not None and use_kfold:
            if wandb.run is not None:
                wandb.finish()
            wandb.init(project=wandb_project, name=prefix,
                       config=wandb_config, reinit=True, resume="allow")

        logger.info(f"=== Training: {prefix} ===")
        logger.info(f"optimizer={eff_optimizer}  scheduler={eff_scheduler}  "
                    f"loss={eff_loss}  loader={eff_loader}")

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val,   Y_val   = X[val_idx],   Y[val_idx]

        # DataLoaders
        train_loader = _make_loader(X_train, Y_train, timesteps, stride,
                                    batch_size, shuffle=True,
                                    loader_mode=eff_loader)
        val_loader   = _make_loader(X_val, Y_val, timesteps, stride,
                                    batch_size, shuffle=False,
                                    loader_mode=eff_loader)

        # Model
        input_size = X.shape[1]
        params = dict(model_params) if model_params else {}
        # TCN, CNN_LSTM, CNN_BiLSTM: input_size already set correctly in
        # model_params (= timesteps). Only override for conv_attention which
        # does not pre-set input_size.
        if "input_size" not in params:
            params["input_size"] = input_size
        params["output_size"] = len(class_names)

        model = get_model(model_type, params).to(device)
        if model_type == "cnn_bilstm":
            model.apply(init_weights_normal)

        # input_shape for FLOPs: all models receive (timesteps, features)
        # TCN transposes internally in forward()
        log_flops(model, prefix, model_type, use_kfold,
                  (timesteps, input_size), fold_idx=fold_idx)
        print_summary(model, model_type, input_size, timesteps)

        # Optimizer & Scheduler
        optimizer = _make_optimizer(model, eff_optimizer, lr)
        scheduler = _make_scheduler(optimizer, eff_scheduler, epochs)

        # Loss
        if eff_loss == "nll_w":
            from sklearn.utils.class_weight import compute_class_weight
            cw = compute_class_weight('balanced',
                                      classes=np.unique(Y_train), y=Y_train)
            cw = torch.tensor(cw, dtype=torch.float).to(device)
            criterion = torch.nn.NLLLoss(weight=cw)
        else:
            # "nll" — plain NLLLoss, no class weights (proven best)
            criterion = torch.nn.NLLLoss()

        early_stopping   = EarlyStopping(patience=patience)
        best_f1          = 0.0
        best_model_state = None
        best_val_loss = best_train_loss = None
        best_metrics     = {}

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            model.train()
            train_loss = 0
            for X_batch, Y_batch in tqdm(
                train_loader,
                desc=f"[{prefix}] Epoch {epoch}/{epochs}",
            ):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                train_loss += train_step(model, optimizer, criterion,
                                         X_batch, Y_batch)

            # Validate
            model.eval()
            val_loss = 0
            all_preds, all_labels, all_probs = [], [], []
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)
                    preds, probs, labels, loss = eval_step(model, X_batch, Y_batch)
                    val_loss += loss * X_batch.size(0)
                    all_preds.append(preds)
                    all_labels.append(labels)
                    all_probs.append(probs)

            val_loss       /= len(val_loader.dataset)
            all_preds       = torch.cat(all_preds)
            all_labels      = torch.cat(all_labels)
            all_probs       = torch.cat(all_probs)
            train_loss_avg  = train_loss / len(train_loader)

            (
                f1_macro, f1_fix, f1_sacc, f1_sp, f1_blink,
                prec_fix, prec_sacc, prec_sp, prec_blink,
                rec_fix, rec_sacc, rec_sp, rec_blink,
                roc_auc_micro, roc_auc_macro
            ) = print_scores(
                all_preds, all_probs, all_labels,
                val_loss, train_loss_avg,
                f"{prefix} Epoch {epoch}", class_names, device=device
            )

            # Event-level metrics
            ev_preds, ev_labels = _get_event(all_preds, all_labels)
            ev_f1 = sk_f1(ev_labels, ev_preds, average=None,
                          labels=np.arange(len(class_names)), zero_division=0)
            ev_f1_avg = sk_f1(ev_labels, ev_preds, average='macro', zero_division=0)
            ev_f1 = list(ev_f1) + [0] * (4 - len(ev_f1))  # pad if needed

            # ReduceLROnPlateau requires a metric; CosineAnnealingLR does not
            if eff_scheduler == "plateau":
                scheduler.step(f1_macro)
            else:
                scheduler.step()
            epoch_time = time.time() - epoch_start

            if use_wandb and wandb.run is not None:
                wandb.log({
                    # General
                    "epoch":              epoch,
                    "train_loss":         train_loss_avg,
                    "val_loss":           val_loss,
                    "learning_rate":      optimizer.param_groups[0]['lr'],
                    "time_sec":           epoch_time,
                    # Sample-level F1
                    "sample/F1_avg":      f1_macro,
                    "sample/F1_Fixation": f1_fix,
                    "sample/F1_Saccade":  f1_sacc,
                    "sample/F1_Pursuit":  f1_sp,
                    "sample/F1_Blink":    f1_blink,
                    # Sample-level Precision
                    "sample/Prec_Fixation": prec_fix,
                    "sample/Prec_Saccade":  prec_sacc,
                    "sample/Prec_Pursuit":  prec_sp,
                    "sample/Prec_Blink":    prec_blink,
                    # Sample-level Recall
                    "sample/Rec_Fixation":  rec_fix,
                    "sample/Rec_Saccade":   rec_sacc,
                    "sample/Rec_Pursuit":   rec_sp,
                    "sample/Rec_Blink":     rec_blink,
                    # ROC-AUC
                    "roc_auc_micro":      roc_auc_micro,
                    "roc_auc_macro":      roc_auc_macro,
                    # Event-level F1
                    "event/F1_avg":       ev_f1_avg,
                    "event/F1_Fixation":  ev_f1[0],
                    "event/F1_Saccade":   ev_f1[1],
                    "event/F1_Pursuit":   ev_f1[2],
                    "event/F1_Blink":     ev_f1[3],
                })

            epoch_logs.append({
                "fold":             fold_idx + 1 if use_kfold else 0,
                "epoch":            epoch,
                "train_loss":       train_loss_avg,
                "val_loss":         val_loss,
                "F1_avg":           f1_macro,
                "F1_Fixation":      f1_fix,
                "F1_Saccade":       f1_sacc,
                "F1_Pursuit":       f1_sp,
                "F1_Blink":         f1_blink,
                "Prec_Fixation":    prec_fix,
                "Prec_Saccade":     prec_sacc,
                "Prec_Pursuit":     prec_sp,
                "Prec_Blink":       prec_blink,
                "Rec_Fixation":     rec_fix,
                "Rec_Saccade":      rec_sacc,
                "Rec_Pursuit":      rec_sp,
                "Rec_Blink":        rec_blink,
                "roc_auc_micro":    roc_auc_micro,
                "roc_auc_macro":    roc_auc_macro,
                "ev_F1_avg":        ev_f1_avg,
                "ev_F1_Fixation":   ev_f1[0],
                "ev_F1_Saccade":    ev_f1[1],
                "ev_F1_Pursuit":    ev_f1[2],
                "ev_F1_Blink":      ev_f1[3],
            })

            # Monitor -val_loss (minimize loss) — matches original working
            # code that achieved 87% SP. Monitoring f1_macro caused
            # premature stopping before SP converged.
            previous_score = early_stopping.best_score
            early_stopping(-val_loss, model)

            if previous_score is not None and early_stopping.counter > 0:
                logger.info(
                    f"Early stopping counter: {early_stopping.counter}/{patience} | "
                    f"Improvement: {(-val_loss) - previous_score:.4f}"
                )
            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            if f1_macro > best_f1:
                best_f1          = f1_macro
                best_preds       = all_preds.clone()
                best_labels      = all_labels.clone()
                best_probs       = all_probs.clone()
                best_model_state = model.state_dict()
                best_val_loss    = val_loss
                best_train_loss  = train_loss_avg
                # Save all per-class metrics from best epoch
                best_metrics = {
                    # Sample-level F1
                    "F1_avg":           float(f1_macro),
                    "F1_Fixation":      float(f1_fix),
                    "F1_Saccade":       float(f1_sacc),
                    "F1_Pursuit":       float(f1_sp),
                    "F1_Blink":         float(f1_blink),
                    # Sample-level Precision
                    "Prec_Fixation":    float(prec_fix),
                    "Prec_Saccade":     float(prec_sacc),
                    "Prec_Pursuit":     float(prec_sp),
                    "Prec_Blink":       float(prec_blink),
                    # Sample-level Recall
                    "Rec_Fixation":     float(rec_fix),
                    "Rec_Saccade":      float(rec_sacc),
                    "Rec_Pursuit":      float(rec_sp),
                    "Rec_Blink":        float(rec_blink),
                    # Event-level F1
                    "ev_F1_avg":        float(ev_f1_avg),
                    "ev_F1_Fixation":   float(ev_f1[0]),
                    "ev_F1_Saccade":    float(ev_f1[1]),
                    "ev_F1_Pursuit":    float(ev_f1[2]),
                    "ev_F1_Blink":      float(ev_f1[3]),
                    # ROC-AUC
                    "roc_auc_micro":    float(roc_auc_micro) if roc_auc_micro else None,
                    "roc_auc_macro":    float(roc_auc_macro) if roc_auc_macro else None,
                }

            if checkpoint:
                save_checkpoint(model, optimizer, epoch, prefix,
                                use_kfold, fold_idx, base_dir, model_type)

        # Post-fold
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Best model restored — F1: {best_f1:.4f}")
            print_scores(
                best_preds, best_probs, best_labels,
                best_val_loss, best_train_loss,
                f"{prefix} FINAL BEST", class_names,
                log_detail=True, device=device
            )

        results_dict = {"preds": best_preds, "labels": best_labels,
                        "probs": best_probs}
        model_path   = save_model(model, prefix, use_kfold, fold_idx, base_dir, model_type)
        results_path = save_results(results_dict, prefix, use_kfold, fold_idx,
                                    base_dir, model_type)
        json_path    = save_json(epoch_logs, prefix, use_kfold, fold_idx,
                                 base_dir, model_type)

        all_metrics.append({
            "fold":         fold_idx + 1 if use_kfold else 0,
            "val_loss":     float(best_val_loss) if best_val_loss else None,
            "train_loss":   float(best_train_loss) if best_train_loss else None,
            "epochs_run":   epoch,
            **best_metrics,   # all per-class metrics from best epoch
        })
        csv_path = save_csv(all_metrics, prefix, use_kfold, fold_idx,
                            base_dir, model_type)

        # Upload artifacts to WandB
        if use_wandb and wandb.run is not None:
            artifact = wandb.Artifact(
                name=prefix,
                type="model",
                description=f"Best model for {prefix} — F1: {best_f1:.4f}",
                metadata={**best_metrics, "fold": fold_idx + 1 if use_kfold else 0},
            )
            artifact.add_file(model_path,   name="model.pt")
            artifact.add_file(results_path, name="results.pt")
            artifact.add_file(json_path,    name="epoch_logs.json")
            artifact.add_file(csv_path,     name="metrics.csv")
            wandb.run.log_artifact(artifact)
            logger.info(f"WandB artifact uploaded: {prefix}")

            # Set best metrics as WandB run summary
            # (visible in WandB runs table without opening each run)
            for k, v in best_metrics.items():
                wandb.run.summary[k] = v
            wandb.run.summary["val_loss"]   = best_val_loss
            wandb.run.summary["train_loss"] = best_train_loss
            wandb.run.summary["epochs_run"] = epoch
            wandb.run.summary["fold"]       = fold_idx + 1 if use_kfold else 0

        if plot_result:
            plot_cmcount(
                prefix, best_labels, best_preds, class_names,
                use_kfold, fold_idx, base_dir, wandb.run, model_type)
            plot_cmpercent(
                prefix, best_labels, best_preds, class_names,
                use_kfold, fold_idx, base_dir, wandb.run, model_type)
            plot_roc(
                prefix, best_labels, best_probs, class_names,
                use_kfold, fold_idx, base_dir, wandb.run, model_type)
            plot_pr(
                prefix, best_labels, best_probs, class_names,
                use_kfold, fold_idx, base_dir, wandb.run, model_type)

        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()
        gc.collect()

        if use_wandb and wandb.run is not None and (use_kfold or close_wandb):
            wandb.finish()

        if not use_kfold:
            break

    return all_metrics, best_labels, best_preds, best_probs, class_names, run_name


# ------------------------------------------------------------------ #
# Public entry point
# ------------------------------------------------------------------ #

def main_kfold(
    X, Y, model_type="conv_attention", resume_path=None,
    class_names=["Fixation", "Saccade", "Pursuit", "Blink"],
    run_name=None, timesteps=5, d_model=256, dropout=0.3,
    lr=0.001, num_heads=4, kernel_size=3,
    tcn_kernel_size=5, tcn_channel_size=30, tcn_num_levels=4,
    optimizer_type=None, scheduler_type=None, loss_type=None,
    loader_mode="lookahead",
    batch_size=2048, epochs=20, patience=10,
    wandb_project="oemc", checkpoint=True, plot_result=True,
    close_wandb=True, use_kfold=True, n_splits=5,
    start_fold=0, max_folds=5, use_wandb=True,
):
    if run_name is None:
        run_name = f"{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model_params = {}
    if model_type == "conv_attention":
        model_params = {
            "d_model":    d_model,
            "num_heads":  num_heads,
            "kernel_size":kernel_size,
            "dropout":    dropout,
            "output_size":len(class_names),
        }
    elif model_type == "tcn":
        # Bai et al. (2018): num_channels=[30]*4, kernel_size=8
        # input_size = timesteps — matches source paper exactly:
        #   TCN(args.timesteps, n_classes, layers, kernel_size, dropout)
        # Conv1d slides over the feature axis (no transpose in forward())
        model_params = {
            "input_size":   timesteps,
            "output_size":  len(class_names),
            "num_channels": [tcn_channel_size] * tcn_num_levels,
            "kernel_size":  tcn_kernel_size,
            "dropout":      dropout,
        }
    elif model_type == "cnn_lstm":
        # Source: CNN_LSTM(input_size, output_size, kernel_size, dropout,
        #                  features, lstm_layers, conv_filters)
        # input_size = timesteps (Conv1d channel dim, matches source)
        # features   = X.shape[1] (LSTM input size)
        model_params = {
            "input_size":   timesteps,
            "output_size":  len(class_names),
            "kernel_size":  kernel_size,
            "dropout":      dropout,
            "features":     X.shape[1],
            "lstm_layers":  2,
            "conv_filters": (32, 16, 8),
        }
    elif model_type == "cnn_bilstm":
        # Source: CNN_BiLSTM(input_size, output_size, kernel_size, dropout,
        #                    features, blstm_layers, conv_filters)
        # input_size = timesteps (Conv1d channel dim, matches source)
        # features   = X.shape[1] (BiLSTM input size)
        model_params = {
            "input_size":    timesteps,
            "output_size":   len(class_names),
            "kernel_size":   kernel_size,
            "dropout":       dropout,
            "features":      X.shape[1],
            "blstm_layers":  2,
            "conv_filters":  (32, 16, 8),
        }

    if use_wandb and wandb.run is not None:
        wandb_config = {"model_type": model_type, "epochs": epochs,
                        "batch_size": batch_size, "timesteps": timesteps,
                        "patience": patience, "optimizer": optimizer_type,
                        "scheduler": scheduler_type, "loss": loss_type,
                        "loader_mode": loader_mode}
        wandb_config.update(model_params)
        wandb.config.update(wandb_config)
        log_config(wandb_config, run_name, use_kfold, model_type)

    return train_model(
        X=X, Y=Y, resume_path=resume_path, class_names=class_names,
        timesteps=timesteps, d_model=d_model, dropout=dropout,
        epochs=epochs, batch_size=batch_size, lr=lr,
        num_heads=num_heads, kernel_size=kernel_size,
        tcn_kernel_size=tcn_kernel_size, tcn_channel_size=tcn_channel_size,
        tcn_num_levels=tcn_num_levels,
        optimizer_type=optimizer_type, scheduler_type=scheduler_type,
        loss_type=loss_type, loader_mode=loader_mode,
        patience=patience, wandb_project=wandb_project,
        checkpoint=checkpoint, plot_result=plot_result,
        close_wandb=close_wandb, use_kfold=use_kfold,
        n_splits=n_splits, start_fold=start_fold, max_folds=max_folds,
        model_type=model_type, model_params=model_params,
        run_name=run_name, use_wandb=use_wandb,
    )