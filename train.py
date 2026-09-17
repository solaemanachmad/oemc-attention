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
from utils.helpers import get_device, is_directml_device

# NOTE: `device` is no longer computed once at module level. Some model
# types (cnn_lstm, cnn_bilstm) use nn.LSTM, which is unsupported on
# torch-directml (missing aten::_thnn_fused_lstm_cell). `device` is now
# computed per-call inside train_model(), forcing CPU for those model
# types specifically while other models (tcn, conv_attention) still use
# the detected GPU backend. See get_device(force_cpu=...) in helpers.py.


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
    #                  optimizer    scheduler          loss
    # Source paper (Elmadjian et al. 2023, main.py get_optimizer):
    #   TCN             -> Adamax
    #   CNN_LSTM/BiLSTM -> RMSprop
    # Scheduler: paper's exact rule (per github.com/elmadjian/OEMC/main.py):
    #   if len(scores) >= 3 and (abs(scores[-1]-scores[-3]) < 0.1
    #                             or scores[-1] < scores[-3]): lr /= 2
    #   Replicated exactly via "elmadjian" (see _elmadjian_lr_step).
    #   "plateau" (PyTorch ReduceLROnPlateau) remains available as an
    #   approximation for models/experiments that don't need exact
    #   replication of the original rule.
    "conv_attention":  ("adamw",    "cosine",           "nll"),   # proposed model, not in paper
    "cnn_lstm":        ("rmsprop",  "elmadjian", "nll"),   # matches source paper exactly
    "cnn_bilstm":      ("rmsprop",  "elmadjian", "nll"),   # matches source paper exactly
    "tcn":             ("adamax",   "elmadjian", "nll"),   # matches source paper exactly
    # Wang et al. 2025 (BSPC) Table 3: optimizer=Adam, lr=0.001. No LR
    # scheduler / decay rule is reported in the paper, so we default to
    # "cosine" (this project's general-purpose scheduler) rather than
    # inventing an unreported rule. No public code exists for this
    # model — see skip_attseqnet.py docstring for reimplementation notes.
    "skip_attseqnet":  ("adam",     "cosine",           "nll"),   # best-effort reimplementation, no public code
}


def _resolve(value, model_type, key):
    """Return explicit CLI value if given, otherwise use model-family default."""
    if value is not None:
        return value
    return _MODEL_DEFAULTS[model_type][{"optimizer": 0, "scheduler": 1, "loss": 2}[key]]


# ------------------------------------------------------------------ #
# Train / eval steps
# ------------------------------------------------------------------ #

def train_step(model, optimizer, criterion, x, y, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss   = criterion(output, y)
    loss.backward()
    # Gradient clipping — stabilizes training for RNN/LSTM-based models
    # (cnn_lstm, cnn_bilstm, skip_attseqnet), which can otherwise suffer
    # exploding gradients and collapse to predicting only the majority
    # class (observed empirically: some folds converged to Recall=1.0
    # for Fixation and 0.0 for every other class). This is a no-op for
    # already-stable gradients (norm < max_grad_norm), so it does not
    # change behavior for models/folds that were already training
    # normally.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
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

# num_workers=4 crashes on Windows for large in-memory tensors: Windows'
# spawn-based multiprocessing must pickle the whole dataset to send to
# each worker process, which fails with "OSError: [Errno 22] Invalid
# argument" / "pickle data was truncated" for datasets of this size
# (millions of samples). Linux (Kaggle, most servers) uses fork instead,
# which shares memory via copy-on-write and doesn't hit this limit, so
# workers stay enabled there for speed.
import platform
_NUM_WORKERS = 0 if platform.system() == "Windows" else 4


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
                      shuffle=shuffle, num_workers=_NUM_WORKERS)


def _make_optimizer(model, optimizer_type, lr):
    """
    adamw   : AdamW + weight_decay=1e-4  (default)
    adamax  : Adamax — TCN paper default
    rmsprop : RMSprop — CNN-LSTM/CNN-BiLSTM paper default
    adam    : plain Adam, no weight decay — Skip-AttSeqNet paper default
    """
    if optimizer_type == "adamax":
        return torch.optim.Adamax(model.parameters(), lr=lr)
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def _make_scheduler(optimizer, scheduler_type, epochs):
    """
    cosine            : CosineAnnealingLR    — step()
    plateau           : ReduceLROnPlateau    — step(metric); approximates
                         the paper's manual rule via PyTorch's built-in
                         cumulative-patience mechanism (factor=0.5 matches
                         the paper's lr/=2, but the trigger condition
                         differs — see 'elmadjian' for an exact
                         replication of the original rule).
    step              : StepLR every 10 ep   — step()
    elmadjian  : No-op here — this scheduler type is handled
                         entirely inside the training loop (see
                         _elmadjian_lr_step below), since the original
                         rule needs a 3-epoch lookback over validation
                         F1-macro history, which doesn't fit PyTorch's
                         standard scheduler.step(metric) interface.
                         Returns None; the optimizer's lr is mutated
                         in-place by _elmadjian_lr_step each epoch.
    """
    if scheduler_type == "elmadjian":
        return None
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    return CosineAnnealingLR(optimizer, T_max=epochs)


def _elmadjian_lr_step(optimizer, f1_history, prefix, epoch):
    """
    Exact replication of the manual LR decay rule from the original
    OEMC codebase (github.com/elmadjian/OEMC/blob/main/main.py):

        if len(scores) >= 3 and (abs(scores[-1]-scores[-3]) < 0.1
                                  or scores[-1] < scores[-3]):
            lr /= 2

    where `scores` is the running list of validation F1-macro values
    IN PERCENTAGE SCALE (e.g. 85.3, not 0.853) — the 0.1 threshold is
    calibrated to that scale in the original code, so f1_history must
    also be passed in percentage scale to match behaviour exactly.

    Called once per epoch, after f1_macro for the current epoch has
    been appended to f1_history.
    """
    if len(f1_history) < 3:
        return
    plateaued = abs(f1_history[-1] - f1_history[-3]) < 0.1
    declined  = f1_history[-1] < f1_history[-3]
    if plateaued or declined:
        for param_group in optimizer.param_groups:
            param_group["lr"] /= 2
        logger.info(
            f"[{prefix}] Epoch {epoch}: F1-macro plateaued/declined "
            f"(prev={f1_history[-3]:.2f}%, now={f1_history[-1]:.2f}%) "
            f"— halving LR to {optimizer.param_groups[0]['lr']:.6f} "
            f"(Elmadjian et al. manual rule)"
        )


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
    base_dir="results", dataset=None, grad_clip_norm=1.0, warmup_epochs=1,
):
    eff_optimizer = _resolve(optimizer_type, model_type, "optimizer")
    eff_scheduler = _resolve(scheduler_type, model_type, "scheduler")
    eff_loss      = _resolve(loss_type,      model_type, "loss")
    eff_loader    = loader_mode

    # cnn_lstm / cnn_bilstm / skip_attseqnet all use nn.LSTM (the latter
    # via its BLSTM decoder). torch-directml specifically cannot run
    # nn.LSTM (missing aten::_thnn_fused_lstm_cell) — CUDA, MPS, and
    # plain CPU all support it natively. So we only override to CPU
    # when the *actually detected* device is DirectML, not just
    # because the model_type is LSTM-based (fixes a bug where this
    # used to force CPU unconditionally for these model types even on
    # CUDA machines like Kaggle's T4, wasting the GPU for no reason).
    device = get_device()
    if model_type in ("cnn_lstm", "cnn_bilstm", "skip_attseqnet") and is_directml_device(device):
        logger.info(
            f"Model '{model_type}' uses nn.LSTM, which torch-directml "
            f"cannot run (missing aten::_thnn_fused_lstm_cell) — "
            f"switching from DirectML to CPU for this model only."
        )
        device = torch.device("cpu")

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
            # cnn_lstm / cnn_bilstm / skip_attseqnet
            lstm_layers=2 if model_type in ("cnn_lstm", "cnn_bilstm") else None,
            conv_filters=(
                (32, 16, 8) if model_type in ("cnn_lstm", "cnn_bilstm")
                else (32, 16, 8, 8) if model_type == "skip_attseqnet"
                else None
            ),
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
                  (timesteps, input_size), fold_idx=fold_idx, dataset=dataset)
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
        # F1-macro history in PERCENTAGE scale, used only by
        # eff_scheduler == "elmadjian" (see _elmadjian_lr_step).
        f1_history_pct   = []

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Linear LR warmup: for the first `warmup_epochs`, override the
            # optimizer's LR to ramp up linearly from a small fraction of
            # the target LR. Adaptive optimizers (RMSprop especially) start
            # with a zero running average of squared gradients, so early
            # updates can take an effectively huge step even with a
            # "normal" raw gradient (unlike exploding gradients, this is
            # NOT caught by gradient-norm clipping). This can push an
            # LSTM's gates into saturation before it has learned anything,
            # from which it may never recover (observed empirically:
            # collapse to predicting only the majority class, fold-
            # dependent since each fold's model gets a different random
            # initialization). The scheduler is intentionally not
            # stepped during warmup — its schedule begins once warmup ends.
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                warmup_lr = lr * epoch / (warmup_epochs + 1)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

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
                                         X_batch, Y_batch,
                                         max_grad_norm=grad_clip_norm)

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

            # ReduceLROnPlateau requires a metric; CosineAnnealingLR does not;
            # elmadjian mutates optimizer lr directly (no scheduler object).
            # During warmup, the scheduler is intentionally NOT stepped —
            # its schedule (and elmadjian's plateau-history) starts fresh
            # once warmup ends, rather than being offset by warmup epochs.
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                pass
            elif eff_scheduler == "elmadjian":
                f1_history_pct.append(f1_macro * 100)
                _elmadjian_lr_step(optimizer, f1_history_pct, prefix, epoch)
            elif eff_scheduler == "plateau":
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
                                use_kfold, fold_idx, base_dir, model_type,
                                dataset=dataset)

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
        model_path   = save_model(model, prefix, use_kfold, fold_idx, base_dir,
                                  model_type, dataset=dataset)
        results_path = save_results(results_dict, prefix, use_kfold, fold_idx,
                                    base_dir, model_type, dataset=dataset)
        json_path    = save_json(epoch_logs, prefix, use_kfold, fold_idx,
                                 base_dir, model_type, dataset=dataset)

        all_metrics.append({
            "fold":         fold_idx + 1 if use_kfold else 0,
            "val_loss":     float(best_val_loss) if best_val_loss else None,
            "train_loss":   float(best_train_loss) if best_train_loss else None,
            "epochs_run":   epoch,
            **best_metrics,   # all per-class metrics from best epoch
        })
        csv_path = save_csv(all_metrics, prefix, use_kfold, fold_idx,
                            base_dir, model_type, dataset=dataset)

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
                use_kfold, fold_idx, base_dir, wandb.run, model_type,
                dataset=dataset)
            plot_cmpercent(
                prefix, best_labels, best_preds, class_names,
                use_kfold, fold_idx, base_dir, wandb.run, model_type,
                dataset=dataset)
            plot_roc(
                prefix, best_labels, best_probs, class_names,
                use_kfold, fold_idx, base_dir, wandb.run, model_type,
                dataset=dataset)
            plot_pr(
                prefix, best_labels, best_probs, class_names,
                use_kfold, fold_idx, base_dir, wandb.run, model_type,
                dataset=dataset)

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
    skip_cnn_dropout=0.2, skip_rnn_dropout=0.3,
    optimizer_type=None, scheduler_type=None, loss_type=None,
    loader_mode="lookahead",
    batch_size=2048, epochs=20, patience=10,
    wandb_project="oemc", checkpoint=True, plot_result=True,
    close_wandb=True, use_kfold=True, n_splits=5,
    start_fold=0, max_folds=5, use_wandb=True, dataset=None,
    grad_clip_norm=1.0, warmup_epochs=1,
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
    elif model_type == "skip_attseqnet":
        # Wang et al. 2025 (BSPC), Table 3 — best-effort reimplementation,
        # no public code available (see skip_attseqnet.py docstring).
        # input_size here is used only to validate that timesteps is
        # long enough to survive 4 VALID-padding conv layers (kernel=3
        # each removes 2 steps -> 8 total; timesteps must be > 8).
        model_params = {
            "input_size":   timesteps,
            "output_size":  len(class_names),
            "features":     X.shape[1],
            "conv_filters": (32, 16, 8, 8),
            "kernel_size":  kernel_size,
            "cnn_dropout":  skip_cnn_dropout,
            "rnn_hidden":   16,
            "rnn_layers":   2,
            "rnn_dropout":  skip_rnn_dropout,
        }

    if use_wandb and wandb.run is not None:
        wandb_config = {"model_type": model_type, "epochs": epochs,
                        "batch_size": batch_size, "timesteps": timesteps,
                        "patience": patience, "optimizer": optimizer_type,
                        "scheduler": scheduler_type, "loss": loss_type,
                        "loader_mode": loader_mode}
        wandb_config.update(model_params)
        wandb.config.update(wandb_config)
        log_config(wandb_config, run_name, use_kfold, model_type, dataset=dataset)

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
        run_name=run_name, use_wandb=use_wandb, dataset=dataset,
        grad_clip_norm=grad_clip_norm, warmup_epochs=warmup_epochs,
    )